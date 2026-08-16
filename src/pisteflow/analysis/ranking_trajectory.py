"""Chronological next-snapshot ranking trajectory baseline."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pisteflow.provenance import sha256_file, write_json

RANDOM_SEED = 20260815
DEFAULT_CUTOFF = pd.Timestamp("2020-01-01")
MAX_HORIZON_DAYS = 180
_POINT_COLUMNS = ["rolling_points", "domestic_points", "international_points"]


def _signed_log(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def build_transitions(
    frame: pd.DataFrame,
    *,
    max_horizon_days: int = MAX_HORIZON_DAYS,
) -> tuple[pd.DataFrame, int]:
    required = {
        "athlete_series_id",
        "snapshot_date",
        "rank",
        "weapon",
        "gender",
        "age_category",
        "training_eligible",
        *_POINT_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing ranking fields: {sorted(missing)}")
    eligible = frame.loc[frame["training_eligible"].astype(bool)].copy()
    eligible["snapshot_date"] = pd.to_datetime(eligible["snapshot_date"], errors="raise")
    eligible = eligible.sort_values(
        ["athlete_series_id", "snapshot_date"], kind="stable"
    ).reset_index(drop=True)
    if eligible.duplicated(["athlete_series_id", "snapshot_date"]).any():
        raise ValueError("Duplicate ranking series-date keys remain after eligibility filtering")
    grouped = eligible.groupby("athlete_series_id", sort=False)
    eligible["target_date"] = grouped["snapshot_date"].shift(-1)
    eligible["target_rank"] = grouped["rank"].shift(-1)
    eligible["horizon_days"] = (eligible["target_date"] - eligible["snapshot_date"]).dt.days
    eligible["previous_date"] = grouped["snapshot_date"].shift(1)
    eligible["days_since_previous"] = (
        eligible["snapshot_date"] - eligible["previous_date"]
    ).dt.days
    eligible["rank_delta_from_previous"] = eligible["rank"].astype(float).sub(
        grouped["rank"].shift(1).astype(float)
    )
    for column in _POINT_COLUMNS:
        eligible[f"{column}_delta_from_previous"] = eligible[column].astype(float).sub(
            grouped[column].shift(1).astype(float)
        )
        eligible[f"signed_log_{column}"] = _signed_log(eligible[column])
    eligible["log_current_rank"] = np.log1p(eligible["rank"].astype(float))
    eligible["snapshot_year"] = eligible["snapshot_date"].dt.year.astype(float)
    month_angle = 2 * np.pi * (eligible["snapshot_date"].dt.month.astype(float) - 1) / 12
    eligible["snapshot_month_sin"] = np.sin(month_angle)
    eligible["snapshot_month_cos"] = np.cos(month_angle)
    eligible["has_previous_snapshot"] = eligible["previous_date"].notna().astype(float)

    all_transitions = int(eligible["target_rank"].notna().sum())
    transitions = eligible.loc[
        eligible["target_rank"].notna()
        & eligible["horizon_days"].between(1, max_horizon_days, inclusive="both")
    ].copy()
    return transitions.reset_index(drop=True), all_transitions


def feature_matrix(transitions: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "log_current_rank",
        "signed_log_rolling_points",
        "signed_log_domestic_points",
        "signed_log_international_points",
        "rank_delta_from_previous",
        "rolling_points_delta_from_previous",
        "domestic_points_delta_from_previous",
        "international_points_delta_from_previous",
        "days_since_previous",
        "snapshot_year",
        "snapshot_month_sin",
        "snapshot_month_cos",
        "has_previous_snapshot",
    ]
    numeric = transitions[numeric_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(0.0)
    categories = pd.get_dummies(
        transitions[["weapon", "gender", "age_category"]].astype(str),
        prefix=["weapon", "gender", "age"],
        dtype=float,
    )
    return pd.concat([numeric.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)


def _metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    correlation = spearmanr(truth, predicted).statistic
    return {
        "mae_rank_places": float(mean_absolute_error(truth, predicted)),
        "rmse_rank_places": float(mean_squared_error(truth, predicted) ** 0.5),
        "r2": float(r2_score(truth, predicted)),
        "spearman_rho": float(correlation),
    }


def _model(train_rows: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=max(10, min(60, train_rows // 300)),
        l2_regularization=3.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def evaluate_trajectory(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp = DEFAULT_CUTOFF,
    max_horizon_days: int = MAX_HORIZON_DAYS,
) -> tuple[dict[str, Any], HistGradientBoostingRegressor, list[str]]:
    transitions, all_transitions = build_transitions(
        frame, max_horizon_days=max_horizon_days
    )
    matrix = feature_matrix(transitions)
    train_mask = transitions["target_date"].lt(cutoff).to_numpy()
    test_mask = transitions["target_date"].ge(cutoff).to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Chronological ranking split has an empty train or test partition")
    truth = transitions["target_rank"].astype(float).to_numpy()
    model = _model(int(train_mask.sum()))
    model.fit(matrix.loc[train_mask], np.log1p(truth[train_mask]))
    predicted = np.maximum(1.0, np.expm1(model.predict(matrix.loc[test_mask])))
    baseline = transitions.loc[test_mask, "rank"].astype(float).to_numpy()
    test_truth = truth[test_mask]
    test_rows = transitions.loc[test_mask].reset_index(drop=True)
    by_weapon = {}
    for weapon in ("epee", "foil", "sabre"):
        weapon_mask = test_rows["weapon"].astype(str).eq(weapon).to_numpy()
        by_weapon[weapon] = {
            "transitions": int(weapon_mask.sum()),
            "model": _metrics(test_truth[weapon_mask], predicted[weapon_mask]),
            "persistence_baseline": _metrics(test_truth[weapon_mask], baseline[weapon_mask]),
        }
    horizon_bins = pd.cut(
        test_rows["horizon_days"],
        bins=[0, 14, 30, 90, max_horizon_days],
        labels=["1-14", "15-30", "31-90", f"91-{max_horizon_days}"],
        include_lowest=True,
    )
    by_horizon = {}
    for label in horizon_bins.cat.categories:
        mask = horizon_bins.eq(label).to_numpy()
        if mask.any():
            by_horizon[str(label)] = {
                "transitions": int(mask.sum()),
                "model_mae_rank_places": float(
                    mean_absolute_error(test_truth[mask], predicted[mask])
                ),
                "persistence_mae_rank_places": float(
                    mean_absolute_error(test_truth[mask], baseline[mask])
                ),
            }
    report = {
        "claim_boundary": (
            "Chronological internal next-observed-snapshot baseline over a third-party USA team "
            "ranking archive. It uses only current and prior ranking fields, excludes gaps over "
            "180 days, and evaluates target dates from 2020 onward against a current-rank "
            "persistence baseline. Series can occur on both sides of the date split, category "
            "series are not verified unique people, and this is not a current official ranking "
            "feed, selection forecast, or causal athlete-development model."
        ),
        "evaluation_design": "target-date chronological holdout; train before 2020, test 2020-2023",
        "random_seed": RANDOM_SEED,
        "raw_eligible_rows": int(frame["training_eligible"].astype(bool).sum()),
        "category_specific_series": int(frame["athlete_series_id"].nunique()),
        "all_next_observation_transitions": all_transitions,
        "eligible_transitions_1_to_180_days": len(transitions),
        "excluded_long_or_invalid_horizon_transitions": all_transitions - len(transitions),
        "train_transitions": int(train_mask.sum()),
        "test_transitions": int(test_mask.sum()),
        "test_target_date_min": test_rows["target_date"].min().date().isoformat(),
        "test_target_date_max": test_rows["target_date"].max().date().isoformat(),
        "feature_count": matrix.shape[1],
        "feature_policy": (
            "current rank and points, prior-snapshot deltas, prior gap, date seasonality, "
            "weapon, gender, and age category; no future scores or athlete-series identity"
        ),
        "test": {
            "model": _metrics(test_truth, predicted),
            "persistence_baseline": _metrics(test_truth, baseline),
            "by_weapon": by_weapon,
            "by_horizon_days": by_horizon,
        },
    }
    final_model = _model(len(transitions))
    final_model.fit(matrix, np.log1p(truth))
    return report, final_model, list(matrix.columns)


def train_and_report(input_path: Path, model_path: Path, report_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, model, features = evaluate_trajectory(frame)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "features": features}, handle)
    report.update(
        {
            "input_sha256": sha256_file(input_path),
            "model_sha256": sha256_file(model_path),
            "model_artifact_policy": "local-only; upstream archive license is unknown",
        }
    )
    write_json(report_path, report)
    return report
