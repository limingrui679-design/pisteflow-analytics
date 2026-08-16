"""Chronological, identity-free outcome baseline for the OSF junior pool corpus."""

from __future__ import annotations

import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pisteflow.provenance import sha256_file, write_json

VALIDATION_YEAR = 2024
TEST_START_YEAR = 2025
C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]
NUMERIC_FEATURES = ["signed_log_seed_ratio"]
CATEGORICAL_FEATURES = ["weapon", "gender"]


def _flip_bout(competition_key: str, source_row: int) -> bool:
    identity = f"{competition_key}:{source_row}"
    digest = sha256(f"pisteflow-osf-pool-orientation:{identity}".encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 1


def prepare_bouts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "source_row",
        "year",
        "competition_key",
        "athlete_a_key",
        "athlete_b_key",
        "A_seed",
        "B_seed",
        "weapon",
        "gender",
        "outcome_eligible",
        "athlete_a_won",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OSF pool fields: {sorted(missing)}")
    source_rows = len(frame)
    eligible = frame.loc[frame["outcome_eligible"].fillna(False)].copy()
    eligible = eligible.loc[
        eligible["athlete_a_key"].notna()
        & eligible["athlete_b_key"].notna()
        & eligible["athlete_a_key"].ne(eligible["athlete_b_key"])
        & eligible["A_seed"].notna()
        & eligible["B_seed"].notna()
        & eligible["athlete_a_won"].notna()
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible OSF pool bouts")
    if eligible[["competition_key", "source_row"]].duplicated().any():
        raise ValueError("OSF pool bout keys are not unique")

    oriented_rows: list[dict[str, Any]] = []
    for row in eligible.itertuples(index=False):
        flip = _flip_bout(str(row.competition_key), int(row.source_row))
        left_seed = int(row.B_seed if flip else row.A_seed)
        right_seed = int(row.A_seed if flip else row.B_seed)
        athlete_a_won = bool(row.athlete_a_won)
        oriented_rows.append(
            {
                "source_row": int(row.source_row),
                "year": int(row.year),
                "competition_key": str(row.competition_key),
                "weapon": str(row.weapon),
                "gender": str(row.gender),
                "signed_log_seed_ratio": float(
                    np.log1p(right_seed) - np.log1p(left_seed)
                ),
                "target_left_win": int(not athlete_a_won if flip else athlete_a_won),
                "orientation_flipped": flip,
                "same_seed": left_seed == right_seed,
            }
        )
    oriented = pd.DataFrame(oriented_rows).sort_values(
        ["year", "competition_key", "source_row"], kind="stable"
    )
    audit = {
        "source_rows": source_rows,
        "eligible_decisive_bouts": len(oriented),
        "excluded_rows": source_rows - len(oriented),
        "source_orientation_flipped_share": float(oriented["orientation_flipped"].mean()),
        "same_seed_bouts": int(oriented["same_seed"].sum()),
    }
    return oriented.reset_index(drop=True), audit


def _pipeline(c_value: float) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("seed", StandardScaler(), NUMERIC_FEATURES),
            (
                "context",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "logistic",
                LogisticRegression(C=c_value, max_iter=2_000, random_state=20260816),
            ),
        ]
    )


def _metrics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= 0.5
    result = {
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "accuracy": float(accuracy_score(truth, predicted)),
    }
    if len(np.unique(truth)) == 2:
        result["roc_auc"] = float(roc_auc_score(truth, probabilities))
    return result


def _evaluate_rows(rows: pd.DataFrame, probability_column: str) -> dict[str, Any]:
    truth = rows["target_left_win"].astype(int).to_numpy()
    probabilities = rows[probability_column].astype(float).to_numpy()
    return {"bouts": len(rows), **_metrics(truth, probabilities)}


def _higher_seed_pick_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the deterministic lower-seed-number pick without fake probabilities."""

    comparable = rows.loc[rows["signed_log_seed_ratio"].ne(0)].copy()
    truth = comparable["target_left_win"].astype(int).to_numpy()
    score = comparable["signed_log_seed_ratio"].astype(float).to_numpy()
    result: dict[str, Any] = {
        "bouts": len(comparable),
        "excluded_same_seed_bouts": len(rows) - len(comparable),
        "accuracy": float(accuracy_score(truth, score > 0)),
    }
    if len(np.unique(truth)) == 2:
        result["roc_auc_from_signed_seed_difference"] = float(
            roc_auc_score(truth, score)
        )
    return result


def evaluate_pool_outcomes(
    frame: pd.DataFrame,
    *,
    validation_year: int = VALIDATION_YEAR,
    test_start_year: int = TEST_START_YEAR,
    c_candidates: list[float] = C_CANDIDATES,
) -> tuple[dict[str, Any], Pipeline]:
    bouts, audit = prepare_bouts(frame)
    train = bouts.loc[bouts["year"].lt(validation_year)].copy()
    validation = bouts.loc[bouts["year"].eq(validation_year)].copy()
    test = bouts.loc[bouts["year"].ge(test_start_year)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("OSF pool data do not span the requested train/validation/test years")
    split_keys = {
        "train": set(train["competition_key"]),
        "validation": set(validation["competition_key"]),
        "test": set(test["competition_key"]),
    }
    if (
        split_keys["train"] & split_keys["validation"]
        or split_keys["train"] & split_keys["test"]
        or split_keys["validation"] & split_keys["test"]
    ):
        raise ValueError("Competition keys cross chronological split boundaries")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    candidates: list[dict[str, float]] = []
    for c_value in c_candidates:
        model = _pipeline(c_value)
        model.fit(train[features], train["target_left_win"].astype(int))
        probabilities = model.predict_proba(validation[features])[:, 1]
        metrics = _metrics(validation["target_left_win"].astype(int).to_numpy(), probabilities)
        candidates.append(
            {
                "c": c_value,
                "validation_log_loss": metrics["log_loss"],
                "validation_brier_score": metrics["brier_score"],
            }
        )
    selected = min(candidates, key=lambda row: (row["validation_log_loss"], row["c"]))
    selected_c = float(selected["c"])
    fit_rows = pd.concat([train, validation], ignore_index=True)
    model = _pipeline(selected_c)
    model.fit(fit_rows[features], fit_rows["target_left_win"].astype(int))
    test["model_probability"] = model.predict_proba(test[features])[:, 1]
    test["uninformative_probability"] = 0.5

    by_weapon = {
        weapon: _evaluate_rows(rows, "model_probability")
        for weapon, rows in test.groupby("weapon", sort=True)
    }
    by_year = {
        str(year): _evaluate_rows(rows, "model_probability")
        for year, rows in test.groupby("year", sort=True)
    }
    train_athletes = set(frame.loc[frame["year"].lt(test_start_year), "athlete_a_key"]) | set(
        frame.loc[frame["year"].lt(test_start_year), "athlete_b_key"]
    )
    test_athletes = set(frame.loc[frame["year"].ge(test_start_year), "athlete_a_key"]) | set(
        frame.loc[frame["year"].ge(test_start_year), "athlete_b_key"]
    )
    report: dict[str, Any] = {
        "claim_boundary": (
            "Identity-free chronological baseline over a third-party transcription of junior "
            "pool bouts across epee, foil, and sabre. It uses only pre-event seed, weapon, and "
            "gender; athlete keys, scores, pool positions, names, nationality, and future rows "
            "are excluded from predictors. This evaluates association and held-out historical "
            "prediction, not causal coaching impact, senior or international generalization, "
            "or official FIE data quality."
        ),
        "evaluation_design": (
            f"train before {validation_year}; select C on {validation_year}; untouched test "
            f"from {test_start_year}; competition keys isolated across splits"
        ),
        "feature_policy": "signed log seed ratio + weapon + gender; no athlete identity or score",
        "selected_c": selected_c,
        "candidate_selection": candidates,
        **audit,
        "train_bouts": len(train),
        "validation_bouts": len(validation),
        "test_bouts": len(test),
        "train_years": [int(train["year"].min()), int(train["year"].max())],
        "test_years": [int(test["year"].min()), int(test["year"].max())],
        "split_competition_counts": {name: len(keys) for name, keys in split_keys.items()},
        "competition_key_overlap": 0,
        "athlete_key_overlap_is_reported_not_used": len(train_athletes & test_athletes),
        "test": {
            "model": _evaluate_rows(test, "model_probability"),
            "uninformative_baseline": _evaluate_rows(test, "uninformative_probability"),
            "higher_seed_pick_nonprobabilistic": _higher_seed_pick_metrics(test),
            "by_weapon": by_weapon,
            "by_year": by_year,
        },
    }
    return report, model


def train_and_report(input_path: Path, model_path: Path, report_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, model = evaluate_pool_outcomes(frame)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "selected_c": report["selected_c"],
            },
            handle,
        )
    report.update(
        {
            "input_sha256": sha256_file(input_path),
            "model_sha256": sha256_file(model_path),
            "model_artifact_policy": "local-only; upstream row-level results are local-only",
        }
    )
    write_json(report_path, report)
    return report
