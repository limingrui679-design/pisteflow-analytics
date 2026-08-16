"""Nested-CV strength and lunge-velocity association baseline."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pisteflow.provenance import sha256_file, write_json

RANDOM_SEED = 20260815
TARGETS = ["static_lunge_velocity_m_s", "advance_lunge_velocity_m_s"]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
_NON_FEATURES = {
    "participant_code",
    "source_doi",
    "population",
    *TARGETS,
}


def _feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        column
        for column in frame
        if column not in _NON_FEATURES
        and column != "weapon"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return numeric, ["weapon"]


def _pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "weapon",
                OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge())])


def _metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(mean_squared_error(truth, predicted) ** 0.5),
        "r2": float(r2_score(truth, predicted)),
    }


def _top_correlations(frame: pd.DataFrame, numeric: list[str], target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in numeric:
        result = spearmanr(frame[feature], frame[target], nan_policy="omit")
        coefficient = float(result.statistic)
        p_value = float(result.pvalue)
        if not np.isfinite(coefficient):
            continue
        rows.append(
            {
                "feature": feature,
                "spearman_rho": coefficient,
                "unadjusted_p_value": p_value,
            }
        )
    return sorted(rows, key=lambda row: (-abs(row["spearman_rho"]), row["feature"]))[:8]


def evaluate_strength_velocity(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[dict[str, Any], dict[str, Pipeline], list[str]]:
    required = {"participant_code", "weapon", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing strength fields: {sorted(missing)}")
    if frame["participant_code"].duplicated().any():
        raise ValueError("Strength table must contain one row per participant")
    if set(frame["weapon"].astype(str)) != {"epee", "foil", "sabre"}:
        raise ValueError("Strength table must contain all three weapons")
    if len(frame) < n_splits * 3:
        raise ValueError("Strength table is too small for the requested outer folds")
    numeric, categorical = _feature_columns(frame)
    if not numeric:
        raise ValueError("No numeric strength features found")
    model_frame = frame[numeric + categorical]
    stratifier = frame["weapon"].astype(str).to_numpy()
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    target_reports: dict[str, Any] = {}
    final_models: dict[str, Pipeline] = {}

    for target in TARGETS:
        truth = frame[target].astype(float).to_numpy()
        predictions = np.zeros(len(frame), dtype=float)
        baseline = np.zeros(len(frame), dtype=float)
        folds: list[dict[str, Any]] = []
        for fold_number, (train_index, test_index) in enumerate(
            outer.split(model_frame, stratifier), start=1
        ):
            inner_splits = min(4, max(2, len(train_index) // 8))
            search = GridSearchCV(
                _pipeline(numeric, categorical),
                {"ridge__alpha": ALPHAS},
                cv=KFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED),
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            search.fit(model_frame.iloc[train_index], truth[train_index])
            predictions[test_index] = search.predict(model_frame.iloc[test_index])
            baseline[test_index] = float(np.mean(truth[train_index]))
            folds.append(
                {
                    "fold": fold_number,
                    "train_participants": len(train_index),
                    "test_participants": len(test_index),
                    "selected_alpha": float(search.best_params_["ridge__alpha"]),
                    "test_by_weapon": dict(
                        sorted(
                            frame.iloc[test_index]["weapon"]
                            .astype(str)
                            .value_counts()
                            .astype(int)
                            .to_dict()
                            .items()
                        )
                    ),
                }
            )
        final_search = GridSearchCV(
            _pipeline(numeric, categorical),
            {"ridge__alpha": ALPHAS},
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED),
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        final_search.fit(model_frame, truth)
        final_models[target] = final_search.best_estimator_
        by_weapon = {
            weapon: {
                "participants": int(mask.sum()),
                "mae": float(mean_absolute_error(truth[mask], predictions[mask])),
                "rmse": float(mean_squared_error(truth[mask], predictions[mask]) ** 0.5),
            }
            for weapon in ("epee", "foil", "sabre")
            if (mask := stratifier == weapon).any()
        }
        target_reports[target] = {
            "nested_cross_validation": _metrics(truth, predictions),
            "training_mean_baseline": _metrics(truth, baseline),
            "final_selected_alpha": float(final_search.best_params_["ridge__alpha"]),
            "by_weapon_oof_error": by_weapon,
            "top_descriptive_spearman_correlations": _top_correlations(frame, numeric, target),
            "folds": folds,
        }

    report = {
        "claim_boundary": (
            "Nested five-fold participant-level internal analysis of 45 elite female Chinese "
            "fencers measured once across epee, foil, and sabre. It estimates cross-sectional "
            "associations between contemporaneous strength or jump tests and lunge velocity. "
            "It is not a longitudinal training intervention, causal effect, population norm, "
            "or externally validated athlete prescription. Spearman p-values are descriptive "
            "and unadjusted for multiple comparisons."
        ),
        "evaluation_design": "outer weapon-stratified 5-fold CV with inner alpha selection",
        "random_seed": RANDOM_SEED,
        "participants": len(frame),
        "by_weapon": dict(
            sorted(frame["weapon"].astype(str).value_counts().astype(int).to_dict().items())
        ),
        "numeric_feature_count": len(numeric),
        "feature_policy": (
            "age, body size, training years, jump, IMTP, and strength measures plus weapon; "
            "both lunge-velocity outcomes are excluded from predictors"
        ),
        "targets": target_reports,
    }
    return report, final_models, numeric


def train_and_report(input_path: Path, model_path: Path, report_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, models, numeric = evaluate_strength_velocity(frame)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"models": models, "numeric_features": numeric, "targets": TARGETS}, handle)
    report.update(
        {
            "input_sha256": sha256_file(input_path),
            "model_sha256": sha256_file(model_path),
            "model_artifact_policy": "local-only baseline artifact",
        }
    )
    write_json(report_path, report)
    return report
