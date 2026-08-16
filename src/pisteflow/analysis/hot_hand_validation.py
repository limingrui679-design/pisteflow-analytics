"""Pool-held-out validation of sequential features in one junior foil event."""

from __future__ import annotations

import pickle
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pisteflow.provenance import sha256_file, write_json

RANDOM_SEED = 20260815
MODEL_FEATURES = {
    "rating_only": ["rating_diff"],
    "rating_plus_previous_outcome": ["rating_diff", "previous_outcome_diff"],
    "rating_plus_recent_form": ["rating_diff", "recent_form_diff"],
}


def _flip(bout_id: str) -> bool:
    digest = sha256(f"pisteflow-hot-hand-orientation:{bout_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 1


def _model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
            ),
        ]
    )


def prepare_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "bout_id",
        "pool_id",
        "winner_side",
        "rating_a_pre",
        "rating_b_pre",
        "prev_win_a",
        "prev_win_b",
        "recent_form_3_a",
        "recent_form_3_b",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing hot-hand analysis fields: {sorted(missing)}")
    if frame["bout_id"].duplicated().any():
        raise ValueError("hot-hand analysis input has duplicate bout IDs")
    if not set(frame["winner_side"]).issubset({"a", "b"}):
        raise ValueError("hot-hand analysis input has an unknown winner side")
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        flipped = _flip(str(row.bout_id))
        sign = -1.0 if flipped else 1.0
        target_a = 1 if row.winner_side == "a" else 0
        rows.append(
            {
                "bout_id": str(row.bout_id),
                "pool_id": str(row.pool_id),
                "orientation_flipped": flipped,
                "target_left_win": 1 - target_a if flipped else target_a,
                "rating_diff": sign * (float(row.rating_a_pre) - float(row.rating_b_pre))
                if pd.notna(row.rating_a_pre) and pd.notna(row.rating_b_pre)
                else np.nan,
                "previous_outcome_diff": sign * (float(row.prev_win_a) - float(row.prev_win_b))
                if pd.notna(row.prev_win_a) and pd.notna(row.prev_win_b)
                else np.nan,
                "recent_form_diff": sign
                * (float(row.recent_form_3_a) - float(row.recent_form_3_b))
                if pd.notna(row.recent_form_3_a) and pd.notna(row.recent_form_3_b)
                else np.nan,
            }
        )
    oriented = pd.DataFrame(rows)
    complete_features = sorted(
        {feature for features in MODEL_FEATURES.values() for feature in features}
    )
    common = oriented.dropna(subset=complete_features)
    common = common.reset_index(drop=True)
    if common["pool_id"].nunique() < 3:
        raise ValueError("hot-hand analysis needs at least three pools")
    if common["target_left_win"].nunique() != 2:
        raise ValueError("hot-hand analysis needs both outcome classes")
    return common, {
        "source_unique_bouts": len(frame),
        "source_pools": int(frame["pool_id"].nunique()),
        "common_complete_case_bouts": len(common),
        "common_complete_case_pools": int(common["pool_id"].nunique()),
        "excluded_incomplete_feature_bouts": len(frame) - len(common),
    }


def _metrics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, probabilities)),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    repetitions: int = 2000,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    pools = np.asarray(sorted(frame["pool_id"].unique()), dtype=object)
    pool_indexes = {
        pool: np.flatnonzero(frame["pool_id"].to_numpy() == pool) for pool in pools
    }
    auc_deltas: list[float] = []
    log_loss_deltas: list[float] = []
    truth = frame["target_left_win"].astype(int).to_numpy()
    for _ in range(repetitions):
        sampled = rng.choice(pools, size=len(pools), replace=True)
        indexes = np.concatenate([pool_indexes[pool] for pool in sampled])
        sampled_truth = truth[indexes]
        if len(np.unique(sampled_truth)) < 2:
            continue
        auc_deltas.append(
            float(
                roc_auc_score(sampled_truth, candidate[indexes])
                - roc_auc_score(sampled_truth, baseline[indexes])
            )
        )
        log_loss_deltas.append(
            float(
                log_loss(sampled_truth, candidate[indexes], labels=[0, 1])
                - log_loss(sampled_truth, baseline[indexes], labels=[0, 1])
            )
        )
    if not auc_deltas:
        raise ValueError("cluster bootstrap produced no valid samples")
    return {
        "unit": "pool",
        "requested_repetitions": repetitions,
        "valid_repetitions": len(auc_deltas),
        "auc_delta_candidate_minus_rating_only": {
            "point": float(
                roc_auc_score(truth, candidate) - roc_auc_score(truth, baseline)
            ),
            "percentile_95_interval": [
                float(np.percentile(auc_deltas, 2.5)),
                float(np.percentile(auc_deltas, 97.5)),
            ],
        },
        "log_loss_delta_candidate_minus_rating_only": {
            "point": float(
                log_loss(truth, candidate, labels=[0, 1])
                - log_loss(truth, baseline, labels=[0, 1])
            ),
            "percentile_95_interval": [
                float(np.percentile(log_loss_deltas, 2.5)),
                float(np.percentile(log_loss_deltas, 97.5)),
            ],
        },
    }


def evaluate_grouped(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Pipeline]]:
    analysis, audit = prepare_analysis(frame)
    truth = analysis["target_left_win"].astype(int).to_numpy()
    groups = analysis["pool_id"].astype(str).to_numpy()
    predictions = {
        name: np.full(len(analysis), np.nan, dtype=float) for name in MODEL_FEATURES
    }
    fold_rows: list[dict[str, Any]] = []
    for fold, held_out_pool in enumerate(sorted(set(groups)), start=1):
        test_index = np.flatnonzero(groups == held_out_pool)
        train_index = np.flatnonzero(groups != held_out_pool)
        if set(groups[train_index]) & set(groups[test_index]):
            raise ValueError(f"pool leakage in fold {fold}")
        if len(np.unique(truth[train_index])) != 2:
            raise ValueError(f"training fold {fold} lacks both outcome classes")
        for name, features in MODEL_FEATURES.items():
            model = _model()
            model.fit(analysis.iloc[train_index][features], truth[train_index])
            predictions[name][test_index] = model.predict_proba(
                analysis.iloc[test_index][features]
            )[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "held_out_pool": held_out_pool,
                "train_pools": len(set(groups[train_index])),
                "test_pools": 1,
                "pool_overlap": 0,
                "train_bouts": len(train_index),
                "test_bouts": len(test_index),
            }
        )
    if any(np.isnan(values).any() for values in predictions.values()):
        raise ValueError("hot-hand grouped evaluation left missing predictions")

    fitted: dict[str, Pipeline] = {}
    model_reports: dict[str, Any] = {}
    for name, features in MODEL_FEATURES.items():
        model = _model()
        model.fit(analysis[features], truth)
        fitted[name] = model
        classifier = model.named_steps["classifier"]
        model_reports[name] = {
            "features": features,
            "out_of_pool_metrics": _metrics(truth, predictions[name]),
            "final_all_sample_standardized_coefficients": {
                feature: float(coefficient)
                for feature, coefficient in zip(features, classifier.coef_[0], strict=True)
            },
        }

    baseline = predictions["rating_only"]
    comparisons = {
        name: _cluster_bootstrap(analysis, baseline, predictions[name])
        for name in ("rating_plus_previous_outcome", "rating_plus_recent_form")
    }
    oof = analysis[["bout_id", "pool_id", "orientation_flipped", "target_left_win"]].copy()
    for name, values in predictions.items():
        oof[f"probability_{name}"] = values
    report: dict[str, Any] = {
        "claim_boundary": (
            f"Leave-one-pool-out internal validation on {len(analysis)} complete-case bouts from "
            f"{analysis['pool_id'].nunique()} pools with complete features, drawn from "
            f"{audit['source_pools']} source pools in one 2026 February NAC Junior Men's Foil "
            "event. Every actual bout appears once; "
            "presentation sides are deterministically flipped to prevent pool-grid position from "
            "becoming a predictor. All models use the identical complete-case sample, and "
            "uncertainty resamples whole pools. This analysis tests incremental prediction over "
            "pre-event rating within one event; it is not causal evidence of or against a hot "
            "hand, an athlete-independent external validation, or a coaching prescription."
        ),
        "evaluation_design": (
            "leave-one-pool-out cross-validation with a common complete-case sample"
        ),
        "random_seed": RANDOM_SEED,
        **audit,
        "pools": int(analysis["pool_id"].nunique()),
        "orientation_flipped_share": float(analysis["orientation_flipped"].mean()),
        "pool_overlap_all_folds": 0,
        "models": model_reports,
        "incremental_comparisons": comparisons,
        "folds": fold_rows,
        "interpretation": (
            "In this internal sample, adding previous-bout outcome or three-bout recent form does "
            "not improve held-out-pool AUC or log loss over pre-event rating alone. The conclusion "
            "is deliberately limited to predictive lift in this one event."
        ),
    }
    return report, oof, fitted


def train_and_report(
    input_path: Path,
    oof_path: Path,
    model_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, oof, models = evaluate_grouped(frame)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"models": models, "features": MODEL_FEATURES}, handle)
    report.update(
        {
            "input_sha256": sha256_file(input_path),
            "oof_sha256": sha256_file(oof_path),
            "model_sha256": sha256_file(model_path),
            "model_artifact_policy": "local-only; the pinned event snapshot has no data license",
        }
    )
    write_json(report_path, report)
    return report
