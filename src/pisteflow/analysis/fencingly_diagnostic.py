"""Transformation-grouped label diagnostic for deduplicated FENCINGLY poses."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from pisteflow.provenance import sha256_file, write_json

LABELS = ["left", "right"]
RANDOM_SEED = 20260815


def select_dynamic_features(frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for column in frame:
        if not column.startswith(("pose_", "pair_")):
            continue
        if column.endswith(
            (
                "_std",
                "_range",
                "_delta",
                "_fraction",
                "_speed_mean",
                "_speed_p90",
                "_speed_max",
            )
        ):
            selected.append(column)
    return selected


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def _metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    right_probability: np.ndarray,
) -> dict[str, Any]:
    binary_truth = (truth == "right").astype(int)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro")),
        "roc_auc_right": float(roc_auc_score(binary_truth, right_probability)),
        "log_loss": float(
            log_loss(
                truth,
                np.column_stack([1.0 - right_probability, right_probability]),
                labels=LABELS,
            )
        ),
        "brier_score_right": float(brier_score_loss(binary_truth, right_probability)),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
    }


def evaluate_grouped(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame, HistGradientBoostingClassifier, list[str]]:
    required = {"clip_id", "geometry_group_id", "upstream_side_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing FENCINGLY diagnostic fields: {sorted(missing)}")
    if set(frame["upstream_side_label"].astype(str)) != set(LABELS):
        raise ValueError("FENCINGLY side-label vocabulary changed")
    if frame["clip_id"].duplicated().any():
        raise ValueError("FENCINGLY diagnostic input contains duplicate exact sequences")
    features = select_dynamic_features(frame)
    if not features:
        raise ValueError("No dynamic FENCINGLY pose features were selected")

    matrix = frame[features].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    truth = frame["upstream_side_label"].astype(str).to_numpy()
    groups = frame["geometry_group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.empty(len(frame), dtype=object)
    right_probability = np.zeros(len(frame), dtype=float)
    baseline_predictions = np.empty(len(frame), dtype=object)
    baseline_right_probability = np.zeros(len(frame), dtype=float)
    folds: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(matrix, truth, groups), start=1
    ):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise ValueError(f"FENCINGLY geometry-group leakage in fold {fold}")
        model = _model()
        model.fit(matrix.iloc[train_index], truth[train_index])
        probability = model.predict_proba(matrix.iloc[test_index])
        class_index = {str(label): index for index, label in enumerate(model.classes_)}
        right_probability[test_index] = probability[:, class_index["right"]]
        predictions[test_index] = np.where(
            right_probability[test_index] >= 0.5, "right", "left"
        )

        counts = pd.Series(truth[train_index]).value_counts()
        right_share = float(counts.get("right", 0)) / len(train_index)
        baseline_right_probability[test_index] = right_share
        baseline_predictions[test_index] = "right" if right_share >= 0.5 else "left"
        folds.append(
            {
                "fold": fold,
                "train_sequences": len(train_index),
                "test_sequences": len(test_index),
                "train_geometry_groups": len(train_groups),
                "test_geometry_groups": len(test_groups),
                "group_overlap": 0,
                "test_label_counts": dict(
                    sorted(
                        pd.Series(truth[test_index]).value_counts().astype(int).to_dict().items()
                    )
                ),
            }
        )

    final_model = _model()
    final_model.fit(matrix, truth)
    oof = frame[["clip_id", "geometry_group_id", "upstream_side_label"]].copy()
    oof["predicted_side_label"] = predictions.astype(str)
    oof["probability_right"] = right_probability
    report: dict[str, Any] = {
        "claim_boundary": (
            "Internal five-fold diagnostic over 28,727 exact-unique pose sequences. All 415 "
            "sequences flagged as possible translated, scaled, or mirrored equivalents are kept "
            "with their geometry group, and static center/size means are excluded. The upstream "
            "left/right label meaning is undocumented, and original video, event, athlete, weapon, "
            "frame-rate, and source-video grouping are absent. Therefore this is label-"
            "predictability evidence only: it is not a referee model, an athlete-held-out or "
            "source-video-held-out benchmark, a fencing-side semantic validation, or deployment "
            "evidence."
        ),
        "evaluation_design": (
            "5-fold StratifiedGroupKFold by conservative transformation-invariant geometry hash"
        ),
        "random_seed": RANDOM_SEED,
        "exact_unique_sequences": len(frame),
        "geometry_groups": int(frame["geometry_group_id"].nunique()),
        "group_overlap_all_folds": 0,
        "dynamic_feature_count": len(features),
        "feature_policy": (
            "pose/pair visibility, motion, dispersion, range, and temporal delta; no static "
            "center or body-size means"
        ),
        "label_counts": dict(
            sorted(
                frame["upstream_side_label"].astype(str).value_counts().astype(int).to_dict().items()
            )
        ),
        "model": _metrics(truth, predictions.astype(str), right_probability),
        "empirical_probability_baseline": _metrics(
            truth,
            baseline_predictions.astype(str),
            baseline_right_probability,
        ),
        "folds": folds,
    }
    return report, oof, final_model, features


def train_and_report(
    input_path: Path,
    oof_path: Path,
    model_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, oof, model, features = evaluate_grouped(frame)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "features": features, "labels": LABELS}, handle)
    report.update(
        {
            "input_sha256": sha256_file(input_path),
            "oof_sha256": sha256_file(oof_path),
            "model_sha256": sha256_file(model_path),
            "model_artifact_policy": "local-only; upstream pose provenance is undocumented",
        }
    )
    write_json(report_path, report)
    return report
