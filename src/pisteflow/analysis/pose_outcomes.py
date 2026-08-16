"""Source-video-grouped foil pose outcome baseline."""

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
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold

from pisteflow.provenance import sha256_file, write_json

LABELS = ["class_001", "class_010", "class_100"]
RANDOM_SEED = 20260815
_KINEMATIC_SUFFIXES = (
    "_std",
    "_range",
    "_delta",
    "_visible_fraction",
    "_speed_mean",
    "_speed_max",
)


def select_kinematic_features(frame: pd.DataFrame) -> list[str]:
    """Use body-motion summaries while excluding lights, labels, and static position means."""

    selected: list[str] = []
    for column in frame:
        if column in {
            "pose_s1_tracker_payload_frame_fraction",
            "pose_s2_tracker_payload_frame_fraction",
        }:
            selected.append(column)
            continue
        if column.startswith("pair_"):
            selected.append(column)
            continue
        if column.startswith(
            ("pose_s1_center_", "pose_s1_span_", "pose_s2_center_", "pose_s2_span_")
        ):
            if column.endswith(("_std", "_range", "_delta")):
                selected.append(column)
            continue
        if not column.startswith(("pose_s1_j", "pose_s2_j")):
            continue
        try:
            joint = int(column.split("_j", 1)[1][:2])
        except (IndexError, ValueError):
            continue
        if 2 <= joint <= 15 and column.endswith(_KINEMATIC_SUFFIXES):
            selected.append(column)
    return selected


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def _metric_block(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro")),
        "log_loss": float(log_loss(truth, probabilities, labels=LABELS)),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
    }


def evaluate_grouped(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[dict[str, Any], HistGradientBoostingClassifier, list[str]]:
    required = {"group_id", "outcome_class", "light_1_code", "light_2_code"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing foil-pose fields: {sorted(missing)}")
    observed = set(frame["outcome_class"].astype(str))
    if observed != set(LABELS):
        raise ValueError(f"Unexpected foil-pose outcome classes: {sorted(observed)}")
    features = select_kinematic_features(frame)
    if not features:
        raise ValueError("No leakage-safe pose features were selected")
    forbidden = {"light_1_code", "light_2_code", "outcome_code", "outcome_class"}
    if forbidden.intersection(features):
        raise ValueError("Scoring-light or target columns entered the feature matrix")

    matrix = frame[features].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    truth = frame["outcome_class"].astype(str).to_numpy()
    groups = frame["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.empty(len(frame), dtype=object)
    probabilities = np.zeros((len(frame), len(LABELS)), dtype=float)
    baseline_predictions = np.empty(len(frame), dtype=object)
    baseline_probabilities = np.zeros((len(frame), len(LABELS)), dtype=float)
    folds: list[dict[str, Any]] = []

    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(matrix, truth, groups), start=1
    ):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise ValueError(f"Source-video leakage in fold {fold_number}")
        model = _model()
        model.fit(matrix.iloc[train_index], truth[train_index])
        fold_probabilities = model.predict_proba(matrix.iloc[test_index])
        class_index = {str(label): index for index, label in enumerate(model.classes_)}
        for label_index, label in enumerate(LABELS):
            probabilities[test_index, label_index] = fold_probabilities[:, class_index[label]]
        predictions[test_index] = np.asarray(LABELS)[
            np.argmax(probabilities[test_index], axis=1)
        ]

        train_counts = pd.Series(truth[train_index]).value_counts()
        majority_label = str(train_counts.idxmax())
        empirical = np.asarray(
            [float(train_counts.get(label, 0)) / len(train_index) for label in LABELS]
        )
        baseline_predictions[test_index] = majority_label
        baseline_probabilities[test_index] = empirical
        folds.append(
            {
                "fold": fold_number,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "train_source_video_groups": len(train_groups),
                "test_source_video_groups": len(test_groups),
                "group_overlap": 0,
                "test_class_counts": dict(
                    sorted(pd.Series(truth[test_index]).value_counts().astype(int).to_dict().items())
                ),
            }
        )

    final_model = _model()
    final_model.fit(matrix, truth)
    report = {
        "claim_boundary": (
            "Five-fold source-video-grouped internal baseline over 9,328 real 75-frame foil "
            "pose clips. Scoring-light fields, target fields, and static joint-position means "
            "are excluded. Athlete identities are unavailable, so this is not athlete-held-out "
            "validation, a referee decision system, or evidence of deployment. Outcome class "
            "codes remain neutral because the archive does not document left/right semantics."
        ),
        "evaluation_design": "5-fold StratifiedGroupKFold by hashed source-video group",
        "random_seed": RANDOM_SEED,
        "rows": len(frame),
        "source_video_groups": int(frame["group_id"].nunique()),
        "feature_count": len(features),
        "feature_policy": (
            "BODY_25 neck-through-ankle motion/visibility summaries, tracker coverage, "
            "body-center dynamics, and pair dynamics; no lights, labels, or static joint means"
        ),
        "class_counts": dict(
            sorted(frame["outcome_class"].astype(str).value_counts().astype(int).to_dict().items())
        ),
        "group_overlap_all_folds": 0,
        "model": _metric_block(truth, predictions.astype(str), probabilities),
        "majority_empirical_probability_baseline": _metric_block(
            truth,
            baseline_predictions.astype(str),
            baseline_probabilities,
        ),
        "folds": folds,
    }
    return report, final_model, features


def train_and_report(
    input_path: Path,
    model_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, model, features = evaluate_grouped(frame)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "features": features, "labels": LABELS}, handle)
    report.update(
        {
            "input_sha256": sha256_file(input_path),
            "model_sha256": sha256_file(model_path),
            "model_artifact_policy": "local-only; upstream archive license is unknown",
        }
    )
    write_json(report_path, report)
    return report
