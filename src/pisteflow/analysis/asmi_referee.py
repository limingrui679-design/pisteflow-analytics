"""Filename-grouped internal baseline for the Asmi three-label clip snapshot."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pisteflow.provenance import sha256_file, write_json

LABELS = ["left", "right", "simultaneous"]
RANDOM_SEED = 20260815


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if column.startswith(("frame_", "flow_"))]


def _model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _metrics(
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
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline, list[str]]:
    required = {"clip_id", "source_group_id", "upstream_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Asmi referee feature fields: {sorted(missing)}")
    if set(frame["upstream_label"].astype(str)) != set(LABELS):
        raise ValueError("Asmi referee label vocabulary changed")
    if frame["clip_id"].duplicated().any():
        raise ValueError("Asmi referee feature rows contain duplicate clips")
    features = _feature_columns(frame)
    if not features:
        raise ValueError("No Asmi referee video features were found")

    matrix = frame[features].astype("float64").replace([np.inf, -np.inf], np.nan)
    truth = frame["upstream_label"].astype(str).to_numpy()
    groups = frame["source_group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.empty(len(frame), dtype=object)
    probabilities = np.zeros((len(frame), len(LABELS)), dtype=float)
    baseline_predictions = np.empty(len(frame), dtype=object)
    baseline_probabilities = np.zeros((len(frame), len(LABELS)), dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(matrix, truth, groups), start=1
    ):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise ValueError(f"Asmi referee filename-group leakage in fold {fold}")
        model = _model()
        model.fit(matrix.iloc[train_index], truth[train_index])
        fold_probability = model.predict_proba(matrix.iloc[test_index])
        class_index = {str(label): index for index, label in enumerate(model.classes_)}
        for label_index, label in enumerate(LABELS):
            probabilities[test_index, label_index] = fold_probability[:, class_index[label]]
        predictions[test_index] = np.asarray(LABELS)[
            np.argmax(probabilities[test_index], axis=1)
        ]

        counts = pd.Series(truth[train_index]).value_counts()
        majority = str(counts.idxmax())
        empirical = np.asarray(
            [float(counts.get(label, 0)) / len(train_index) for label in LABELS]
        )
        baseline_predictions[test_index] = majority
        baseline_probabilities[test_index] = empirical
        fold_rows.append(
            {
                "fold": fold,
                "train_clips": len(train_index),
                "test_clips": len(test_index),
                "train_filename_source_groups": len(train_groups),
                "test_filename_source_groups": len(test_groups),
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
    oof = frame[["clip_id", "source_group_id", "upstream_label"]].copy()
    oof["predicted_label"] = predictions.astype(str)
    for index, label in enumerate(LABELS):
        oof[f"probability_{label}"] = probabilities[:, index]

    report: dict[str, Any] = {
        "claim_boundary": (
            "Five-fold internal evaluation over all 395 exact-unique clips, keeping every clip "
            "with the same filename source number in one fold. These 55 numbers are the strongest "
            "available grouping signal but are not documented original-video IDs. Labels are "
            "unverified upstream left/right/simultaneous folders; clip-level weapon, athletes, "
            "events, source URLs, and rater data are absent. Therefore the score measures internal "
            "label predictability only, not referee correctness, unseen-event generalization, "
            "human agreement, deployment, or a validated saber system."
        ),
        "evaluation_design": "5-fold StratifiedGroupKFold by filename source number",
        "random_seed": RANDOM_SEED,
        "clips": len(frame),
        "filename_source_groups": int(frame["source_group_id"].nunique()),
        "group_overlap_all_folds": 0,
        "feature_count": len(features),
        "label_counts": dict(
            sorted(frame["upstream_label"].astype(str).value_counts().astype(int).to_dict().items())
        ),
        "model": _metrics(truth, predictions.astype(str), probabilities),
        "majority_empirical_probability_baseline": _metrics(
            truth,
            baseline_predictions.astype(str),
            baseline_probabilities,
        ),
        "folds": fold_rows,
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
            "model_artifact_policy": "local-only; upstream media have no license or provenance",
        }
    )
    write_json(report_path, report)
    return report
