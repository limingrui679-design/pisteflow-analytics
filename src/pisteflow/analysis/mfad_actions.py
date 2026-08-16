"""Participant-held-out action classification over MFAD skeleton summaries."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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

LABELS = ["fleche", "lunge", "one_step_forward_lunge"]
RANDOM_SEED = 20260815


def select_features(frame: pd.DataFrame) -> list[str]:
    prefixes = ("time_steps", "duration_", "body_", "centroid_", "normalized_", "joint_")
    return [column for column in frame if column.startswith(prefixes)]


def _model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
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
) -> tuple[dict[str, Any], Pipeline, list[str]]:
    required = {"subject_group_id", "skill_level", "action"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing MFAD action fields: {sorted(missing)}")
    observed = set(frame["action"].astype(str))
    if observed != set(LABELS):
        raise ValueError(f"Unexpected MFAD action labels: {sorted(observed)}")
    features = select_features(frame)
    if not features:
        raise ValueError("No MFAD skeleton features were selected")
    forbidden = {"subject_group_id", "skill_level", "action", "trial_index_within_subject_action"}
    if forbidden.intersection(features):
        raise ValueError("MFAD identity, target, or trial index entered the feature matrix")

    matrix = frame[features].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    truth = frame["action"].astype(str).to_numpy()
    groups = frame["subject_group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
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
        if train_groups & test_groups:
            raise ValueError(f"MFAD participant leakage in fold {fold_number}")
        model = _model()
        model.fit(matrix.iloc[train_index], truth[train_index])
        fold_probabilities = model.predict_proba(matrix.iloc[test_index])
        classifier = model.named_steps["classifier"]
        class_index = {
            str(label): index for index, label in enumerate(classifier.classes_)
        }
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
                "train_sequences": len(train_index),
                "test_sequences": len(test_index),
                "train_participants": len(train_groups),
                "test_participants": len(test_groups),
                "participant_overlap": 0,
                "test_action_counts": dict(
                    sorted(
                        pd.Series(truth[test_index])
                        .value_counts()
                        .astype(int)
                        .to_dict()
                        .items()
                    )
                ),
            }
        )

    model_metrics = _metrics(truth, predictions.astype(str), probabilities)
    baseline_metrics = _metrics(
        truth,
        baseline_predictions.astype(str),
        baseline_probabilities,
    )
    by_skill_level = {}
    for skill_level, indexes in frame.groupby("skill_level", sort=True).groups.items():
        selected = np.asarray(list(indexes), dtype=int)
        by_skill_level[str(skill_level)] = {
            "sequences": len(selected),
            "participants": int(frame.iloc[selected]["subject_group_id"].nunique()),
            "model": _metrics(
                truth[selected],
                predictions[selected].astype(str),
                probabilities[selected],
            ),
        }

    final_model = _model()
    final_model.fit(matrix, truth)
    report: dict[str, Any] = {
        "claim_boundary": (
            "Five-fold participant-held-out internal action classification over 928 MFAD "
            "depth-camera skeleton sequences from 27 author-coded participants. Complete "
            "participants, not trials, are held out. Identity, skill-level label, and trial index "
            "are excluded from predictors. This is not external validation, a coaching "
            "prescription, weapon identification, or evidence that author-assigned skill levels "
            "are correct. The small four-participant amateur subgroup makes its slice especially "
            "uncertain."
        ),
        "evaluation_design": "5-fold StratifiedGroupKFold by pseudonymous participant",
        "random_seed": RANDOM_SEED,
        "sequences": len(frame),
        "participants": int(frame["subject_group_id"].nunique()),
        "feature_count": len(features),
        "feature_policy": (
            "translation- and scale-normalized 18-joint trajectory summaries, body-scale and "
            "centroid dynamics, and observed sequence length; no identity, level, or trial index"
        ),
        "class_counts": dict(
            sorted(frame["action"].astype(str).value_counts().astype(int).to_dict().items())
        ),
        "participant_overlap_all_folds": 0,
        "model": model_metrics,
        "majority_empirical_probability_baseline": baseline_metrics,
        "by_author_skill_level": by_skill_level,
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
            "model_artifact_policy": "local-only; upstream MFAD repository has no license",
        }
    )
    write_json(report_path, report)
    return report
