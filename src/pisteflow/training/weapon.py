"""A transparent three-weapon video baseline with held-out source files."""

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
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pisteflow.provenance import sha256_file, write_json


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if column.startswith(("frame_", "flow_"))]


def _model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=20260814,
                ),
            ),
        ]
    )


def _video_predictions(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    classes = np.asarray(classifier.classes_)
    probabilities = model.predict_proba(frame[feature_columns])
    probability_frame = pd.DataFrame(probabilities, columns=classes, index=frame.index)
    probability_frame["video_id"] = frame["video_id"].to_numpy()
    grouped = probability_frame.groupby("video_id", sort=True)[classes.tolist()].mean()
    predicted = classes[np.argmax(grouped.to_numpy(), axis=1)]
    truth = frame.groupby("video_id", sort=True)["weapon"].first()
    return pd.DataFrame({"truth": truth, "predicted": predicted}, index=grouped.index)


def _metrics(truth: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    labels = ["epee", "foil", "sabre"]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "labels": labels,
        "classification_report": classification_report(
            truth,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_weapon_classifier(features_path: Path, output_directory: Path) -> dict[str, Any]:
    frame = pd.read_parquet(features_path)
    required = {"video_id", "weapon", "split", "group_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    split_videos = {
        split: set(rows["video_id"]) for split, rows in frame.groupby("split", sort=True)
    }
    split_groups = {
        split: set(rows["group_id"]) for split, rows in frame.groupby("split", sort=True)
    }
    split_names = list(split_videos)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            video_overlap = split_videos[left].intersection(split_videos[right])
            if video_overlap:
                raise ValueError(
                    f"Video leakage between {left} and {right}: {sorted(video_overlap)}"
                )
            group_overlap = split_groups[left].intersection(split_groups[right])
            if group_overlap:
                raise ValueError(
                    f"Source-family leakage between {left} and {right}: {sorted(group_overlap)}"
                )

    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    if set(train["weapon"]) != {"epee", "foil", "sabre"}:
        raise ValueError("Training split must contain all three weapons")
    if test.empty:
        raise ValueError("A source-held-out test split is required")

    feature_columns = _feature_columns(frame)
    candidates: list[dict[str, float]] = []
    selected_c = 1.0
    if not validation.empty:
        for c_value in (0.01, 0.1, 1.0, 10.0):
            candidate = _model(c_value)
            candidate.fit(train[feature_columns], train["weapon"])
            prediction = candidate.predict(validation[feature_columns])
            score = float(balanced_accuracy_score(validation["weapon"], prediction))
            candidates.append({"c": c_value, "validation_balanced_accuracy": score})
        selected_c = max(
            candidates,
            key=lambda row: (row["validation_balanced_accuracy"], -row["c"]),
        )["c"]

    training_rows = pd.concat([train, validation], ignore_index=True)
    model = _model(selected_c)
    model.fit(training_rows[feature_columns], training_rows["weapon"])
    window_prediction = model.predict(test[feature_columns])
    video_prediction = _video_predictions(model, test, feature_columns)

    report = {
        "claim_boundary": (
            "Internal source-family-held-out baseline on 43 verified-open Commons videos. The "
            "test contains only three independent videos, one per weapon, so video-level metrics "
            "are a smoke test rather than a stable generalization estimate. This is not a referee, "
            "athlete-performance, or in-the-wild deployment result."
        ),
        "features_sha256": sha256_file(features_path),
        "feature_count": len(feature_columns),
        "selected_c": selected_c,
        "candidate_selection": candidates,
        "train": {
            "videos": int(training_rows["video_id"].nunique()),
            "source_groups": int(training_rows["group_id"].nunique()),
            "windows": len(training_rows),
            "by_weapon": training_rows.groupby("weapon")["video_id"].nunique().to_dict(),
        },
        "test": {
            "videos": int(test["video_id"].nunique()),
            "source_groups": int(test["group_id"].nunique()),
            "windows": len(test),
            "by_weapon": test.groupby("weapon")["video_id"].nunique().to_dict(),
            "window_metrics": _metrics(test["weapon"], window_prediction),
            "video_metrics": _metrics(
                video_prediction["truth"],
                video_prediction["predicted"].to_numpy(),
            ),
            "predictions": video_prediction.reset_index().to_dict(orient="records"),
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "weapon-baseline.pkl").open("wb") as handle:
        pickle.dump({"model": model, "features": feature_columns}, handle)
    write_json(output_directory / "weapon-baseline-report.json", report)
    return report
