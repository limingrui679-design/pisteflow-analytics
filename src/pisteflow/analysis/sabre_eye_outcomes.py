"""Participant-isolated sabre touch-outcome baseline from deposited eye-tracking data."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pisteflow.provenance import sha256_file, write_json

C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]
NUMERIC_FEATURES = [
    "log_quiet_eye_duration_ms",
    "fixation_count_numeric",
    "location_count_numeric",
]
CATEGORICAL_FEATURES = ["offensive_phase", "left_eye_validity", "right_eye_validity"]


def prepare_touches(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "participant_key",
        "recording_key",
        "touch_index_within_participant",
        "quiet_eye_duration_ms",
        "won_touch",
        "offensive_phase",
        "fixation_count_numeric",
        "location_count_numeric",
        "left_eye_validity",
        "right_eye_validity",
        "supervised_touch_eligible",
        "weapon",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing sabre eye-tracking fields: {sorted(missing)}")
    source_rows = len(frame)
    eligible = frame.loc[frame["supervised_touch_eligible"].fillna(False)].copy()
    eligible = eligible.loc[
        eligible["participant_key"].notna()
        & eligible["won_touch"].isin([0, 1])
        & eligible["quiet_eye_duration_ms"].notna()
        & eligible["offensive_phase"].isin([0, 1])
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible sabre eye-tracking touches")
    if set(eligible["weapon"].astype(str)) != {"sabre"}:
        raise ValueError("Eye-tracking outcome table must be sabre-only")
    if eligible[["participant_key", "touch_index_within_participant"]].duplicated().any():
        raise ValueError("Participant touch keys are not unique")
    eligible["log_quiet_eye_duration_ms"] = np.log1p(
        eligible["quiet_eye_duration_ms"].astype(float)
    )
    eligible["offensive_phase"] = eligible["offensive_phase"].astype(str)
    audit = {
        "source_label_rows": source_rows,
        "eligible_supervised_touches": len(eligible),
        "excluded_placeholder_or_incomplete_rows": source_rows - len(eligible),
        "participants": int(eligible["participant_key"].nunique()),
        "recordings": int(eligible["recording_key"].nunique()),
        "won_touches": int(eligible["won_touch"].sum()),
        "lost_touches": int(len(eligible) - eligible["won_touch"].sum()),
    }
    return eligible.reset_index(drop=True), audit


def _pipeline(c_value: float, *, phase_only: bool = False) -> tuple[Pipeline, list[str]]:
    numeric = [] if phase_only else NUMERIC_FEATURES
    categorical = ["offensive_phase"] if phase_only else CATEGORICAL_FEATURES
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    transformers.append(
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        )
    )
    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "logistic",
                LogisticRegression(C=c_value, max_iter=2_000, random_state=20260816),
            ),
        ]
    )
    return model, numeric + categorical


def _metrics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "accuracy": float(accuracy_score(truth, probabilities >= 0.5)),
    }
    if len(np.unique(truth)) == 2:
        result["roc_auc"] = float(roc_auc_score(truth, probabilities))
    return result


def _loso_predictions(
    rows: pd.DataFrame,
    *,
    c_value: float,
    phase_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    truth = rows["won_touch"].astype(int).to_numpy()
    groups = rows["participant_key"].astype(str).to_numpy()
    probabilities = np.zeros(len(rows), dtype=float)
    prevalence = np.zeros(len(rows), dtype=float)
    folds: list[dict[str, Any]] = []
    splitter = LeaveOneGroupOut()
    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(rows, truth, groups), start=1
    ):
        model, features = _pipeline(c_value, phase_only=phase_only)
        model.fit(rows.iloc[train_index][features], truth[train_index])
        probabilities[test_index] = model.predict_proba(rows.iloc[test_index][features])[:, 1]
        prevalence[test_index] = float(np.mean(truth[train_index]))
        test_groups = sorted(set(groups[test_index]))
        folds.append(
            {
                "fold": fold_number,
                "train_participants": len(set(groups[train_index])),
                "test_participant_key": test_groups[0],
                "test_touches": len(test_index),
            }
        )
    return probabilities, prevalence, folds


def _select_c_loso(
    rows: pd.DataFrame,
    *,
    c_candidates: list[float],
    phase_only: bool = False,
) -> tuple[float, list[dict[str, float]]]:
    truth = rows["won_touch"].astype(int).to_numpy()
    candidates: list[dict[str, float]] = []
    for c_value in c_candidates:
        probabilities, _, _ = _loso_predictions(
            rows, c_value=c_value, phase_only=phase_only
        )
        metrics = _metrics(truth, probabilities)
        candidates.append(
            {
                "c": c_value,
                "loso_log_loss": metrics["log_loss"],
                "loso_brier_score": metrics["brier_score"],
            }
        )
    selected = min(candidates, key=lambda row: (row["loso_log_loss"], row["c"]))
    return float(selected["c"]), candidates


def _nested_loso_predictions(
    rows: pd.DataFrame,
    *,
    c_candidates: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    truth = rows["won_touch"].astype(int).to_numpy()
    groups = rows["participant_key"].astype(str).to_numpy()
    probabilities = np.zeros(len(rows), dtype=float)
    phase_probabilities = np.zeros(len(rows), dtype=float)
    prevalence = np.zeros(len(rows), dtype=float)
    folds: list[dict[str, Any]] = []
    outer = LeaveOneGroupOut()
    for fold_number, (train_index, test_index) in enumerate(
        outer.split(rows, truth, groups), start=1
    ):
        train_rows = rows.iloc[train_index]
        test_rows = rows.iloc[test_index]
        selected_c, _ = _select_c_loso(train_rows, c_candidates=c_candidates)
        selected_phase_c, _ = _select_c_loso(
            train_rows, c_candidates=c_candidates, phase_only=True
        )
        model, features = _pipeline(selected_c)
        model.fit(train_rows[features], truth[train_index])
        probabilities[test_index] = model.predict_proba(test_rows[features])[:, 1]
        phase_model, phase_features = _pipeline(selected_phase_c, phase_only=True)
        phase_model.fit(train_rows[phase_features], truth[train_index])
        phase_probabilities[test_index] = phase_model.predict_proba(
            test_rows[phase_features]
        )[:, 1]
        prevalence[test_index] = float(np.mean(truth[train_index]))
        test_groups = sorted(set(groups[test_index]))
        folds.append(
            {
                "fold": fold_number,
                "train_participants": len(set(groups[train_index])),
                "test_participant_key": test_groups[0],
                "test_touches": len(test_index),
                "selected_c_from_inner_loso": selected_c,
                "selected_phase_only_c_from_inner_loso": selected_phase_c,
            }
        )
    return probabilities, phase_probabilities, prevalence, folds


def evaluate_sabre_eye_outcomes(
    frame: pd.DataFrame,
    *,
    c_candidates: list[float] = C_CANDIDATES,
) -> tuple[dict[str, Any], Pipeline]:
    rows, audit = prepare_touches(frame)
    if rows["participant_key"].nunique() < 5:
        raise ValueError("At least five participants are required for LOSO evaluation")
    truth = rows["won_touch"].astype(int).to_numpy()
    probabilities, phase_probabilities, prevalence, folds = _nested_loso_predictions(
        rows, c_candidates=c_candidates
    )
    selected_c, candidates = _select_c_loso(rows, c_candidates=c_candidates)
    final_model, features = _pipeline(selected_c)
    final_model.fit(rows[features], truth)

    fold_metrics: list[dict[str, Any]] = []
    groups = rows["participant_key"].astype(str).to_numpy()
    for fold in folds:
        mask = groups == fold["test_participant_key"]
        fold_metrics.append(
            {
                **fold,
                "model": _metrics(truth[mask], probabilities[mask]),
                "training_prevalence_baseline": _metrics(truth[mask], prevalence[mask]),
            }
        )
    report: dict[str, Any] = {
        "claim_boundary": (
            "Leave-one-participant-out internal association baseline for 379 labelled touches "
            "from ten national-level sabre athletes in one deposited study. Eye and phase "
            "features from each held-out athlete are never used to fit that fold. This is not "
            "external validation, causal evidence that gaze training improves touches, a live "
            "referee, or a model for epee or foil. Pilot rows are excluded because participant "
            "overlap with the main sample cannot be resolved from the deposit."
        ),
        "evaluation_design": (
            "nested leave-one-participant-out; each outer fold selects C using inner "
            "participant-held-out log loss"
        ),
        "feature_policy": (
            "log quiet-eye duration, fixation/location counts, offensive/defensive phase, and "
            "left/right eye validity; no participant ID, timestamp, or outcome-derived score"
        ),
        "pilot_policy": "575 pilot touches excluded because main-pilot identity overlap is unknown",
        "selected_c": selected_c,
        "final_fit_candidate_selection_not_used_for_outer_test_predictions": candidates,
        **audit,
        "nested_loso": {
            "model": _metrics(truth, probabilities),
            "phase_only_baseline": _metrics(truth, phase_probabilities),
            "training_prevalence_baseline": _metrics(truth, prevalence),
            "folds": fold_metrics,
        },
    }
    return report, final_model


def train_and_report(input_path: Path, model_path: Path, report_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, model = evaluate_sabre_eye_outcomes(frame)
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
            "model_artifact_policy": "local-only; upstream row-level eye data are local-only",
        }
    )
    write_json(report_path, report)
    return report
