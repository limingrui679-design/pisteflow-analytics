"""Chronological Elo replication on the privacy-minimized New Zealand ledger."""

from __future__ import annotations

import pickle
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from pisteflow.provenance import sha256_file, write_json

INITIAL_RATING = 1500.0
RATING_SCALE = 400.0
K_CANDIDATES = [8.0, 16.0, 32.0, 64.0]
VALIDATION_START = pd.Timestamp("2025-01-01")
TEST_START = pd.Timestamp("2026-01-01")


def _flip_bout(bout_id: str) -> bool:
    digest = sha256(f"pisteflow-nz-orientation:{bout_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 1


def _expected(left_rating: float, right_rating: float) -> float:
    exponent = np.clip((right_rating - left_rating) / RATING_SCALE, -8.0, 8.0)
    return float(1.0 / (1.0 + 10.0**exponent))


def _rating_key(row: Any, athlete_id: str, *, split_streams: bool) -> tuple[str, ...]:
    base = (str(row.weapon), str(row.gender))
    if split_streams:
        return (*base, str(row.bout_type), athlete_id)
    return (*base, athlete_id)


def prepare_bouts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "bout_id",
        "date",
        "weapon",
        "bout_type",
        "gender",
        "fencer_a_id",
        "fencer_b_id",
        "score_a",
        "score_b",
        "winner_side",
        "model_eligible",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing FencingStatsNZ model fields: {sorted(missing)}")
    source_rows = len(frame)
    eligible = frame.loc[
        frame["model_eligible"].eq(True)
        & frame["winner_side"].isin(["a", "b"])
        & frame["fencer_a_id"].notna()
        & frame["fencer_b_id"].notna()
        & frame["fencer_a_id"].ne(frame["fencer_b_id"])
        & frame["score_a"].ne(frame["score_b"])
    ].copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="raise")
    eligible = eligible.sort_values(["date", "bout_id"], kind="stable").reset_index(drop=True)
    if eligible["bout_id"].duplicated().any():
        raise ValueError("FencingStatsNZ model bout IDs are duplicated")
    return eligible, {
        "source_rows": source_rows,
        "eligible_decisive_bouts": len(eligible),
        "excluded_withdrawal_tied_or_invalid_rows": source_rows - len(eligible),
    }


def replay(
    bouts: pd.DataFrame,
    *,
    k_factor: float,
    split_streams: bool,
) -> tuple[pd.DataFrame, dict[tuple[str, ...], float]]:
    ratings: dict[tuple[str, ...], float] = {}
    prediction_rows: list[dict[str, Any]] = []
    for date, date_rows in bouts.groupby("date", sort=True):
        updates: defaultdict[tuple[str, ...], float] = defaultdict(float)
        for row in date_rows.itertuples(index=False):
            if row.winner_side == "a":
                winner_id = str(row.fencer_a_id)
                loser_id = str(row.fencer_b_id)
            else:
                winner_id = str(row.fencer_b_id)
                loser_id = str(row.fencer_a_id)
            winner_key = _rating_key(row, winner_id, split_streams=split_streams)
            loser_key = _rating_key(row, loser_id, split_streams=split_streams)
            winner_rating = ratings.get(winner_key, INITIAL_RATING)
            loser_rating = ratings.get(loser_key, INITIAL_RATING)
            expected_winner = _expected(winner_rating, loser_rating)
            flipped = _flip_bout(str(row.bout_id))
            probability_left = 1.0 - expected_winner if flipped else expected_winner
            target_left = 0 if flipped else 1
            prediction_rows.append(
                {
                    "bout_id": str(row.bout_id),
                    "date": date,
                    "weapon": str(row.weapon),
                    "bout_type": str(row.bout_type),
                    "gender": str(row.gender),
                    "target_left_win": target_left,
                    "probability_left_win": probability_left,
                    "orientation_flipped": flipped,
                    "both_previously_unrated": (
                        winner_key not in ratings and loser_key not in ratings
                    ),
                }
            )
            change = k_factor * (1.0 - expected_winner)
            updates[winner_key] += change
            updates[loser_key] -= change
        for key, change in updates.items():
            ratings[key] = ratings.get(key, INITIAL_RATING) + change
    return pd.DataFrame(prediction_rows), ratings


def _metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    truth = rows["target_left_win"].astype(int).to_numpy()
    probabilities = rows["probability_left_win"].astype(float).to_numpy()
    predicted = probabilities >= 0.5
    return {
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "roc_auc": (
            float(roc_auc_score(truth, probabilities)) if len(np.unique(truth)) == 2 else None
        ),
    }


def _baseline_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    baseline = rows.copy()
    baseline["probability_left_win"] = 0.5
    return _metrics(baseline)


def evaluate_nz_elo(
    frame: pd.DataFrame,
    *,
    validation_start: pd.Timestamp = VALIDATION_START,
    test_start: pd.Timestamp = TEST_START,
    k_candidates: list[float] = K_CANDIDATES,
) -> tuple[dict[str, Any], dict[tuple[str, ...], float]]:
    bouts, audit = prepare_bouts(frame)
    if bouts["date"].min() >= validation_start or bouts["date"].max() < test_start:
        raise ValueError("FencingStatsNZ data do not span the validation/test cutoffs")
    candidates: list[dict[str, Any]] = []
    replays: dict[tuple[float, bool], tuple[pd.DataFrame, dict[tuple[str, ...], float]]] = {}
    for split_streams in (False, True):
        for k_factor in k_candidates:
            predictions, ratings = replay(
                bouts,
                k_factor=k_factor,
                split_streams=split_streams,
            )
            replays[(k_factor, split_streams)] = predictions, ratings
            validation = predictions.loc[
                predictions["date"].ge(validation_start)
                & predictions["date"].lt(test_start)
            ]
            candidates.append(
                {
                    "k_factor": k_factor,
                    "split_pool_and_de_ratings": split_streams,
                    "validation_bouts": len(validation),
                    "validation_log_loss": _metrics(validation)["log_loss"],
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["validation_log_loss"]),
            bool(row["split_pool_and_de_ratings"]),
            float(row["k_factor"]),
        ),
    )
    selected_k = float(selected["k_factor"])
    selected_split = bool(selected["split_pool_and_de_ratings"])
    predictions, ratings = replays[(selected_k, selected_split)]
    test = predictions.loc[predictions["date"].ge(test_start)].reset_index(drop=True)
    if test.empty:
        raise ValueError("FencingStatsNZ chronological test period is empty")

    def grouped_metrics(column: str) -> dict[str, Any]:
        return {
            str(value): {
                "bouts": len(rows),
                "elo": _metrics(rows),
                "uninformative_baseline": _baseline_metrics(rows),
            }
            for value, rows in test.groupby(column, sort=True)
        }

    report: dict[str, Any] = {
        "claim_boundary": (
            "Pre-day Elo replay over decisive 2024-2026 New Zealand public-result bouts. "
            "Ratings enter each date unchanged and update only after all same-day bouts. K and "
            "whether pool/DE ratings are separate are selected on 2025, then evaluated on an "
            "untouched 2026 period. Presentation orientation is deterministically flipped for "
            "about half the test bouts. This is an internal regional replication, not an official "
            "FeNZ rating, current forecast, causal training model, external human review, or "
            "deployed decision system."
        ),
        "evaluation_design": (
            "pre-day batch Elo; K and shared-vs-stream rating key selected on 2025; "
            "untouched chronological test from 2026-01-01"
        ),
        "initial_rating": INITIAL_RATING,
        "rating_scale": RATING_SCALE,
        "selected_k_factor": selected_k,
        "selected_split_pool_and_de_ratings": selected_split,
        "rating_key": (
            "weapon + gender + bout type + pseudonymous fencer ID"
            if selected_split
            else "weapon + gender + pseudonymous fencer ID"
        ),
        "candidate_selection": candidates,
        **audit,
        "eligible_by_weapon": dict(
            sorted(bouts["weapon"].astype(str).value_counts().astype(int).to_dict().items())
        ),
        "eligible_date_min": bouts["date"].min().date().isoformat(),
        "eligible_date_max": bouts["date"].max().date().isoformat(),
        "test_bouts": len(test),
        "test_date_min": test["date"].min().date().isoformat(),
        "test_date_max": test["date"].max().date().isoformat(),
        "test_orientation_flipped_share": float(test["orientation_flipped"].mean()),
        "test_previously_unrated_pair_share": float(test["both_previously_unrated"].mean()),
        "test": {
            "elo": _metrics(test),
            "uninformative_baseline": _baseline_metrics(test),
            "by_weapon": grouped_metrics("weapon"),
            "by_bout_type": grouped_metrics("bout_type"),
            "by_gender": grouped_metrics("gender"),
        },
        "final_rating_keys": len(ratings),
    }
    return report, ratings


def train_and_report(input_path: Path, model_path: Path, report_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, ratings = evaluate_nz_elo(frame)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "ratings": ratings,
                "k_factor": report["selected_k_factor"],
                "split_pool_and_de_ratings": report["selected_split_pool_and_de_ratings"],
                "initial_rating": INITIAL_RATING,
                "rating_scale": RATING_SCALE,
            },
            handle,
        )
    report.update(
        {
            "input_sha256": sha256_file(input_path),
            "model_sha256": sha256_file(model_path),
            "model_artifact_policy": "local-only; underlying public-result rights are unresolved",
        }
    )
    write_json(report_path, report)
    return report
