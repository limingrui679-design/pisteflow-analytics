"""Pre-day Elo replay for real three-weapon USA Fencing bouts."""

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
VALIDATION_START = pd.Timestamp("2019-01-01")
TEST_START = pd.Timestamp("2020-01-01")


def _flip_bout(bout_id: str) -> bool:
    return int(sha256(f"pisteflow-orientation:{bout_id}".encode()).hexdigest()[:8], 16) % 2 == 1


def _expected(left_rating: float, right_rating: float) -> float:
    exponent = np.clip((right_rating - left_rating) / RATING_SCALE, -8.0, 8.0)
    return float(1.0 / (1.0 + 10.0**exponent))


def _rating_key(row: Any, athlete_id: str) -> tuple[str, str, str]:
    return str(row.weapon), str(row.gender), athlete_id


def prepare_bouts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    required = {
        "bout_id",
        "event_date",
        "stage",
        "left_fencer_id",
        "right_fencer_id",
        "left_score",
        "right_score",
        "left_won",
        "weapon",
        "gender",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing competition fields: {sorted(missing)}")
    source_rows = len(frame)
    valid = frame.loc[
        frame["left_won"].eq(1)
        & frame["left_score"].notna()
        & frame["right_score"].notna()
        & frame["left_score"].gt(frame["right_score"])
        & frame["left_fencer_id"].notna()
        & frame["right_fencer_id"].notna()
        & frame["left_fencer_id"].ne(frame["right_fencer_id"])
    ].copy()
    valid["event_date"] = pd.to_datetime(valid["event_date"], errors="raise")
    valid = valid.sort_values(["event_date", "bout_id"], kind="stable").reset_index(drop=True)
    if valid["bout_id"].duplicated().any():
        raise ValueError("Competition bout IDs are not unique")
    audit = {
        "source_rows": source_rows,
        "eligible_decisive_bouts": len(valid),
        "excluded_tied_missing_or_contradictory_rows": source_rows - len(valid),
        "source_left_winner_share_eligible": float(valid["left_won"].mean()),
    }
    return valid, audit


def replay(
    bouts: pd.DataFrame,
    *,
    k_factor: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], float]]:
    ratings: dict[tuple[str, str, str], float] = {}
    prediction_rows: list[dict[str, Any]] = []
    for event_date, date_rows in bouts.groupby("event_date", sort=True):
        updates: defaultdict[tuple[str, str, str], float] = defaultdict(float)
        for row in date_rows.itertuples(index=False):
            winner_id = str(row.left_fencer_id)
            loser_id = str(row.right_fencer_id)
            winner_key = _rating_key(row, winner_id)
            loser_key = _rating_key(row, loser_id)
            winner_rating = ratings.get(winner_key, INITIAL_RATING)
            loser_rating = ratings.get(loser_key, INITIAL_RATING)
            expected_winner = _expected(winner_rating, loser_rating)
            flipped = _flip_bout(str(row.bout_id))
            probability_left = 1.0 - expected_winner if flipped else expected_winner
            target_left = 0 if flipped else 1
            prediction_rows.append(
                {
                    "bout_id": str(row.bout_id),
                    "event_date": event_date,
                    "weapon": str(row.weapon),
                    "stage": str(row.stage),
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


def _metrics(rows: pd.DataFrame) -> dict[str, float]:
    truth = rows["target_left_win"].astype(int).to_numpy()
    probabilities = rows["probability_left_win"].astype(float).to_numpy()
    predicted = probabilities >= 0.5
    return {
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
    }


def _baseline_metrics(rows: pd.DataFrame) -> dict[str, float]:
    baseline = rows.copy()
    baseline["probability_left_win"] = 0.5
    return _metrics(baseline)


def evaluate_competition_elo(
    frame: pd.DataFrame,
    *,
    validation_start: pd.Timestamp = VALIDATION_START,
    test_start: pd.Timestamp = TEST_START,
    k_candidates: list[float] = K_CANDIDATES,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], float]]:
    bouts, audit = prepare_bouts(frame)
    if bouts["event_date"].min() >= validation_start or bouts["event_date"].max() < test_start:
        raise ValueError("Competition data do not span the chronological validation/test cutoffs")
    candidates: list[dict[str, float]] = []
    replays: dict[float, tuple[pd.DataFrame, dict[tuple[str, str, str], float]]] = {}
    for k_factor in k_candidates:
        predictions, ratings = replay(bouts, k_factor=k_factor)
        replays[k_factor] = predictions, ratings
        validation = predictions.loc[
            predictions["event_date"].ge(validation_start)
            & predictions["event_date"].lt(test_start)
        ]
        candidates.append(
            {
                "k_factor": k_factor,
                "validation_bouts": len(validation),
                "validation_log_loss": _metrics(validation)["log_loss"],
            }
        )
    selected = min(candidates, key=lambda row: (row["validation_log_loss"], row["k_factor"]))
    selected_k = float(selected["k_factor"])
    predictions, ratings = replays[selected_k]
    test = predictions.loc[predictions["event_date"].ge(test_start)].reset_index(drop=True)
    if test.empty:
        raise ValueError("Competition test period is empty")
    by_weapon = {
        weapon: {
            "bouts": len(rows),
            "elo": _metrics(rows),
            "uninformative_baseline": _baseline_metrics(rows),
        }
        for weapon, rows in test.groupby("weapon", sort=True)
    }
    by_stage = {
        stage: {
            "bouts": len(rows),
            "elo": _metrics(rows),
            "uninformative_baseline": _baseline_metrics(rows),
        }
        for stage, rows in test.groupby("stage", sort=True)
    }
    report: dict[str, Any] = {
        "claim_boundary": (
            "Pre-day Elo replay over 374,088 decisive USA Fencing national-event bouts. The "
            "source stores the winner on the left, so presentation orientation is "
            "deterministically "
            "flipped for roughly half of bouts before evaluation; source ordering is never a "
            "predictor. All bouts on a date are predicted from ratings entering that date and "
            "updated only afterward. K is selected on 2019 and evaluated on February-March 2020. "
            "This third-party historical snapshot is not a current official forecast, causal "
            "training model, or athlete-independent external validation."
        ),
        "evaluation_design": (
            "pre-day batch Elo; K selected on 2019; untouched chronological test from 2020-01-01"
        ),
        "rating_key": "weapon + gender + pseudonymous fencer ID",
        "initial_rating": INITIAL_RATING,
        "rating_scale": RATING_SCALE,
        "selected_k_factor": selected_k,
        "candidate_selection": candidates,
        **audit,
        "eligible_by_weapon": dict(
            sorted(bouts["weapon"].astype(str).value_counts().astype(int).to_dict().items())
        ),
        "eligible_date_min": bouts["event_date"].min().date().isoformat(),
        "eligible_date_max": bouts["event_date"].max().date().isoformat(),
        "test_bouts": len(test),
        "test_date_min": test["event_date"].min().date().isoformat(),
        "test_date_max": test["event_date"].max().date().isoformat(),
        "test_orientation_flipped_share": float(test["orientation_flipped"].mean()),
        "test_previously_unrated_pair_share": float(test["both_previously_unrated"].mean()),
        "test": {
            "elo": _metrics(test),
            "uninformative_baseline": _baseline_metrics(test),
            "by_weapon": by_weapon,
            "by_stage": by_stage,
        },
        "final_rating_keys": len(ratings),
    }
    return report, ratings


def train_and_report(input_path: Path, model_path: Path, report_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    report, ratings = evaluate_competition_elo(frame)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "ratings": ratings,
                "k_factor": report["selected_k_factor"],
                "initial_rating": INITIAL_RATING,
                "rating_scale": RATING_SCALE,
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
