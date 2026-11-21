"""Score changes across an illustrative rank boundary on next observed transitions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _event_probability(deltas: NDArray[np.float64], rank: float, boundary: int) -> float:
    # The sorted signed movement distribution is shared by all origins in the cell.
    below = np.searchsorted(deltas, boundary - rank, side="right") / len(deltas)
    return float(1 - below if rank <= boundary else below)


def _slot_weights(scores: NDArray[np.float64], slots: int) -> NDArray[np.float64]:
    threshold = np.partition(scores, len(scores) - slots)[-slots]
    above, tied = scores > threshold, scores == threshold
    return np.asarray(above + tied * ((slots - above.sum()) / tied.sum()), dtype=float)


def _expected_hits(scores: NDArray[np.float64], events: NDArray[np.bool_], slots: int) -> float:
    return float(_slot_weights(scores, slots) @ events)


def predict_boundary_crossings(
    development: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    boundary: int = 16,
    prior_count: float = 20.0,
) -> NDArray[np.float64]:
    """Forecast crossings using only rank, weapon and date from each new snapshot."""
    if development.empty or origins.empty:
        raise ValueError("Require development transitions and forecast origins")
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 1:
        raise ValueError("Require a positive integer boundary")
    if not math.isfinite(prior_count) or prior_count <= 0:
        raise ValueError("Require finite positive smoothing mass")
    target_dates = pd.to_datetime(development["target_date"], errors="raise")
    origin_dates = pd.to_datetime(origins["snapshot_date"], errors="raise")
    if (
        target_dates.isna().any()
        or origin_dates.isna().any()
        or target_dates.max() >= origin_dates.min()
    ):
        raise ValueError("Development outcomes must precede every forecast origin")
    for ranks in (
        development[["rank", "target_rank"]].to_numpy(dtype=float),
        origins["rank"].to_numpy(dtype=float),
    ):
        if not np.isfinite(ranks).all() or (ranks < 1).any() or (ranks != np.floor(ranks)).any():
            raise ValueError("Require finite positive integer ranks")
    hist = development.assign(
        rank_band=np.floor(np.log2(development["rank"].astype(float))),
        movement=development["target_rank"].astype(float) - development["rank"].astype(float),
    )
    global_changes = np.sort(hist["movement"].to_numpy())
    cells = {
        key: np.sort(group["movement"].to_numpy())
        for key, group in hist.groupby(["weapon", "rank_band"], observed=True)
    }
    # Shrink each empirical CDF toward the pooled movement CDF to handle sparse cells.
    cache = {}
    probabilities = []
    for weapon, rank_value in zip(origins["weapon"], origins["rank"], strict=True):
        rank = float(rank_value)
        key = (weapon, rank)
        if key not in cache:
            pooled = _event_probability(global_changes, rank, boundary)
            cell = cells.get((weapon, math.floor(math.log2(rank))))
            if cell is None:
                cache[key] = pooled
            else:
                local = _event_probability(cell, rank, boundary)
                cache[key] = (len(cell) * local + prior_count * pooled) / (len(cell) + prior_count)
        probabilities.append(cache[key])

    return np.asarray(probabilities, dtype=float)


def evaluate_boundary_review(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    boundary: int = 16,
    capacity_fraction: float = 0.05,
    prior_count: float = 20.0,
) -> dict[str, Any]:
    """Replay the historically trained risk queue against proximity and same-date random review."""
    if not 0 < capacity_fraction <= 1:
        raise ValueError("Capacity fraction must be in (0, 1]")
    risk = predict_boundary_crossings(
        development, evaluation, boundary=boundary, prior_count=prior_count
    )
    outcomes = evaluation["target_rank"].to_numpy(dtype=float)
    if (
        not np.isfinite(outcomes).all()
        or (outcomes < 1).any()
        or (outcomes != np.floor(outcomes)).any()
    ):
        raise ValueError("Require finite positive integer target ranks")
    ranks = evaluation["rank"].to_numpy(dtype=float)
    events = (ranks <= boundary) != (outcomes <= boundary)
    nearest = -np.abs(ranks - (boundary + 0.5))
    slots_used = 0
    risk_hits = proximity_hits = random_hits = 0.0
    month_gains: dict[str, float] = {}
    for date, indices in evaluation.groupby("snapshot_date", sort=True).indices.items():
        slots = math.ceil(capacity_fraction * len(indices))
        slots_used += slots
        day_risk = _expected_hits(risk[indices], events[indices], slots)
        day_proximity = _expected_hits(nearest[indices], events[indices], slots)
        month = pd.Timestamp(date).strftime("%Y-%m")
        month_gains[month] = month_gains.get(month, 0.0) + day_risk - day_proximity
        risk_hits += day_risk
        proximity_hits += day_proximity
        random_hits += slots * float(events[indices].mean())
    historical_events = (development["rank"] <= boundary) != (
        development["target_rank"] <= boundary
    )
    return {
        "boundary": boundary,
        "capacity_fraction": capacity_fraction,
        "prior_count": prior_count,
        "transitions": len(risk),
        "crossings": int(events.sum()),
        "review_slots": slots_used,
        "risk_brier": float(np.square(risk - events).mean()),
        "historical_prevalence_brier": float(np.square(historical_events.mean() - events).mean()),
        "persistence_brier": float(events.mean()),
        "risk_expected_hits": risk_hits,
        "proximity_expected_hits": proximity_hits,
        "random_expected_hits": random_hits,
        "evaluated_calendar_months": len(month_gains),
        "months_risk_exceeded_proximity": sum(value > 1e-10 for value in month_gains.values()),
        "months_proximity_exceeded_risk": sum(value < -1e-10 for value in month_gains.values()),
        "leave_one_month_out_gain_range": [
            sum(month_gains.values()) - max(month_gains.values()),
            sum(month_gains.values()) - min(month_gains.values()),
        ]
        if len(month_gains) > 1
        else None,
    }


def evaluate_origin_roster(
    frame: pd.DataFrame,
    development: pd.DataFrame,
    observed_transitions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    max_horizon_days: int,
    boundary: int = 16,
    capacity_fraction: float = 0.05,
) -> dict[str, Any]:
    """Allocate slots before looking up follow-up and bound unresolved crossing outcomes."""
    if not 0 < capacity_fraction <= 1 or max_horizon_days < 1:
        raise ValueError("Require a positive horizon and a capacity fraction in (0, 1]")
    keys = ["athlete_series_id", "snapshot_date"]
    category = ["weapon", "gender", "age_category"]
    roster = frame.loc[frame["training_eligible"].astype(bool), [*keys, *category, "rank"]].copy()
    roster["snapshot_date"] = pd.to_datetime(roster["snapshot_date"])
    # Limit backtesting dates to the part of each category archive with a full horizon.
    ends = roster.groupby(category, observed=True, dropna=False)["snapshot_date"].transform("max")
    mature = roster["snapshot_date"] + pd.Timedelta(days=max_horizon_days) <= ends
    origins = roster.loc[mature & roster["snapshot_date"].ge(cutoff)].reset_index(drop=True)
    if origins.empty:
        return {"status": "no_mature_origins", "origins": 0}

    ranks = origins["rank"].to_numpy(dtype=float)
    risk = predict_boundary_crossings(development, origins, boundary=boundary)
    scores = {"risk": risk, "proximity": -np.abs(ranks - (boundary + 0.5))}
    weights = {policy: np.zeros(len(origins)) for policy in [*scores, "random"]}
    total_slots = 0
    dates = origins.groupby("snapshot_date").indices
    for indices in dates.values():
        slots = math.ceil(capacity_fraction * len(indices))
        total_slots += slots
        for policy, values in scores.items():
            weights[policy][indices] = _slot_weights(values[indices], slots)
        weights["random"][indices] = slots / len(indices)

    targets = (
        origins[keys]
        .merge(
            observed_transitions[[*keys, "target_rank"]], on=keys, how="left", validate="one_to_one"
        )["target_rank"]
        .to_numpy(dtype=float)
    )
    missing = np.isnan(targets)
    known_crossings = ((ranks <= boundary) != (targets <= boundary)) & ~missing

    def bounds(weight: NDArray[np.float64]) -> list[float]:
        # The same missing label applies to both policies when weight is their difference.
        known_hits = float(weight @ known_crossings)
        unresolved = weight[missing]
        return [
            known_hits + float(np.minimum(unresolved, 0).sum()),
            known_hits + float(np.maximum(unresolved, 0).sum()),
        ]

    return {
        "status": "evaluated",
        "boundary": boundary,
        "capacity_fraction": capacity_fraction,
        "max_horizon_days": max_horizon_days,
        "dates": len(dates),
        "origins": len(origins),
        "known_outcomes": int((~missing).sum()),
        "unobserved_outcomes": int(missing.sum()),
        "known_crossings": int(known_crossings.sum()),
        "review_slots": total_slots,
        "expected_hit_bounds": {policy: bounds(weight) for policy, weight in weights.items()},
        "paired_gain_bounds_vs_proximity": bounds(weights["risk"] - weights["proximity"]),
    }
