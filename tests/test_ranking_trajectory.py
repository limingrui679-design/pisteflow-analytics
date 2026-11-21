from __future__ import annotations

import numpy as np
import pandas as pd

from pisteflow.analysis import ranking_trajectory


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_index in range(30):
        weapon = ("epee", "foil", "sabre")[series_index % 3]
        for snapshot_index, date in enumerate(pd.date_range("2018-01-01", periods=16, freq="120D")):
            rank = 20 + series_index + snapshot_index % 4
            domestic = 100.0 - rank
            international = float(series_index % 5)
            rows.append(
                {
                    "athlete_series_id": f"series-{series_index:02d}",
                    "snapshot_date": date,
                    "rank": rank,
                    "rolling_points": domestic + international,
                    "domestic_points": domestic,
                    "international_points": international,
                    "weapon": weapon,
                    "gender": ("men", "women")[series_index % 2],
                    "age_category": ("cadet", "junior", "senior")[series_index % 3],
                    "training_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_build_transitions_uses_forward_target_and_horizon() -> None:
    transitions, total = ranking_trajectory.build_transitions(_frame(), max_horizon_days=180)

    assert total == 30 * 15
    assert len(transitions) == total
    assert transitions["horizon_days"].eq(120).all()
    assert (transitions["target_date"] > transitions["snapshot_date"]).all()


def test_chronological_evaluation_covers_three_weapons() -> None:
    report, model, features = ranking_trajectory.evaluate_trajectory(
        _frame(), cutoff=pd.Timestamp("2020-01-01"), max_horizon_days=180
    )

    assert report["train_transitions"] > 0
    assert report["test_transitions"] > 0
    assert set(report["test"]["by_weapon"]) == {"epee", "foil", "sabre"}
    assert "target_rank" not in features
    assert np.isfinite(report["test"]["model"]["mae_rank_places"])
    assert model.n_features_in_ == len(features)


def test_purges_crossing_transitions_and_estimates_paired_series_gain() -> None:
    frame = _frame()
    report, _, _ = ranking_trajectory.evaluate_trajectory(frame)
    assert report["last_training_target"] < report["cutoff"] <= report["first_test_origin"]
    assert report["purged_crossing_transitions"] == 30
    assert (
        sum(
            report[x]
            for x in ("train_transitions", "test_transitions", "purged_crossing_transitions")
        )
        == report["eligible_transitions"]
    )
    skill = ranking_trajectory.paired_series_skill(
        pd.Series(["a", "a", "a", "b"]),
        np.ones(4),
        np.array([1.0, 1.0, 1.0, 5.0]),
        np.array([3.0, 3.0, 3.0, 1.0]),
    )
    assert skill["transition_weighted_mae_gain"] == 0.5
    assert skill["series_balanced_mae_gain"] == -1.0


def test_weight_selection_uses_only_pre_cutoff_targets_and_equal_scores_favour_persistence():
    frame = _frame()
    frame["rank"] = 1
    report, predictor, _ = ranking_trajectory.evaluate_trajectory(frame)
    assert predictor.weight == 0
    folds = report["development_selection"]["folds"]
    assert folds
    for fold in folds:
        assert fold["last_fit_target"] < fold["first_validation_origin"]
        assert fold["last_validation_target"] < report["cutoff"]
    transitions, _ = ranking_trajectory.build_transitions(frame)
    weight, evidence = ranking_trajectory.select_weight(transitions.iloc[:1])
    assert weight == 0 and evidence["fallback"]
