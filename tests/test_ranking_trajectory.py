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
