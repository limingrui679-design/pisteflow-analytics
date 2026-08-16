from __future__ import annotations

import numpy as np
import pandas as pd

from pisteflow.analysis import competition_elo


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bout_index = 0
    athletes = [f"f-{index:02d}" for index in range(18)]
    for date in pd.date_range("2018-01-01", "2020-06-01", freq="30D"):
        for pair_index in range(0, len(athletes), 2):
            winner = athletes[pair_index]
            loser = athletes[pair_index + 1]
            rows.append(
                {
                    "bout_id": f"bout-{bout_index:04d}",
                    "event_date": date,
                    "stage": "pool" if bout_index % 2 else "direct_elimination",
                    "left_fencer_id": winner,
                    "right_fencer_id": loser,
                    "left_score": 5,
                    "right_score": 2,
                    "left_won": 1,
                    "weapon": ("epee", "foil", "sabre")[(pair_index // 2) % 3],
                    "gender": "women" if pair_index % 4 else "men",
                }
            )
            bout_index += 1
    return pd.DataFrame(rows)


def test_replay_randomizes_presentation_and_batches_dates() -> None:
    bouts, audit = competition_elo.prepare_bouts(_frame())
    predictions, ratings = competition_elo.replay(bouts, k_factor=16.0)

    assert audit["eligible_decisive_bouts"] == len(bouts)
    assert 0.3 < predictions["orientation_flipped"].mean() < 0.7
    assert set(predictions["target_left_win"]) == {0, 1}
    assert ratings


def test_competition_elo_chronological_test() -> None:
    report, ratings = competition_elo.evaluate_competition_elo(
        _frame(),
        validation_start=pd.Timestamp("2019-01-01"),
        test_start=pd.Timestamp("2020-01-01"),
        k_candidates=[8.0, 16.0],
    )

    assert report["test_bouts"] > 0
    assert set(report["test"]["by_weapon"]) == {"epee", "foil", "sabre"}
    assert np.isfinite(report["test"]["elo"]["log_loss"])
    assert 0.3 < report["test_orientation_flipped_share"] < 0.7
    assert ratings
