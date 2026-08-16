from __future__ import annotations

import pandas as pd

from pisteflow.analysis.nz_elo import prepare_bouts, replay


def test_nz_elo_updates_only_after_all_same_day_bouts() -> None:
    frame = pd.DataFrame(
        [
            {
                "bout_id": "bout-1",
                "date": "2024-01-01",
                "weapon": "foil",
                "bout_type": "pool",
                "gender": "men",
                "fencer_a_id": "winner",
                "fencer_b_id": "loser",
                "score_a": 5,
                "score_b": 2,
                "winner_side": "a",
                "model_eligible": True,
            },
            {
                "bout_id": "bout-2",
                "date": "2024-01-01",
                "weapon": "foil",
                "bout_type": "pool",
                "gender": "men",
                "fencer_a_id": "winner",
                "fencer_b_id": "loser",
                "score_a": 5,
                "score_b": 1,
                "winner_side": "a",
                "model_eligible": True,
            },
            {
                "bout_id": "bout-3",
                "date": "2024-01-02",
                "weapon": "foil",
                "bout_type": "pool",
                "gender": "men",
                "fencer_a_id": "winner",
                "fencer_b_id": "loser",
                "score_a": 5,
                "score_b": 3,
                "winner_side": "a",
                "model_eligible": True,
            },
            {
                "bout_id": "withdrawal",
                "date": "2024-01-02",
                "weapon": "foil",
                "bout_type": "de",
                "gender": "men",
                "fencer_a_id": "winner",
                "fencer_b_id": "loser",
                "score_a": 0,
                "score_b": 0,
                "winner_side": None,
                "model_eligible": False,
            },
        ]
    )
    bouts, audit = prepare_bouts(frame)
    predictions, _ratings = replay(bouts, k_factor=32.0, split_streams=False)

    assert audit["eligible_decisive_bouts"] == 3
    assert audit["excluded_withdrawal_tied_or_invalid_rows"] == 1
    assert set(predictions.loc[:1, "probability_left_win"]) == {0.5}
    assert predictions.loc[2, "probability_left_win"] != 0.5
