from __future__ import annotations

import pandas as pd

from pisteflow.analysis.hot_hand_validation import evaluate_grouped


def test_hot_hand_validation_holds_out_complete_pools() -> None:
    rows = []
    for pool in range(4):
        for bout in range(8):
            left_win = (pool + bout) % 2 == 0
            rating_a = 1800 + 50 * ((bout + pool) % 4)
            rating_b = 1800 + 50 * ((bout + pool + 1) % 4)
            rows.append(
                {
                    "bout_id": f"p{pool}-b{bout}",
                    "pool_id": f"pool_{pool}",
                    "winner_side": "a" if left_win else "b",
                    "rating_a_pre": rating_a,
                    "rating_b_pre": rating_b,
                    "prev_win_a": float((bout + 1) % 2),
                    "prev_win_b": float(bout % 2),
                    "recent_form_3_a": float((bout % 3) / 3),
                    "recent_form_3_b": float(((bout + 1) % 3) / 3),
                }
            )

    report, oof, models = evaluate_grouped(pd.DataFrame(rows))

    assert report["pools"] == 4
    assert report["pool_overlap_all_folds"] == 0
    assert len(report["folds"]) == 4
    assert len(oof) == 32
    assert set(models) == {
        "rating_only",
        "rating_plus_previous_outcome",
        "rating_plus_recent_form",
    }
