import pandas as pd

from pisteflow.sources.foil_history import normalize_bouts


def test_normalize_bouts_pseudonymizes_ids_and_preserves_bad_age() -> None:
    frame = pd.DataFrame(
        {
            "fencer_ID": [10],
            "opp_ID": [20],
            "fencer_age": [121],
            "opp_age": [24],
            "fencer_score": [5],
            "opp_score": [3],
            "winner_ID": [10],
            "fencer_curr_pts": [20.5],
            "opp_curr_pts": [10.0],
            "tournament_ID": ["2021-1"],
            "pool_ID": [2],
            "upset": [False],
            "date": ["2021-01-02"],
        }
    )

    result = normalize_bouts(frame, gender="women")

    assert result.loc[0, "left_fencer_id"] != "10"
    assert result.loc[0, "left_won"] == 1
    assert result.loc[0, "left_age_upstream"] == 121
    assert bool(result.loc[0, "ages_plausible"]) is False
