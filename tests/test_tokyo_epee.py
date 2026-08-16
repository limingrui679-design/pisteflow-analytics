import pandas as pd

from pisteflow.sources.tokyo_epee import _COLUMNS, normalize


def test_normalize_removes_names_and_treats_double_as_both() -> None:
    frame = pd.DataFrame(
        [
            [
                "Gold Medal",
                "First Fencer",
                "Second Person",
                "M",
                "arm",
                3,
                False,
                "center",
                10,
                1,
                "Fencer",
            ],
            [
                "Gold Medal",
                "First Fencer",
                "Second Person",
                "M",
                "chest",
                2,
                True,
                "center",
                5,
                1,
                "Second Person",
            ],
            [
                "Gold Medal",
                "First Fencer",
                "Second Person",
                "M",
                None,
                0,
                False,
                "center",
                0,
                1,
                None,
            ],
        ],
        columns=_COLUMNS,
    )

    result = normalize(frame)

    assert "player1" not in result.columns
    assert "scorer" not in result.columns
    assert result.loc[0, "reported_scorer_side"] == "player_1"
    assert result.loc[1, "awarded_side"] == "both"
    assert result.loc[2, "event_type"] == "uncoded_or_no_touch_exchange"
    assert result["bout_id"].nunique() == 1
