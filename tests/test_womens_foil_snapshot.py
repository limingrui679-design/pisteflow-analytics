import pandas as pd

from pisteflow.sources.womens_foil_snapshot import (
    normalize_bios,
    normalize_rankings,
    normalize_tournaments,
)


def test_bios_remove_direct_identity_and_preserve_upstream_flags() -> None:
    frame = pd.DataFrame(
        {
            "id": [123],
            "name": ["Example Athlete"],
            "country_code": ["GBR"],
            "country": ["GREAT BRITAIN"],
            "hand": [None],
            "age": [87],
            "url": ["https://fie.org/athletes/123"],
            "date_accessed": ["2021-05-13 12:00:00"],
        }
    )

    result = normalize_bios(frame)

    assert not {"id", "name", "url"}.intersection(result.columns)
    assert result.loc[0, "athlete_id"].startswith("wfs-athlete:")
    assert result.loc[0, "handedness"] == "unknown"
    assert bool(result.loc[0, "age_between_13_and_80"]) is False


def test_rankings_keep_cross_weapon_rows_and_mark_exact_repeat() -> None:
    frame = pd.DataFrame(
        {
            "id": [123, 123, 123],
            "weapon": ["Foil", "Foil", "Epee"],
            "category": ["Senior", "Senior", "Senior"],
            "season": ["2020/2021", "2020/2021", "2020/2021"],
            "rank": [5, 5, 5],
            "points": [10.0, 10.0, 10.0],
        }
    )

    result = normalize_rankings(frame)

    assert set(result["weapon"]) == {"foil", "epee"}
    assert result["training_eligible"].sum() == 2
    assert result["exact_repeated_row"].sum() == 2


def test_tournament_keeps_and_flags_inverted_upstream_dates() -> None:
    frame = pd.DataFrame(
        {
            "competition_ID": [1],
            "season": [2018],
            "name": ["Example"],
            "category": ["Senior"],
            "country": ["EXAMPLE"],
            "start_date": ["2018-05-27"],
            "end_date": ["2017-05-27"],
            "weapon": ["Foil"],
            "gender": ["Womens"],
            "timezone": ["UTC"],
            "url": ["https://example.test/event"],
            "unique_ID": ["2018-1"],
            "missing_results_flag": [None],
        }
    )

    result = normalize_tournaments(frame)

    assert bool(result.loc[0, "upstream_date_order_valid"]) is False
    assert result.loc[0, "start_date"].year == 2018
    assert result.loc[0, "end_date"].year == 2017
