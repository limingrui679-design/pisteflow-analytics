from __future__ import annotations

import pandas as pd

from pisteflow.sources import frontiers_handedness_2021 as source


def _ranking_frame() -> pd.DataFrame:
    rows = [
        [
            "fencing",
            "junior",
            "female",
            "épée",
            "2015/2016",
            2016,
            1,
            120,
            7,
            "USA",
            "United States",
            0,
            "2000-06-04",
            "L",
            6,
            2,
            6,
            2,
        ],
        [
            "fencing",
            "senior",
            "male",
            "sabre",
            "2015/2016",
            2016,
            1,
            110,
            8,
            "ITA",
            "Italy",
            0,
            None,
            None,
            1,
            1,
            1,
            1,
        ],
        [
            "tennis",
            "senior",
            "male",
            None,
            "2015/2016",
            2016,
            1,
            100,
            9,
            "GBR",
            "United Kingdom",
            1,
            "1990-01-01",
            "R",
            1,
            1,
            1,
            1,
        ],
    ]
    return pd.DataFrame(rows, columns=source._RANKING_COLUMNS)


def _athlete_frame() -> pd.DataFrame:
    rows: list[list[object]] = []
    for raw_id, weapon, age_level, flags in [
        (7, "épée", "junior", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        (8, "sabre", "senior", [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
    ]:
        rows.append(
            [
                "fencing",
                age_level,
                "female" if raw_id == 7 else "male",
                weapon,
                raw_id,
                "USA" if raw_id == 7 else "ITA",
                "United States" if raw_id == 7 else "Italy",
                0,
                "2000-06-04" if raw_id == 7 else None,
                "L" if raw_id == 7 else None,
                6 if raw_id == 7 else 1,
                2 if raw_id == 7 else 1,
                6 if raw_id == 7 else 1,
                2 if raw_id == 7 else 1,
                *flags,
                sum(flags),
            ]
        )
    return pd.DataFrame(rows, columns=source._ATHLETE_COLUMNS)


def test_normalize_rankings_removes_direct_identity_and_exact_birth_date() -> None:
    normalized = source.normalize_rankings(_ranking_frame())

    assert len(normalized) == 2
    assert normalized["weapon"].tolist() == ["epee", "sabre"]
    assert normalized["handedness"].tolist() == ["left", "unknown"]
    assert normalized["athlete_key"].str.startswith("FRA-").all()
    assert "athlete_ID" not in normalized
    assert "DOB" not in normalized
    assert "nation_full" not in normalized
    assert normalized["birth_month_january_cutoff"].tolist() == [6, 1]


def test_normalize_athlete_memberships_preserves_annual_flags() -> None:
    normalized = source.normalize_athlete_memberships(_athlete_frame())

    assert len(normalized) == 2
    assert normalized["all_years"].tolist() == [1, 2]
    assert normalized[["listed_2015", "listed_2016"]].values.tolist() == [[0, 1], [1, 1]]
    assert "athlete_ID" not in normalized
    assert "DOB" not in normalized


def test_athlete_memberships_reject_inconsistent_all_years() -> None:
    frame = _athlete_frame()
    frame.loc[0, "all_years"] = 2
    try:
        source.normalize_athlete_memberships(frame)
    except ValueError as exc:
        assert "all_years" in str(exc)
    else:
        raise AssertionError("Inconsistent annual membership total was accepted")
