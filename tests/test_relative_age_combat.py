import pandas as pd

from pisteflow.sources import relative_age_combat as source


def test_normalize_frame_filters_and_removes_direct_identifiers() -> None:
    frame = pd.DataFrame(
        [
            [
                "ID-1",
                0,
                12.34,
                "12/10/2019",
                "01/06/2007",
                2,
                1,
                "U-13",
                "---",
                "Fencing",
                "Epee",
                3,
            ],
            [
                "ID-2",
                1,
                15.67,
                "13/10/2019",
                "01/02/2004",
                1,
                1,
                "U-17",
                "---",
                "Fencing",
                "Saber",
                11,
            ],
            [
                "ID-3",
                0,
                14.0,
                "31/12/2019",
                "01/01/2006",
                1,
                1,
                "U-14",
                "---",
                "Karate",
                "Kata",
                1,
            ],
        ],
        columns=source.SOURCE_COLUMNS,
    )

    result = source.normalize_frame(frame)

    assert len(result) == 2
    assert result["weapon"].tolist() == ["epee", "sabre"]
    assert result["sex_reported"].tolist() == ["male", "female"]
    assert result["age_years_rounded_0_1"].tolist() == [12.3, 15.7]
    assert result["national_ranking_position_2019"].tolist() == [3, 11]
    assert result["athlete_code"].str.startswith("raf-").all()
    assert "Birth Date" not in result
    assert "Valuation Date" not in result
    assert "ID - Participant" not in result
