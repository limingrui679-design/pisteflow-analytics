# ruff: noqa: RUF001 - fixtures reproduce exact upstream workbook headers

import pandas as pd

from pisteflow.sources.personalized_strength import normalize_measurements, normalize_profiles


def test_profiles_remove_birth_dates_and_flag_minors() -> None:
    raw = pd.DataFrame(
        [
            [
                None,
                "Name",
                "Group",
                "Birth",
                "Age（year）",
                "Weight（kg）",
                "Height（cm）",
                "Years of sports experience（year）",
                "Sports Level",
                "Injury Side",
            ],
            [None] * 10,
            ["CG1", 1, 1, "2008.7.1", 14, 62, 182, 7, "Second Level", "left"],
        ]
    )

    normalized = normalize_profiles(raw)

    assert "Birth" not in normalized
    assert "Name" not in normalized
    assert normalized.loc[0, "group"] == "control"
    assert normalized.loc[0, "age_below_18"]


def test_measurements_preserve_duplicate_headers_without_merging() -> None:
    raw = pd.DataFrame(
        [
            [None, "Group", "Name", "Total Score（left）", "Total Score（left）"],
            [None] * 5,
            ["IG1", 3, 1, 90, 91],
        ]
    )

    normalized = normalize_measurements({"Y-BT（after)": raw})

    assert len(normalized) == 2
    assert normalized["upstream_duplicate_header"].all()
    assert normalized["value_numeric"].tolist() == [90, 91]
