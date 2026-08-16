from datetime import date, time

import pandas as pd

from pisteflow.sources.sleep_wake import normalize_frame


def test_sleep_normalization_flags_duplicate_dates_without_dropping_rows() -> None:
    p1 = [
        "P1",
        date(2022, 1, 1),
        date(2022, 1, 1),
        time(23),
        date(2022, 1, 2),
        time(7),
        100,
        200,
        80,
        10,
        0,
        380,
    ]
    p2 = [
        "P2",
        date(2022, 1, 2),
        date(2022, 1, 1),
        time(23),
        date(2022, 1, 2),
        time(7),
        90,
        210,
        70,
        10,
        20,
        390,
    ]
    rows = [p1, p1.copy(), p2]
    columns = [
        "ID",
        "date time",
        "Date of sleep",
        "sleep onset time",
        "Date of wake-up",
        "wake-up time",
        "deep sleep duration(min)",
        " light sleep duration(min)",
        "REM duration(min)",
        "awake duration(min)",
        "nap duration",
        "total sleep duration(min)",
    ]

    frame = normalize_frame(pd.DataFrame(rows, columns=columns))

    assert len(frame) == 3
    assert frame["exact_duplicate_group"].tolist() == [True, True, False]
    assert frame["participant_date_collision"].tolist() == [True, True, False]
    assert frame.loc[0, "component_total_difference_minutes"] == 10
    assert set(frame["weapon"]) == {"unknown"}
