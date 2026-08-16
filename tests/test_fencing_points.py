import pandas as pd

from pisteflow.sources.fencing_points import normalize


def test_normalize_removes_identity_and_exact_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "fie_ID": [1, 1],
            "fencer_name": ["Example", "Example"],
            "age": [92, 92],
            "dob": ["1930-01-01", "1930-01-01"],
            "nationality": ["AAA", "AAA"],
            "rank": [4, 4],
            "host_city": ["Paris", "Paris"],
            "event_type": ["World Cup", "World Cup"],
            "points_earned": [16.0, 16.0],
            "date": ["2024-01-01", "2024-01-01"],
        }
    )

    result, duplicates = normalize(frame)

    assert duplicates == 1
    assert len(result) == 1
    assert "fencer_name" not in result.columns
    assert result.loc[0, "athlete_id"] != "1"
    assert bool(result.loc[0, "age_plausible"]) is False
