import pandas as pd

from pisteflow.sources.handedness_rankings import normalize_frame


def test_ranking_normalization_filters_sport_and_removes_names() -> None:
    frame = pd.DataFrame(
        [
            ["fencing", "female", "épée", 2023, 1, "Athlete One", "L"],
            ["fencing", "male", "sabre", 2023, 1, "Athlete Two", None],
            ["tennis", "female", None, 2023, 1, "Athlete Three", "R"],
        ],
        columns=["sport", "gender", "weapon", "year", "rank", "name", "hand"],
    )

    normalized = normalize_frame(frame)

    assert len(normalized) == 2
    assert "name" not in normalized
    assert normalized["weapon"].tolist() == ["epee", "sabre"]
    assert normalized["handedness"].tolist() == ["left", "unknown"]
    assert normalized["athlete_name_key"].str.startswith("FHR-").all()
