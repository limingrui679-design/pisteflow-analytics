import pandas as pd

from pisteflow.sources.wheelchair_oxygen import (
    normalize_f3_competition,
    normalize_f7_competition,
)


def test_f3_bout_state_is_boolean_and_epee_is_explicit() -> None:
    frame = pd.DataFrame(
        {
            "Fencer": ["F3", "F3"],
            "Crono_hms": ["00:00:00", "00:00:01"],
            "Clock_time": ["15:06:10", "15:06:11"],
            "HR_bpm": [107, 108],
            "VO2est_mL_kg_min": [13.4, 13.7],
            "Bout": ["No", "Yes"],
            "Phase": ["POULE", "POULE"],
        }
    )

    normalized = normalize_f3_competition(frame)

    assert normalized["in_bout"].tolist() == [False, True]
    assert set(normalized["weapon"]) == {"epee"}


def test_f7_limited_schema_is_not_augmented_with_fake_heart_rate() -> None:
    frame = pd.DataFrame(
        {"ID": ["F7"], "sec": [0], "vo2": [4.8343], "bout": ["No"]}
    )

    normalized = normalize_f7_competition(frame)

    assert "heart_rate_bpm" not in normalized
    assert normalized.loc[0, "participant_code"] == "F7"
