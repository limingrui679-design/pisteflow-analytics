from __future__ import annotations

import pandas as pd

from pisteflow.sources.physionet_fencer_cardio import normalize


def test_normalize_filters_fencers_removes_ids_and_derives_stages() -> None:
    subjects = pd.DataFrame(
        [
            [1, 15, 55.0, 165, 80, 140, "fencing"],
            [2, 16, 70.0, 175, 100, 180, "kayak"],
        ],
        columns=["ID", "age", "Weight", "Height", "P_vt1", "P_vt2", "sport"],
    )
    measurements = pd.DataFrame(
        [
            [1, -1.0, 800.0, 0.3, 0],
            [1, 1.0, 600.0, 1.2, 100],
            [1, 10.0, 900.0, 0.5, 0],
            [2, -1.0, None, 0.2, 0],
        ],
        columns=["ID", "time", "RR", "VO2", "power"],
    )

    profiles, series = normalize(subjects, measurements)

    assert len(profiles) == 1
    assert len(series) == 3
    assert profiles["age_band"].tolist() == ["14_15"]
    assert series["stage_derived"].tolist() == ["pre_exercise", "effort", "recovery"]
    assert "ID" not in profiles and "age" not in profiles and "ID" not in series
    assert profiles["participant_key"].iloc[0] == series["participant_key"].iloc[0]
