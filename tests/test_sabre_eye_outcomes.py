from __future__ import annotations

import numpy as np
import pandas as pd

from pisteflow.analysis import sabre_eye_outcomes


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(11)
    for participant in range(6):
        for touch in range(24):
            duration = 100 + 15 * touch + 10 * participant
            offensive = touch % 2
            won = rng.random() < (0.35 + 0.2 * offensive + 0.0002 * duration)
            rows.append(
                {
                    "participant_key": f"p-{participant}",
                    "recording_key": f"r-{participant}",
                    "touch_index_within_participant": touch + 1,
                    "quiet_eye_duration_ms": duration,
                    "won_touch": int(won),
                    "offensive_phase": offensive,
                    "fixation_count_numeric": 1 + touch % 3,
                    "location_count_numeric": 1 + touch % 2,
                    "left_eye_validity": "Valid" if touch % 5 else "Invalid",
                    "right_eye_validity": "Valid",
                    "supervised_touch_eligible": True,
                    "weapon": "sabre",
                }
            )
    return pd.DataFrame(rows)


def test_sabre_eye_loso_isolates_participants() -> None:
    report, model = sabre_eye_outcomes.evaluate_sabre_eye_outcomes(
        _frame(), c_candidates=[0.1, 1.0]
    )

    assert report["participants"] == 6
    assert len(report["nested_loso"]["folds"]) == 6
    assert all(
        fold["train_participants"] == 5 for fold in report["nested_loso"]["folds"]
    )
    assert np.isfinite(report["nested_loso"]["model"]["log_loss"])
    assert model


def test_sabre_eye_rejects_mixed_weapon_rows() -> None:
    frame = _frame()
    frame.loc[0, "weapon"] = "foil"
    try:
        sabre_eye_outcomes.prepare_touches(frame)
    except ValueError as exc:
        assert "sabre-only" in str(exc)
    else:
        raise AssertionError("Mixed-weapon eye table was accepted")
