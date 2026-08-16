from __future__ import annotations

import numpy as np
import pandas as pd

from pisteflow.analysis import strength_velocity


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    for index in range(36):
        weapon = ("epee", "foil", "sabre")[index % 3]
        strength = 10.0 + index / 10
        rows.append(
            {
                "participant_code": f"p-{index:02d}",
                "weapon": weapon,
                "age_years": 20 + index % 5,
                "training_years": 5 + index % 6,
                "imtp_relative_peak_force_n_kg": strength,
                "sj_jump_height_m": 0.25 + index / 1000,
                "static_lunge_velocity_m_s": 1.5 + 0.04 * strength + rng.normal(0, 0.02),
                "advance_lunge_velocity_m_s": 1.8 + 0.05 * strength + rng.normal(0, 0.02),
                "source_doi": "example",
                "population": "test",
            }
        )
    return pd.DataFrame(rows)


def test_strength_velocity_nested_evaluation() -> None:
    report, models, features = strength_velocity.evaluate_strength_velocity(
        _frame(), n_splits=3
    )

    assert report["participants"] == 36
    assert report["by_weapon"] == {"epee": 12, "foil": 12, "sabre": 12}
    assert set(models) == set(strength_velocity.TARGETS)
    assert "static_lunge_velocity_m_s" not in features
    assert "advance_lunge_velocity_m_s" not in features
    for target in strength_velocity.TARGETS:
        metrics = report["targets"][target]["nested_cross_validation"]
        assert np.isfinite(metrics["mae"])
        assert len(report["targets"][target]["folds"]) == 3


def test_strength_velocity_rejects_duplicate_participant() -> None:
    frame = _frame()
    frame.loc[1, "participant_code"] = frame.loc[0, "participant_code"]

    try:
        strength_velocity.evaluate_strength_velocity(frame, n_splits=3)
    except ValueError as exc:
        assert "one row per participant" in str(exc)
    else:
        raise AssertionError("Duplicate participant was accepted")
