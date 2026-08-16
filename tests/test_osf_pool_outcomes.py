from __future__ import annotations

import numpy as np
import pandas as pd

from pisteflow.analysis import osf_pool_outcomes


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    source_row = 0
    for year in range(2021, 2026):
        for competition in range(3):
            for bout in range(30):
                source_row += 1
                a_seed = 1 + bout % 15
                b_seed = 1 + (bout * 7) % 17
                probability_a = 1 / (1 + np.exp(-0.7 * (np.log1p(b_seed) - np.log1p(a_seed))))
                rows.append(
                    {
                        "source_row": source_row,
                        "year": year,
                        "competition_key": f"{year}:{competition}",
                        "athlete_a_key": f"a-{competition}-{bout % 10}",
                        "athlete_b_key": f"b-{competition}-{bout % 11}",
                        "A_seed": a_seed,
                        "B_seed": b_seed,
                        "weapon": ("epee", "foil", "sabre")[bout % 3],
                        "gender": ("F", "M")[bout % 2],
                        "outcome_eligible": True,
                        "athlete_a_won": bool(rng.random() < probability_a),
                    }
                )
    return pd.DataFrame(rows)


def test_pool_outcomes_chronological_competition_isolation() -> None:
    report, model = osf_pool_outcomes.evaluate_pool_outcomes(
        _frame(), validation_year=2024, test_start_year=2025, c_candidates=[0.1, 1.0]
    )

    assert report["competition_key_overlap"] == 0
    assert report["test_bouts"] == 90
    assert set(report["test"]["by_weapon"]) == {"epee", "foil", "sabre"}
    assert 0.3 < report["source_orientation_flipped_share"] < 0.7
    assert np.isfinite(report["test"]["model"]["log_loss"])
    seed_baseline = report["test"]["higher_seed_pick_nonprobabilistic"]
    assert "log_loss" not in seed_baseline
    assert seed_baseline["bouts"] + seed_baseline["excluded_same_seed_bouts"] == 90
    assert model


def test_pool_outcomes_rejects_duplicate_keys() -> None:
    frame = _frame()
    frame.loc[1, ["competition_key", "source_row"]] = frame.loc[
        0, ["competition_key", "source_row"]
    ]
    try:
        osf_pool_outcomes.prepare_bouts(frame)
    except ValueError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("Duplicate bout key was accepted")
