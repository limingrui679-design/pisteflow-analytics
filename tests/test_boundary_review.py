import numpy as np
import pandas as pd
import pytest

from pisteflow.analysis.boundary_review import (
    evaluate_boundary_review,
    evaluate_origin_roster,
    predict_boundary_crossings,
)


def test_unlabelled_forecasts_unknown_weapon_and_first_origin():
    development = pd.DataFrame(
        {
            "rank": [16, 16, 18],
            "target_rank": [17, 15, 16],
            "weapon": ["foil"] * 3,
            "target_date": [pd.Timestamp("2019-12-01")] * 3,
        }
    )
    origins = pd.DataFrame(
        {
            "rank": [16, 17],
            "weapon": ["foil", "sabre"],
            "snapshot_date": [pd.Timestamp("2020-01-01")] * 2,
        }
    )
    probs = predict_boundary_crossings(development, origins)
    assert probs == pytest.approx([1 / 3, 2 / 3])
    labeled = origins.assign(target_rank=[17, 14])
    r = evaluate_boundary_review(development, labeled, capacity_fraction=0.5)
    assert r["review_slots"] == 1
    assert r["crossings"] == 2
    assert r["risk_brier"] == pytest.approx(((2 / 3) ** 2 + (1 / 3) ** 2) / 2)
    assert r["risk_expected_hits"] == r["proximity_expected_hits"] == r["random_expected_hits"] == 1
    with pytest.raises(ValueError, match="precede"):
        predict_boundary_crossings(
            development, origins.assign(snapshot_date=pd.Timestamp("2019-12-01"))
        )


def test_tie_expectation_and_row_order():
    dates = pd.to_datetime(["2019-12-01"] * 3)
    train = pd.DataFrame(
        {
            "rank": [16] * 3,
            "target_rank": [17, 16, 16],
            "weapon": ["foil"] * 3,
            "target_date": dates,
        }
    )
    test = train.assign(snapshot_date=pd.Timestamp("2020-01-01"))
    first = evaluate_boundary_review(train, test)
    reversed_rows = evaluate_boundary_review(train, test.iloc[::-1])
    assert first == reversed_rows
    assert first["review_slots"] == 1
    assert first["risk_expected_hits"] == first["proximity_expected_hits"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="integer"):
        evaluate_boundary_review(train, test.assign(target_rank=np.nan))
    with pytest.raises(ValueError, match="Capacity"):
        evaluate_boundary_review(train, test, capacity_fraction=0)


def test_blind_roster_retains_disappearing_series_and_bounds_shared_unknown_outcomes():
    day = pd.Timestamp("2020-01-01")
    train = pd.DataFrame(
        {
            "rank": [17, 18],
            "target_rank": [15, 16],
            "weapon": "foil",
            "target_date": day - pd.Timedelta(days=1),
        }
    )
    frame = pd.DataFrame(
        {
            "athlete_series_id": ["disappeared", "returned", "archive-end", "other"],
            "snapshot_date": [day, day, day + pd.Timedelta(days=180), day],
            "rank": [17, 18, 5, 2],
            "weapon": "foil",
            "gender": "women",
            "age_category": ["senior"] * 3 + ["junior"],
            "training_eligible": True,
        }
    )
    observed = pd.DataFrame(
        {"athlete_series_id": ["returned"], "snapshot_date": [day], "target_rank": [16]}
    )

    def replay(data, outcomes, start):
        return evaluate_origin_roster(
            data, train, outcomes, cutoff=start, max_horizon_days=180, capacity_fraction=0.5
        )

    result = replay(frame, observed, day)
    assert result["origins"] == 2
    assert result["review_slots"] == result["dates"] == result["unobserved_outcomes"] == 1
    assert result["expected_hit_bounds"]["risk"] == [0.5, 1.0]
    assert result["expected_hit_bounds"]["proximity"] == [0.0, 1.0]
    assert result["paired_gain_bounds_vs_proximity"] == [0.0, 0.5]
    # Revealing either possible outcome must attain one of the paired sharp bounds.
    for rank, gain in [(16, 0.0), (17, 0.5)]:
        completed = pd.concat(
            [observed, observed.assign(athlete_series_id="disappeared", target_rank=rank)]
        )
        assert replay(frame.iloc[::-1], completed, day)["paired_gain_bounds_vs_proximity"] == [
            gain,
            gain,
        ]
    assert replay(frame, observed, day + pd.Timedelta(days=1)) == {
        "status": "no_mature_origins",
        "origins": 0,
    }
