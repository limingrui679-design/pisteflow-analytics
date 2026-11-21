import numpy as np
import pandas as pd
import pytest

from pisteflow.analysis.boundary_review import evaluate_boundary_review, predict_boundary_crossings


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
