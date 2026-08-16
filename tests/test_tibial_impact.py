import pandas as pd

from pisteflow.sources.tibial_impact import normalize_frame


def test_tibial_impact_creates_three_weapon_memberships_and_long_trials() -> None:
    row: dict[str, object] = {
        "Gender": "f",
        "Age": 17,
        "Height (m)": 1.7,
        "Weight (kg)": 60,
        "BMI": 20.8,
        "Weapon": "F,E,S",
        "Handedness": "L",
        "Years Fencing": 5,
        "Hours a week": 8,
    }
    for surface in "ABCDEF":
        row.update(
            {
                f"{surface}1": 9.0,
                f"{surface}2": 10.0,
                f"{surface}3": 11.0,
                f"{surface}_Av": 10.0,
            }
        )

    participants, trials, averages = normalize_frame(pd.DataFrame([row]))

    assert participants.loc[0, "weapon"] == "epee+foil+sabre"
    assert participants.loc[0, ["epee", "foil", "sabre"]].all()
    assert participants.loc[0, "age_below_18"]
    assert len(trials) == 18
    assert len(averages) == 6
    assert averages["reported_recomputed_difference_g"].eq(0).all()
