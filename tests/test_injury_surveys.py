import pandas as pd

from pisteflow.sources.injury_surveys import normalize_general_frame, normalize_hand_frame


def test_general_injuries_are_long_without_creating_controls() -> None:
    row: dict[str, object] = {
        "Age": 30,
        "Epee": "Epee",
        "Foil": "Foil",
        "Saber": None,
        "Gender": "Female",
        "Injured_While_Fencing": "Yes",
        "Number_of_Injuries": 2,
    }
    for number in range(1, 6):
        for field in [
            "Body_Part",
            "Body_Side",
            "Injury_Type",
            "Mechanism",
            "Weapon_Part",
            "CompvPrac",
            "Time_Off",
            "Treatments",
            "Recovery",
        ]:
            row[f"Injury_{number}_{field}"] = None
    row["Injury_1_Body_Part"] = "Ankle"
    row["Injury_2_Body_Part"] = "Wrist"

    participants, injuries = normalize_general_frame(pd.DataFrame([row]))

    assert participants.loc[0, "multi_weapon"]
    assert injuries["body_part"].tolist() == ["Ankle", "Wrist"]
    assert participants.loc[0, "Injured_While_Fencing"] == "Yes"


def test_hand_survey_omits_narratives_and_flags_empty_rows() -> None:
    columns = [
        "age ",
        "weapon",
        "Which Injury(ies) have you had to your hands? - Tendon Injury, "
        "please elaborate on which tendon(s) - Text",
        "Which Injury(ies) have you had to your hands? - Nerve Injury, "
        "please elaborate on which nerve(s) - Text",
        "Which Injury(ies) have you had to your hands? - Muscle injury, "
        "please elaborate on which muscle(s) - Text",
        "Which Injury(ies) have you had to your hands? - Ligament injury, "
        "please elaborate on which ligament(s) - Text",
        "Which Injury(ies) have you had to your hands? - Other injury not listed - Text",
        "Mechanism of Injury",
    ]
    frame = pd.DataFrame(
        [[17, "Sabre", "private", None, None, None, None, "private"], [None] * 8],
        columns=columns,
    )

    normalized = normalize_hand_frame(frame)

    assert "Mechanism of Injury" not in normalized
    assert normalized["empty_source_row"].tolist() == [False, True]
    assert normalized["age_below_18"].tolist() == [True, False]
