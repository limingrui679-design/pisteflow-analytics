from __future__ import annotations

import pandas as pd

from pisteflow.sources import youth_athletes


def _row(label: str) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in youth_athletes._SOURCE_COLUMNS}
    row.update(
        {
            "Sport": label,
            "Sport_Number": "2",
            "Sex": "0",
            "Test_age_T1": "180",
            "Body_height_cm_T1": "175.5",
            "Eye_hand_reaction_time": "205",
            "Back_strength_kg_T1": "101",
            "Urine_protein_T1": "-1",
            "Menarche_yr_mth_T1": "sensitive",
        }
    )
    return row


def test_normalize_frame_filters_and_minimizes() -> None:
    frame = pd.DataFrame([_row("basketball-1"), _row("fencing-1"), _row("fencing-2")])

    result = youth_athletes.normalize_frame(frame)

    assert len(result) == 2
    assert result["participant_code"].nunique() == 2
    assert result["age_months"].tolist() == [180, 180]
    assert result["body_height_cm"].tolist() == [175.5, 175.5]
    assert "Sport" not in result
    assert "Sex" not in result
    assert "urine_protein" not in " ".join(result.columns)
    assert "menarche" not in " ".join(result.columns)


def test_normalize_frame_rejects_schema_change() -> None:
    frame = pd.DataFrame([_row("fencing-1")]).drop(columns=["Sex"])

    try:
        youth_athletes.normalize_frame(frame)
    except ValueError as exc:
        assert "columns changed" in str(exc)
    else:
        raise AssertionError("Schema change was accepted")
