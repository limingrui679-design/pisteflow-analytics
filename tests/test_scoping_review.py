import pandas as pd

from pisteflow.sources.scoping_review import _COLUMNS, normalize_frame


def test_scoping_review_is_labeled_as_catalog_metadata() -> None:
    row = {column: None for column in _COLUMNS}
    row.update(
        {
            "Title": "Example study",
            "Year": 2024,
            "N fencers": "4,652",
            "Weapon": "Saber",
            "Able Bodied": "Wheelchair",
        }
    )

    normalized = normalize_frame(pd.DataFrame([row]))

    assert normalized.loc[0, "study_code"] == "FSR-0001"
    assert normalized.loc[0, "weapon_normalized"] == "sabre"
    assert normalized.loc[0, "adaptive_scope"] == "wheelchair"
    assert normalized.loc[0, "reported_n_fencers_numeric"] == 4_652
