import pandas as pd

from pisteflow.sources.wheelchair_foil_temporal import normalize_frame


def test_temporal_normalization_keeps_source_and_adds_bout_fields() -> None:
    frame = pd.DataFrame(
        {
            "Orden": [1],
            "Cat_Discapa": ["A"],
            "Mano": ["DD"],
            "Fase": ["FIN"],
            "MO_FIN_ASAL_PERIODO": ["PER2"],
            "MO_FIN_ASAL_PER_MINUT": ["PER2M1"],
            "Consume_tpo_asalto": ["NO"],
            "TTASAL": [100],
            "TTACC": [30],
            "TTPAU": [70],
            "TTDES": [60],
            "STACC": [10],
            "STPAU": [10],
            "STDES": [1],
            "TSACC": [3.0],
            "TSPAU": [7.0],
            "TSDES": [60.0],
        }
    )

    normalized = normalize_frame(frame)

    assert normalized.loc[0, "bout_code"] == "WFF-001"
    assert normalized.loc[0, "total_equals_action_plus_pause"]
    assert normalized.loc[0, "weapon"] == "foil"
    assert "TTACC" in normalized
