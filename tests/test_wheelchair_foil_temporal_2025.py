from __future__ import annotations

import pandas as pd

from pisteflow.sources import wheelchair_foil_temporal_2025 as source


def test_normalize_removes_order_and_maps_handedness(monkeypatch) -> None:
    monkeypatch.setattr(source, "EXPECTED_ROWS", 4)
    rows = []
    for order, category, handedness, minute in [
        (1, "A", "DD", "PER2M0"),
        (2, "A", "ZD", "PER2M1"),
        (3, "B", "ZZ", "PER2M2"),
        (4, "B", "DZ", "PER2M2"),
    ]:
        row = {column: 0.0 for column in source.EXPECTED_COLUMNS}
        row.update(
            {
                "Orden": float(order),
                "Cat_Discapa": category,
                "Mano": handedness,
                "Fase": "FIN",
                "MO_FIN_ASAL_PERIODO": "PER2",
                "MO_FIN_ASAL_PER_MINUT": minute,
                "Consume_tpo_asalto": "NO",
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows, columns=source.EXPECTED_COLUMNS)

    normalized = source.normalize(frame)

    assert "Orden" not in normalized
    assert normalized["handedness_matchup"].tolist() == [
        "right_vs_right",
        "left_vs_right",
        "left_vs_left",
        "right_vs_left",
    ]
    assert normalized["weapon"].eq("foil").all()
