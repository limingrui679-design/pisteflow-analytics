from pathlib import Path

import pandas as pd

from pisteflow.sources import peerj_strength


def _write_workbook(path: Path, columns: dict[str, str], rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={normalized: upstream for upstream, normalized in columns.items()})
    frame.to_excel(path, index=False)


def test_normalize_primary_preserves_three_weapon_values(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    weapon_rows = [("épée", 17), ("foil", 12), ("saber", 16)]
    counter = 0
    for weapon, count in weapon_rows:
        for _ in range(count):
            counter += 1
            row = {column: float(counter) for column in peerj_strength._PRIMARY_COLUMNS.values()}
            row["participant_code"] = f"A{counter}"
            row["weapon"] = weapon
            rows.append(row)
    path = tmp_path / "primary.xlsx"
    _write_workbook(path, peerj_strength._PRIMARY_COLUMNS, rows)

    frame = peerj_strength.normalize_primary(path)

    assert len(frame) == 45
    assert frame["weapon"].value_counts().to_dict() == {"epee": 17, "sabre": 16, "foil": 12}
    assert frame.loc[0, "static_lunge_velocity_m_s"] == 1.0


def test_normalize_reliability_keeps_repeated_trials_separate(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    weapon_codes = [("épée", 13), ("foil", 12), ("saber", 15)]
    counter = 0
    for weapon, count in weapon_codes:
        for _ in range(count):
            counter += 1
            for test_number in (1, 2, 3):
                row = {
                    column: float(counter * 10 + test_number)
                    for column in peerj_strength._RELIABILITY_COLUMNS.values()
                }
                row["participant_code"] = f"R{counter}"
                row["weapon"] = weapon
                row["test_number"] = test_number
                rows.append(row)
    path = tmp_path / "reliability.xlsx"
    _write_workbook(path, peerj_strength._RELIABILITY_COLUMNS, rows)

    frame = peerj_strength.normalize_reliability(path)

    assert len(frame) == 120
    assert frame["participant_code"].nunique() == 40
    assert frame.groupby(["weapon", "participant_code"]).size().eq(3).all()
