from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pandas as pd
from pytest import MonkeyPatch

from pisteflow.provenance import sha256_file
from pisteflow.sources import hot_hand


def test_hot_hand_deduplicates_mirrored_rows_and_rebuilds_scores(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    values = [
        [
            "Feb NAC 2026", "pool_1", 1, "Alpha", 2, "Beta", 1, "pool", "V5",
            1, 5, 5, 0, 2000, 1800, 200, 1, 1, None, None,
        ],
        [
            "Feb NAC 2026", "pool_1", 2, "Beta", 1, "Alpha", 1, "pool", "D3",
            0, 3, 5, -2, 1800, 2000, -200, 0, 1, None, None,
        ],
    ]
    frame = pd.DataFrame(values, columns=hot_hand.EXPECTED_COLUMNS)
    csv_bytes = frame.to_csv(index=False).encode()
    archive_path = tmp_path / "source.tar.gz"
    member_name = f"snapshot/{hot_hand.CSV_SUFFIX}"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(csv_bytes)
        archive.addfile(info, io.BytesIO(csv_bytes))
    monkeypatch.setattr(hot_hand, "EXPECTED_BYTES", archive_path.stat().st_size)
    monkeypatch.setattr(hot_hand, "EXPECTED_SHA256", sha256_file(archive_path))

    output = tmp_path / "bouts.parquet"
    summary = hot_hand.prepare(archive_path, output, tmp_path / "summary.json")
    result = pd.read_parquet(output)

    assert summary["source_perspective_rows"] == 2
    assert summary["retained_unique_bouts"] == 1
    assert summary["upstream_winner_side_score_margin_rows_corrected"] == 1
    assert (result.loc[0, "score_a"], result.loc[0, "score_b"]) == (5, 3)
    assert result.loc[0, "winner_side"] == "a"
    payload = output.read_bytes().decode("latin1", errors="ignore")
    assert "Alpha" not in payload
    assert "Beta" not in payload
