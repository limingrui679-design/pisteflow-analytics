from __future__ import annotations

from pathlib import Path

import pandas as pd
from pytest import MonkeyPatch

from pisteflow.provenance import sha256_file
from pisteflow.sources import fencingstatsnz


def test_prepare_fencingstatsnz_minimizes_identity_and_excludes_ties(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "bouts.csv"
    frame = pd.DataFrame(
        [
            [
                "2026-01-01", "Example", "foil", "pool", "Alpha", "Club A",
                "Beta", "Club B", 5, 3, None, "Mens", 10, 20, None,
            ],
            [
                "2026-01-01", "Example", "foil", "de", "Alpha", "Club A",
                "Gamma", "Club C", 0, 0, "Final", "Mens", None, 30, "wd_a",
            ],
        ],
        columns=fencingstatsnz.EXPECTED_COLUMNS,
    )
    frame.to_csv(source, index=False)
    monkeypatch.setattr(fencingstatsnz, "EXPECTED_BYTES", source.stat().st_size)
    monkeypatch.setattr(fencingstatsnz, "EXPECTED_SHA256", sha256_file(source))

    output = tmp_path / "bouts.parquet"
    summary = fencingstatsnz.prepare(source, output, tmp_path / "summary.json")
    result = pd.read_parquet(output)

    assert summary["retained_name_free_bouts"] == 2
    assert summary["model_eligible_decisive_bouts"] == 1
    assert summary["missing_uid_slots_recovered_by_unambiguous_name_crosswalk"] == 1
    assert set(result["result_status"]) == {"withdrawal_a", "decisive"}
    payload = output.read_bytes().decode("latin1", errors="ignore")
    assert "Alpha" not in payload
    assert "Club A" not in payload
