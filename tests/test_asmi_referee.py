from __future__ import annotations

from pathlib import Path

import pytest

from pisteflow.sources.asmi_referee import parse_clip_path


def test_parse_asmi_referee_path_preserves_filename_group() -> None:
    root = Path("data")
    parsed = parse_clip_path(root / "simul" / "Fencing-S12-5 (30).mp4", root)

    assert parsed["label"] == "simultaneous"
    assert parsed["source_clip_number"] == 5
    assert str(parsed["source_group_id"]).startswith("asmi-source:")


def test_parse_asmi_referee_rejects_label_conflict() -> None:
    root = Path("data")
    with pytest.raises(ValueError, match="label conflict"):
        parse_clip_path(root / "left" / "Fencing-R12-5 (30).mp4", root)
