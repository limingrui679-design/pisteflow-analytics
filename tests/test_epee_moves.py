import json
from pathlib import Path

import pandas as pd

from pisteflow.sources.epee_moves import prepare


def test_prepare_epee_metadata_preserves_source_groups(tmp_path: Path) -> None:
    annotations = {
        "fl_001": {
            "class": "fleche",
            "ranka": 1,
            "taikinio_zona": 3,
            "fencer": "left",
            "reversed": False,
            "source_video": "104.mp4",
            "start_frame": 2,
            "end_frame": 5,
        },
        "ld_001": {
            "class": "lunge_direct",
            "ranka": 0,
            "taikinio_zona": 1,
            "fencer": "right",
            "reversed": True,
            "source_video": "104.mp4",
            "start_frame": 9,
            "end_frame": 12,
        },
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    output = tmp_path / "clips.parquet"

    summary = prepare(annotations_path, output, tmp_path / "summary.json")
    frame = pd.read_parquet(output)

    assert summary["annotated_clips"] == 2
    assert summary["source_video_groups"] == 1
    assert summary["media_audit"]["metadata_only"] is True
    assert frame["source_group_id"].nunique() == 1
    assert frame["source_frame_count_inclusive"].tolist() == [4, 4]
