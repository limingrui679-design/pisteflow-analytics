from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pisteflow.sources.fencing_ref_metadata import prepare


def _row(video_id: str, label: str, start_frame: int, split: str) -> dict[str, object]:
    return {
        "annotations": {"label": label, "segment": [start_frame / 25, start_frame / 25 + 4]},
        "annotations_frame": {
            "label": label,
            "segment": [start_frame, start_frame + 100],
        },
        "duration": 4.0,
        "subset": split,
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


def test_prepare_fencing_ref_flags_source_group_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "dataset").mkdir(parents=True)
    (source / "per_vid_labels").mkdir()
    train = {"video-a_0": _row("video-a", "left", 100, "train")}
    valid = {"video-a_1": _row("video-a", "right", 300, "valid")}
    (source / "dataset" / "train.json").write_text(json.dumps(train))
    (source / "dataset" / "valid.json").write_text(json.dumps(valid))
    (source / "per_vid_labels" / "video-a.json").write_text(json.dumps({**train, **valid}))

    output = tmp_path / "rows.parquet"
    summary = prepare(
        source,
        output,
        tmp_path / "links.jsonl",
        tmp_path / "summary.json",
    )
    frame = pd.read_parquet(output)

    assert len(frame) == 2
    assert summary["train_valid_source_video_overlap"] == 1
    assert summary["valid_videos_already_seen_in_train_fraction"] == 1.0
    assert summary["source_video_union_with_fencing_matches_ai"] == 1
    assert frame["group_id"].nunique() == 1
