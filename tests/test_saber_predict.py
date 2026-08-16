from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pisteflow.sources.saber_predict import prepare


def test_prepare_saber_predict_exactly_joins_labels_and_groups(tmp_path: Path) -> None:
    source = tmp_path / "source"
    videos = source / "Dataset" / "Videos"
    (videos / "Clips").mkdir(parents=True)
    (videos / "Timestamps").mkdir()
    (videos / "manifest.csv").write_text(
        "id,event,name,url\n"
        "1,Event A,Bout A,https://www.youtube.com/watch?v=video-a\n"
        "2,Event B,Bout B,https://www.youtube.com/watch?v=video-b\n"
    )
    (videos / "Clips" / "metadata.csv").write_text(
        "source_id,clip_id,label\n"
        "1,1_Left.mp4,Left\n"
        "1,2_Right.mp4,Right\n"
    )
    (videos / "Timestamps" / "1.csv").write_text(
        "00:00:01.000,00:00:02.500,1_Left\n"
        "00:00:03.000,00:00:05.000,2_Right\n"
    )
    fie = tmp_path / "fie.jsonl"
    fie.write_text(
        json.dumps({"video_id": "youtube:video-a"}) + "\n"
        + json.dumps({"video_id": "youtube:video-b"})
        + "\n"
    )

    output = tmp_path / "rows.parquet"
    summary = prepare(
        source,
        output,
        tmp_path / "links.jsonl",
        tmp_path / "summary.json",
        fie_manifest_path=fie,
    )
    frame = pd.read_parquet(output)

    assert summary["timestamp_windows"] == 2
    assert summary["source_video_links"] == 2
    assert summary["catalog_videos_without_timestamps"] == 1
    assert summary["videos_already_in_fie_youtube_catalog"] == 2
    assert summary["new_video_groups_beyond_fie_youtube_catalog"] == 0
    assert summary["overlapping_windows_within_source"] == 0
    assert frame["group_id"].nunique() == 1
    assert set(frame["upstream_side_label"]) == {"left", "right"}
    assert frame["duration_seconds"].sum() == 3.5
