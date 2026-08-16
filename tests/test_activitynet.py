from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from pisteflow.provenance import sha256_file
from pisteflow.sources import activitynet


def _annotation(path: Path) -> None:
    payload = {
        "version": "VERSION 1.3",
        "taxonomy": [],
        "database": {
            "video-a": {
                "duration": 12.0,
                "subset": "training",
                "url": "https://www.youtube.com/watch?v=video-a",
                "annotations": [
                    {"label": "Doing fencing", "segment": [5.0, 9.0]},
                    {"label": "Painting fence", "segment": [0.0, 4.0]},
                    {"label": "Doing fencing", "segment": [1.0, 3.0]},
                ],
            },
            "video-b": {
                "duration": 20.0,
                "subset": "validation",
                "url": "https://www.youtube.com/watch?v=video-b",
                "annotations": [{"label": "Doing fencing", "segment": [2.5, 8.5]}],
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_activitynet_keeps_source_video_groups_and_exact_label(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "activitynet.json"
    _annotation(source)
    monkeypatch.setattr(activitynet, "EXPECTED_BYTES", source.stat().st_size)
    monkeypatch.setattr(activitynet, "EXPECTED_SHA256", sha256_file(source))
    monkeypatch.setattr(activitynet, "EXPECTED_INTERVALS", 3)
    monkeypatch.setattr(activitynet, "EXPECTED_VIDEOS", 2)

    rows = activitynet.parse_annotations(source)

    assert [row["start_seconds"] for row in rows if row["youtube_id"] == "video-a"] == [
        1.0,
        5.0,
    ]
    assert {row["class_label"] for row in rows} == {"Doing fencing"}
    assert len({row["source_group_id"] for row in rows}) == 2
    assert all(row["weapon"] == "unknown" for row in rows)
    assert all(not row["media_downloaded"] for row in rows)


def test_activitynet_reports_cross_source_video_overlap(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "activitynet.json"
    _annotation(source)
    monkeypatch.setattr(activitynet, "EXPECTED_BYTES", source.stat().st_size)
    monkeypatch.setattr(activitynet, "EXPECTED_SHA256", sha256_file(source))
    monkeypatch.setattr(activitynet, "EXPECTED_INTERVALS", 3)
    monkeypatch.setattr(activitynet, "EXPECTED_VIDEOS", 2)
    comparison = tmp_path / "comparison.jsonl"
    comparison.write_text('{"video_id":"youtube:video-b"}\n', encoding="utf-8")

    summary = activitynet.prepare(
        source,
        tmp_path / "manifest.jsonl",
        tmp_path / "summary.json",
        comparison_manifests={"example": comparison},
    )

    assert summary["intervals"] == 3
    assert summary["unique_source_videos"] == 2
    assert summary["cross_source_video_overlap"] == {"example": 1}
