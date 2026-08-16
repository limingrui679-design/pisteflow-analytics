from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from pytest import MonkeyPatch

from pisteflow.provenance import sha256_file
from pisteflow.sources import hacs


def _archive(path: Path) -> None:
    payload = {
        "version": "v1.1",
        "taxonomy": [],
        "database": {
            "video-one": {
                "duration": "20",
                "subset": "training",
                "url": "https://www.youtube.com/watch?v=video-one",
                "annotations": [
                    {"label": "Doing fencing", "segment": [2, 8]},
                    {"label": "Painting fence", "segment": [10, 15]},
                ],
            }
        },
    }
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=["classname", "youtube_id", "subset", "start", "end", "label"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "classname": "Doing fencing",
            "youtube_id": "video-one",
            "subset": "training",
            "start": 3,
            "end": 5,
            "label": 1,
        }
    )
    writer.writerow(
        {
            "classname": "Doing fencing",
            "youtube_id": "video-two",
            "subset": "validation",
            "start": 7,
            "end": 9,
            "label": -1,
        }
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(hacs.SEGMENT_MEMBER, json.dumps(payload))
        archive.writestr(hacs.CLIP_MEMBER, stream.getvalue())


def _patch_expected(path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(hacs, "ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setattr(hacs, "ARCHIVE_SHA256", sha256_file(path))
    monkeypatch.setattr(hacs, "EXPECTED_SEGMENTS", 1)
    monkeypatch.setattr(hacs, "EXPECTED_SEGMENT_VIDEOS", 1)
    monkeypatch.setattr(hacs, "EXPECTED_POSITIVE_CLIPS", 1)
    monkeypatch.setattr(hacs, "EXPECTED_NEGATIVE_CLIPS", 1)
    monkeypatch.setattr(hacs, "EXPECTED_CLIP_VIDEOS", 2)
    monkeypatch.setattr(hacs, "EXPECTED_POSITIVE_CLIP_VIDEOS", 1)


def test_hacs_keeps_exact_fencing_label_and_hard_negatives(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = tmp_path / "hacs.zip"
    _archive(source)
    _patch_expected(source, monkeypatch)

    segments = hacs.parse_segments(source)
    clips = hacs.parse_clips(source)

    assert len(segments) == 1
    assert segments[0]["class_label"] == "Doing fencing"
    assert [row["is_positive"] for row in clips] == [True, False]
    assert {row["annotation_kind"] for row in clips} == {
        "sparse_positive_clip",
        "hard_negative_clip",
    }
    assert all(row["weapon"] == "unknown" for row in [*segments, *clips])
    assert all(not row["media_downloaded"] for row in [*segments, *clips])


def test_hacs_union_collapses_overlapping_clip_and_segment(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = tmp_path / "hacs.zip"
    _archive(source)
    _patch_expected(source, monkeypatch)
    monkeypatch.setattr(hacs, "EXPECTED_UNION_VIDEOS", 1)
    monkeypatch.setattr(hacs, "EXPECTED_UNION_INTERVALS", 1)
    monkeypatch.setattr(hacs, "EXPECTED_UNION_SECONDS", 6.0)

    summary = hacs.prepare(
        source,
        tmp_path / "segments.jsonl",
        tmp_path / "clips.jsonl",
        tmp_path / "summary.json",
    )

    assert summary["positive_union_source_videos"] == 1
    assert summary["positive_union_intervals"] == 1
    assert summary["positive_union_hours"] == round(6 / 3600, 6)
    assert summary["dense_segments_ending_after_reported_source_duration"] == 0
    assert (
        summary[
            "dense_segments_ending_more_than_0_05_seconds_after_reported_source_duration"
        ]
        == 0
    )
