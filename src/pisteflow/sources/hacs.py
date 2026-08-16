"""Metadata-only fencing clips and temporal segments from HACS v1.1.1."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pisteflow.provenance import read_jsonl, sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

DATASET_PAGE = "https://github.com/hangzhaomit/HACS-dataset"
SNAPSHOT = "ca100f92392bfbeaf8c3dc753988dc805c7cab54"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/hangzhaomit/HACS-dataset/"
    f"{SNAPSHOT}/HACS_v1.1.1.zip"
)
ARCHIVE_BYTES = 16_183_542
ARCHIVE_SHA256 = "a7bea563c598b4b96c7958f156a1fbae3e93ef642bf8473f378358cb2c6a2cbc"
SEGMENT_MEMBER = "HACS_v1.1.1/HACS_segments_v1.1.1.json"
CLIP_MEMBER = "HACS_v1.1.1/HACS_clips_v1.1.1.csv"
TARGET_LABEL = "Doing fencing"

EXPECTED_SEGMENTS = 1_458
EXPECTED_SEGMENT_VIDEOS = 257
EXPECTED_POSITIVE_CLIPS = 5_690
EXPECTED_NEGATIVE_CLIPS = 1_701
EXPECTED_CLIP_VIDEOS = 2_521
EXPECTED_POSITIVE_CLIP_VIDEOS = 2_242
EXPECTED_UNION_VIDEOS = 2_243
EXPECTED_UNION_INTERVALS = 6_457
EXPECTED_UNION_SECONDS = 34_521.53


def acquire(destination: Path, *, user_agent: str, force: bool = False) -> Path:
    """Acquire the byte-pinned HACS annotation bundle, never linked media."""

    if force and destination.exists():
        destination.unlink()
    download_resumable(
        ARCHIVE_URL,
        destination,
        user_agent=user_agent,
        expected_bytes=ARCHIVE_BYTES,
    )
    actual = sha256_file(destination)
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"HACS archive checksum changed: {actual}")
    return destination


def _validated_zip(archive_path: Path) -> zipfile.ZipFile:
    if archive_path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("HACS archive byte size changed")
    actual = sha256_file(archive_path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"HACS archive checksum changed: {actual}")
    archive = zipfile.ZipFile(archive_path)
    missing = {SEGMENT_MEMBER, CLIP_MEMBER}.difference(archive.namelist())
    if missing:
        archive.close()
        raise ValueError(f"HACS archive members missing: {sorted(missing)}")
    if archive.testzip() is not None:
        archive.close()
        raise ValueError("HACS archive CRC validation failed")
    return archive


def parse_segments(archive_path: Path) -> list[dict[str, Any]]:
    """Parse exact ``Doing fencing`` temporal segments."""

    with _validated_zip(archive_path) as archive:
        payload = json.loads(archive.read(SEGMENT_MEMBER))
    if payload.get("version") != "v1.1" or not isinstance(payload.get("database"), dict):
        raise ValueError("HACS segment schema or version changed")

    rows: list[dict[str, Any]] = []
    for youtube_id, video in payload["database"].items():
        if not isinstance(video, dict):
            raise ValueError("HACS segment video record is malformed")
        subset = str(video.get("subset"))
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(f"HACS segment subset changed: {subset}")
        duration = float(video.get("duration", 0.0))
        if duration <= 0:
            raise ValueError("HACS segment source duration is invalid")
        annotations = video.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("HACS segment annotations are malformed")
        selected: list[tuple[float, float]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict) or annotation.get("label") != TARGET_LABEL:
                continue
            segment = annotation.get("segment")
            if not isinstance(segment, list) or len(segment) != 2:
                raise ValueError("HACS fencing segment is malformed")
            start, end = float(segment[0]), float(segment[1])
            if start < 0 or end <= start:
                raise ValueError("HACS fencing segment interval is invalid")
            selected.append((start, end))
        for interval_index, (start, end) in enumerate(sorted(selected), start=1):
            rows.append(
                {
                    "record_id": f"hacs-segment:{youtube_id}:{interval_index:03d}",
                    "annotation_kind": "dense_positive_segment",
                    "class_label": TARGET_LABEL,
                    "is_positive": True,
                    "youtube_id": str(youtube_id),
                    "source_group_id": f"youtube:{youtube_id}",
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": round(end - start, 6),
                    "source_video_duration_seconds": duration,
                    "end_exceeds_source_duration_seconds": round(max(0.0, end - duration), 6),
                    "split": subset,
                    "source_url": str(
                        video.get("url")
                        or f"https://www.youtube.com/watch?v={youtube_id}"
                    ),
                    "weapon": "unknown",
                    "media_downloaded": False,
                    "annotation_license": "BSD-3-Clause",
                    "media_license_status": "link_only_video_license_not_asserted",
                }
            )
    rows.sort(key=lambda row: str(row["record_id"]))
    if len(rows) != EXPECTED_SEGMENTS:
        raise ValueError(f"HACS fencing segment count changed: {len(rows)}")
    if len({str(row["youtube_id"]) for row in rows}) != EXPECTED_SEGMENT_VIDEOS:
        raise ValueError("HACS fencing segment source-video count changed")
    return rows


def parse_clips(archive_path: Path) -> list[dict[str, Any]]:
    """Parse exact HACS fencing positive and hard-negative 2-second clips."""

    with _validated_zip(archive_path) as archive:
        stream = io.TextIOWrapper(archive.open(CLIP_MEMBER), encoding="utf-8")
        source_rows = list(csv.DictReader(stream))

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        if source.get("classname") != TARGET_LABEL:
            continue
        label = str(source.get("label"))
        if label not in {"1", "-1"}:
            raise ValueError(f"HACS fencing clip label changed: {label}")
        subset = str(source.get("subset"))
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(f"HACS fencing clip subset changed: {subset}")
        start, end = float(source["start"]), float(source["end"])
        if start < 0 or end <= start:
            raise ValueError("HACS fencing clip interval is invalid")
        youtube_id = str(source["youtube_id"])
        rows.append(
            {
                "record_id": f"hacs-clip:{youtube_id}:{start:.3f}:{end:.3f}",
                "annotation_kind": "sparse_positive_clip" if label == "1" else "hard_negative_clip",
                "class_label": TARGET_LABEL,
                "is_positive": label == "1",
                "upstream_label": int(label),
                "youtube_id": youtube_id,
                "source_group_id": f"youtube:{youtube_id}",
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": round(end - start, 6),
                "split": subset,
                "source_url": f"https://www.youtube.com/watch?v={youtube_id}",
                "weapon": "unknown",
                "media_downloaded": False,
                "annotation_license": "BSD-3-Clause",
                "media_license_status": "link_only_video_license_not_asserted",
            }
        )
    rows.sort(key=lambda row: str(row["record_id"]))
    positives = [row for row in rows if bool(row["is_positive"])]
    negatives = [row for row in rows if not bool(row["is_positive"])]
    if len(positives) != EXPECTED_POSITIVE_CLIPS:
        raise ValueError(f"HACS positive fencing clip count changed: {len(positives)}")
    if len(negatives) != EXPECTED_NEGATIVE_CLIPS:
        raise ValueError(f"HACS negative fencing clip count changed: {len(negatives)}")
    if len({str(row["youtube_id"]) for row in rows}) != EXPECTED_CLIP_VIDEOS:
        raise ValueError("HACS fencing clip source-video count changed")
    if len({str(row["youtube_id"]) for row in positives}) != EXPECTED_POSITIVE_CLIP_VIDEOS:
        raise ValueError("HACS positive fencing clip source-video count changed")
    return rows


def _positive_union(
    segments: list[dict[str, Any]], clips: list[dict[str, Any]]
) -> tuple[int, int, float]:
    intervals_by_video: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in [*segments, *clips]:
        if bool(row["is_positive"]):
            intervals_by_video[str(row["youtube_id"])].append(
                (float(row["start_seconds"]), float(row["end_seconds"]))
            )
    interval_count = 0
    seconds = 0.0
    for intervals in intervals_by_video.values():
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        interval_count += len(merged)
        seconds += sum(end - start for start, end in merged)
    return len(intervals_by_video), interval_count, round(seconds, 6)


def _manifest_youtube_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        for key in ("youtube_id", "video_id", "source_group_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                if value.startswith("youtube:"):
                    value = value.removeprefix("youtube:")
                if len(value) == 11:
                    ids.add(value)
    return ids


def prepare(
    archive_path: Path,
    segment_output: Path,
    clip_output: Path,
    report_path: Path,
    *,
    comparison_manifests: dict[str, Path] | None = None,
) -> dict[str, Any]:
    segments = parse_segments(archive_path)
    clips = parse_clips(archive_path)
    segment_sha256 = write_jsonl(segment_output, segments)
    clip_sha256 = write_jsonl(clip_output, clips)

    union_videos, union_intervals, union_seconds = _positive_union(segments, clips)
    if union_videos != EXPECTED_UNION_VIDEOS:
        raise ValueError(f"HACS positive union source-video count changed: {union_videos}")
    if union_intervals != EXPECTED_UNION_INTERVALS:
        raise ValueError(f"HACS positive union interval count changed: {union_intervals}")
    if abs(union_seconds - EXPECTED_UNION_SECONDS) > 0.001:
        raise ValueError(f"HACS positive union duration changed: {union_seconds}")

    all_video_ids = {
        str(row["youtube_id"]) for row in [*segments, *clips] if bool(row["is_positive"])
    }
    overlaps = {
        name: len(all_video_ids.intersection(_manifest_youtube_ids(path))) if path.exists() else 0
        for name, path in sorted((comparison_manifests or {}).items())
    }
    clip_labels = Counter("positive" if row["is_positive"] else "hard_negative" for row in clips)
    summary: dict[str, Any] = {
        "dataset_page": DATASET_PAGE,
        "snapshot": SNAPSHOT,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "annotation_license": "BSD-3-Clause",
        "media_downloaded": False,
        "media_license_status": "linked YouTube media rights are not asserted by HACS",
        "dense_positive_segments": len(segments),
        "dense_segment_source_videos": len({str(row["youtube_id"]) for row in segments}),
        "dense_positive_segment_hours": round(
            sum(float(row["duration_seconds"]) for row in segments) / 3600, 6
        ),
        "dense_segments_ending_after_reported_source_duration": sum(
            float(row["end_exceeds_source_duration_seconds"]) > 0 for row in segments
        ),
        "dense_segments_ending_more_than_0_05_seconds_after_reported_source_duration": sum(
            float(row["end_exceeds_source_duration_seconds"]) > 0.05 for row in segments
        ),
        "maximum_end_overrun_seconds": max(
            float(row["end_exceeds_source_duration_seconds"]) for row in segments
        ),
        "sparse_clips": len(clips),
        "sparse_clip_labels": dict(sorted(clip_labels.items())),
        "sparse_clip_source_videos": len({str(row["youtube_id"]) for row in clips}),
        "positive_sparse_clip_source_videos": len(
            {str(row["youtube_id"]) for row in clips if bool(row["is_positive"])}
        ),
        "positive_sparse_clip_hours": round(
            sum(float(row["duration_seconds"]) for row in clips if bool(row["is_positive"]))
            / 3600,
            6,
        ),
        "positive_union_source_videos": union_videos,
        "positive_union_intervals": union_intervals,
        "positive_union_hours": round(union_seconds / 3600, 6),
        "cross_source_video_overlap": overlaps,
        "weapon_label": "unknown; HACS labels the activity but not epee, foil, or sabre",
        "grouping_rule": "hold out complete source_group_id values (one YouTube video)",
        "outputs": {
            "segments": "data/manifests/hacs-fencing-segments.jsonl",
            "clips": "data/manifests/hacs-fencing-clips.jsonl",
        },
        "output_sha256": {"segments": segment_sha256, "clips": clip_sha256},
        "claim_boundary": (
            "HACS v1.1.1 contributes BSD-licensed annotation metadata for real public-video "
            "fencing intervals. The 1,458 dense segments and 5,690 positive sparse clips "
            "overlap, so only the merged 6,457 intervals across 2,243 source videos and 9.589 "
            "hours may be used as a union total. Thirteen dense segments exceed the upstream "
            "reported source duration by some positive amount; eight exceed it by more than "
            "0.05 seconds (0.06-0.55 seconds). Original timestamps are retained and flagged "
            "rather than silently clipped. The linked media are not downloaded or "
            "represented as reusable, and HACS does not label weapon, athlete, touch, or bout."
        ),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    segment_output: Path,
    clip_output: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
    comparison_manifests: dict[str, Path] | None = None,
) -> dict[str, Any]:
    acquire(archive_path, user_agent=user_agent, force=force)
    return prepare(
        archive_path,
        segment_output,
        clip_output,
        report_path,
        comparison_manifests=comparison_manifests,
    )
