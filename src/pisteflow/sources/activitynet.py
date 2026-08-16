"""Metadata-only fencing intervals from ActivityNet 1.3 official annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pisteflow.provenance import read_jsonl, sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

DATASET_PAGE = "https://activity-net.org/download.html"
OFFICIAL_ANNOTATION_URL = (
    "http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json"
)
MIRROR_SNAPSHOT = "1a20776262cc1fe3761588843cad26b62c2b7125"
MIRROR_URL = (
    "https://huggingface.co/datasets/YimuWang/ActivityNet/resolve/"
    f"{MIRROR_SNAPSHOT}/activity_net.v1-3.min.json"
)
EXPECTED_BYTES = 4_938_159
EXPECTED_SHA256 = "4c29d5b1561e1cbff9ac69816c159e60d417a18f16db8acc0fc254c377fa9ae3"
TARGET_LABEL = "Doing fencing"
EXPECTED_INTERVALS = 169
EXPECTED_VIDEOS = 76


def acquire(destination: Path, *, user_agent: str, force: bool = False) -> Path:
    """Acquire the byte-pinned official annotation mirror, never the linked videos."""

    if force and destination.exists():
        destination.unlink()
    download_resumable(
        MIRROR_URL,
        destination,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    actual = sha256_file(destination)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"ActivityNet annotation checksum changed: {actual}")
    return destination


def parse_annotations(source_path: Path) -> list[dict[str, Any]]:
    """Extract exact ``Doing fencing`` intervals with one group per source video."""

    if source_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("ActivityNet annotation byte size changed")
    actual = sha256_file(source_path)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"ActivityNet annotation checksum changed: {actual}")

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("version") != "VERSION 1.3":
        raise ValueError("ActivityNet version changed")
    database = payload.get("database")
    if not isinstance(database, dict):
        raise ValueError("ActivityNet database is missing")

    rows: list[dict[str, Any]] = []
    for youtube_id, raw_video in database.items():
        if not isinstance(raw_video, dict):
            raise ValueError("ActivityNet video metadata is malformed")
        subset = raw_video.get("subset")
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(f"ActivityNet subset changed: {subset}")
        duration = float(raw_video.get("duration", 0.0))
        if duration <= 0:
            raise ValueError("ActivityNet video duration is invalid")
        source_url = str(raw_video.get("url") or f"https://www.youtube.com/watch?v={youtube_id}")
        annotations = raw_video.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("ActivityNet annotations are malformed")
        selected: list[tuple[float, float]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict) or annotation.get("label") != TARGET_LABEL:
                continue
            segment = annotation.get("segment")
            if not isinstance(segment, list) or len(segment) != 2:
                raise ValueError("ActivityNet fencing segment is malformed")
            start, end = (float(segment[0]), float(segment[1]))
            if start < 0 or end <= start or end > duration + 0.05:
                raise ValueError("ActivityNet fencing segment is outside video bounds")
            selected.append((start, end))

        for interval_index, (start, end) in enumerate(sorted(selected), start=1):
            rows.append(
                {
                    "clip_id": f"activitynet:{youtube_id}:{interval_index:02d}",
                    "source_group_id": f"youtube:{youtube_id}",
                    "youtube_id": str(youtube_id),
                    "class_label": TARGET_LABEL,
                    "weapon": "unknown",
                    "start_seconds": round(start, 6),
                    "end_seconds": round(end, 6),
                    "duration_seconds": round(end - start, 6),
                    "source_video_duration_seconds": duration,
                    "split": str(subset),
                    "source_url": source_url,
                    "dataset_page": DATASET_PAGE,
                    "media_downloaded": False,
                    "license_status": "link_only_video_license_not_asserted",
                }
            )

    rows.sort(key=lambda row: str(row["clip_id"]))
    if len(rows) != EXPECTED_INTERVALS:
        raise ValueError(f"ActivityNet fencing interval count changed: {len(rows)}")
    if len({str(row["source_group_id"]) for row in rows}) != EXPECTED_VIDEOS:
        raise ValueError("ActivityNet fencing source-video count changed")
    return rows


def _manifest_youtube_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        identifier = row.get("youtube_id")
        if isinstance(identifier, str) and identifier:
            ids.add(identifier)
        video_id = row.get("video_id")
        if isinstance(video_id, str) and video_id.startswith("youtube:"):
            ids.add(video_id.removeprefix("youtube:"))
    return ids


def prepare(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    comparison_manifests: dict[str, Path] | None = None,
) -> dict[str, Any]:
    rows = parse_annotations(source_path)
    manifest_sha256 = write_jsonl(output_path, rows)
    source_ids = {str(row["youtube_id"]) for row in rows}
    overlaps: dict[str, int] = {}
    for name, path in sorted((comparison_manifests or {}).items()):
        overlaps[name] = (
            len(source_ids.intersection(_manifest_youtube_ids(path))) if path.exists() else 0
        )

    by_split = Counter(str(row["split"]) for row in rows)
    summary: dict[str, Any] = {
        "dataset_page": DATASET_PAGE,
        "official_annotation_url": OFFICIAL_ANNOTATION_URL,
        "mirror_url": MIRROR_URL,
        "mirror_snapshot": MIRROR_SNAPSHOT,
        "annotation_bytes": source_path.stat().st_size,
        "annotation_sha256": sha256_file(source_path),
        "manifest_sha256": manifest_sha256,
        "intervals": len(rows),
        "unique_source_videos": len(source_ids),
        "hours": round(sum(float(row["duration_seconds"]) for row in rows) / 3600, 3),
        "by_split": dict(sorted(by_split.items())),
        "cross_source_video_overlap": overlaps,
        "weapon_label": "unknown; ActivityNet labels the activity but not the weapon",
        "grouping_rule": "hold out complete source_group_id values (one YouTube video)",
        "media_downloaded": False,
        "output": "data/manifests/activitynet-fencing.jsonl",
        "claim_boundary": (
            "ActivityNet 1.3 annotation metadata identifies real public-video intervals labelled "
            "Doing fencing. Linked media are neither downloaded nor represented as reusable; "
            "the annotations do not identify epee, foil, sabre, athletes, touches, or bout "
            "outcomes. All intervals from one source video must remain in one split."
        ),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
    comparison_manifests: dict[str, Path] | None = None,
) -> dict[str, Any]:
    acquire(source_path, user_agent=user_agent, force=force)
    return prepare(
        source_path,
        output_path,
        report_path,
        comparison_manifests=comparison_manifests,
    )
