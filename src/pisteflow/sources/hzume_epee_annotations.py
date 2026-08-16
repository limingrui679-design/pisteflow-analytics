"""Audit the pinned Hzume epee temporal-action annotation snapshot.

The repository contains one VIA project with temporal annotations, but it does not
contain the 22 referenced videos.  The adapter therefore preserves annotation rows
locally and publishes only hashes and aggregate audit results.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/hzume/epee-action-recognition"
SNAPSHOT = "fb83a2272e60f1ccfdc7190dd2fa5f732e9b4199"
ARCHIVE_URL = f"https://codeload.github.com/hzume/epee-action-recognition/tar.gz/{SNAPSHOT}"
EXPECTED_ARCHIVE_BYTES = 351_368
EXPECTED_ARCHIVE_SHA256 = "71c05cdbe6c2c9735b8ae73bf871a510c040c92c06019eef033cfd2f597879d7"
EXPECTED_SOURCE_VIDEOS = 22
EXPECTED_ANNOTATED_VIDEOS = 21
EXPECTED_ANNOTATIONS = 105

LICENSE_BOUNDARY = (
    "The pinned repository has no license and does not include the 22 referenced source videos. "
    "The annotation JSON, row-level derivatives, filenames, and trained artifacts stay local; "
    "only parser code, snapshot hashes, aggregate counts, and limitations are published."
)

_LABEL = re.compile(
    r"^(?P<side>[lr])_(?P<action>lunge|fleche|counter|parry|prime)_"
    r"(?P<outcome>hit|miss|success|fail)$"
)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def parse_label(value: str) -> tuple[str, str, str]:
    """Parse a strict upstream side/action/outcome label."""

    match = _LABEL.fullmatch(value)
    if match is None:
        raise ValueError(f"Unexpected Hzume epee label: {value!r}")
    return match.group("side"), match.group("action"), match.group("outcome")


def merged_interval_seconds(intervals: list[tuple[float, float]]) -> float:
    """Return the interval union, avoiding double-counting paired fencer labels."""

    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _safe_extract_stripped(archive_path: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        roots = {
            PurePosixPath(member.name).parts[0]
            for member in members
            if PurePosixPath(member.name).parts
        }
        if len(roots) != 1:
            raise ValueError("Hzume epee archive does not have one stable root")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) <= 1:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Hzume epee archive contains an unsafe member path")
            destination = (resolved_root / relative).resolve()
            if resolved_root not in destination.parents:
                raise ValueError("Hzume epee archive escapes its extraction root")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Hzume epee archive contains a non-regular member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Hzume epee archive member could not be opened")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def acquire(
    archive_path: Path,
    output_directory: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> Path:
    """Download, verify, and safely extract the pinned source snapshot."""

    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_ARCHIVE_BYTES,
    )
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Hzume epee archive checksum changed")
    marker = output_directory / "input/raw_annotation/via_project_13Jan2025_01h17m28s.json"
    if force or not marker.is_file():
        _safe_extract_stripped(archive_path, output_directory)
    return output_directory


def _load_project(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"project", "file", "metadata", "attribute", "config", "view"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unexpected Hzume VIA project schema")
    for key in ("project", "file", "metadata"):
        if not isinstance(payload[key], dict):
            raise ValueError(f"Malformed Hzume VIA {key} section")
    return payload


def prepare(
    source_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Create a privacy-minimized local row table and a public-safe summary."""

    project_path = (
        source_directory / "input/raw_annotation/via_project_13Jan2025_01h17m28s.json"
    )
    payload = _load_project(project_path)
    project = payload["project"]
    files = payload["file"]
    metadata = payload["metadata"]
    video_ids = [str(value) for value in project.get("vid_list", [])]
    if len(video_ids) != EXPECTED_SOURCE_VIDEOS or set(video_ids) != set(files):
        raise ValueError("Hzume epee source-video inventory changed")
    if len(metadata) != EXPECTED_ANNOTATIONS:
        raise ValueError("Hzume epee annotation count changed")

    filename_by_id: dict[str, str] = {}
    for video_id in video_ids:
        file_row = files[video_id]
        if not isinstance(file_row, dict):
            raise ValueError("Malformed Hzume VIA file row")
        filename = str(file_row.get("fname", ""))
        if not filename.endswith(".mp4") or str(file_row.get("src", "")):
            raise ValueError("Unexpected Hzume source-video reference")
        filename_by_id[video_id] = filename

    rows: list[dict[str, Any]] = []
    exact_signatures: list[tuple[str, float, float, str]] = []
    intervals_by_video: dict[str, list[tuple[float, float]]] = {
        video_id: [] for video_id in video_ids
    }
    for source_annotation_id, value in sorted(metadata.items()):
        if not isinstance(value, dict):
            raise ValueError("Malformed Hzume VIA annotation row")
        video_id = str(value.get("vid", ""))
        if video_id not in filename_by_id:
            raise ValueError("Hzume annotation references an unknown video")
        interval = value.get("z")
        attributes = value.get("av")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not isinstance(attributes, dict)
            or set(attributes) != {"1"}
        ):
            raise ValueError("Unexpected Hzume temporal-annotation schema")
        start_seconds, end_seconds = float(interval[0]), float(interval[1])
        if start_seconds < 0 or end_seconds <= start_seconds:
            raise ValueError("Invalid Hzume temporal interval")
        upstream_label = str(attributes["1"])
        side, action, outcome = parse_label(upstream_label)
        pseudonymous_video_id = _stable_id("hzume-epee-video", filename_by_id[video_id])
        annotation_id = _stable_id(
            "hzume-epee-annotation", f"{source_annotation_id}:{upstream_label}"
        )
        rows.append(
            {
                "annotation_id": annotation_id,
                "source_group_id": pseudonymous_video_id,
                "weapon": "epee",
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": end_seconds - start_seconds,
                "side": side,
                "action": action,
                "outcome": outcome,
                "upstream_label": upstream_label,
                "human_review_status": "upstream_annotation_protocol_not_documented",
                "source_media_available": False,
            }
        )
        exact_signatures.append((video_id, start_seconds, end_seconds, upstream_label))
        intervals_by_video[video_id].append((start_seconds, end_seconds))

    annotated_video_ids = {row[0] for row in exact_signatures}
    if len(annotated_video_ids) != EXPECTED_ANNOTATED_VIDEOS:
        raise ValueError("Hzume epee annotated-video count changed")
    duplicate_rows = len(exact_signatures) - len(set(exact_signatures))
    if duplicate_rows:
        raise ValueError("Hzume epee snapshot contains exact duplicate annotations")

    frame = pd.DataFrame(rows).sort_values(
        ["source_group_id", "start_seconds", "end_seconds", "upstream_label"],
        kind="stable",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    union_seconds = sum(merged_interval_seconds(value) for value in intervals_by_video.values())
    report: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "archive_bytes": archive_path.stat().st_size if archive_path else None,
        "archive_sha256": sha256_file(archive_path) if archive_path else None,
        "annotation_json_bytes": project_path.stat().st_size,
        "annotation_json_sha256": sha256_file(project_path),
        "source_video_references": len(video_ids),
        "source_videos_with_annotations": len(annotated_video_ids),
        "source_videos_without_annotations": len(video_ids) - len(annotated_video_ids),
        "source_video_media_files_acquired": 0,
        "annotation_rows": len(rows),
        "exact_duplicate_annotation_rows": duplicate_rows,
        "sum_of_row_durations_seconds": round(float(frame["duration_seconds"].sum()), 6),
        "merged_annotation_union_seconds": round(union_seconds, 6),
        "side_counts": dict(sorted(Counter(frame["side"]).items())),
        "action_counts": dict(sorted(Counter(frame["action"]).items())),
        "outcome_counts": dict(sorted(Counter(frame["outcome"]).items())),
        "label_counts": dict(sorted(Counter(frame["upstream_label"]).items())),
        "output": "data/processed/hzume-epee-action-annotations.parquet",
        "output_sha256": sha256_file(output_path),
        "license_boundary": LICENSE_BOUNDARY,
        "claim_boundary": (
            "The pinned VIA project contributes 105 temporal side/action/outcome rows from 21 "
            "annotated epee video references, while a 22nd referenced file has no annotations. "
            "The source videos are absent, the annotation protocol and rater agreement are not "
            "documented, and overlapping left/right rows describe the same time ranges. The row "
            "duration sum is therefore not a unique-video duration; the merged interval union is "
            "reported separately. These labels are local research inputs, not externally verified "
            "referee decisions or evidence of athlete-independent generalization."
        ),
    }
    write_json(report_path, report)
    return report


def fetch_and_prepare(
    archive_path: Path,
    source_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    acquire(archive_path, source_directory, user_agent=user_agent, force=force)
    return prepare(
        source_directory,
        output_path,
        report_path,
        archive_path=archive_path,
    )
