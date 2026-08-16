"""Audit recent sabre side-labelled windows without downloading YouTube media."""

from __future__ import annotations

import csv
import hashlib
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/MistakesWereMaed/SaberPredict"
SNAPSHOT = "ed9e0ca9a095d2598a33acfcae935a895ff288ce"
ARCHIVE_URL = f"https://codeload.github.com/MistakesWereMaed/SaberPredict/tar.gz/{SNAPSHOT}"
EXPECTED_ARCHIVE_BYTES = 3_858_307
EXPECTED_ARCHIVE_SHA256 = "0c1c040b03118bd13ad8cabbb07f14a5b70c29c563b9c0a78bbd4817af7df4fd"
LICENSE = "No repository or footage license found; local analysis and link-only media"
EXPECTED_LABELS = {"Left", "Right"}


def _safe_extract(archive_path: Path, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (root / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError("SaberPredict archive contains an unsafe member path")
            if member.issym() or member.islnk():
                raise ValueError("SaberPredict archive contains a link member")
        archive.extractall(output_directory, members=members, filter="data")
    expected = output_directory / f"SaberPredict-{SNAPSHOT}"
    if not expected.is_dir():
        raise ValueError("SaberPredict archive root changed")
    return expected


def acquire(
    archive_path: Path,
    output_directory: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> Path:
    """Download and safely extract the pinned metadata-only repository snapshot."""

    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_ARCHIVE_BYTES,
    )
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("SaberPredict archive checksum changed")
    expected = output_directory / f"SaberPredict-{SNAPSHOT}"
    if force or not expected.is_dir():
        return _safe_extract(archive_path, output_directory)
    return expected


def _youtube_id(url: str) -> str:
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("v", [])
    if parsed.hostname not in {"youtube.com", "www.youtube.com"} or len(values) != 1:
        raise ValueError(f"Unexpected SaberPredict video URL: {url}")
    return values[0]


def _seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected SaberPredict timestamp: {value}")
    hours, minutes, seconds = parts
    parsed = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if parsed < 0:
        raise ValueError("SaberPredict timestamp is negative")
    return parsed


def _group_id(video_id: str) -> str:
    digest = hashlib.sha256(f"pisteflow:saber-predict:{video_id}".encode()).hexdigest()[:20]
    return f"saber-predict-video:{digest}"


def _read_catalog(source_directory: Path) -> list[dict[str, str]]:
    path = source_directory / "Dataset" / "Videos" / "manifest.csv"
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"id", "event", "name", "url"}:
        raise ValueError("SaberPredict source-video catalog schema changed")
    ids = [int(row["id"]) for row in rows]
    if ids != list(range(1, len(rows) + 1)):
        raise ValueError("SaberPredict source-video IDs are not contiguous")
    if len({_youtube_id(row["url"]) for row in rows}) != len(rows):
        raise ValueError("SaberPredict source-video catalog contains duplicate URLs")
    return rows


def _read_upstream_metadata(source_directory: Path) -> pd.DataFrame:
    path = source_directory / "Dataset" / "Videos" / "Clips" / "metadata.csv"
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns) != ["source_id", "clip_id", "label"]:
        raise ValueError("SaberPredict clip metadata schema changed")
    if set(frame["label"]) != EXPECTED_LABELS:
        raise ValueError("SaberPredict side-label vocabulary changed")
    if frame.duplicated(["source_id", "clip_id"]).any():
        raise ValueError("SaberPredict clip metadata keys are duplicated")
    return frame


def _read_timestamps(source_directory: Path) -> pd.DataFrame:
    directory = source_directory / "Dataset" / "Videos" / "Timestamps"
    files = sorted(directory.glob("*.csv"), key=lambda path: int(path.stem))
    if not files:
        raise ValueError("SaberPredict timestamp directory is empty")
    rows: list[dict[str, Any]] = []
    for path in files:
        source_id = path.stem
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) != 3:
                    raise ValueError(
                        "SaberPredict timestamp row has "
                        f"{len(row)} fields: {path.name}:{line_number}"
                    )
                start_text, end_text, raw_label = row
                start_seconds = _seconds(start_text)
                end_seconds = _seconds(end_text)
                if end_seconds <= start_seconds:
                    raise ValueError("SaberPredict timestamp window is empty or reversed")
                label = raw_label.rsplit("_", maxsplit=1)[-1]
                if label not in EXPECTED_LABELS:
                    raise ValueError("SaberPredict timestamp side-label vocabulary changed")
                rows.append(
                    {
                        "source_id": source_id,
                        "clip_id": f"{raw_label}.mp4",
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "duration_seconds": end_seconds - start_seconds,
                        "timestamp_label": label,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["source_id", "clip_id"]).any():
        raise ValueError("SaberPredict timestamp keys are duplicated")
    return frame


def _fie_video_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            import json

            value = json.loads(line)
            video_id = str(value.get("video_id", "")).removeprefix("youtube:")
            if video_id:
                ids.add(video_id)
    return ids


def prepare(
    source_directory: Path,
    output_path: Path,
    link_manifest_path: Path,
    report_path: Path,
    *,
    archive_path: Path | None = None,
    fie_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Join the exact upstream side labels to their non-overlapping timestamp windows."""

    catalog = _read_catalog(source_directory)
    metadata = _read_upstream_metadata(source_directory)
    timestamps = _read_timestamps(source_directory)
    joined = timestamps.merge(
        metadata,
        on=["source_id", "clip_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("SaberPredict metadata and timestamp keys do not match exactly")
    if not joined["timestamp_label"].eq(joined["label"]).all():
        raise ValueError("SaberPredict filename and metadata labels disagree")
    catalog_by_id = {row["id"]: row for row in catalog}
    unknown_sources = set(joined["source_id"]) - set(catalog_by_id)
    if unknown_sources:
        raise ValueError(f"SaberPredict timestamps reference unknown videos: {unknown_sources}")

    overlap_windows = 0
    for _, group in joined.groupby("source_id"):
        ordered = group.sort_values("start_seconds")
        previous_end: float | None = None
        for row in ordered.itertuples(index=False):
            if previous_end is not None and float(row.start_seconds) < previous_end:
                overlap_windows += 1
            previous_end = max(previous_end or 0.0, float(row.end_seconds))

    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        catalog_row = catalog_by_id[str(row.source_id)]
        video_id = _youtube_id(catalog_row["url"])
        sample_key = f"{row.source_id}:{row.clip_id}"
        rows.append(
            {
                "sample_id": (
                    "saber-predict:"
                    + hashlib.sha256(sample_key.encode()).hexdigest()[:20]
                ),
                "group_id": _group_id(video_id),
                "source_video_index": int(row.source_id),
                "start_seconds": float(row.start_seconds),
                "end_seconds": float(row.end_seconds),
                "duration_seconds": float(row.duration_seconds),
                "upstream_side_label": str(row.label).lower(),
                "weapon": "sabre",
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["source_video_index", "start_seconds", "sample_id"], kind="stable"
    )
    if output["sample_id"].duplicated().any():
        raise ValueError("SaberPredict generated duplicate sample IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    annotated_source_ids = set(output["source_video_index"].astype(str))
    link_rows = []
    for row in catalog:
        video_id = _youtube_id(row["url"])
        source_rows = output[output["source_video_index"].eq(int(row["id"]))]
        link_rows.append(
            {
                "source_video_index": int(row["id"]),
                "video_id": video_id,
                "source_url": row["url"],
                "weapon": "sabre",
                "timestamp_windows": len(source_rows),
                "annotated_duration_seconds": round(
                    float(source_rows["duration_seconds"].sum()), 3
                ),
                "media_acquired": False,
                "license_status": "unknown",
            }
        )
    link_manifest_sha256 = write_jsonl(link_manifest_path, link_rows)

    catalog_ids = {_youtube_id(row["url"]) for row in catalog}
    fie_ids = _fie_video_ids(fie_manifest_path)
    label_counts = Counter(output["upstream_side_label"])
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        "source_video_links": len(catalog),
        "annotated_source_videos": len(annotated_source_ids),
        "catalog_videos_without_timestamps": len(catalog) - len(annotated_source_ids),
        "timestamp_windows": len(output),
        "annotated_duration_seconds": round(float(output["duration_seconds"].sum()), 3),
        "minimum_window_seconds": round(float(output["duration_seconds"].min()), 3),
        "maximum_window_seconds": round(float(output["duration_seconds"].max()), 3),
        "side_label_counts": dict(sorted(label_counts.items())),
        "overlapping_windows_within_source": overlap_windows,
        "videos_already_in_fie_youtube_catalog": len(catalog_ids & fie_ids),
        "new_video_groups_beyond_fie_youtube_catalog": len(catalog_ids - fie_ids),
        "media_files_acquired": 0,
        "archive_bytes": archive_path.stat().st_size if archive_path else None,
        "archive_sha256": sha256_file(archive_path) if archive_path else None,
        "output_sha256": sha256_file(output_path),
        "link_manifest_sha256": link_manifest_sha256,
        "claim_boundary": (
            "Recent sabre competition links plus 250 upstream side-labelled timestamp windows. "
            "All 22 source videos were already present in the broader FIE YouTube catalog when "
            "that manifest is supplied, so only the timestamp/side annotations are additive. "
            "The labels are filename/metadata side labels and are not independently verified "
            "right-of-way calls, referee decisions, action classes, athlete identities, or model "
            "outcomes. No video media is downloaded or redistributed, and the repository has no "
            "license. Windows from one source video must stay in the same evaluation group."
        ),
    }
    write_json(report_path, summary)
    return summary
