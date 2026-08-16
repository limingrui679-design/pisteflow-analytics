"""Audit the pinned Asmi Sawant three-label fencing clip repository."""

from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import pandas as pd

from pisteflow.models import LicenseStatus, SourceKind, VideoRecord, Weapon
from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable
from pisteflow.video.features import VideoFeatureConfig, extract_video_windows

SOURCE_URL = "https://github.com/asmisawant/fencing-ai-referee"
SNAPSHOT = "ef9ace1ce267c11d1ab68698685d143179e3eaa4"
ARCHIVE_URL = f"https://codeload.github.com/asmisawant/fencing-ai-referee/tar.gz/{SNAPSHOT}"
EXPECTED_ARCHIVE_BYTES = 350_319_339
EXPECTED_ARCHIVE_SHA256 = "db051964b0bb5f70865b7fd6e240658a35472307362ace29cec42c1847a03320"
LICENSE = "No repository or dataset license found; media, rows, features, and models stay local"
EXPECTED_MEDIA_FILES = 395
_CLIP_PATTERN = re.compile(r"^Fencing-([LRS])(\d+)-(\d+) \(30\)\.mp4$")
_DIRECTORY_LABELS = {"left": "left", "right": "right", "simul": "simultaneous"}
_CODE_LABELS = {"L": "left", "R": "right", "S": "simultaneous"}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _safe_extract_stripped(archive_path: Path, output_directory: Path) -> None:
    """Extract regular files after removing the single archive-root component."""

    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        top_levels = {
            PurePosixPath(member.name).parts[0]
            for member in members
            if PurePosixPath(member.name).parts
        }
        if len(top_levels) != 1:
            raise ValueError("Asmi referee archive does not have one stable root")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) <= 1:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Asmi referee archive contains an unsafe member path")
            destination = (resolved_root / relative).resolve()
            if resolved_root not in destination.parents:
                raise ValueError("Asmi referee archive escapes its extraction root")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Asmi referee archive contains a non-regular member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Asmi referee archive member could not be opened")
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
    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_ARCHIVE_BYTES,
    )
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Asmi referee archive checksum changed")
    if force or not (output_directory / "data").is_dir():
        _safe_extract_stripped(archive_path, output_directory)
    return output_directory


def parse_clip_path(path: Path, media_root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(media_root)
    except ValueError as exc:
        raise ValueError("Asmi clip is outside the media root") from exc
    if len(relative.parts) != 2 or relative.parts[0] not in _DIRECTORY_LABELS:
        raise ValueError(f"Unexpected Asmi referee clip path: {relative.as_posix()}")
    match = _CLIP_PATTERN.fullmatch(relative.name)
    if match is None:
        raise ValueError(f"Unexpected Asmi referee clip filename: {relative.name}")
    label = _DIRECTORY_LABELS[relative.parts[0]]
    if _CODE_LABELS[match.group(1)] != label:
        raise ValueError(f"Asmi referee directory/filename label conflict: {relative}")
    source_number = int(match.group(2))
    return {
        "label": label,
        "source_group_id": _stable_id("asmi-source", str(source_number)),
        "source_clip_number": int(match.group(3)),
    }


def _media_metadata(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok_first, _ = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ok_middle, _ = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
        ok_last, _ = capture.read()
    finally:
        capture.release()
    if not (
        ok_first
        and ok_middle
        and ok_last
        and frame_count > 0
        and fps > 0
        and width > 0
        and height > 0
    ):
        raise ValueError(f"Asmi referee clip does not decode at three positions: {path.name}")
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
    }


def _clip_features(
    path: Path,
    clip_id: str,
    group_id: str,
    label: str,
) -> dict[str, Any]:
    record = VideoRecord.model_validate(
        {
            "video_id": clip_id,
            "title": "Pinned local Asmi referee label clip",
            "weapon": Weapon.UNKNOWN,
            "source_kind": SourceKind.DATASET,
            "source_url": SOURCE_URL,
            "license_status": LicenseStatus.UNKNOWN,
            "discovered_at": datetime(2026, 8, 15, tzinfo=UTC),
            "group_id": group_id,
        }
    )
    rows = extract_video_windows(
        path,
        record,
        VideoFeatureConfig(window_seconds=2.0, stride_seconds=2.0, frames_per_window=12),
    )
    if len(rows) != 1:
        raise ValueError(f"Asmi referee clip did not produce exactly one feature row: {path.name}")
    selected = {
        key: value
        for key, value in rows[0].items()
        if key.startswith(("frame_", "flow_"))
    }
    return {
        "clip_id": clip_id,
        "source_group_id": group_id,
        "upstream_label": label,
        **selected,
    }


def prepare(
    source_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    media_root = source_directory / "data"
    paths = sorted(media_root.glob("*/*.mp4"))
    if len(paths) != EXPECTED_MEDIA_FILES:
        raise ValueError(
            f"Asmi referee media count changed: {len(paths)} != {EXPECTED_MEDIA_FILES}"
        )

    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    tree_digest = hashlib.sha256()
    for path in paths:
        parsed = parse_clip_path(path, media_root)
        file_sha256 = sha256_file(path)
        hashes[file_sha256].append(path.relative_to(media_root).as_posix())
        metadata = _media_metadata(path)
        clip_id = _stable_id("asmi-clip", file_sha256)
        tree_digest.update(path.relative_to(media_root).as_posix().encode())
        tree_digest.update(b"\0")
        tree_digest.update(file_sha256.encode())
        tree_digest.update(b"\n")
        rows.append(
            {
                "clip_id": clip_id,
                "source_group_id": parsed["source_group_id"],
                "upstream_label": parsed["label"],
                "source_clip_number": parsed["source_clip_number"],
                "bytes": path.stat().st_size,
                "sha256": file_sha256,
                **metadata,
            }
        )
        feature_rows.append(
            _clip_features(
                path,
                clip_id,
                str(parsed["source_group_id"]),
                str(parsed["label"]),
            )
        )

    duplicate_hash_groups = [values for values in hashes.values() if len(values) > 1]
    if duplicate_hash_groups:
        raise ValueError("Asmi referee snapshot contains unexpected exact duplicate videos")
    frame = pd.DataFrame(rows)
    features = pd.DataFrame(feature_rows)
    if frame["clip_id"].duplicated().any() or features["clip_id"].duplicated().any():
        raise ValueError("Asmi referee clip identifiers are not unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    labels = Counter(frame["upstream_label"].astype(str))
    groups_by_label = {
        label: int(frame.loc[frame["upstream_label"].eq(label), "source_group_id"].nunique())
        for label in sorted(labels)
    }
    group_label_coverage = Counter(
        "+".join(sorted(set(group_rows["upstream_label"].astype(str))))
        for _group, group_rows in frame.groupby("source_group_id", sort=True)
    )
    feature_count = sum(column.startswith(("frame_", "flow_")) for column in features)
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        "archive_bytes": archive_path.stat().st_size if archive_path is not None else None,
        "archive_sha256": sha256_file(archive_path) if archive_path is not None else None,
        "verified_media_files": len(frame),
        "verified_media_bytes": int(frame["bytes"].sum()),
        "exact_unique_media_files": len(hashes),
        "exact_duplicate_media_files": len(frame) - len(hashes),
        "decoded_frames": int(frame["frame_count"].sum()),
        "duration_seconds": float(frame["duration_seconds"].sum()),
        "fps_values": sorted(float(value) for value in frame["fps"].unique()),
        "frame_count_values": sorted(int(value) for value in frame["frame_count"].unique()),
        "resolutions": {
            f"{int(width)}x{int(height)}": len(group)
            for (width, height), group in frame.groupby(["width", "height"], sort=True)
        },
        "label_counts": dict(sorted(labels.items())),
        "filename_source_groups": int(frame["source_group_id"].nunique()),
        "groups_by_label": groups_by_label,
        "group_label_coverage": dict(sorted(group_label_coverage.items())),
        "feature_rows": len(features),
        "feature_count": feature_count,
        "media_tree_sha256": tree_digest.hexdigest(),
        "feature_output": "data/processed/asmi-referee-video-features.parquet",
        "feature_output_sha256": sha256_file(output_path),
        "claim_boundary": (
            "The pinned repository contains 395 decodable 640x360, 30 fps, two-second clips, "
            "not the README's approximately 1,700 three-second collection. Directory labels are "
            "left, right, and simultaneous; they are preserved as upstream labels, not accepted "
            "as verified referee decisions. The README discusses saber and names foil/epee as "
            "future expansion, but no clip has source URL, event, athlete, or clip-level weapon "
            "provenance. Its approximate 70% human-agreement statement has no label protocol, "
            "predictions, or rater table in the archive and is not reproduced here. Filename "
            "source numbers define the 55 best-available groups. With no license, media, row "
            "features, and trained artifacts remain local."
        ),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    source_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    source = acquire(
        archive_path,
        source_directory,
        user_agent=user_agent,
        force=force,
    )
    return prepare(source, output_path, report_path, archive_path=archive_path)
