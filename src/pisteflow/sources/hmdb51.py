"""Extract and audit HMDB51's broad, mixed-origin ``fencing`` action class."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import cv2

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = "https://www.kaggle.com/datasets/nguyenbon/hmdb51"
ORIGINAL_PROJECT = "https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/"
ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/nguyenbon/hmdb51"
LICENSE = "MIT in Kaggle metadata; original clip-level rights are not established"
EXPECTED_BYTES = 2_117_581_749


def source_group(filename: str) -> str:
    """Keep clips cut from one source video in the same evaluation partition."""

    stem = PurePosixPath(filename).stem
    base = stem.rsplit("_fencing_", 1)[0] if "_fencing_" in stem else stem
    return base.removesuffix("_avi").casefold()


def split_for_group(group: str) -> str:
    bucket = int(sha256(group.encode()).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _video_metadata(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"HMDB51 clip is not decodable: {path.name}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"HMDB51 clip has invalid video metadata: {path.name}")
    return {
        "frames": frames,
        "fps": fps,
        "duration_seconds": frames / fps,
        "width": width,
        "height": height,
    }


def _weapon_hint(filename: str) -> str | None:
    lowered = filename.casefold()
    if "epee" in lowered or "épée" in lowered:
        return "epee"
    if "sabre" in lowered or "saber" in lowered:
        return "sabre"
    if "foil" in lowered:
        return "foil"
    return None


def prepare(
    archive_path: Path,
    output_directory: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Extract only the fencing class and write source-grouped metadata."""

    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    all_class_counts: Counter[str] = Counter()
    first_by_digest: dict[str, str] = {}
    with ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"HMDB51 archive has a corrupt member: {bad}")
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 3 or parts[0] != "hmdb51" or not parts[2].lower().endswith(".avi"):
                continue
            all_class_counts[parts[1]] += 1
            if parts[1] != "fencing":
                continue
            payload = archive.read(member)
            digest = sha256(payload).hexdigest()
            safe_name = f"{digest[:20]}.avi"
            destination = output_directory / safe_name
            if not destination.exists():
                destination.write_bytes(payload)
            group = source_group(parts[2])
            metadata = _video_metadata(destination)
            duplicate_of = first_by_digest.get(digest)
            if duplicate_of is None:
                first_by_digest[digest] = f"hmdb51:{digest[:20]}"
            rows.append(
                {
                    "clip_id": f"hmdb51:{digest[:20]}:{len(rows) + 1:03d}",
                    "source_member": member.filename,
                    "local_filename": safe_name,
                    "action_label": "fencing",
                    "weapon": "unknown",
                    "weapon_keyword_hint": _weapon_hint(parts[2]),
                    "weapon_hint_is_label": False,
                    "source_group": group,
                    "group_id": "hmdb51-source:" + sha256(group.encode()).hexdigest()[:20],
                    "split": split_for_group(group),
                    "sha256": digest,
                    "bytes": len(payload),
                    "is_exact_duplicate": duplicate_of is not None,
                    "duplicate_of": duplicate_of,
                    "source_url": SOURCE_PAGE,
                    "source_license": LICENSE,
                    **metadata,
                }
            )
    if not rows:
        raise ValueError("HMDB51 archive contains no fencing clips")
    rows.sort(key=lambda row: str(row["source_member"]))
    manifest_sha256 = write_jsonl(manifest_path, rows)
    unique_rows = [row for row in rows if not bool(row["is_exact_duplicate"])]
    summary: dict[str, Any] = {
        "claim_boundary": (
            "HMDB51's broad action class mixes sport fencing, historical/rapier practice, "
            "stage combat, and film scenes. It supports generic action-recognition stress tests, "
            "not modern three-weapon labels, athlete evaluation, or bout analysis. Kaggle's "
            "dataset-level MIT label does not establish rights for each source clip, so extracted "
            "media remains local research material."
        ),
        "source_url": SOURCE_PAGE,
        "original_project_url": ORIGINAL_PROJECT,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "all_action_classes": len(all_class_counts),
        "all_action_clips": int(sum(all_class_counts.values())),
        "fencing_clips": len(rows),
        "unique_fencing_clips": len(unique_rows),
        "exact_duplicate_rows": len(rows) - len(unique_rows),
        "source_groups": len({str(row["group_id"]) for row in rows}),
        "duration_hours_unique": round(
            sum(float(row["duration_seconds"]) for row in unique_rows) / 3600, 6
        ),
        "by_split_unique": dict(
            sorted(Counter(str(row["split"]) for row in unique_rows).items())
        ),
        "weapon_keyword_hints": dict(
            sorted(
                Counter(
                    str(row["weapon_keyword_hint"])
                    for row in unique_rows
                    if row["weapon_keyword_hint"] is not None
                ).items()
            )
        ),
        "manifest_sha256": manifest_sha256,
        "output_directory": str(output_directory),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    output_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    return prepare(archive_path, output_directory, manifest_path, report_path)
