"""Extract the fencing class from a CC0-labelled multi-sport image archive."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from PIL import Image

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = (
    "https://www.kaggle.com/datasets/sheikhzaib/sports-image-image-classification"
)
ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sheikhzaib/sports-image-image-classification"
)
LICENSE = "CC0: Public Domain (Kaggle dataset metadata)"
# Kaggle's API reports the uncompressed member total separately; this is the ZIP size.
EXPECTED_BYTES = 489_378_873
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _split_from_digest(digest: str) -> str:
    bucket = int(digest[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def prepare(
    archive_path: Path,
    output_directory: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Validate the archive and extract only unique, decodable fencing images."""

    output_directory.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    with ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Sports-image archive has a corrupt member: {bad}")
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 3 or parts[0] != "sports":
                continue
            if PurePosixPath(member.filename).suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            class_counts[parts[1]] += 1
            if parts[1] != "fencing":
                continue
            payload = archive.read(member)
            digest = sha256(payload).hexdigest()
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                mode = image.mode
            extension = PurePosixPath(member.filename).suffix.lower()
            destination = output_directory / f"{digest}{extension}"
            if not destination.exists():
                destination.write_bytes(payload)
            rows.append(
                {
                    "image_id": f"sports-images:{digest[:20]}",
                    "source_member": member.filename,
                    "local_filename": destination.name,
                    "label": "fencing",
                    "weapon": "unknown",
                    "split": _split_from_digest(digest),
                    "group_id": f"sha256:{digest}",
                    "sha256": digest,
                    "bytes": len(payload),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "source_url": SOURCE_PAGE,
                    "source_license": LICENSE,
                }
            )
    if not rows:
        raise ValueError("Sports-image archive contains no fencing images")
    rows.sort(key=lambda row: str(row["image_id"]))
    manifest_sha256 = write_jsonl(manifest_path, rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Generic real-world fencing still images from a multi-sport Kaggle archive. "
            "Kaggle labels the archive CC0, but the dataset card does not document each "
            "image's original provenance. Images have no weapon, athlete, action, or bout labels."
        ),
        "source_url": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "all_sport_images": int(sum(class_counts.values())),
        "sport_classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "fencing_images": len(rows),
        "unique_fencing_images": len(seen_digests),
        "fencing_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "by_split": dict(sorted(split_counts.items())),
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
    """Acquire the dataset snapshot and prepare its fencing subset."""

    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    return prepare(archive_path, output_directory, manifest_path, report_path)
