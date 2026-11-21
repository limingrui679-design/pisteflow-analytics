"""Metadata and acquisition for the public RT-Fencing touch-time image dataset."""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from pisteflow.provenance import sha256_file, write_json, write_jsonl

DATASET_PAGE = "https://github.com/CodLiver/RT-Fencing"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1IHlRLp2y5sltFbCKqQoKaFS5xLk9gGB4"
DRIVE_FILE_ID = "10BvzyYMBl_8zGQ0NGFZtEyP3oKLC75wj"
EXPECTED_BYTES = 936_840_403
VERIFIED_SNAPSHOT_SHA256 = "af2e0ddd1f48f18a6499d485fba23071ced80c28d3d88c6cda575d5347586b67"
LICENSE = "Creative Commons non-commercial; exact version not stated upstream"
CLASS_NAMES = {
    "CTR": "counter_attack",
    "LNG": "lunge",
    "NON": "invalid_move",
    "NON2": "referee",
    "PREP": "preparation_to_attack",
}
_NAME = re.compile(
    r"^(?P<augmented>f-)?(?P<match>\d+)\.(?P<round>\d+)\.(?P<fencer>\d+)"
    r"(?P<copy>\(1\)| - Copy)?\.png$"
)


def parse_member(name: str, *, file_size: int, compressed_size: int, crc32: int) -> dict[str, Any]:
    path = PurePosixPath(name)
    if len(path.parts) != 3 or path.suffix.casefold() != ".png":
        raise ValueError(f"Unexpected RT-Fencing member path: {name}")
    upstream_split, class_code, filename = path.parts
    if upstream_split not in {"train", "test"} or class_code not in CLASS_NAMES:
        raise ValueError(f"Unexpected RT-Fencing split/class: {name}")
    match = _NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unexpected RT-Fencing filename: {name}")
    source_match = match.group("match")
    fencer_code = match.group("fencer")
    copy_suffix = match.group("copy")
    copy_variant = {None: None, "(1)": "paren_1", " - Copy": "copy"}[copy_suffix]
    return {
        "sample_id": f"rtf:{upstream_split}:{class_code}:{path.stem}",
        "archive_member": name,
        "upstream_split": upstream_split,
        "class_code": class_code,
        "action_label": CLASS_NAMES[class_code],
        "is_augmented": bool(match.group("augmented")),
        "is_packaging_duplicate": copy_variant is not None,
        "copy_variant": copy_variant,
        "source_match_code": source_match,
        "group_id": f"rtf:match:{source_match}",
        "round_number": int(match.group("round")),
        "fencer_code": fencer_code,
        "fencer_side": {"0": "left", "1": "right"}.get(fencer_code, f"other_{fencer_code}"),
        "bytes": file_size,
        "compressed_bytes": compressed_size,
        "crc32": f"{crc32:08x}",
        "dataset_page": DATASET_PAGE,
        "download_page": DRIVE_FOLDER,
        "source_video_channel": "https://www.youtube.com/channel/UCA40s4GODjkaJ9JeM0fBcdg",
        "license": LICENSE,
        "weapon": "unknown",
        "claim_boundary": (
            "A touch-time crop with an upstream action label; not a weapon label, full phrase, "
            "referee decision, athlete identity, or competition outcome."
        ),
    }


def acquire(archive_path: Path, *, force: bool = False) -> Path:
    """Download the verified public Google Drive snapshot."""

    if archive_path.exists() and not force and archive_path.stat().st_size == EXPECTED_BYTES:
        return archive_path
    try:
        from importlib import import_module

        gdown = import_module("gdown")
    except ImportError as exc:  # pragma: no cover - optional collector
        raise RuntimeError("Install collectors: pip install -e '.[collectors]'") from exc
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(
        id=DRIVE_FILE_ID,
        output=str(archive_path),
        quiet=False,
        resume=True,
    )
    if downloaded is None or not archive_path.exists():
        raise RuntimeError("RT-Fencing Google Drive download did not produce an archive")
    if archive_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"RT-Fencing archive size mismatch: expected {EXPECTED_BYTES}, "
            f"got {archive_path.stat().st_size}"
        )
    return archive_path


def prepare(archive_path: Path, manifest_path: Path, report_path: Path) -> dict[str, Any]:
    """Audit every image and preserve source-match groups for leakage-safe evaluation."""

    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != VERIFIED_SNAPSHOT_SHA256:
        raise ValueError(
            "RT-Fencing snapshot checksum mismatch: "
            f"expected {VERIFIED_SNAPSHOT_SHA256}, got {actual_sha256}"
        )
    records: list[dict[str, Any]] = []
    with ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"RT-Fencing archive has a corrupt member: {bad}")
        for info in archive.infolist():
            if info.is_dir():
                continue
            row = parse_member(
                info.filename,
                file_size=info.file_size,
                compressed_size=info.compress_size,
                crc32=info.CRC,
            )
            row["sha256"] = sha256(archive.read(info)).hexdigest()
            records.append(row)
    records.sort(key=lambda row: str(row["sample_id"]))
    manifest_sha256 = write_jsonl(manifest_path, records)
    train_groups = {str(row["group_id"]) for row in records if row["upstream_split"] == "train"}
    test_groups = {str(row["group_id"]) for row in records if row["upstream_split"] == "test"}
    originals = [
        row for row in records if not row["is_augmented"] and not row["is_packaging_duplicate"]
    ]
    image_hashes = Counter(str(row["sha256"]) for row in records)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Real touch-time crops from fencing videos with five upstream classes. Weapon, "
            "athlete, and outcome labels are unavailable. The upstream train/test split shares "
            "source-match codes, so PisteFlow uses match-grouped splits for formal evaluation."
        ),
        "source_url": DATASET_PAGE,
        "download_url": DRIVE_FOLDER,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_sha256,
        "manifest_sha256": manifest_sha256,
        "images": len(records),
        "unique_image_sha256": len(image_hashes),
        "exact_duplicate_hash_groups": sum(count > 1 for count in image_hashes.values()),
        "exact_duplicate_extra_images": sum(count - 1 for count in image_hashes.values()),
        "original_images": len(originals),
        "mirrored_augmented_images": sum(bool(row["is_augmented"]) for row in records),
        "packaging_duplicate_images": sum(bool(row["is_packaging_duplicate"]) for row in records),
        "source_match_codes": len({str(row["group_id"]) for row in records}),
        "upstream_group_overlap": len(train_groups & test_groups),
        "by_class_all": dict(sorted(Counter(str(row["action_label"]) for row in records).items())),
        "by_class_original": dict(
            sorted(Counter(str(row["action_label"]) for row in originals).items())
        ),
        "by_upstream_split": dict(
            sorted(Counter(str(row["upstream_split"]) for row in records).items())
        ),
        "by_fencer_code": dict(sorted(Counter(str(row["fencer_code"]) for row in records).items())),
        "unexpected_fencer_codes": sorted(
            {str(row["fencer_code"]) for row in records} - {"0", "1"}
        ),
        "output": str(manifest_path),
    }
    write_json(report_path, summary)
    return summary
