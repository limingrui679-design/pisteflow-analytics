"""Extract and audit the broad Fencing class from the WomenSports image archive."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.fft import dctn

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://data.mendeley.com/datasets/m39ymdv48s/3"
SOURCE_DOI = "https://doi.org/10.17632/m39ymdv48s.3"
DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/m39ymdv48s/download/3"
DATASET_LICENSE = "CC BY 4.0 at the Mendeley dataset level"
OUTER_MEMBER = "WomenSports/WSPORTS2024.zip"
OUTER_BYTES = 735_256_568
OUTER_SHA256 = "2f6e63f99d063205b6f864299965371ec355c55520ca145f9f5212644df22555"
INNER_BYTES = 736_265_110
INNER_SHA256 = "3d1e972ee82980a6912c84048fe64f03022d686932639768ba25ff70b9a898c9"
EXPECTED_TOTAL_FILES = 12_666
EXPECTED_FENCING_FILES = 204
EXPECTED_SPLITS = {"test": 171, "train": 33}
PAPER_REPORTED_IMAGES = 33_504
PAPER_REPORTED_CLASSES = 50


def _extract_inner(outer_path: Path, inner_path: Path) -> None:
    if (
        inner_path.exists()
        and inner_path.stat().st_size == INNER_BYTES
        and sha256_file(inner_path) == INNER_SHA256
    ):
        return
    inner_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = inner_path.with_suffix(inner_path.suffix + ".part")
    with ZipFile(outer_path) as archive:
        names = archive.namelist()
        if names != [OUTER_MEMBER]:
            raise ValueError("WomenSports outer wrapper members changed")
        with archive.open(OUTER_MEMBER) as source, temporary.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
    if temporary.stat().st_size != INNER_BYTES or sha256_file(temporary) != INNER_SHA256:
        raise ValueError("WomenSports inner archive checksum changed")
    temporary.replace(inner_path)


def _phash(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("Pillow is required for the WomenSports collector") from exc

    with Image.open(path) as image:
        grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        values = np.asarray(grayscale, dtype=np.float64)
    low_frequency = dctn(values, type=2, norm="ortho")[:8, :8]
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = (low_frequency > median).flatten()
    number = 0
    for bit in bits:
        number = (number << 1) | int(bit)
    return f"{number:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def perceptual_groups(hashes: list[str], *, threshold: int = 4) -> list[int]:
    """Return deterministic connected components for near-identical image hashes."""

    if threshold < 0 or threshold > 12:
        raise ValueError("WomenSports perceptual threshold must be between zero and twelve")
    parents = list(range(len(hashes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if _hamming(hashes[left], hashes[right]) <= threshold:
                union(left, right)
    roots = [find(index) for index in range(len(hashes))]
    canonical = {root: position for position, root in enumerate(sorted(set(roots)))}
    return [canonical[root] for root in roots]


def _image_metadata(path: Path) -> tuple[int, int, str]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for the WomenSports collector") from exc
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return int(image.width), int(image.height), str(image.format)


def extract_fencing(inner_path: Path, output_directory: Path) -> list[dict[str, Any]]:
    """Extract only Fencing-class JPEGs and retain every source row for audit."""

    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ZipFile(inner_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != EXPECTED_TOTAL_FILES:
            raise ValueError(f"WomenSports archive file count changed: {len(infos)}")
        fencing = [info for info in infos if "/Fencing/" in info.filename]
        if len(fencing) != EXPECTED_FENCING_FILES:
            raise ValueError(f"WomenSports Fencing class count changed: {len(fencing)}")
        for info in fencing:
            match = re.fullmatch(
                r"WSPORTS2024/(train|test)/Fencing/Fencing \((\d+)\)\.jpg",
                info.filename,
            )
            if match is None:
                raise ValueError(f"Unexpected WomenSports fencing member: {info.filename}")
            split, source_number = match.groups()
            destination = output_directory / f"womensports-{split}-{int(source_number):04d}.jpg"
            payload = archive.read(info)
            if not destination.exists() or destination.read_bytes() != payload:
                destination.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            width, height, image_format = _image_metadata(destination)
            if image_format != "JPEG":
                raise ValueError("WomenSports fencing member is not a decodable JPEG")
            rows.append(
                {
                    "sample_id": f"womensports:{split}:{int(source_number):04d}",
                    "source_member": info.filename,
                    "source_number": int(source_number),
                    "declared_split": split,
                    "class_label": "Fencing",
                    "weapon": "unknown",
                    "gender_scope": "women_sports_dataset_class",
                    "width": width,
                    "height": height,
                    "bytes": len(payload),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": digest,
                    "perceptual_hash": _phash(destination),
                    "local_path": str(destination),
                    "source_url": SOURCE_URL,
                    "source_doi": SOURCE_DOI,
                    "dataset_license": DATASET_LICENSE,
                }
            )

    split_counts = Counter(str(row["declared_split"]) for row in rows)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise ValueError(f"WomenSports fencing split counts changed: {dict(split_counts)}")
    exact_first: dict[str, str] = {}
    for row in rows:
        digest = str(row["sha256"])
        row["exact_duplicate_of"] = exact_first.get(digest)
        exact_first.setdefault(digest, str(row["sample_id"]))
        row["eligible_exact_unique"] = row["exact_duplicate_of"] is None

    group_numbers = perceptual_groups([str(row["perceptual_hash"]) for row in rows])
    group_sizes = Counter(group_numbers)
    for row, group_number in zip(rows, group_numbers, strict=True):
        row["perceptual_group_id"] = f"womensports-phash:{group_number:04d}"
        row["perceptual_group_size"] = group_sizes[group_number]
    rows.sort(key=lambda row: (str(row["declared_split"]), int(row["source_number"])))
    return rows


def fetch_and_prepare(
    outer_path: Path,
    inner_path: Path,
    output_directory: Path,
    manifest_path: Path,
    summary_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        outer_path.unlink(missing_ok=True)
        inner_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        outer_path,
        user_agent=user_agent,
        expected_bytes=OUTER_BYTES,
    )
    if sha256_file(outer_path) != OUTER_SHA256:
        raise ValueError("WomenSports outer archive checksum changed")
    _extract_inner(outer_path, inner_path)
    rows = extract_fencing(inner_path, output_directory)
    manifest_sha256 = write_jsonl(manifest_path, rows)
    exact_counts = Counter(str(row["sha256"]) for row in rows)
    perceptual_counts = Counter(str(row["perceptual_group_id"]) for row in rows)
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "dataset_license": DATASET_LICENSE,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "outer_bytes": outer_path.stat().st_size,
        "outer_sha256": sha256_file(outer_path),
        "inner_bytes": inner_path.stat().st_size,
        "inner_sha256": sha256_file(inner_path),
        "archive_total_files": EXPECTED_TOTAL_FILES,
        "archive_fencing_files": len(rows),
        "archive_fencing_files_by_declared_split": dict(
            Counter(str(row["declared_split"]) for row in rows)
        ),
        "archive_fencing_bytes": sum(int(row["bytes"]) for row in rows),
        "exact_unique_images": len(exact_counts),
        "exact_duplicate_rows": sum(count - 1 for count in exact_counts.values()),
        "perceptual_groups_hamming_le_4": len(perceptual_counts),
        "rows_in_multi_image_perceptual_groups": sum(
            count for count in perceptual_counts.values() if count > 1
        ),
        "paper_reported_total_images": PAPER_REPORTED_IMAGES,
        "paper_reported_classes": PAPER_REPORTED_CLASSES,
        "archive_total_minus_paper_report": EXPECTED_TOTAL_FILES - PAPER_REPORTED_IMAGES,
        "manifest_sha256": manifest_sha256,
        "claim_boundary": (
            "WomenSports supplies a manually labelled broad Fencing image class for women in "
            "sport, but no weapon, athlete, event, bout, touch, or action labels. The deposited "
            "archive has 12,666 images although the later paper reports 33,504; the discrepancy "
            "is retained. The original images were collected from public websites and open-source "
            "datasets, so dataset-level CC BY does not eliminate possible third-party rights. "
            "Images remain local research inputs. Exact duplicates are excluded and perceptual "
            "groups must stay intact in any evaluation split."
        ),
        "repository_policy": (
            "local images only; publish code, source links, hashes, counts, and aggregate audits"
        ),
    }
    write_json(summary_path, summary)
    return summary
