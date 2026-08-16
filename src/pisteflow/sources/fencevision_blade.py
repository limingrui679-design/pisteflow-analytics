"""Audit the CC BY 4.0 blade-localization subset embedded in FenceVision."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pisteflow.provenance import sha256_file, write_json, write_jsonl

SOURCE_URL = "https://github.com/Electrotime/fencing"
SNAPSHOT = "34190b034e48579c44bd2eccc9307545a6392965"
DATASET_URL = (
    "https://universe.roboflow.com/ahs-workspace-2qhtk/"
    "my-first-project-gg3o7/dataset/3"
)
DATA_YAML_SHA256 = "7aef98063ef65398264097127f3583ed6a7d8fd4b2289becb2e0bf2495427d36"
LICENSE = "CC BY 4.0 stated in blade_dataset/data.yaml"

EXPECTED_EXPORT_IMAGES = 2_399
EXPECTED_UNIQUE_IMAGE_HASHES = 2_398
EXPECTED_ORIGINAL_FRAMES = 1_000
EXPECTED_SOURCE_VIDEOS = 5
EXPECTED_BOXES = 2_910
EXPECTED_POSITIVE_EXPORT_IMAGES = 1_851
EXPECTED_POSITIVE_ORIGINAL_FRAMES = 764
EXPECTED_SPLITS = {"test": 100, "train": 2_099, "valid": 200}
EXPECTED_VARIANTS_PER_ORIGINAL = {1: 300, 2: 1, 3: 699}

_IMAGE_PATTERN = re.compile(
    r"(?P<source>.+)_f(?P<frame>\d+)_jpg\.rf\.(?P<variant>[0-9a-f]{32})\.jpg"
)


def repository_snapshot(checkout: Path) -> str:
    """Return and validate the exact Git snapshot used by the local checkout."""

    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot = result.stdout.strip()
    if snapshot != SNAPSHOT:
        raise ValueError(f"FenceVision checkout changed: {snapshot} != {SNAPSHOT}")
    return snapshot


def _source_aliases(source_names: set[str]) -> dict[str, str]:
    return {
        source_name: f"foil-source-{index:02d}"
        for index, source_name in enumerate(sorted(source_names), start=1)
    }


def _group_id(namespace: str, value: str) -> str:
    digest = hashlib.sha256(f"fencevision-blade:{namespace}:{value}".encode()).hexdigest()
    return f"fencevision-blade-{namespace}:{digest[:20]}"


def _parse_boxes(path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Unexpected FenceVision YOLO row: {path}")
        values = [float(value) for value in parts[1:]]
        if not all(0 <= value <= 1 for value in values):
            raise ValueError(f"FenceVision YOLO coordinate outside [0, 1]: {path}")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"FenceVision YOLO box has non-positive extent: {path}")
        boxes.append(values)
    return boxes


def records_from_dataset(dataset_directory: Path) -> list[dict[str, Any]]:
    source_names: set[str] = set()
    parsed: list[tuple[str, Path, Path, re.Match[str]]] = []
    for split in ("train", "valid", "test"):
        image_directory = dataset_directory / split / "images"
        label_directory = dataset_directory / split / "labels"
        for image_path in sorted(image_directory.glob("*.jpg")):
            match = _IMAGE_PATTERN.fullmatch(image_path.name)
            if match is None:
                raise ValueError(f"Unexpected FenceVision image name: {image_path.name}")
            label_path = label_directory / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"Missing FenceVision label for {image_path.name}")
            source_names.add(match.group("source"))
            parsed.append((split, image_path, label_path, match))

        labels = {path.stem for path in label_directory.glob("*.txt")}
        images = {path.stem for path in image_directory.glob("*.jpg")}
        if labels != images:
            raise ValueError(f"FenceVision image/label pairing changed in {split}")

    aliases = _source_aliases(source_names)
    rows: list[dict[str, Any]] = []
    for split, image_path, label_path, match in parsed:
        source_name = match.group("source")
        frame_number = int(match.group("frame"))
        original_token = f"{source_name}:{frame_number}"
        boxes = _parse_boxes(label_path)
        relative_path = image_path.relative_to(dataset_directory).as_posix()
        rows.append(
            {
                "sample_id": _group_id("export", relative_path),
                "original_frame_id": _group_id("frame", original_token),
                "source_group_id": _group_id("video", source_name),
                "source_alias": aliases[source_name],
                "source_frame_number": frame_number,
                "upstream_split": split,
                "export_variant_id": match.group("variant"),
                "relative_image_path": relative_path,
                "image_bytes": image_path.stat().st_size,
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
                "blade_boxes": len(boxes),
                "has_blade_box": bool(boxes),
                "weapon": "foil",
                "weapon_evidence": (
                    "manual multi-frame visual review of all five source-video groups; "
                    "foil target lame, guard geometry, event graphics, or identified foil "
                    "competitors"
                ),
                "dataset_license": LICENSE,
                "media_policy": (
                    "local training only; publish hashes and aggregate audit because the "
                    "embedded CC BY 4.0 export does not document underlying broadcast rights"
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["relative_image_path"]))


def _validate_data_card(dataset_directory: Path) -> None:
    path = dataset_directory / "data.yaml"
    if sha256_file(path) != DATA_YAML_SHA256:
        raise ValueError("FenceVision blade_dataset/data.yaml checksum changed")
    text = path.read_text(encoding="utf-8")
    required = [
        "names: ['blade']",
        "license: CC BY 4.0",
        "version: 3",
        DATASET_URL,
    ]
    if not all(value in text for value in required):
        raise ValueError("FenceVision blade dataset card fields changed")


def prepare(
    dataset_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    repository_revision: str,
) -> dict[str, Any]:
    if repository_revision != SNAPSHOT:
        raise ValueError("FenceVision repository snapshot is not pinned")
    _validate_data_card(dataset_directory)
    rows = records_from_dataset(dataset_directory)

    source_frames: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    source_videos: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_frames[str(row["original_frame_id"])].append(row)
        source_videos[str(row["source_group_id"])].append(row)

    split_counts = dict(sorted(Counter(str(row["upstream_split"]) for row in rows).items()))
    variants_per_original = dict(
        sorted(Counter(len(members) for members in source_frames.values()).items())
    )
    positive_originals = sum(
        any(bool(row["has_blade_box"]) for row in members)
        for members in source_frames.values()
    )
    source_split_overlap = sum(
        len({str(row["upstream_split"]) for row in members}) > 1
        for members in source_videos.values()
    )
    frame_split_overlap = sum(
        len({str(row["upstream_split"]) for row in members}) > 1
        for members in source_frames.values()
    )
    image_hash_counts = Counter(str(row["image_sha256"]) for row in rows)

    positive_export_images = sum(bool(row["has_blade_box"]) for row in rows)
    actual = {
        "export_images": len(rows),
        "unique_image_hashes": len(image_hash_counts),
        "original_frames": len(source_frames),
        "source_videos": len(source_videos),
        "blade_boxes": sum(int(row["blade_boxes"]) for row in rows),
        "positive_export_images": positive_export_images,
        "positive_original_frames": positive_originals,
        "split_counts": split_counts,
        "variants_per_original_frame": variants_per_original,
    }
    expected = {
        "export_images": EXPECTED_EXPORT_IMAGES,
        "unique_image_hashes": EXPECTED_UNIQUE_IMAGE_HASHES,
        "original_frames": EXPECTED_ORIGINAL_FRAMES,
        "source_videos": EXPECTED_SOURCE_VIDEOS,
        "blade_boxes": EXPECTED_BOXES,
        "positive_export_images": EXPECTED_POSITIVE_EXPORT_IMAGES,
        "positive_original_frames": EXPECTED_POSITIVE_ORIGINAL_FRAMES,
        "split_counts": EXPECTED_SPLITS,
        "variants_per_original_frame": EXPECTED_VARIANTS_PER_ORIGINAL,
    }
    if actual != expected:
        raise ValueError(f"FenceVision blade counts changed: {actual} != {expected}")

    by_source = {}
    for group_id, members in sorted(source_videos.items()):
        original_ids = {str(row["original_frame_id"]) for row in members}
        by_source[str(members[0]["source_alias"])] = {
            "source_group_id": group_id,
            "export_images": len(members),
            "original_frames": len(original_ids),
            "positive_original_frames": sum(
                any(bool(row["has_blade_box"]) for row in source_frames[frame_id])
                for frame_id in original_ids
            ),
            "blade_boxes": sum(int(row["blade_boxes"]) for row in members),
            "upstream_splits": sorted({str(row["upstream_split"]) for row in members}),
        }

    manifest_sha256 = write_jsonl(manifest_path, rows)
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "repository_snapshot": SNAPSHOT,
        "dataset_url": DATASET_URL,
        "dataset_card_sha256": DATA_YAML_SHA256,
        "license": LICENSE,
        **actual,
        "empty_export_images": len(rows) - positive_export_images,
        "empty_original_frames": len(source_frames) - positive_originals,
        "exact_duplicate_image_hash_groups": sum(value > 1 for value in image_hash_counts.values()),
        "exact_duplicate_extra_exports": sum(value - 1 for value in image_hash_counts.values()),
        "original_frame_overlap_between_upstream_splits": frame_split_overlap,
        "source_video_overlap_between_upstream_splits": source_split_overlap,
        "by_source": by_source,
        "weapon": "foil",
        "grouping_rule": (
            "ignore the upstream image split for evaluation and hold out complete source-video "
            "groups; all augmentations of one original frame remain in the same partition"
        ),
        "manifest": "data/manifests/fencevision-blade-images.jsonl",
        "manifest_sha256": manifest_sha256,
        "claim_boundary": (
            "The embedded Roboflow export contains 2,399 image files but only 1,000 original "
            "frames from five real foil source videos. There are 2,910 one-class blade boxes; "
            "236 original frames have no positive box. Every source video spans upstream splits, "
            "so those splits cannot measure source-video generalization. The subdataset card says "
            "CC BY 4.0, while the parent repository has no top-level license and does not document "
            "the underlying broadcast rights; media therefore remain local. Other parent-repo "
            "action labels, code, and checkpoints are not covered by this subdataset audit."
        ),
    }
    write_json(report_path, summary)
    return summary
