"""Audit Harmony4D sword-sparring media without calling it sport fencing data."""

from __future__ import annotations

import json
import ssl
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import truststore

from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json, write_jsonl

API_URL = (
    "https://huggingface.co/api/datasets/Jyun-Ting/Harmony4D/tree/main"
    "?recursive=true&expand=true"
)
DATASET_URL = "https://huggingface.co/datasets/Jyun-Ting/Harmony4D"
DATASET_SNAPSHOT = "3fedb23fd9d1a92541d98ccbce025c695bd752e4"
VOXEL51_DATASET_URL = "https://huggingface.co/datasets/Voxel51/Harmony4D"
VOXEL51_SNAPSHOT = "0efc0187af1377c0c12f0002217c83307c937875"
PROJECT_URL = "https://jyuntins.github.io/harmony4d/"
PAPER_URL = "https://arxiv.org/abs/2410.20294"
EXPECTED_PATHS = {
    "test/05_sword2.zip",
    "test/06_sword3.zip",
    "train/04_sword_part1.zip",
    "train/04_sword_part2.zip",
    "train/05_sword2.zip",
}
VOXEL51_SAMPLES_BYTES = 895_131
VOXEL51_SAMPLES_SHA256 = "bbb4a6580faf7d0e125cd5a0f17f66df241a2667debc70b89ac7e622014c89a5"
VOXEL51_METADATA_BYTES = 16_722
VOXEL51_METADATA_SHA256 = "0163dbb5cc16dec7e05ef8562dcc7203a06c9e329619dcfaefd7c266f0c9762b"
VOXEL51_EXPECTED_RECORDS = 887
VOXEL51_EXPECTED_SWORD_RECORDS = 204
VOXEL51_EXPECTED_SWORD_SEQUENCES = {
    "001_sword3",
    "002_sword3",
    "003_sword3",
    "004_sword3",
    "005_sword3",
    "006_sword3",
    "009_sword2",
    "010_sword2",
}
VOXEL51_REMOTE_SWORD_BYTES = 21_224_165_737
VOXEL51_REMOTE_REPOSITORY_BYTES = 53_762_984_709
PREVIEW_BYTES = 103_083_815
PREVIEW_SHA256 = "2a13b450e921ad2d8a56b97c7985f2745ad6af05ea941192c3f40915b169937c"


def parse_inventory(payload: bytes) -> list[dict[str, Any]]:
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("Harmony4D API response is not a file list")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path", ""))
        if path not in EXPECTED_PATHS:
            continue
        lfs = item.get("lfs")
        if not isinstance(lfs, Mapping):
            raise ValueError(f"Harmony4D archive has no LFS metadata: {path}")
        size = int(lfs["size"])
        sha256 = str(lfs["oid"])
        if size <= 0 or len(sha256) != 64:
            raise ValueError(f"Invalid Harmony4D LFS metadata: {path}")
        rows.append(
            {
                "archive_id": f"harmony4d:{path}",
                "path": path,
                "split": path.split("/", 1)[0],
                "activity_label": "sword",
                "fencing_relevance": "adjacent_sword_sparring_only",
                "sport_fencing_eligible": False,
                "weapon": None,
                "bytes": size,
                "sha256_lfs": sha256,
                "download_url": (
                    "https://huggingface.co/datasets/Jyun-Ting/Harmony4D/resolve/main/"
                    f"{quote(path, safe='/')}?download=true"
                ),
                "license": "MIT",
                "availability": "remote_not_acquired",
                "athlete_identity_available": False,
                "weapon_label_available": False,
            }
        )
    found = {row["path"] for row in rows}
    if found != EXPECTED_PATHS:
        missing = sorted(EXPECTED_PATHS - found)
        extra = sorted(found - EXPECTED_PATHS)
        raise ValueError(f"Harmony4D sword archive set changed; missing={missing}, extra={extra}")
    return sorted(rows, key=lambda row: row["path"])


def _fetch(client: httpx.Client, cache_path: Path, *, force: bool) -> bytes:
    if cache_path.exists() and not force:
        return cache_path.read_bytes()
    response = client.get(API_URL)
    response.raise_for_status()
    if b"sword" not in response.content:
        raise ValueError("Harmony4D API response contains no sword archives")
    atomic_write_bytes(cache_path, response.content)
    return response.content


def fetch_and_prepare(
    cache_path: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(
        follow_redirects=True,
        trust_env=False,
        verify=ssl_context,
        timeout=httpx.Timeout(60, read=120),
        headers={"User-Agent": user_agent},
    ) as client:
        payload = _fetch(client, cache_path, force=force)
    rows = parse_inventory(payload)
    split_bytes: dict[str, int] = {}
    for split in ("train", "test"):
        split_bytes[split] = sum(row["bytes"] for row in rows if row["split"] == split)
    report: dict[str, Any] = {
        "claim_boundary": (
            "The official repository exposes five archives whose filenames use 'sword', and "
            "the paper uses 'fencing' as an activity label. Visual review of the locally "
            "verified preview shows staged stick/sword sparring without a modern fencing piste, "
            "masks, uniforms, scoring apparatus, or epee/foil/sabre labels. It is therefore "
            "adjacent-domain pose and occlusion pretraining material, not three-weapon sport "
            "fencing data. Archive bytes and LFS SHA-256 values are exact remote inventory; the "
            "five 109 GiB archives were not acquired. Dataset-level counts are context only."
        ),
        "source_url": DATASET_URL,
        "source_snapshot": DATASET_SNAPSHOT,
        "project_url": PROJECT_URL,
        "paper_url": PAPER_URL,
        "license": "MIT",
        "prepared_at": datetime.now(UTC).isoformat(),
        "archives": len(rows),
        "archive_bytes": sum(row["bytes"] for row in rows),
        "archive_gib": round(sum(row["bytes"] for row in rows) / 1024**3, 3),
        "split_archive_counts": {
            split: sum(row["split"] == split for row in rows) for split in ("train", "test")
        },
        "split_bytes": split_bytes,
        "dataset_level_context_not_fencing_subset": {
            "images": 1_660_000,
            "human_instances": 3_320_000,
            "video_sequences": 208,
            "unique_subjects": 24,
            "synchronized_cameras": "more_than_20",
            "capture_resolution": "3840x2160",
            "capture_fps": 60,
            "annotation_fps": 20,
            "subjects_per_sequence": 2,
            "annotations": [
                "camera_parameters",
                "bounding_boxes",
                "person_ids",
                "2d_pose",
                "3d_pose",
                "3d_mesh",
            ],
        },
        "security_note": (
            "The hosting service flags serialized Python objects for review. Treat archives as "
            "trusted-source research inputs and inspect members before deserialization."
        ),
        "cache_sha256": sha256_file(cache_path),
        "manifest": str(manifest_path),
    }
    report["manifest_sha256"] = write_jsonl(manifest_path, rows)
    write_json(report_path, report)
    return report


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "archives": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
    }


def _validate_local_snapshot(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Harmony4D snapshot byte size changed: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Harmony4D snapshot checksum changed: {path}: {actual_sha256}")


def parse_voxel51_samples(payload: bytes) -> list[dict[str, Any]]:
    """Return the privacy-neutral sword subset from the Voxel51 catalog snapshot."""

    root = json.loads(payload)
    if not isinstance(root, Mapping) or not isinstance(root.get("samples"), list):
        raise ValueError("Harmony4D Voxel51 sample catalog is not a samples object")
    samples = root["samples"]
    if len(samples) != VOXEL51_EXPECTED_RECORDS:
        raise ValueError(f"Harmony4D Voxel51 record count changed: {len(samples)}")
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("activity") != "sword":
            continue
        sequence = str(sample.get("sequence_name", ""))
        camera = str(sample.get("camera_name", ""))
        media_type = str(sample.get("_media_type", ""))
        if not sequence or not camera or media_type not in {"video", "3d"}:
            raise ValueError(f"Invalid Harmony4D sword catalog row: {sample}")
        rows.append(
            {
                "sample_id": str(sample.get("_id", {}).get("$oid", "")),
                "filepath": str(sample.get("filepath", "")),
                "split": str(sample.get("split", "")),
                "activity_label": "sword",
                "sequence": sequence,
                "camera": camera,
                "camera_type": str(sample.get("camera_type", "")),
                "media_type": media_type,
                "stream": str(sample.get("stream", "")),
                "source_group_id": f"harmony4d:{sequence}",
                "weapon": "unknown",
                "sport_fencing_eligible": False,
                "training_role": "adjacent_domain_pose_or_occlusion_pretraining",
                "availability": "remote_not_acquired",
            }
        )
    sequences = {str(row["sequence"]) for row in rows}
    if len(rows) != VOXEL51_EXPECTED_SWORD_RECORDS:
        raise ValueError(f"Harmony4D Voxel51 sword count changed: {len(rows)}")
    if sequences != VOXEL51_EXPECTED_SWORD_SEQUENCES:
        raise ValueError(f"Harmony4D Voxel51 sword sequences changed: {sorted(sequences)}")
    return sorted(rows, key=lambda row: (str(row["sequence"]), str(row["camera"])))


def prepare_voxel51_catalog(
    samples_path: Path,
    metadata_path: Path,
    preview_path: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Verify the public catalog and one preview, while preserving its domain boundary."""

    _validate_local_snapshot(
        samples_path,
        expected_bytes=VOXEL51_SAMPLES_BYTES,
        expected_sha256=VOXEL51_SAMPLES_SHA256,
    )
    _validate_local_snapshot(
        metadata_path,
        expected_bytes=VOXEL51_METADATA_BYTES,
        expected_sha256=VOXEL51_METADATA_SHA256,
    )
    _validate_local_snapshot(
        preview_path,
        expected_bytes=PREVIEW_BYTES,
        expected_sha256=PREVIEW_SHA256,
    )
    rows = parse_voxel51_samples(samples_path.read_bytes())
    sequences = {str(row["sequence"]) for row in rows}
    video_rows = [row for row in rows if row["media_type"] == "video"]
    scene_rows = [row for row in rows if row["media_type"] == "3d"]
    camera_type_counts = {
        camera_type: sum(row["camera_type"] == camera_type for row in rows)
        for camera_type in ("exo", "ego", "3d")
    }
    report: dict[str, Any] = {
        "source_url": VOXEL51_DATASET_URL,
        "source_snapshot": VOXEL51_SNAPSHOT,
        "upstream_source_url": DATASET_URL,
        "upstream_source_snapshot": DATASET_SNAPSHOT,
        "license": (
            "The Voxel51 mirror declares no dataset license. The upstream dataset card is "
            "tagged MIT; verify media-level terms before redistribution."
        ),
        "catalog_records": VOXEL51_EXPECTED_RECORDS,
        "sword_records": len(rows),
        "sword_sequences": len(sequences),
        "sword_sequence_ids": sorted(sequences),
        "sword_video_views": len(video_rows),
        "sword_3d_scene_records": len(scene_rows),
        "sword_camera_type_counts": camera_type_counts,
        "remote_sword_bytes": VOXEL51_REMOTE_SWORD_BYTES,
        "remote_sword_gib": round(VOXEL51_REMOTE_SWORD_BYTES / 1024**3, 3),
        "remote_repository_bytes": VOXEL51_REMOTE_REPOSITORY_BYTES,
        "remote_repository_gib": round(VOXEL51_REMOTE_REPOSITORY_BYTES / 1024**3, 3),
        "remote_media_acquired": False,
        "verified_preview": {
            "path": str(preview_path),
            "bytes": preview_path.stat().st_size,
            "sha256": sha256_file(preview_path),
            "duration_seconds": 30.05,
            "frames": 601,
            "fps": 20,
            "resolution": "3840x2160",
            "sequence": "001_sword3",
            "camera": "cam01",
        },
        "visual_qa": {
            "reviewed": True,
            "finding": "adjacent_sword_sparring_not_modern_sport_fencing",
            "observed": (
                "staged close-contact stick/sword sparring in ordinary clothing with protective "
                "glasses; no modern fencing piste, masks, uniforms, scoring box, or weapon label"
            ),
        },
        "sport_fencing_training_eligible": False,
        "permitted_training_role": "adjacent_domain_pose_or_occlusion_pretraining_only",
        "grouping_rule": (
            "hold out complete sequences; every ego/exo view and 3D scene record from one "
            "sequence remains in one split"
        ),
        "samples_sha256": sha256_file(samples_path),
        "metadata_sha256": sha256_file(metadata_path),
        "claim_boundary": (
            "The Voxel51 catalog identifies 204 sword-labelled records across eight staged "
            "sparring sequences: 196 video views and eight 3D scene records. Only one 30.05-second "
            "preview view was acquired. These are not 204 independent bouts, have no modern "
            "epee/foil/sabre labels, and must not be reported as sport-fencing match data."
        ),
    }
    report["manifest_sha256"] = write_jsonl(manifest_path, rows)
    report["manifest"] = str(manifest_path)
    write_json(report_path, report)
    return report
