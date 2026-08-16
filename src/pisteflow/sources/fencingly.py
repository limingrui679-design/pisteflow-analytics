"""Exact-deduplicate and schema-audit the FENCINGLY pose archives."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://huggingface.co/datasets/EXCALOFRIO/FENCINGLY"
SNAPSHOT = "e14907c88e4969c4b38426137179fce2cc9d3056"
LICENSE = (
    "Dataset card declares Apache-2.0, but source-video, event, athlete, weapon, and annotation "
    "provenance are not documented"
)
_SUFFIX = "_output_filtered_data_ordenado.json"
_DIRECTORY_LABELS = {
    "IZQUIERDAO": "left",
    "IZQUIERDAOB": "left",
    "DERECHAO": "right",
    "DERECHAOB": "right",
}


@dataclass(frozen=True)
class ArchiveSpec:
    key: str
    filename: str
    bytes: int
    sha256: str
    json_files: int
    directory_counts: dict[str, int]

    @property
    def url(self) -> str:
        return f"{SOURCE_URL}/resolve/{SNAPSHOT}/{self.filename}"


ARCHIVES = (
    ArchiveSpec(
        key="original",
        filename="JSON_FORMADOS.rar",
        bytes=524_881_957,
        sha256="cc80ac0c3dc3006a03027ae2be68bb5005b41de177d9e3f54e3645a3a605313d",
        json_files=23_184,
        directory_counts={
            "DERECHAO": 7_694,
            "DERECHAOB": 3_898,
            "IZQUIERDAO": 7_694,
            "IZQUIERDAOB": 3_898,
        },
    ),
    ArchiveSpec(
        key="v1",
        filename="JSON_FORMADOS_V1.rar",
        bytes=597_883_341,
        sha256="55ce448c622eb4979b6bae3b6c58401a676937942e926cceefa9c70d528f8ff5",
        json_files=27_942,
        directory_counts={"DERECHAO": 13_971, "IZQUIERDAO": 13_971},
    ),
)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _archive_members(archive_path: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["bsdtar", "-tf", str(archive_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("bsdtar is required to inspect the FENCINGLY RAR archives") from exc
    return [line for line in result.stdout.splitlines() if line]


def _validate_archive_members(spec: ArchiveSpec, members: list[str]) -> None:
    json_members = []
    counts: Counter[str] = Counter()
    for name in members:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"FENCINGLY {spec.key} archive contains an unsafe member")
        if name.lower().endswith(".json"):
            if len(pure.parts) != 2:
                raise ValueError(f"FENCINGLY {spec.key} JSON member layout changed")
            json_members.append(name)
            counts[pure.parts[0]] += 1
    if len(json_members) != spec.json_files:
        raise ValueError(
            f"FENCINGLY {spec.key} JSON count changed: {len(json_members)} != {spec.json_files}"
        )
    if dict(sorted(counts.items())) != dict(sorted(spec.directory_counts.items())):
        raise ValueError(f"FENCINGLY {spec.key} directory counts changed")
    if len(json_members) != len(set(json_members)):
        raise ValueError(f"FENCINGLY {spec.key} archive repeats a JSON member path")


def _extract_archive(archive_path: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["bsdtar", "-xf", str(archive_path), "-C", str(output_directory)],
        check=True,
        capture_output=True,
        text=True,
    )


def acquire(
    archive_directory: Path,
    extraction_root: Path,
    *,
    user_agent: str,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for spec in ARCHIVES:
        archive_path = archive_directory / spec.filename
        download_resumable(
            spec.url,
            archive_path,
            user_agent=user_agent,
            expected_bytes=spec.bytes,
        )
        if sha256_file(archive_path) != spec.sha256:
            raise ValueError(f"FENCINGLY {spec.key} archive checksum changed")
        members = _archive_members(archive_path)
        _validate_archive_members(spec, members)
        output_directory = extraction_root / spec.key
        current_count = sum(1 for _path in output_directory.glob("*/*.json"))
        if current_count != spec.json_files:
            _extract_archive(archive_path, output_directory)
        final_count = sum(1 for _path in output_directory.glob("*/*.json"))
        if final_count != spec.json_files:
            raise ValueError(f"FENCINGLY {spec.key} extraction is incomplete")
        outputs[spec.key] = output_directory
    return outputs


def _series_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_delta": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_range": float(np.max(finite) - np.min(finite)),
        f"{prefix}_delta": float(finite[-1] - finite[0]),
    }


def _pose_features(sequence: np.ndarray) -> dict[str, float]:
    # sequence: frame, person, BODY_25 joint, (x, y, confidence)
    features: dict[str, float] = {}
    centroids = np.full((sequence.shape[0], 2, 2), np.nan, dtype=np.float64)
    for person in range(2):
        coordinates = sequence[:, person, :, :2]
        confidence = sequence[:, person, :, 2]
        visible = (confidence > 0) & (coordinates[..., 0] > 0) & (coordinates[..., 1] > 0)
        span_x = np.full(sequence.shape[0], np.nan, dtype=np.float64)
        span_y = np.full(sequence.shape[0], np.nan, dtype=np.float64)
        for frame_index in range(sequence.shape[0]):
            frame_visible = visible[frame_index]
            if not frame_visible.any():
                continue
            points = coordinates[frame_index, frame_visible]
            centroids[frame_index, person] = np.mean(points, axis=0)
            span_x[frame_index] = float(np.max(points[:, 0]) - np.min(points[:, 0]))
            span_y[frame_index] = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        prefix = f"pose_{person + 1}"
        features[f"{prefix}_visible_joint_fraction"] = float(np.mean(visible))
        features[f"{prefix}_detected_frame_fraction"] = float(
            np.mean(np.isfinite(centroids[:, person, 0]))
        )
        features.update(_series_stats(centroids[:, person, 0], f"{prefix}_center_x"))
        features.update(_series_stats(centroids[:, person, 1], f"{prefix}_center_y"))
        features.update(_series_stats(span_x, f"{prefix}_span_x"))
        features.update(_series_stats(span_y, f"{prefix}_span_y"))
        valid_centroids = centroids[:, person]
        pair_valid = np.isfinite(valid_centroids[:-1]).all(axis=1) & np.isfinite(
            valid_centroids[1:]
        ).all(axis=1)
        speeds = np.linalg.norm(
            valid_centroids[1:][pair_valid] - valid_centroids[:-1][pair_valid], axis=1
        )
        features[f"{prefix}_center_speed_mean"] = (
            float(np.mean(speeds)) if speeds.size else 0.0
        )
        features[f"{prefix}_center_speed_p90"] = (
            float(np.quantile(speeds, 0.9)) if speeds.size else 0.0
        )
        features[f"{prefix}_center_speed_max"] = (
            float(np.max(speeds)) if speeds.size else 0.0
        )

    both = np.isfinite(centroids).all(axis=(1, 2))
    delta = centroids[:, 1] - centroids[:, 0]
    features["pair_both_detected_frame_fraction"] = float(np.mean(both))
    features.update(_series_stats(delta[both, 0], "pair_delta_x"))
    features.update(_series_stats(delta[both, 1], "pair_delta_y"))
    distances = np.linalg.norm(delta[both], axis=1)
    features.update(_series_stats(distances, "pair_centroid_distance"))
    features["pair_second_pose_right_of_first_fraction"] = (
        float(np.mean(delta[both, 0] > 0)) if both.any() else 0.0
    )
    return features


def _geometry_fingerprint(sequence: np.ndarray) -> str:
    """Hash per-frame normalized body geometry, invariant to translation/scale/reflection."""

    joint_indexes = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    upper = np.triu_indices(len(joint_indexes), k=1)
    frame_signatures: list[np.ndarray] = []
    for frame in sequence:
        person_signatures: list[np.ndarray] = []
        for person in range(2):
            selected = frame[person, joint_indexes]
            coordinates = selected[:, :2].astype(np.float64)
            visible = selected[:, 2] > 0
            delta = coordinates[:, None, :] - coordinates[None, :, :]
            distances = np.linalg.norm(delta, axis=2)[upper]
            pair_visible = (visible[:, None] & visible[None, :])[upper]
            positive = distances[pair_visible & (distances > 0)]
            scale = float(np.median(positive)) if positive.size else 1.0
            normalized = np.full(len(distances), -1.0, dtype=np.float64)
            normalized[pair_visible] = distances[pair_visible] / max(scale, 1e-9)
            quantized = np.rint(normalized * 1_000).astype(np.int32)
            signature = np.concatenate([visible.astype(np.int32), quantized])
            person_signatures.append(signature)
        person_signatures.sort(key=lambda values: values.tobytes())
        frame_signatures.extend(person_signatures)
    digest = hashlib.sha256()
    digest.update(str(sequence.shape[0]).encode())
    digest.update(b"\0")
    for signature in frame_signatures:
        digest.update(signature.tobytes())
    return digest.hexdigest()


def parse_pose_file(task: tuple[Path, str]) -> dict[str, Any]:
    path, snapshot = task
    directory = path.parent.name
    if directory not in _DIRECTORY_LABELS or not path.name.endswith(_SUFFIX):
        raise ValueError(f"Unexpected FENCINGLY pose path: {path}")
    source_name = path.name.removesuffix(_SUFFIX)
    blob = path.read_bytes()
    file_sha256 = hashlib.sha256(blob).hexdigest()
    payload = json.loads(blob)
    if not isinstance(payload, dict) or set(payload) != {"frames"}:
        raise ValueError(f"FENCINGLY top-level schema changed: {path.name}")
    frames = payload["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"FENCINGLY frame list is empty: {path.name}")
    selected = frames[:100]
    sequence = np.zeros((len(selected), 2, 25, 3), dtype=np.float32)
    pose_count = 0
    extra_pose_frames = 0
    for frame_index, frame in enumerate(selected):
        if not isinstance(frame, dict) or not isinstance(frame.get("poses"), list):
            raise ValueError(f"FENCINGLY frame schema changed: {path.name}")
        poses = frame["poses"]
        pose_count += min(len(poses), 2)
        extra_pose_frames += int(len(poses) > 2)
        for person_index, pose in enumerate(poses[:2]):
            if not isinstance(pose, dict):
                raise ValueError(f"FENCINGLY pose schema changed: {path.name}")
            values = pose.get("pose_keypoints_2d")
            if not isinstance(values, list) or len(values) < 75:
                raise ValueError(f"FENCINGLY BODY_25 payload is incomplete: {path.name}")
            array = np.asarray(values[:75], dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"FENCINGLY pose payload is nonfinite: {path.name}")
            sequence[frame_index, person_index] = array.reshape(25, 3)
    features = _pose_features(sequence)
    return {
        "snapshot": snapshot,
        "source_directory": directory,
        "upstream_side_label": _DIRECTORY_LABELS[directory],
        "source_name": source_name,
        "bytes": len(blob),
        "sha256": file_sha256,
        "geometry_fingerprint": _geometry_fingerprint(sequence),
        "source_frame_count": len(frames),
        "modeled_frame_count": len(selected),
        "first_two_pose_slots": pose_count,
        "frames_with_more_than_two_poses": extra_pose_frames,
        **features,
    }


def deduplicate_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_hash: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["sha256"])].append(row)
    retained: list[dict[str, Any]] = []
    label_conflict_hashes = 0
    label_conflict_copies = 0
    cross_snapshot_hashes = 0
    for content_hash, copies in sorted(by_hash.items()):
        labels = {str(copy["upstream_side_label"]) for copy in copies}
        snapshots = {str(copy["snapshot"]) for copy in copies}
        cross_snapshot_hashes += int(len(snapshots) > 1)
        if len(labels) != 1:
            label_conflict_hashes += 1
            label_conflict_copies += len(copies)
            continue
        canonical = min(
            copies,
            key=lambda row: (
                str(row["snapshot"]) != "v1",
                str(row["source_directory"]).endswith("B"),
                str(row["source_name"]),
            ),
        ).copy()
        canonical["clip_id"] = _stable_id("fencingly-clip", content_hash)
        canonical["filename_group_id"] = _stable_id(
            "fencingly-name", str(canonical["source_name"])
        )
        canonical["geometry_group_id"] = _stable_id(
            "fencingly-geometry", str(canonical["geometry_fingerprint"])
        )
        canonical["exact_copy_count"] = len(copies)
        canonical["appears_in_both_snapshots"] = len(snapshots) > 1
        retained.append(canonical)
    audit = {
        "file_copies": sum(len(copies) for copies in by_hash.values()),
        "exact_unique_sequences": len(by_hash),
        "exact_duplicate_file_copies": sum(len(copies) - 1 for copies in by_hash.values()),
        "exact_duplicate_hash_groups": sum(len(copies) > 1 for copies in by_hash.values()),
        "cross_snapshot_exact_duplicate_hashes": cross_snapshot_hashes,
        "label_conflict_unique_sequences_excluded": label_conflict_hashes,
        "label_conflict_file_copies_excluded": label_conflict_copies,
        "retained_exact_unique_sequences": len(retained),
    }
    return retained, audit


def prepare(
    extracted: dict[str, Path],
    archive_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    if workers < 1 or workers > 8:
        raise ValueError("FENCINGLY workers must be between 1 and 8")
    tasks: list[tuple[Path, str]] = []
    directory_counts: Counter[str] = Counter()
    for spec in ARCHIVES:
        source = extracted.get(spec.key)
        if source is None:
            raise ValueError(f"Missing extracted FENCINGLY snapshot: {spec.key}")
        paths = sorted(source.glob("*/*.json"))
        if len(paths) != spec.json_files:
            raise ValueError(f"FENCINGLY {spec.key} extracted JSON count changed")
        for path in paths:
            directory_counts[f"{spec.key}:{path.parent.name}"] += 1
            tasks.append((path, spec.key))

    rows: list[dict[str, Any]] = []
    if workers == 1:
        for index, row in enumerate(map(parse_pose_file, tasks), start=1):
            rows.append(row)
            if index % 5_000 == 0 or index == len(tasks):
                print(f"FENCINGLY audit: {index}/{len(tasks)} JSON files", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, row in enumerate(
                executor.map(parse_pose_file, tasks, chunksize=20), start=1
            ):
                rows.append(row)
                if index % 5_000 == 0 or index == len(tasks):
                    print(f"FENCINGLY audit: {index}/{len(tasks)} JSON files", flush=True)

    retained, deduplication = deduplicate_rows(rows)
    feature_prefixes = ("pose_", "pair_")
    public_columns = [
        "clip_id",
        "filename_group_id",
        "geometry_group_id",
        "upstream_side_label",
        "source_frame_count",
        "modeled_frame_count",
        "first_two_pose_slots",
        "frames_with_more_than_two_poses",
        "exact_copy_count",
        "appears_in_both_snapshots",
        *[
            key
            for key in retained[0]
            if key.startswith(feature_prefixes)
        ],
    ] if retained else []
    frame = pd.DataFrame(retained)
    if not frame.empty:
        frame = frame[public_columns].sort_values("clip_id", kind="stable").reset_index(drop=True)
        if frame["clip_id"].duplicated().any():
            raise ValueError("FENCINGLY retained clip IDs are duplicated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    retained_hashes = {str(row["sha256"]) for row in retained}
    unique_bytes = sum(
        int(copies[0]["bytes"])
        for content_hash, copies in _group_rows_by_hash(rows).items()
        if content_hash in retained_hashes
    )
    frame_counts = Counter(int(row["source_frame_count"]) for row in retained)
    source_labels = Counter(str(row["upstream_side_label"]) for row in retained)
    geometry_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        geometry_groups[str(row["geometry_fingerprint"])].append(row)
    repeated_geometry = [copies for copies in geometry_groups.values() if len(copies) > 1]
    geometry_size_distribution = Counter(len(copies) for copies in repeated_geometry)
    geometry_label_conflicts = sum(
        len({str(copy["upstream_side_label"]) for copy in copies}) > 1
        for copies in repeated_geometry
    )
    archive_reports = {
        spec.key: {
            "filename": spec.filename,
            "bytes": (archive_directory / spec.filename).stat().st_size,
            "sha256": sha256_file(archive_directory / spec.filename),
            "json_files": spec.json_files,
        }
        for spec in ARCHIVES
    }
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        "archives": archive_reports,
        "compressed_archive_bytes": sum(report["bytes"] for report in archive_reports.values()),
        "extracted_json_file_copies": len(rows),
        "extracted_json_bytes": sum(int(row["bytes"]) for row in rows),
        "directory_copy_counts": dict(sorted(directory_counts.items())),
        **deduplication,
        "retained_unique_json_bytes": unique_bytes,
        "retained_side_label_counts": dict(sorted(source_labels.items())),
        "retained_source_frame_count_distribution": {
            str(key): value for key, value in sorted(frame_counts.items())
        },
        "retained_source_frames": sum(
            int(row["source_frame_count"]) for row in retained
        ),
        "retained_modeled_frames": sum(
            int(row["modeled_frame_count"]) for row in retained
        ),
        "retained_filename_groups": int(frame["filename_group_id"].nunique())
        if not frame.empty
        else 0,
        "transformation_invariant_geometry_groups": len(geometry_groups),
        "possible_transform_equivalent_groups": len(repeated_geometry),
        "possible_transform_equivalent_extra_sequences": sum(
            len(copies) - 1 for copies in repeated_geometry
        ),
        "possible_transform_equivalent_label_conflict_groups": geometry_label_conflicts,
        "possible_transform_equivalent_group_size_distribution": {
            str(key): value for key, value in sorted(geometry_size_distribution.items())
        },
        "feature_count": sum(column.startswith(feature_prefixes) for column in frame),
        "feature_output": "data/processed/fencingly-pose-features.parquet",
        "feature_output_sha256": sha256_file(output_path),
        "claim_boundary": (
            "The two pinned archives are audited byte-for-byte and exact pose-sequence copies "
            "are collapsed before counting. The dataset card contains only an Apache-2.0 tag. "
            "Its code maps IZQUIERDA to one and DERECHA to zero and performs a random file-level "
            "split, but does not document whether labels mean winner, touch side, referee call, "
            "or another task. Original videos, source-group IDs, events, athletes, weapon, frame "
            "rate, and annotation protocol are absent. Filename labels are therefore retained "
            "without semantic reinterpretation. The compact pose features are local inventory "
            "artifacts, not a leakage-safe external benchmark. A deliberately conservative "
            "translation/scale/reflection-invariant body-geometry fingerprint is supplied so "
            "possible transformed equivalents can stay in one split, but it is not an original-"
            "video identifier and cannot establish independent examples or generalization."
        ),
    }
    write_json(report_path, summary)
    return summary


def _group_rows_by_hash(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sha256"])].append(row)
    return dict(grouped)


def fetch_and_prepare(
    archive_directory: Path,
    extraction_root: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    workers: int = 4,
) -> dict[str, Any]:
    extracted = acquire(archive_directory, extraction_root, user_agent=user_agent)
    return prepare(
        extracted,
        archive_directory,
        output_path,
        report_path,
        workers=workers,
    )
