"""Privacy-minimized features for the multimodal fencing action dataset (MFAD)."""

from __future__ import annotations

import hashlib
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/EMRGSZU/MFAD"
SNAPSHOT = "153c5a630230805a1d469bd9fb46cde2fc128eda"
ARCHIVE_URL = f"https://codeload.github.com/EMRGSZU/MFAD/tar.gz/{SNAPSHOT}"
EXPECTED_ARCHIVE_BYTES = 28_215_044
EXPECTED_ARCHIVE_SHA256 = "a1a5bac22beb970dfd6e04c06ae6ad976c3632d8f4ac162633b11141cf40135f"
LICENSE = "No repository or dataset license found; raw files and row-level derivatives stay local"
SKELETON_PATTERN = re.compile(
    r"(?P<level>[123])_p(?P<subject>\d+)_a(?P<action>[123])_(?P<trial>\d+)\.npy"
)
SENSOR_DIRECTORY_PATTERN = re.compile(
    r"(?P<level>[123])(?P<subject>\d{2})(?P<action>[123])"
)
LEVELS = {"1": "amateur", "2": "subelite", "3": "elite"}
ACTIONS = {"1": "lunge", "2": "one_step_forward_lunge", "3": "fleche"}
PLACEMENTS = {"hand.txt": "hand", "left_foot.txt": "left_foot", "right_foot.txt": "right_foot"}
SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _subject_id(level: str, subject: str) -> str:
    digest = hashlib.sha256(f"pisteflow:mfad:{level}:{int(subject)}".encode()).hexdigest()[:20]
    return f"mfad-subject:{digest}"


def _safe_extract(archive_path: Path, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (root / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError("MFAD archive contains an unsafe member path")
            if member.issym() or member.islnk():
                raise ValueError("MFAD archive contains a link member")
        archive.extractall(output_directory, members=members, filter="data")
    candidates = sorted(path for path in output_directory.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError("MFAD archive root changed")
    return candidates[0]


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
        raise ValueError("MFAD archive checksum changed")
    expected_root = output_directory / f"MFAD-{SNAPSHOT}"
    if force or not expected_root.is_dir():
        return _safe_extract(archive_path, output_directory)
    return expected_root


def _stat(values: np.ndarray, operation: str) -> float:
    if operation == "mean":
        return float(np.mean(values))
    if operation == "std":
        return float(np.std(values))
    if operation == "range":
        return float(np.ptp(values))
    if operation == "delta":
        return float(values[-1] - values[0])
    if operation == "abs_mean":
        return float(np.mean(np.abs(values)))
    if operation == "rms":
        return float(np.sqrt(np.mean(np.square(values))))
    if operation == "max":
        return float(np.max(values))
    raise ValueError(f"Unknown MFAD statistic: {operation}")


def _skeleton_features(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 3 or values.shape[1:] != (18, 3):
        raise ValueError(f"MFAD skeleton shape changed: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("MFAD skeleton contains non-finite values")
    centers = np.mean(values, axis=1, keepdims=True)
    centered = values - centers
    scale = np.sqrt(np.mean(np.square(centered), axis=(1, 2)))
    if np.any(scale <= 0):
        raise ValueError("MFAD skeleton contains a zero-scale frame")
    normalized = centered / scale[:, None, None]

    features: dict[str, float] = {
        "time_steps": float(values.shape[0]),
        "duration_seconds_at_reported_100hz": float(values.shape[0] / 100.0),
        "body_scale_mean": float(np.mean(scale)),
        "body_scale_std": float(np.std(scale)),
        "body_scale_range": float(np.ptp(scale)),
        "body_scale_delta": float(scale[-1] - scale[0]),
    }
    centroid_velocity = np.linalg.norm(np.diff(centers[:, 0, :], axis=0), axis=1)
    normalized_velocity = np.linalg.norm(np.diff(normalized, axis=0), axis=2)
    for name, series in (
        ("centroid_speed", centroid_velocity),
        ("normalized_joint_speed", normalized_velocity.ravel()),
    ):
        features[f"{name}_mean"] = _stat(series, "mean")
        features[f"{name}_std"] = _stat(series, "std")
        features[f"{name}_max"] = _stat(series, "max")
    for joint in range(18):
        for axis, axis_name in enumerate(("x", "y", "z")):
            series = normalized[:, joint, axis]
            for operation in ("mean", "std", "range", "delta"):
                features[f"joint_{joint + 1:02d}_{axis_name}_{operation}"] = _stat(
                    series, operation
                )
    return features


def _prepare_skeleton(source_directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((source_directory / "skeleton").glob("*.npy")):
        match = SKELETON_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"MFAD skeleton filename changed: {path.name}")
        values = np.load(path, allow_pickle=False)
        level = match.group("level")
        subject = match.group("subject")
        action = match.group("action")
        trial = int(match.group("trial"))
        rows.append(
            {
                "sample_id": "mfad-skeleton:"
                + hashlib.sha256(path.name.encode()).hexdigest()[:20],
                "subject_group_id": _subject_id(level, subject),
                "skill_level": LEVELS[level],
                "action": ACTIONS[action],
                "trial_index_within_subject_action": trial,
                **_skeleton_features(values),
            }
        )
    if not rows:
        raise ValueError("MFAD skeleton directory is empty")
    frame = pd.DataFrame(rows).sort_values(
        ["subject_group_id", "action", "trial_index_within_subject_action"], kind="stable"
    )
    if frame["sample_id"].duplicated().any():
        raise ValueError("MFAD skeleton sample IDs are duplicated")
    stats = {
        "skeleton_sequences": len(frame),
        "participants": int(frame["subject_group_id"].nunique()),
        "sequences_by_level": {
            str(key): int(value)
            for key, value in frame["skill_level"].value_counts().sort_index().items()
        },
        "participants_by_level": {
            str(key): int(value)
            for key, value in (
                frame[["subject_group_id", "skill_level"]]
                .drop_duplicates()["skill_level"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "sequences_by_action": {
            str(key): int(value)
            for key, value in frame["action"].value_counts().sort_index().items()
        },
        "skeleton_time_steps": int(frame["time_steps"].sum()),
        "time_steps_min": int(frame["time_steps"].min()),
        "time_steps_max": int(frame["time_steps"].max()),
        "reported_depth_camera_hz": 100,
    }
    return frame, stats


def _delimiter(values: np.ndarray) -> bool:
    return bool(np.all(values[:5] == 0) and values[5] == 1000)


def _parse_sensor_file(path: Path) -> tuple[list[tuple[list[str], np.ndarray]], int, int]:
    segments: list[tuple[list[str], np.ndarray]] = []
    timestamps: list[str] = []
    values: list[np.ndarray] = []
    delimiter_rows = 0
    blank_rows = 0
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = line.strip().split()
        if not parts:
            blank_rows += 1
            continue
        if len(parts) != 7:
            raise ValueError(f"MFAD sensor row has {len(parts)} fields: {path}")
        row = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        if not np.isfinite(row).all():
            raise ValueError(f"MFAD sensor row is non-finite: {path}")
        if _delimiter(row):
            delimiter_rows += 1
            if values:
                segments.append((timestamps, np.vstack(values)))
                timestamps = []
                values = []
            continue
        timestamps.append(parts[0])
        values.append(row)
    if values:
        segments.append((timestamps, np.vstack(values)))
    return segments, delimiter_rows, blank_rows


def _sensor_features(values: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    for index, column in enumerate(SENSOR_COLUMNS):
        series = values[:, index]
        for operation in ("mean", "std", "range", "delta", "abs_mean", "rms"):
            features[f"{column}_{operation}"] = _stat(series, operation)
    return features


def _prepare_sensors(source_directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delimiter_rows = 0
    blank_rows = 0
    source_directories = sorted(
        path for path in (source_directory / "sensor").iterdir() if path.is_dir()
    )
    missing_files: Counter[str] = Counter()
    for directory in source_directories:
        match = SENSOR_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"MFAD sensor directory name changed: {directory.name}")
        level = match.group("level")
        subject = match.group("subject")
        action = match.group("action")
        for filename, placement in PLACEMENTS.items():
            path = directory / filename
            if not path.is_file():
                missing_files[placement] += 1
                continue
            segments, file_delimiters, file_blanks = _parse_sensor_file(path)
            delimiter_rows += file_delimiters
            blank_rows += file_blanks
            for segment_index, (timestamps, values) in enumerate(segments, start=1):
                parsed_times = pd.to_datetime(
                    timestamps,
                    format="%Y-%m-%d-%H:%M:%S-%f",
                    errors="raise",
                )
                elapsed_seconds = float(
                    (parsed_times[-1] - parsed_times[0]).total_seconds()
                )
                rows.append(
                    {
                        "segment_id": (
                            "mfad-sensor:"
                            + hashlib.sha256(
                                f"{directory.name}:{placement}:{segment_index}".encode()
                            ).hexdigest()[:20]
                        ),
                        "subject_group_id": _subject_id(level, subject),
                        "skill_level": LEVELS[level],
                        "action": ACTIONS[action],
                        "sensor_placement": placement,
                        "segment_index_within_file": segment_index,
                        "observations": len(values),
                        "elapsed_seconds_from_timestamps": elapsed_seconds,
                        "estimated_observations_per_second": (
                            (len(values) - 1) / elapsed_seconds if elapsed_seconds > 0 else 0.0
                        ),
                        **_sensor_features(values),
                    }
                )
    if not rows:
        raise ValueError("MFAD sensor directory has no numeric segments")
    frame = pd.DataFrame(rows).sort_values(
        ["subject_group_id", "action", "sensor_placement", "segment_index_within_file"],
        kind="stable",
    )
    median_observations = float(frame["observations"].median())
    outlier_threshold = median_observations * 5
    frame["long_segment_audit_flag"] = frame["observations"].gt(outlier_threshold)
    if frame["segment_id"].duplicated().any():
        raise ValueError("MFAD sensor segment IDs are duplicated")
    stats = {
        "sensor_action_directories": len(source_directories),
        "sensor_files": sum(
            (directory / filename).is_file()
            for directory in source_directories
            for filename in PLACEMENTS
        ),
        "missing_sensor_files_by_placement": dict(sorted(missing_files.items())),
        "sensor_stream_segments": len(frame),
        "sensor_segments_by_placement": {
            str(key): int(value)
            for key, value in frame["sensor_placement"].value_counts().sort_index().items()
        },
        "sensor_numeric_observations": int(frame["observations"].sum()),
        "sensor_delimiter_rows": delimiter_rows,
        "blank_sensor_rows_ignored": blank_rows,
        "median_segment_observations": median_observations,
        "long_segment_audit_threshold_observations": outlier_threshold,
        "long_segment_audit_flags": int(frame["long_segment_audit_flag"].sum()),
        "reported_imu_hz": 50,
    }
    return frame, stats


def _source_tree_sha256(source_directory: Path) -> str:
    paths = sorted((source_directory / "skeleton").glob("*.npy"))
    paths.extend(sorted((source_directory / "sensor").rglob("*.txt")))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(source_directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prepare(
    source_directory: Path,
    skeleton_output_path: Path,
    sensor_output_path: Path,
    report_path: Path,
    *,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    skeleton, skeleton_stats = _prepare_skeleton(source_directory)
    sensors, sensor_stats = _prepare_sensors(source_directory)
    if set(skeleton["subject_group_id"]) != set(sensors["subject_group_id"]):
        raise ValueError("MFAD skeleton and sensor participant sets differ")
    skeleton_output_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_output_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton.to_parquet(skeleton_output_path, index=False)
    sensors.to_parquet(sensor_output_path, index=False)

    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        **skeleton_stats,
        **sensor_stats,
        "source_tree_sha256": _source_tree_sha256(source_directory),
        "archive_bytes": archive_path.stat().st_size if archive_path is not None else None,
        "archive_sha256": sha256_file(archive_path) if archive_path is not None else None,
        "skeleton_output_sha256": sha256_file(skeleton_output_path),
        "sensor_output_sha256": sha256_file(sensor_output_path),
        "skeleton_output": "data/processed/mfad-skeleton-features.parquet",
        "sensor_output": "data/processed/mfad-sensor-features.parquet",
        "claim_boundary": (
            "Twenty-seven author-coded participants and 928 depth-camera skeleton sequences "
            "across lunge, one-step-forward-lunge, and fleche actions, plus partially missing "
            "hand/foot IMU streams. The author-defined amateur, subelite, and elite labels are "
            "preserved, not independently validated. Participant codes are pseudonymized. Sensor "
            "segments are not asserted to align one-to-one with skeleton trials, and unusually "
            "long segments remain flagged rather than silently deleted. The source does not "
            "identify weapon, sex, age, handedness, or competition outcome and declares no "
            "license, so raw files and row-level derivatives stay local. Any model split must "
            "hold out complete participant groups."
        ),
    }
    write_json(report_path, summary)
    return summary
