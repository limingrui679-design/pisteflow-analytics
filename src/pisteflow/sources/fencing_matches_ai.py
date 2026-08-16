"""Local-only audit and compact features for the public foil pose archive."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = "https://www.kaggle.com/datasets/alexpgd/sport-fencing-matches-ai"
ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "alexpgd/sport-fencing-matches-ai?datasetVersionNumber=1"
)
EXPECTED_BYTES = 855_286_232
EXPECTED_MD5 = "8e3036e04bf8fc459fd73bceecb00e51"
LICENSE = "Unknown (Kaggle dataset metadata)"
_DESCRIPTOR = "Kaggle/ClipDescriptorKaggle.csv"
_COLUMNS = [
    "id",
    "ClipName",
    "URL",
    "Start_frame",
    "End_frame",
    "Light1",
    "Light2",
    "Hit_label",
]
_OUTCOMES = {"1-0-0": "class_100", "0-1-0": "class_010", "0-0-1": "class_001"}


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _youtube_id(value: str) -> str:
    parsed = urlparse(value)
    values = parse_qs(parsed.query).get("v", [])
    if parsed.hostname != "www.youtube.com" or parsed.path != "/watch" or len(values) != 1:
        raise ValueError(f"Unexpected source-video URL in foil pose archive: {value}")
    return values[0]


def _group_id(youtube_id: str) -> str:
    digest = sha256(f"fencing-matches-ai:{youtube_id}".encode()).hexdigest()[:20]
    return f"fma-source-video:{digest}"


def normalize_descriptor(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate source rows and mark duplicate source-video windows."""

    if list(frame.columns) != _COLUMNS:
        raise ValueError("Fencing Matches AI descriptor columns changed")
    result = frame.copy()
    if result.isna().any().any():
        raise ValueError("Fencing Matches AI descriptor contains missing values")
    result["id"] = pd.to_numeric(result["id"], errors="raise").astype("int32")
    result["Start_frame"] = pd.to_numeric(result["Start_frame"], errors="raise").astype("int32")
    result["End_frame"] = pd.to_numeric(result["End_frame"], errors="raise").astype("int32")
    if not result["End_frame"].sub(result["Start_frame"]).eq(75).all():
        raise ValueError("Fencing Matches AI clips are no longer uniformly 75 frames")
    for column in ["Light1", "Light2"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int8")
        if not set(result[column].unique()).issubset({1, 2}):
            raise ValueError(f"Fencing Matches AI {column} codes changed")
    observed_outcomes = set(result["Hit_label"].astype(str).unique())
    if not observed_outcomes or not observed_outcomes.issubset(_OUTCOMES):
        raise ValueError("Fencing Matches AI outcome codes changed")
    if result["ClipName"].duplicated().any() or result["id"].duplicated().any():
        raise ValueError("Fencing Matches AI clip identifiers are not unique")
    result["youtube_id"] = result["URL"].astype(str).map(_youtube_id)
    result["group_id"] = result["youtube_id"].map(_group_id)
    result["outcome_class"] = result["Hit_label"].map(_OUTCOMES)
    window_key = ["youtube_id", "Start_frame", "End_frame"]
    result["duplicate_source_window"] = result.duplicated(window_key, keep=False)
    result["training_eligible"] = ~result.duplicated(window_key, keep="first")
    duplicate_groups = result.loc[result["duplicate_source_window"], window_key].drop_duplicates()
    stats = {
        "descriptor_rows": len(result),
        "source_video_groups": int(result["group_id"].nunique()),
        "duplicate_source_window_groups": len(duplicate_groups),
        "duplicate_source_window_rows_after_first": int((~result["training_eligible"]).sum()),
        "training_eligible_unique_windows": int(result["training_eligible"].sum()),
    }
    return result, stats


def _stat(values: np.ndarray, operation: str) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0
    if operation == "mean":
        return float(np.mean(finite))
    if operation == "std":
        return float(np.std(finite))
    if operation == "range":
        return float(np.max(finite) - np.min(finite))
    if operation == "delta":
        return float(finite[-1] - finite[0])
    if operation == "max":
        return float(np.max(finite))
    raise ValueError(f"Unknown pose statistic: {operation}")


def _decode_pose_sequence(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(payload)
    if not isinstance(raw, list) or len(raw) != 75:
        raise ValueError("Tracked-pose sequence is not 75 frames")
    values = np.zeros((75, 2, 25, 3), dtype=np.float32)
    tracker_present = np.zeros((75, 2), dtype=bool)
    for frame_index, frame in enumerate(raw):
        if not isinstance(frame, list) or len(frame) != 2:
            raise ValueError(f"Tracked-pose frame {frame_index} does not contain two slots")
        for side_index, pose in enumerate(frame):
            if pose == []:
                continue
            pose_values = np.asarray(pose, dtype=np.float32)
            if pose_values.shape != (25, 3) or not np.isfinite(pose_values).all():
                raise ValueError(
                    f"Unexpected tracked pose at frame {frame_index}, side {side_index}: "
                    f"{pose_values.shape}"
                )
            values[frame_index, side_index] = pose_values
            tracker_present[frame_index, side_index] = True
    return values, tracker_present


def extract_pose_features(payload: bytes) -> dict[str, float]:
    """Reduce one 75x2x25 OpenPose sequence to leakage-safe motion summaries."""

    values, tracker_present = _decode_pose_sequence(payload)
    coordinates = values[..., :2]
    confidence = values[..., 2]
    visible = (confidence > 0.05) & np.any(coordinates != 0, axis=-1)
    masked = np.where(visible[..., None], coordinates, np.nan)
    centers = np.zeros((75, 2, 2), dtype=np.float32)
    spans = np.ones((75, 2), dtype=np.float32)
    normalized = np.full_like(masked, np.nan)
    for frame_index in range(75):
        for side_index in range(2):
            points = masked[frame_index, side_index]
            valid_points = points[np.isfinite(points).all(axis=1)]
            if not valid_points.size:
                centers[frame_index, side_index] = np.nan
                spans[frame_index, side_index] = np.nan
                continue
            center = np.mean(valid_points, axis=0)
            span = max(float(np.ptp(valid_points[:, 0])), float(np.ptp(valid_points[:, 1])), 1.0)
            centers[frame_index, side_index] = center
            spans[frame_index, side_index] = span
            normalized[frame_index, side_index] = (points - center) / span

    features: dict[str, float] = {}
    for side_index in range(2):
        side = side_index + 1
        features[f"pose_s{side}_tracker_payload_frame_fraction"] = float(
            np.mean(tracker_present[:, side_index])
        )
        for joint_index in range(25):
            joint = joint_index + 1
            for dimension_index, dimension in enumerate(["x", "y"]):
                series = normalized[:, side_index, joint_index, dimension_index]
                for operation in ["mean", "std", "range", "delta"]:
                    features[f"pose_s{side}_j{joint:02d}_{dimension}_{operation}"] = _stat(
                        series, operation
                    )
            joint_confidence = confidence[:, side_index, joint_index]
            features[f"pose_s{side}_j{joint:02d}_confidence_mean"] = float(
                np.mean(joint_confidence)
            )
            features[f"pose_s{side}_j{joint:02d}_visible_fraction"] = float(
                np.mean(visible[:, side_index, joint_index])
            )
            joint_xy = normalized[:, side_index, joint_index]
            speed = np.linalg.norm(np.diff(joint_xy, axis=0), axis=1)
            features[f"pose_s{side}_j{joint:02d}_speed_mean"] = _stat(speed, "mean")
            features[f"pose_s{side}_j{joint:02d}_speed_max"] = _stat(speed, "max")

        center_x = centers[:, side_index, 0] / 640.0
        center_y = centers[:, side_index, 1] / 360.0
        span_series = spans[:, side_index] / 640.0
        for name, series in [
            ("center_x", center_x),
            ("center_y", center_y),
            ("span", span_series),
        ]:
            for operation in ["mean", "std", "range", "delta"]:
                features[f"pose_s{side}_{name}_{operation}"] = _stat(series, operation)

    separation_x = (centers[:, 1, 0] - centers[:, 0, 0]) / 640.0
    separation_y = (centers[:, 1, 1] - centers[:, 0, 1]) / 360.0
    separation_distance = np.sqrt(separation_x**2 + separation_y**2)
    for name, series in [
        ("separation_x", separation_x),
        ("separation_y", separation_y),
        ("separation_distance", separation_distance),
    ]:
        for operation in ["mean", "std", "range", "delta"]:
            features[f"pair_{name}_{operation}"] = _stat(series, operation)
    features["pair_visible_frame_fraction"] = float(np.mean(np.isfinite(centers).all(axis=(1, 2))))
    return features


def prepare(archive_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if archive_path.stat().st_size != EXPECTED_BYTES or _md5_file(archive_path) != EXPECTED_MD5:
        raise ValueError("Fencing Matches AI archive size or checksum changed")
    feature_rows: list[dict[str, Any]] = []
    duplicate_payload_matches = 0
    clips_with_missing_tracker_slots = 0
    missing_tracker_frame_slots = 0
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        fencer_members = {
            name.removeprefix("Kaggle/FencersKeyPoints/").removesuffix(".json"): name
            for name in names
            if name.startswith("Kaggle/FencersKeyPoints/") and name.endswith(".json")
        }
        raw_members = {
            name.removeprefix("Kaggle/KeyPoints/").removesuffix(".json"): name
            for name in names
            if name.startswith("Kaggle/KeyPoints/") and name.endswith(".json")
        }
        descriptor = pd.read_csv(BytesIO(archive.read(_DESCRIPTOR)), encoding="utf-8-sig")
        normalized, stats = normalize_descriptor(descriptor)
        if set(normalized["Hit_label"].astype(str).unique()) != set(_OUTCOMES):
            raise ValueError("Fencing Matches AI full archive outcome classes changed")
        clip_names = set(normalized["ClipName"].astype(str))
        if set(fencer_members) != clip_names or set(raw_members) != clip_names:
            raise ValueError("Fencing Matches AI descriptor/member crosswalk changed")
        if len(names) != 1 + 2 * len(normalized):
            raise ValueError("Fencing Matches AI archive member inventory changed")

        duplicate_rows = normalized[normalized["duplicate_source_window"]]
        for _window, group in duplicate_rows.groupby(["youtube_id", "Start_frame", "End_frame"]):
            payload_hashes = {
                sha256(archive.read(fencer_members[str(clip_name)])).hexdigest()
                for clip_name in group["ClipName"]
            }
            duplicate_payload_matches += int(len(payload_hashes) == 1)

        for row in normalized.loc[normalized["training_eligible"]].itertuples(index=False):
            clip_name = str(row.ClipName)
            features = extract_pose_features(archive.read(fencer_members[clip_name]))
            tracker_slots = [
                features["pose_s1_tracker_payload_frame_fraction"],
                features["pose_s2_tracker_payload_frame_fraction"],
            ]
            missing_slots = round((2.0 - sum(tracker_slots)) * 75)
            missing_tracker_frame_slots += missing_slots
            clips_with_missing_tracker_slots += int(missing_slots > 0)
            feature_rows.append(
                {
                    "sample_id": "fma-" + sha256(clip_name.encode()).hexdigest()[:20],
                    "group_id": row.group_id,
                    "start_frame": int(row.Start_frame),
                    "end_frame": int(row.End_frame),
                    "outcome_code": str(row.Hit_label),
                    "outcome_class": str(row.outcome_class),
                    "light_1_code": int(row.Light1),
                    "light_2_code": int(row.Light2),
                    "weapon": "foil",
                    **features,
                }
            )
        uncompressed_bytes = sum(info.file_size for info in archive.infolist())

    result = pd.DataFrame(feature_rows).sort_values(
        ["group_id", "start_frame", "sample_id"], kind="stable"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    feature_columns = [column for column in result if column.startswith(("pose_", "pair_"))]
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Real 75-frame foil pose sequences derived from public YouTube competition videos. "
            "The archive has an unknown license and remains local-only. Forty-two repeated "
            "source-video windows are excluded from training. Evaluation must group all clips "
            "from the same one of 1,332 source videos. Outcome codes are preserved without "
            "claiming an undocumented left/right ordering, and scoring-light fields are not "
            "eligible model inputs. Athlete identities are unavailable, so video-group holdout "
            "is not proof of athlete-independent generalization."
        ),
        "source_url": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": sha256_file(archive_path),
        "archive_integrity_check": (
            "Exact byte size and MD5 matched the pinned Kaggle archive; every required member "
            "was read while validating the descriptor crosswalk and extracting features."
        ),
        "uncompressed_member_bytes": uncompressed_bytes,
        "tracked_pose_json_files": (
            len(feature_rows) + stats["duplicate_source_window_rows_after_first"]
        ),
        "raw_openpose_json_files": (
            len(feature_rows) + stats["duplicate_source_window_rows_after_first"]
        ),
        "compact_feature_rows": len(result),
        "motion_feature_columns": len(feature_columns),
        "frames_per_clip": 75,
        "tracked_fencers_per_frame": 2,
        "joints_per_fencer": 25,
        "tracked_pose_frame_rows": len(result) * 75,
        "tracked_joint_triplets": len(result) * 75 * 2 * 25,
        "clips_with_missing_tracker_frame_slots": clips_with_missing_tracker_slots,
        "missing_tracker_frame_slots": missing_tracker_frame_slots,
        "duplicate_window_groups_with_identical_tracked_payloads": duplicate_payload_matches,
        "outcome_distribution": dict(
            sorted(Counter(result["outcome_class"].astype(str)).items())
        ),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
        **stats,
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    output_path: Path,
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
    return prepare(archive_path, output_path, report_path)
