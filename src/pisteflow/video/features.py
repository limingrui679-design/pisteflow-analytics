"""Leakage-aware, source-video-level features for the three-weapon baseline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pisteflow.models import VideoRecord
from pisteflow.sources.commons import load_manifest


@dataclass(frozen=True)
class VideoFeatureConfig:
    window_seconds: float = 4.0
    stride_seconds: float = 4.0
    frames_per_window: int = 12
    width: int = 160
    height: int = 90
    minimum_frames: int = 4


def media_path(media_root: Path, record: VideoRecord) -> Path:
    if record.direct_media_url is None:
        raise ValueError(f"{record.video_id} has no downloadable open media URL")
    extension = Path(str(record.direct_media_url).split("?", 1)[0]).suffix or ".video"
    safe_id = record.video_id.replace(":", "_")
    return media_root / record.weapon.value / f"{safe_id}{extension}"


def _frame_descriptor(
    frame: np.ndarray,
    config: VideoFeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    values: list[float] = []
    for array in (resized, hsv):
        values.extend(np.mean(array, axis=(0, 1)).astype(float).tolist())
        values.extend(np.std(array, axis=(0, 1)).astype(float).tolist())
    for channel, upper in ((0, 180), (1, 256), (2, 256)):
        histogram_raw = cv2.calcHist([hsv], [channel], None, [8], [0, upper]).ravel()
        histogram_normalized = histogram_raw / max(float(histogram_raw.sum()), 1.0)
        values.extend(histogram_normalized.astype(float).tolist())

    edges = cv2.Canny(gray, 70, 160)
    values.append(float(np.mean(edges > 0)))
    horizontal = cv2.resize(gray, (12, 1), interpolation=cv2.INTER_AREA).ravel() / 255.0
    vertical = cv2.resize(gray, (1, 6), interpolation=cv2.INTER_AREA).ravel() / 255.0
    values.extend(horizontal.astype(float).tolist())
    values.extend(vertical.astype(float).tolist())
    return np.asarray(values, dtype=np.float32), gray


def _motion_descriptor(grays: list[np.ndarray]) -> np.ndarray:
    magnitudes: list[np.ndarray] = []
    horizontal: list[float] = []
    vertical: list[float] = []
    for previous, current in pairwise(grays):
        flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            previous,
            current,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(magnitude.ravel())
        horizontal.append(float(np.mean(flow[..., 0])))
        vertical.append(float(np.mean(flow[..., 1])))
    if not magnitudes:
        return np.zeros(8, dtype=np.float32)
    merged = np.concatenate(magnitudes)
    finite = merged[np.isfinite(merged)]
    if not finite.size:
        return np.zeros(8, dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(finite)),
            float(np.std(finite)),
            *np.quantile(finite, [0.5, 0.9, 0.99]).astype(float).tolist(),
            float(np.mean(finite > 1.0)),
            float(np.mean(horizontal)),
            float(np.mean(vertical)),
        ],
        dtype=np.float32,
    )


def _feature_names(frame_width: int) -> list[str]:
    return [
        *(f"frame_mean_{index:03d}" for index in range(frame_width)),
        *(f"frame_std_{index:03d}" for index in range(frame_width)),
        "flow_mean",
        "flow_std",
        "flow_p50",
        "flow_p90",
        "flow_p99",
        "flow_active_share",
        "flow_dx_mean",
        "flow_dy_mean",
    ]


def _read_sampled_frames(
    capture: cv2.VideoCapture,
    frame_indices: np.ndarray,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            frames.append(frame)
    return frames


def _training_intervals(record: VideoRecord, duration: float) -> list[tuple[float, float]]:
    raw_segments = record.metadata.get("training_segments", [])
    if not raw_segments:
        return [(0.0, duration)]
    if not isinstance(raw_segments, list):
        raise ValueError(f"Invalid training segments for {record.video_id}")
    intervals: list[tuple[float, float]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, list) or len(raw_segment) != 2:
            raise ValueError(f"Invalid training segment for {record.video_id}")
        start, end = (float(raw_segment[0]), float(raw_segment[1]))
        if start < 0 or end <= start or end > duration + 0.05:
            raise ValueError(f"Training segment is outside {record.video_id}")
        intervals.append((start, min(end, duration)))
    for previous, current in pairwise(intervals):
        if current[0] < previous[1]:
            raise ValueError(f"Training segments overlap for {record.video_id}")
    return intervals


def extract_video_windows(
    path: Path,
    record: VideoRecord,
    config: VideoFeatureConfig,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid video metadata for {path}: fps={fps}, frames={frame_count}")
        duration = frame_count / fps
        rows: list[dict[str, Any]] = []
        window_index = 0
        for segment_index, (segment_start, segment_end) in enumerate(
            _training_intervals(record, duration)
        ):
            segment_duration = segment_end - segment_start
            window = min(config.window_seconds, segment_duration)
            starts = np.arange(
                segment_start,
                max(segment_end - window, segment_start) + 1e-9,
                config.stride_seconds,
            )
            if not starts.size:
                starts = np.asarray([segment_start])
            for start_seconds in starts:
                end_seconds = min(float(start_seconds + window), segment_end)
                start_frame = int(start_seconds * fps)
                end_frame = max(start_frame + 1, int(end_seconds * fps) - 1)
                indices = np.linspace(
                    start_frame,
                    end_frame,
                    num=config.frames_per_window,
                    dtype=int,
                )
                frames = _read_sampled_frames(capture, np.unique(indices))
                if len(frames) < config.minimum_frames:
                    continue
                frame_descriptors: list[np.ndarray] = []
                grays: list[np.ndarray] = []
                for frame in frames:
                    descriptor, gray = _frame_descriptor(frame, config)
                    frame_descriptors.append(descriptor)
                    grays.append(gray)
                matrix = np.stack(frame_descriptors)
                feature_values = np.concatenate(
                    [np.mean(matrix, axis=0), np.std(matrix, axis=0), _motion_descriptor(grays)]
                )
                names = _feature_names(matrix.shape[1])
                row: dict[str, Any] = {
                    "video_id": record.video_id,
                    "title": record.title,
                    "weapon": record.weapon.value,
                    "split": record.split,
                    "group_id": record.group_id or record.video_id,
                    "segment_index": segment_index,
                    "window_index": window_index,
                    "window_start": float(start_seconds),
                    "window_end": end_seconds,
                    "fps": fps,
                    "video_duration": duration,
                    "source_path": str(path),
                }
                row.update(dict(zip(names, feature_values.astype(float), strict=True)))
                rows.append(row)
                window_index += 1
        return rows
    finally:
        capture.release()


def extract_manifest_features(
    manifest_path: Path,
    media_root: Path,
    output_path: Path,
    *,
    config: VideoFeatureConfig | None = None,
) -> dict[str, Any]:
    active_config = config or VideoFeatureConfig()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in load_manifest(manifest_path):
        try:
            path = media_path(media_root, record)
            if not path.exists():
                raise FileNotFoundError(path)
            rows.extend(extract_video_windows(path, record, active_config))
        except (FileNotFoundError, ValueError, cv2.error) as exc:
            failures.append({"video_id": record.video_id, "error": str(exc)})
    if not rows:
        raise RuntimeError("No video windows were extracted")
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {
        "videos": int(frame["video_id"].nunique()),
        "windows": len(frame),
        "features": sum(column.startswith(("frame_", "flow_")) for column in frame.columns),
        "by_weapon": frame.groupby("weapon")["video_id"].nunique().to_dict(),
        "by_split": frame.groupby("split")["video_id"].nunique().to_dict(),
        "failures": failures,
        "output": str(output_path),
    }
