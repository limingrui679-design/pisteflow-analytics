"""Audit a few-shot world-level epee movement clip repository."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json

SOURCE_URL = "https://github.com/d-mali/fie-epee-moves-clasificator"
LICENSE = "No repository license; local analysis only"
CLASSES = {
    "fleche",
    "jump",
    "lunge_direct",
    "lunge_jumping",
    "squat",
    "step_backward",
    "step_forward",
}
ANNOTATION_KEYS = {
    "class",
    "ranka",
    "taikinio_zona",
    "fencer",
    "reversed",
    "source_video",
    "start_frame",
    "end_frame",
}


def _group_id(source_video: str) -> str:
    digest = hashlib.sha256(f"pisteflow-epee-source:{source_video}".encode()).hexdigest()[:20]
    return f"epee-source:{digest}"


def _media_audit(classification_directory: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional media dependency
        raise RuntimeError("Install the video extra to validate epee clips") from exc

    video_bytes = 0
    array_bytes = 0
    decoded_frames = 0
    array_shapes: Counter[str] = Counter()
    source_files: list[Path] = []
    for row in rows:
        clip_id = str(row["clip_id"])
        action_class = str(row["movement_class"])
        base = classification_directory / action_class / "raw" / clip_id
        video_path = base.with_suffix(".mp4")
        array_path = base.with_suffix(".npy")
        if not video_path.is_file() or not array_path.is_file():
            raise ValueError(f"Epee movement pair is missing for {clip_id}")
        source_files.extend([video_path, array_path])
        video_bytes += video_path.stat().st_size
        array_bytes += array_path.stat().st_size

        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _frame = capture.read()
        capture.release()
        if not ok or frame_count < 1 or fps <= 0 or width < 1 or height < 1:
            raise ValueError(f"Epee movement video does not decode: {clip_id}")
        decoded_frames += frame_count

        array = np.load(array_path, allow_pickle=False)
        if array.size < 1 or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Epee point-tracking array is empty or nonnumeric: {clip_id}")
        if not np.isfinite(array).all():
            raise ValueError(f"Epee point-tracking array contains nonfinite values: {clip_id}")
        array_shapes[str(tuple(int(value) for value in array.shape))] += 1

    digest = hashlib.sha256()
    for path in sorted(source_files):
        digest.update(path.relative_to(classification_directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return {
        "media_pairs_verified": len(rows),
        "video_bytes": video_bytes,
        "point_tracking_bytes": array_bytes,
        "decoded_video_frames": decoded_frames,
        "point_tracking_shapes": dict(sorted(array_shapes.items())),
        "media_tree_sha256": digest.hexdigest(),
    }


def prepare(
    annotations_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    classification_directory: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Epee movement annotations are not a nonempty object")

    rows: list[dict[str, Any]] = []
    for clip_id, annotation in sorted(payload.items()):
        if not isinstance(clip_id, str) or not isinstance(annotation, dict):
            raise ValueError("Epee movement annotation row is malformed")
        if set(annotation) != ANNOTATION_KEYS:
            raise ValueError(f"Epee movement annotation schema changed: {clip_id}")
        action_class = str(annotation["class"])
        if action_class not in CLASSES:
            raise ValueError(f"Unknown epee movement class: {action_class}")
        side = str(annotation["fencer"])
        if side not in {"left", "right"}:
            raise ValueError(f"Unknown epee fencer side: {side}")
        source_attribute = int(annotation["ranka"])
        target_zone = int(annotation["taikinio_zona"])
        start_frame = int(annotation["start_frame"])
        end_frame = int(annotation["end_frame"])
        if source_attribute not in {0, 1} or target_zone not in {1, 2, 3}:
            raise ValueError(f"Epee annotation source codes changed: {clip_id}")
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(f"Epee annotation frame range is invalid: {clip_id}")
        source_video = str(annotation["source_video"])
        rows.append(
            {
                "clip_id": clip_id,
                "weapon": "epee",
                "movement_class": action_class,
                "source_group_id": _group_id(source_video),
                "source_start_frame": start_frame,
                "source_end_frame": end_frame,
                "source_frame_count_inclusive": end_frame - start_frame + 1,
                "fencer_side": side,
                "source_binary_attribute": source_attribute,
                "source_target_zone_code": target_zone,
                "source_marked_reversed": bool(annotation["reversed"]),
            }
        )

    frame = pd.DataFrame(rows)
    if frame["clip_id"].duplicated().any():
        raise ValueError("Epee movement clip IDs are not unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    media_summary = (
        _media_audit(classification_directory, rows)
        if classification_directory is not None
        else {"media_pairs_verified": 0, "metadata_only": True}
    )
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "weapon": "epee",
        "annotated_clips": len(frame),
        "source_video_groups": int(frame["source_group_id"].nunique()),
        "annotated_source_frames_inclusive": int(frame["source_frame_count_inclusive"].sum()),
        "class_counts": {
            str(key): int(value)
            for key, value in frame["movement_class"].value_counts().sort_index().items()
        },
        "fencer_side_counts": {
            str(key): int(value)
            for key, value in frame["fencer_side"].value_counts().sort_index().items()
        },
        "target_zone_source_code_counts": {
            str(key): int(value)
            for key, value in frame["source_target_zone_code"].value_counts().sort_index().items()
        },
        "source_marked_reversed": int(frame["source_marked_reversed"].sum()),
        "annotations_sha256": sha256_file(annotations_path),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
        "media_audit": media_summary,
        "claim_boundary": (
            "Few-shot epee movement clips collected upstream from world-level competition "
            "footage. All clips from the same one of 61 source videos must stay in one split. "
            "The seven upstream action labels and numeric source attributes are preserved without "
            "reinterpreting referee calls, target anatomy, outcomes, or athlete identity. The "
            "repository has no license, so clips, arrays, row metadata, and models remain local."
        ),
    }
    write_json(report_path, summary)
    return summary
