"""Acquire the two CC BY 4.0 fencing-motion clips from G1 Moves.

These are useful adjacent-domain motion-pretraining inputs, not verified sport
fencing, weapon-specific actions, athletes, competition bouts, or performance
outcomes.  The pinned snapshot also conflicts on performer name: its root
manifest says Maya Coro while both per-clip READMEs say Mitch Chaiet.  The
adapter preserves that conflict and never resolves identity by guessing.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://huggingface.co/datasets/exptech/g1-moves"
SNAPSHOT = "895d064b2385725e6aabf540461c219afc3e0916"
RESOLVE_ROOT = f"https://huggingface.co/datasets/exptech/g1-moves/resolve/{SNAPSHOT}"


@dataclass(frozen=True)
class FileSpec:
    expected_bytes: int
    expected_sha256: str


FILES: dict[str, FileSpec] = {
    "README.md": FileSpec(
        14_135,
        "673070cd21a725428f19d73c648dd6243a8911ed69bca270d836fc8d7f907629",
    ),
    "manifest.json": FileSpec(
        112_323,
        "69b58916ecd22f237e6424c96f9299b15e7deef4db262d2fa50aefb2b07a846b",
    ),
    "quality_report.json": FileSpec(
        37_965,
        "ae78d3a3aed26249503973a0daa9ab6a5050a9367404dd223ac49cea25aa5c28",
    ),
    "bonus/B_Fence1/README.md": FileSpec(
        878,
        "c0f82fa8192c0a53e6ea8f2fccd07c6143f0513d8ee3a7cd7fe76050d6880a6c",
    ),
    "bonus/B_Fence1/capture/B_Fence1.bvh": FileSpec(
        1_888_399,
        "a8160d429fa6d285171cd566a938afaa4acb8b72b50106de2dff894666e2efe7",
    ),
    "bonus/B_Fence1/capture/B_Fence1.mp4": FileSpec(
        365_988,
        "08dd24315d31598f83ac1dae2088e59c1499b4ab60cf48937b748be781982961",
    ),
    "bonus/B_Fence1/retarget/B_Fence1.csv": FileSpec(
        548_648,
        "ff2c752598fae9175db6f42524208d1b76c94eec1b7378b5272332b50aa1e537",
    ),
    "bonus/B_Fence1/training/B_Fence1.npz": FileSpec(
        2_903_042,
        "4637ec5d69229e50d513f1bf2cf4993411b900398227016024b1a204d10edb57",
    ),
    "bonus/B_Fence2/README.md": FileSpec(
        874,
        "58284087cd803c211cc4b73108afc44413bd0f1a37af98651d5dc0e02a8d071a",
    ),
    "bonus/B_Fence2/capture/B_Fence2.bvh": FileSpec(
        755_184,
        "aea0da934697c13e18b1b7a0439d377b84d284675a67aec26902f0aec1172a23",
    ),
    "bonus/B_Fence2/capture/B_Fence2.mp4": FileSpec(
        272_116,
        "06136c243c5dd82a390229d22abca9e721fbd4ca63ac9e4ff5fa5eff288c7fed",
    ),
    "bonus/B_Fence2/retarget/B_Fence2.csv": FileSpec(
        308_877,
        "18ddaea965952141592edbf50377aaeecb601bf5c309949a99117a50321a4b7a",
    ),
    "bonus/B_Fence2/training/B_Fence2.npz": FileSpec(
        1_143_298,
        "80cce94cfe37542f248085472f82eabc4d3bd31652bb3d6740958568d1fbaef7",
    ),
}

CLIPS = ("B_Fence1", "B_Fence2")
EXPECTED_MANIFEST = {
    "B_Fence1": {
        "frames": 1_620,
        "training_frames": 1_619,
        "fps": 60,
        "duration_s": 27.0,
        "bvh_joints": 51,
        "retarget_dof": 29,
    },
    "B_Fence2": {
        "frames": 638,
        "training_frames": 637,
        "fps": 60,
        "duration_s": 10.63,
        "bvh_joints": 51,
        "retarget_dof": 29,
    },
}

README_PERFORMER = "Mitch Chaiet"
MANIFEST_PERFORMER = "Maya Coro"
SOURCE_GROUP_ID = "g1-moves:fencing-performer-label-conflict-b"


def acquire(
    output_directory: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> Path:
    """Download and verify the pinned metadata and two fencing-motion clips."""

    for relative_path, spec in FILES.items():
        destination = output_directory / relative_path
        if force:
            destination.unlink(missing_ok=True)
        download_resumable(
            f"{RESOLVE_ROOT}/{relative_path}",
            destination,
            user_agent=user_agent,
            expected_bytes=spec.expected_bytes,
        )
        if sha256_file(destination) != spec.expected_sha256:
            raise ValueError(f"G1 Moves checksum changed: {relative_path}")
    return output_directory


def parse_bvh_header(text: str) -> tuple[int, float, int]:
    """Return declared frames, frame time, and total channel count."""

    frames_match = re.search(r"^Frames:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    time_match = re.search(
        r"^Frame Time:\s*([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE
    )
    if frames_match is None or time_match is None:
        raise ValueError("BVH motion header is incomplete")
    channels = sum(
        int(value)
        for value in re.findall(r"^\s*CHANNELS\s+(\d+)\s+", text, flags=re.MULTILINE)
    )
    if channels <= 0:
        raise ValueError("BVH hierarchy has no channels")
    return int(frames_match.group(1)), float(time_match.group(1)), channels


def _inspect_video(path: Path) -> dict[str, int | float]:
    try:
        cv2: Any = importlib.import_module("cv2")
    except ImportError as error:  # pragma: no cover - environment-specific guidance
        raise RuntimeError("Install pisteflow-analytics[video] to audit preview media") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open G1 Moves preview: {path}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    readable_frames = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        readable_frames += 1
    capture.release()
    if reported_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid G1 Moves preview metadata: {path}")
    return {
        "preview_reported_frames": reported_frames,
        "preview_readable_frames": readable_frames,
        "preview_fps": fps,
        "preview_width": width,
        "preview_height": height,
    }


def _parse_readme_performer(text: str) -> str:
    match = re.search(r"\*\*Performer:\*\*\s*([^|\n]+)", text)
    if match is None:
        raise ValueError("Per-clip README performer field is absent")
    return match.group(1).strip()


def _inspect_clip(
    source_directory: Path,
    clip: str,
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    expected = EXPECTED_MANIFEST[clip]
    for key, value in expected.items():
        if manifest_row.get(key) != value:
            raise ValueError(f"G1 Moves manifest changed for {clip}: {key}")
    if manifest_row.get("source") != "mocap":
        raise ValueError(f"G1 Moves source type changed for {clip}")

    clip_root = source_directory / "bonus" / clip
    readme_performer = _parse_readme_performer(
        (clip_root / "README.md").read_text(encoding="utf-8")
    )
    manifest_performer = str(manifest_row.get("performer"))
    if readme_performer != README_PERFORMER or manifest_performer != MANIFEST_PERFORMER:
        raise ValueError(f"G1 Moves performer-label conflict changed for {clip}")

    bvh_path = clip_root / "capture" / f"{clip}.bvh"
    bvh_frames, bvh_frame_time, bvh_channels = parse_bvh_header(
        bvh_path.read_text(encoding="utf-8")
    )
    if bvh_frames != expected["frames"] or abs((1 / bvh_frame_time) - 60.0) > 0.01:
        raise ValueError(f"G1 Moves BVH timing changed for {clip}")

    csv_path = clip_root / "retarget" / f"{clip}.csv"
    retarget = pd.read_csv(csv_path, header=None)
    # The CSV stores root position (3), root quaternion (4), and 29 joint
    # angles, despite the per-clip README describing it simply as joint angles.
    if retarget.shape != (expected["frames"], 36):
        raise ValueError(f"G1 Moves retarget CSV shape changed for {clip}")
    if not np.isfinite(retarget.to_numpy(dtype=float)).all():
        raise ValueError(f"G1 Moves retarget CSV contains non-finite values: {clip}")

    npz_path = clip_root / "training" / f"{clip}.npz"
    with np.load(npz_path, allow_pickle=False) as arrays:
        expected_shapes = {
            "joint_pos": (expected["training_frames"], 29),
            "joint_vel": (expected["training_frames"], 29),
            "body_pos_w": (expected["training_frames"], 30, 3),
            "body_quat_w": (expected["training_frames"], 30, 4),
        }
        for key, shape in expected_shapes.items():
            if key not in arrays or arrays[key].shape != shape:
                raise ValueError(f"G1 Moves training array changed for {clip}: {key}")
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"G1 Moves training array is non-finite for {clip}: {key}")
        array_names = sorted(arrays.files)

    return {
        "clip_id": clip,
        "source_group_id": SOURCE_GROUP_ID,
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "dataset_license": "CC BY 4.0",
        "domain": "adjacent_generic_fencing_motion",
        "weapon": "unknown",
        "verified_athlete": False,
        "capture_frames": int(expected["frames"]),
        "training_frames": int(expected["training_frames"]),
        "fps": int(expected["fps"]),
        "duration_seconds": float(expected["duration_s"]),
        "bvh_joints_upstream": int(expected["bvh_joints"]),
        "bvh_channels_parsed": bvh_channels,
        "retarget_dof": int(expected["retarget_dof"]),
        "retarget_csv_columns": int(retarget.shape[1]),
        "training_body_count": 30,
        "training_array_names": json.dumps(array_names),
        "manifest_performer": manifest_performer,
        "per_clip_readme_performer": readme_performer,
        "performer_label_conflict": manifest_performer != readme_performer,
        "bvh_sha256": sha256_file(bvh_path),
        "retarget_csv_sha256": sha256_file(csv_path),
        "training_npz_sha256": sha256_file(npz_path),
        **_inspect_video(clip_root / "capture" / f"{clip}.mp4"),
    }


def prepare(source_directory: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    """Validate both motion clips and publish an aggregate audit."""

    manifest = json.loads((source_directory / "manifest.json").read_text(encoding="utf-8"))
    clips = manifest.get("clips")
    if not isinstance(clips, dict):
        raise ValueError("G1 Moves manifest clips field is invalid")
    rows = [_inspect_clip(source_directory, clip, clips[clip]) for clip in CLIPS]
    frame = pd.DataFrame(rows).sort_values("clip_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    report: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "dataset_license": "CC BY 4.0",
        "clips": len(frame),
        "capture_frames": int(frame["capture_frames"].sum()),
        "training_frames": int(frame["training_frames"].sum()),
        "duration_seconds": round(float(frame["duration_seconds"].sum()), 6),
        "source_groups": int(frame["source_group_id"].nunique()),
        "bvh_joints_upstream": sorted(set(int(value) for value in frame["bvh_joints_upstream"])),
        "retarget_dof": sorted(set(int(value) for value in frame["retarget_dof"])),
        "retarget_csv_columns": sorted(
            set(int(value) for value in frame["retarget_csv_columns"])
        ),
        "training_body_count": sorted(
            set(int(value) for value in frame["training_body_count"])
        ),
        "performer_label_conflicts": int(frame["performer_label_conflict"].sum()),
        "manifest_performer": MANIFEST_PERFORMER,
        "per_clip_readme_performer": README_PERFORMER,
        "preview_decode": {
            "reported_frames": int(frame["preview_reported_frames"].sum()),
            "readable_frames": int(frame["preview_readable_frames"].sum()),
        },
        "output": "data/processed/g1-moves-fencing-motion.parquet",
        "output_sha256": sha256_file(output_path),
        "claim_boundary": (
            "Two fencing-named motion clips provide 2,258 raw BVH frames and 2,256 "
            "retargeted training frames, but both remain one conservative performer group. "
            "They are adjacent-domain generic fencing motion with no weapon, action, bout, "
            "touch, score, competition, or athlete-performance ground truth. The pinned "
            "manifest and clip READMEs disagree on performer name, so identity is unresolved."
        ),
    }
    write_json(report_path, report)
    return report


def fetch_and_prepare(
    source_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire and validate the pinned two-clip fencing subset."""

    acquire(source_directory, user_agent=user_agent, force=force)
    return prepare(source_directory, output_path, report_path)
