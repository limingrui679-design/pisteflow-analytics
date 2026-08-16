"""Pinned local-only acquisition of a small fencing-interaction robot dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = (
    "https://huggingface.co/datasets/ashishrao7/"
    "fencing-lerobot-realsense-depth-and-selfie-cam"
)
SNAPSHOT = "bef33f9163a7675f003a4c69a89c0eed75d83c1a"
LICENSE = "No license declared in dataset card; local analysis only"


@dataclass(frozen=True)
class FileSpec:
    path: str
    bytes: int
    sha256: str


FILES = (
    FileSpec(
        "meta/info.json",
        3438,
        "483800cd34ec8effe3e49367fb8e3ce504c0074b9e356c41a68874368f2b78cb",
    ),
    FileSpec(
        "meta/episodes.jsonl",
        317,
        "a3699f2de86c797427342323b1de449cebecd496dbef14aa8c41b2ad52a729b3",
    ),
    FileSpec(
        "meta/tasks.jsonl",
        87,
        "6d8436de613568897815534eeebd0ec7dc55a6ba3ad2085fe54e12b96ecf9870",
    ),
    FileSpec(
        "data/chunk-000/episode_000000.parquet",
        7976,
        "cfe3fe935b55bd1d8e22e3e8e3138e95130facd24acf82105a23124ec78d2713",
    ),
    FileSpec(
        "data/chunk-000/episode_000001.parquet",
        30307,
        "5ca65850a7cb65420a9b216bfddb465e5f2ffb04d974cb372d7edbb8f6157820",
    ),
    FileSpec(
        "data/chunk-000/episode_000002.parquet",
        26857,
        "4eb35460fd70cba547e0c59fddcc30bc0dbf0a0381f6d7d57b4c2840aa08d4ca",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.main/episode_000000.mp4",
        96541,
        "73f3f1c25d06f29eaca5fc21c8c0201900be24aee94b451484409800cd5ecd09",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.main/episode_000001.mp4",
        1730312,
        "17d9d1236434e99673f4c7e04035e039f9aead6585e05c1dbf3b097b90d460c2",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.main/episode_000002.mp4",
        1677844,
        "5a301b8f4956db30b82fe1b13af5d5e78e447e6df9804fd81fd7d82195bd1f95",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.secondary_0/episode_000000.mp4",
        47289,
        "d2334dd3ac5aa575af84998fbd827ef649bb3312d0f886319bfed224648e3396",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.secondary_0/episode_000001.mp4",
        749391,
        "8da1b59762bad5c4e3d8ae207b487a643ef4bd83bd91e3510b70fb57d8c59cd9",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.secondary_0/episode_000002.mp4",
        637993,
        "cdc2abfc60db7d30e9b7250b98a9f9b15537393b0c957e6b7b1ecbbe139ad2c6",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.secondary_1/episode_000000.mp4",
        487105,
        "f2167c16417208b0c949b34b264a757e19c444497aca7c475520b814540e2176",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.secondary_1/episode_000001.mp4",
        5811557,
        "02e1ce0fcf77468f0eb1a3d69bb3d35364c7046bf6811ca1fede2143e39494f5",
    ),
    FileSpec(
        "videos/chunk-000/observation.images.secondary_1/episode_000002.mp4",
        4697204,
        "181566f88777d7d43e0903063f75ee8fa576b295e18c4b8da5631ab2d5b3491e",
    ),
)


def _url(path: str) -> str:
    return f"{SOURCE_URL}/resolve/{SNAPSHOT}/{path}"


def acquire(output_directory: Path, *, user_agent: str) -> None:
    for spec in FILES:
        destination = output_directory / spec.path
        download_resumable(
            _url(spec.path),
            destination,
            user_agent=user_agent,
            expected_bytes=spec.bytes,
        )
        actual = sha256_file(destination)
        if actual != spec.sha256:
            raise ValueError(f"LeRobot fencing checksum changed for {spec.path}: {actual}")


def _metadata_summary(output_directory: Path) -> dict[str, Any]:
    info = json.loads((output_directory / "meta/info.json").read_text(encoding="utf-8"))
    episode_text = (output_directory / "meta/episodes.jsonl").read_text(encoding="utf-8")
    episodes = [json.loads(line) for line in episode_text.splitlines() if line.strip()]
    if info.get("total_episodes") != len(episodes):
        raise ValueError("LeRobot fencing episode count does not match metadata")
    lengths = [int(row["length"]) for row in episodes]
    if sum(lengths) != int(info.get("total_frames", -1)):
        raise ValueError("LeRobot fencing frame count does not match episode lengths")
    return {
        "episodes": len(episodes),
        "episode_lengths": lengths,
        "frames": sum(lengths),
        "videos": int(info["total_videos"]),
        "fps": int(info["fps"]),
        "robot_type": str(info["robot_type"]),
        "camera_streams": sum(
            str(value.get("dtype")) == "video"
            for value in info.get("features", {}).values()
            if isinstance(value, dict)
        ),
    }


def prepare(output_directory: Path, report_path: Path) -> dict[str, Any]:
    for spec in FILES:
        path = output_directory / spec.path
        if not path.is_file() or path.stat().st_size != spec.bytes:
            raise ValueError(f"LeRobot fencing snapshot is incomplete: {spec.path}")
        if sha256_file(path) != spec.sha256:
            raise ValueError(f"LeRobot fencing snapshot checksum mismatch: {spec.path}")

    metadata = _metadata_summary(output_directory)
    parquet_rows: list[int] = []
    for episode_index in range(metadata["episodes"]):
        frame = pd.read_parquet(
            output_directory / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        )
        parquet_rows.append(len(frame))
        if len(frame) != metadata["episode_lengths"][episode_index]:
            raise ValueError(f"LeRobot fencing Parquet length mismatch: {episode_index}")
        if not {"action", "observation.state", "timestamp", "episode_index"}.issubset(
            frame.columns
        ):
            raise ValueError("LeRobot fencing Parquet schema changed")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional media dependency
        raise RuntimeError("Install the video extra to validate LeRobot footage") from exc
    video_frame_counts: list[int] = []
    for spec in FILES:
        if not spec.path.endswith(".mp4"):
            continue
        capture = cv2.VideoCapture(str(output_directory / spec.path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, _frame = capture.read()
        capture.release()
        if not ok or frames < 1 or fps <= 0:
            raise ValueError(f"LeRobot fencing video does not decode: {spec.path}")
        video_frame_counts.append(frames)

    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        **metadata,
        "parquet_rows": parquet_rows,
        "verified_files": len(FILES),
        "verified_bytes": sum(spec.bytes for spec in FILES),
        "video_frame_counts": video_frame_counts,
        "claim_boundary": (
            "Three small robot-imitation episodes with synchronized three-camera video, motor "
            "state, and motor action. This is a fencing-themed human-robot interaction recording, "
            "not competition footage, a modern-weapon label source, an athlete cohort, or a "
            "referee benchmark. The dataset card declares no license, so all files and derived "
            "rows remain local."
        ),
    }
    write_json(report_path, summary)
    return summary
