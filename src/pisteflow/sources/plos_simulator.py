"""CC BY supplementary video from the fencing lunge-simulator study."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

DOI = "https://doi.org/10.1371/journal.pone.0218959"
ARTICLE_URL = "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0218959"
VIDEO_URL = (
    "https://journals.plos.org/plosone/article/file?type=supplementary&"
    "id=info:doi/10.1371/journal.pone.0218959.s001"
)
LICENSE = "CC BY 4.0"
EXPECTED_BYTES = 72_685_260


def video_metadata(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("PLOS simulator supplementary video is not decodable")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if min(frames, fps, width, height) <= 0:
        raise ValueError("PLOS simulator supplementary video metadata is invalid")
    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frames / fps,
    }


def prepare(video_path: Path, manifest_path: Path, report_path: Path) -> dict[str, Any]:
    metadata = video_metadata(video_path)
    digest = sha256_file(video_path)
    row = {
        "asset_id": "plos-simulator:s001",
        "asset_kind": "supplementary_demonstration_video",
        "content": "fencing lunge simulator and post-session performance report illustration",
        "weapon": "mixed_study_context_epee_dominant",
        "sha256": digest,
        "bytes": video_path.stat().st_size,
        "source_url": VIDEO_URL,
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "source_license": LICENSE,
        **metadata,
    }
    manifest_sha256 = write_jsonl(manifest_path, [row])
    summary: dict[str, Any] = {
        "claim_boundary": (
            "One CC BY supplementary demonstration video from a controlled lunge-simulator "
            "study, not competition footage and not a clip-per-trial dataset. The article "
            "reports 11 male participants and 1,320 attacks, but does not publish those attacks "
            "as row-level data; the video must not be counted as 1,320 training samples."
        ),
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "video_url": VIDEO_URL,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "media_assets": 1,
        "video_sha256": digest,
        "video_bytes": video_path.stat().st_size,
        **metadata,
        "study_participants_reported": 11,
        "study_participants_by_weapon_reported": {"epee": 10, "foil": 1, "sabre": 0},
        "study_attacks_reported_not_released_as_rows": 1_320,
        "manifest_sha256": manifest_sha256,
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    video_path: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        video_path.unlink(missing_ok=True)
    download_resumable(
        VIDEO_URL,
        video_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    return prepare(video_path, manifest_path, report_path)
