"""Audit the pinned prod-ric fencing-footwork and broadcast-video snapshot.

The repository bundles five directory-labelled demonstration clips and two
unlabelled broadcast clips.  Its README contains an MIT sentence but there is
no licence file or licence text, so raw media and row-level derivatives remain
local.  Visual source grouping is fixed here to prevent clips of the same
demonstrator or bout from crossing an evaluation split.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/prod-ric/fencing-mlproject"
SNAPSHOT = "5da44208176a6cf8e5608d0c76c96811ac1b7075"
ARCHIVE_URL = f"https://codeload.github.com/prod-ric/fencing-mlproject/tar.gz/{SNAPSHOT}"
EXPECTED_ARCHIVE_BYTES = 196_702_016
EXPECTED_ARCHIVE_SHA256 = "390e0f262aa919d469d758404c157463653b43bca5bc9ff7a34de27e9eaa2128"
EXPECTED_README_SHA256 = "c91137ba557a7c1931a1468900626d312fd687876b253bc4fac0ff3a26b548db"


class ClipSpec(NamedTuple):
    partition: str
    action_label: str | None
    weapon: str
    source_group_id: str
    expected_bytes: int
    expected_sha256: str


# The three demonstration groups and one broadcast-bout group were assigned by
# visual inspection of middle frames.  They are conservative anti-leakage
# groups, not verified athlete identities.
EXPECTED_CLIPS: dict[str, ClipSpec] = {
    "src/data/videoes/advance/advance1.mov": ClipSpec(
        "directory_labelled_demonstration",
        "advance",
        "weapon_agnostic_footwork",
        "prod-ric:visual-demo-group-1",
        8_588_933,
        "7a1ae321f53538d074696eba2feea60fbbc1186a52540c739d88331b83d32186",
    ),
    "src/data/videoes/advance/fastadvance.mov": ClipSpec(
        "directory_labelled_demonstration",
        "advance",
        "weapon_agnostic_footwork",
        "prod-ric:visual-demo-group-2",
        3_117_537,
        "0f5048cacf4166d2e040337c908bb5e658945962e0dd792cda3d59a10a585d70",
    ),
    "src/data/videoes/lunge/lunge1.mov": ClipSpec(
        "directory_labelled_demonstration",
        "lunge",
        "weapon_agnostic_footwork",
        "prod-ric:visual-demo-group-2",
        23_001_038,
        "769fff3d37ffb172430f3e20e909f3010a375626dd341c4115cf265ec1df5fa8",
    ),
    "src/data/videoes/retreat/fastretreat.mov": ClipSpec(
        "directory_labelled_demonstration",
        "retreat",
        "weapon_agnostic_footwork",
        "prod-ric:visual-demo-group-3",
        3_852_196,
        "acaad4bda67f6ed19b71eef1b9d6ba076da3096122c4c31c8892a563c3dae4be",
    ),
    "src/data/videoes/retreat/retreat1.mov": ClipSpec(
        "directory_labelled_demonstration",
        "retreat",
        "weapon_agnostic_footwork",
        "prod-ric:visual-demo-group-3",
        6_483_099,
        "56841ad367ed083f65bdab44bb30bcae92affef6d307d4dd7b7c55916b6c1e9d",
    ),
    "testvideoes/testvideo1.mov": ClipSpec(
        "unlabelled_broadcast_test_video",
        None,
        "epee",
        "prod-ric:visual-broadcast-bout-group-1",
        77_909_371,
        "b545f812765722275865d4a8ace30b0051b14315e6574994537d1e076f5db2c1",
    ),
    "testvideoes/testvideo2.mov": ClipSpec(
        "unlabelled_broadcast_test_video",
        None,
        "epee",
        "prod-ric:visual-broadcast-bout-group-1",
        73_793_509,
        "4ffc8c94c34a7df3355d6e8ee9d5547b9953b9baaa0bfce83ac1b04894e369fb",
    ),
}

RIGHTS_BOUNDARY = (
    "The README ends with a one-line MIT statement, but the pinned repository has no "
    "LICENSE file or full licence text and does not document the rights provenance of the "
    "Olympic broadcast clips. Raw videos, extracted frames, pose arrays, row-level metadata, "
    "and trained artefacts remain local; only adapter code, hashes, aggregate counts, and "
    "limitations are publishable."
)


def classify_relative_path(relative_path: str) -> ClipSpec:
    """Return the fixed audit specification for one expected media path."""

    try:
        return EXPECTED_CLIPS[relative_path]
    except KeyError as error:
        raise ValueError(f"Unexpected prod-ric video path: {relative_path!r}") from error


def _safe_extract_stripped(archive_path: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        roots = {
            PurePosixPath(member.name).parts[0]
            for member in members
            if PurePosixPath(member.name).parts
        }
        if len(roots) != 1:
            raise ValueError("prod-ric archive does not have one stable root")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) <= 1:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("prod-ric archive contains an unsafe member path")
            destination = (resolved_root / relative).resolve()
            if resolved_root not in destination.parents:
                raise ValueError("prod-ric archive escapes its extraction root")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("prod-ric archive contains a non-regular member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("prod-ric archive member could not be opened")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def acquire(
    archive_path: Path,
    output_directory: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> Path:
    """Download, verify, and safely extract the pinned source snapshot."""

    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_ARCHIVE_BYTES,
    )
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("prod-ric archive checksum changed")
    marker = output_directory / "README.md"
    if force and output_directory.is_dir():
        shutil.rmtree(output_directory)
    if not marker.is_file():
        _safe_extract_stripped(archive_path, output_directory)
    if sha256_file(marker) != EXPECTED_README_SHA256:
        raise ValueError("prod-ric README checksum changed")
    return output_directory


def _inspect_video(path: Path) -> dict[str, int | float]:
    try:
        cv2: Any = importlib.import_module("cv2")
    except ImportError as error:  # pragma: no cover - environment-specific guidance
        raise RuntimeError("Install pisteflow-analytics[video] to audit video media") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open prod-ric video: {path}")
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
    if reported_frames <= 0 or readable_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid prod-ric video metadata: {path}")
    return {
        "reported_frames": reported_frames,
        "sequentially_readable_frames": readable_frames,
        "reported_minus_readable_frames": reported_frames - readable_frames,
        "fps": fps,
        "duration_seconds": reported_frames / fps,
        "width": width,
        "height": height,
    }


def _stable_video_id(relative_path: str) -> str:
    digest = hashlib.sha256(f"pisteflow:prod-ric:{relative_path}".encode()).hexdigest()[:20]
    return f"prod-ric-video:{digest}"


def prepare(
    source_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Decode every video and write a local-only, source-grouped media inventory."""

    media_paths = sorted(source_directory.rglob("*.mov"))
    relative_paths = {path.relative_to(source_directory).as_posix() for path in media_paths}
    if relative_paths != set(EXPECTED_CLIPS):
        missing = sorted(set(EXPECTED_CLIPS) - relative_paths)
        extra = sorted(relative_paths - set(EXPECTED_CLIPS))
        raise ValueError(f"prod-ric video inventory changed; missing={missing}, extra={extra}")
    if any(source_directory.rglob("LICENSE*")):
        raise ValueError("prod-ric licence-file status changed; review rights policy")

    rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for path in media_paths:
        relative_path = path.relative_to(source_directory).as_posix()
        spec = classify_relative_path(relative_path)
        digest = sha256_file(path)
        if path.stat().st_size != spec.expected_bytes or digest != spec.expected_sha256:
            raise ValueError(f"prod-ric media checksum changed: {relative_path}")
        hashes.append(digest)
        rows.append(
            {
                "video_id": _stable_video_id(relative_path),
                "source_group_id": spec.source_group_id,
                "relative_path": relative_path,
                "partition": spec.partition,
                "weapon": spec.weapon,
                "action_label": spec.action_label,
                "label_source": "upstream_directory_name" if spec.action_label else "unlabelled",
                "bytes": path.stat().st_size,
                "sha256": digest,
                **_inspect_video(path),
            }
        )
    if len(hashes) != len(set(hashes)):
        raise ValueError("prod-ric snapshot contains exact duplicate media files")

    frame = pd.DataFrame(rows).sort_values(["partition", "source_group_id", "video_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    labelled = frame[frame["action_label"].notna()]
    unlabelled = frame[frame["action_label"].isna()]
    action_durations: dict[str, float] = defaultdict(float)
    for row in labelled.itertuples(index=False):
        action_durations[str(row.action_label)] += float(row.duration_seconds)
    report: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "archive_bytes": archive_path.stat().st_size if archive_path else None,
        "archive_sha256": sha256_file(archive_path) if archive_path else None,
        "readme_sha256": sha256_file(source_directory / "README.md"),
        "repository_license_file_present": False,
        "readme_contains_one_line_mit_statement": True,
        "rights_boundary": RIGHTS_BOUNDARY,
        "original_video_files": len(frame),
        "exact_duplicate_video_files": len(hashes) - len(set(hashes)),
        "source_groups": int(frame["source_group_id"].nunique()),
        "directory_labelled_demonstration_videos": len(labelled),
        "unlabelled_broadcast_test_videos": len(unlabelled),
        "reported_frames": int(frame["reported_frames"].sum()),
        "sequentially_readable_frames": int(frame["sequentially_readable_frames"].sum()),
        "reported_minus_readable_frames": int(
            frame["reported_minus_readable_frames"].sum()
        ),
        "reported_duration_seconds": round(float(frame["duration_seconds"].sum()), 6),
        "labelled_duration_seconds": round(float(labelled["duration_seconds"].sum()), 6),
        "unlabelled_duration_seconds": round(float(unlabelled["duration_seconds"].sum()), 6),
        "directory_action_video_counts": dict(
            sorted(Counter(str(value) for value in labelled["action_label"]).items())
        ),
        "directory_action_duration_seconds": {
            key: round(value, 6) for key, value in sorted(action_durations.items())
        },
        "weapon_video_counts": {
            str(key): int(value)
            for key, value in frame["weapon"].value_counts().sort_index().items()
        },
        "manual_visual_source_group_audit": {
            "minimum_visually_distinct_training_demonstrators": 3,
            "training_visual_groups": 3,
            "broadcast_bout_visual_groups": 1,
            "minimum_broadcast_athletes": 2,
            "note": (
                "Middle-frame inspection shows at least three visibly distinct training "
                "demonstrators, contradicting the README's single-fencer limitation. Both test "
                "clips show the same two-athlete Olympic epee bout and are one leakage group. "
                "These are conservative visual groups, not verified athlete identities."
            ),
        },
        "upstream_validation_claim_not_adopted": (
            "The README's 97.14% validation accuracy is not treated as a benchmark result: the "
            "repository describes a 70/15/15 sequence split, source clips can generate overlapping "
            "windows, the bundled videos expose repeated demonstrators, and no frozen split or "
            "evaluation artefact is included."
        ),
        "output": "data/processed/prod-ric-fencing-video-inventory.parquet",
        "output_sha256": sha256_file(output_path),
        "claim_boundary": (
            "Seven original video files contribute 157.9 reported seconds, not hundreds of "
            "independent sequence samples. The five folder labels cover advance, retreat, and "
            "lunge only; idle is documented but absent. The two broadcast files are unlabelled "
            "epee footage from one visual bout group. Evaluation must hold out complete visual "
            "source groups, and no athlete-generalisation claim is supported."
        ),
    }
    write_json(report_path, report)
    return report


def fetch_and_prepare(
    archive_path: Path,
    source_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the pinned source and prepare its audited local inventory."""

    acquire(archive_path, source_directory, user_agent=user_agent, force=force)
    return prepare(source_directory, output_path, report_path, archive_path=archive_path)
