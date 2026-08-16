"""Audit three pinned GitHub snapshots that bundle real fencing media.

The snapshots add sabre, foil, and epee coverage, but none establishes rights
to redistribute the underlying competition broadcasts.  Media, extracted
frames, row-level inventories, and trained artefacts therefore remain local.
The adapter publishes only code, snapshot hashes, aggregate counts, and
leakage-safe source-group boundaries.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_url: str
    snapshot: str
    archive_name: str
    archive_bytes: int
    archive_sha256: str
    readme_sha256: str
    license_status: str
    archive_url: str
    license_relative_path: str | None = None
    license_sha256: str | None = None


class MediaSpec(NamedTuple):
    source_group_id: str
    weapon: str
    media_role: str
    derivative_of: str | None
    expected_bytes: int
    expected_sha256: str


SOURCES: dict[str, SourceSpec] = {
    "fencstats": SourceSpec(
        source_id="fencstats",
        source_url="https://github.com/JanVDB2000/FencStats-AI-Powered-Fencing-Analyzer",
        snapshot="af41a354726046dbc3f56976ff4557088a75242c",
        archive_name="fencstats-af41a35.tar.gz",
        archive_bytes=5_164_621,
        archive_sha256="f8090267ef9e8aa60d952a92558629588a2ce282fa6b5c0a91a3947d394fd8ef",
        readme_sha256="cc62af906b4539c0295d50b580596067a7d2c21c039cc97e76928f561174fc4a",
        license_status="no_repository_license_underlying_broadcast_rights_unverified",
        archive_url=(
            "https://codeload.github.com/JanVDB2000/"
            "FencStats-AI-Powered-Fencing-Analyzer/tar.gz/"
            "af41a354726046dbc3f56976ff4557088a75242c"
        ),
    ),
    "theephop": SourceSpec(
        source_id="theephop",
        source_url="https://github.com/TheephopWS/fencing-ai",
        snapshot="d361e055f406171591c700ec4d1980ec94cd657c",
        archive_name="theephop-fencing-ai-d361e05.tar.gz",
        archive_bytes=20_829_334,
        archive_sha256="d93f01f9a9247eb4445049f59e006e35ab44ea7c46594f1da53e1c058fee8737",
        readme_sha256="5072f563ac36260a7015367d686e8e76e3cb141ce6ec5269d4c454d5fcdc0b11",
        license_status="mit_code_underlying_broadcast_rights_unverified",
        archive_url=(
            "https://codeload.github.com/TheephopWS/fencing-ai/tar.gz/"
            "d361e055f406171591c700ec4d1980ec94cd657c"
        ),
        license_relative_path="LICENSE",
        license_sha256="ddd9303b259bdbf3467e7a1423ffbdea4bca530a10e8d7f4ed11bd04ec60a5e5",
    ),
    "tennis_tool": SourceSpec(
        source_id="tennis_tool",
        source_url="https://github.com/Tennis-Ball/Fencing-Video-Tool",
        snapshot="e98804f904cf0d676a22f7413d1482d2318fab04",
        archive_name="tennis-ball-fencing-video-tool-e98804f.tar.gz",
        archive_bytes=163_480_206,
        archive_sha256="781f3a6e744d3f3849cd9ba4a64370e7f79ef69b614e0244fa11b7c3fb4f9be3",
        readme_sha256="4aac1df1404255e959ef98f10beafca450b847321a7feac4307a204f4b87746d",
        license_status="no_repository_license_underlying_broadcast_rights_unverified",
        archive_url=(
            "https://codeload.github.com/Tennis-Ball/Fencing-Video-Tool/tar.gz/"
            "e98804f904cf0d676a22f7413d1482d2318fab04"
        ),
    ),
}


FENCSTATS_MEDIA: dict[str, MediaSpec] = {
    "data/inference/video/AbandonedCandidAfricanporcupine.mp4": MediaSpec(
        "fencstats:sabre-xu-ferjani",
        "sabre",
        "original_broadcast_excerpt",
        None,
        2_375_448,
        "d742dc2aad1cabf47abb82b2f6bf219bf653554f40b209b239084ddec412622b",
    ),
    "data/inference/video/BarrenSecondhandAcornwoodpecker-mobile.mp4": MediaSpec(
        "fencstats:sabre-kim-navarro",
        "sabre",
        "original_broadcast_excerpt",
        None,
        769_503,
        "78a4caf1cf75ddda3eb0be228530db260f9d9478e71901df26bfbe30af6515c8",
    ),
    "data/inference/video/SlimyDependentIbadanmalimbe-mobile.mp4": MediaSpec(
        "fencstats:sabre-xu-yingming-vu-thanh-an",
        "sabre",
        "original_broadcast_excerpt",
        None,
        512_810,
        "2fe16ef5a9a4160b783dc49eb08c706a7183c6f2ad9a06008493a34b0c1c72b5",
    ),
    "data/inference/video/TangibleVelvetyHellbender-mobile.mp4": MediaSpec(
        "fencstats:sabre-reshetnikov-barloy",
        "sabre",
        "original_broadcast_excerpt",
        None,
        414_388,
        "8d1367294b8b1f678d5d23084df53902a6ebdd9f39c40e0466df3fd123cc3235",
    ),
    "data/inference/video/WigglyAlarmedKitfox-mobile.mp4": MediaSpec(
        "fencstats:sabre-xu-yingming-vu-thanh-an",
        "sabre",
        "original_broadcast_excerpt",
        None,
        503_261,
        "2d1d63e75b6a80295a82e1389fe645f4dfdae24bf3c9887c9e21f21490d1ae7a",
    ),
    (
        "data/inference/image/AbandonedAbandonedFurseal_mp4-1_jpg.rf."
        "b58e5acde13cfc780a015d6bc0a6386d.jpg"
    ): MediaSpec(
        "fencstats:sabre-hwang-ferjani",
        "sabre",
        "original_still_frame",
        None,
        32_108,
        "01b3c4bdf33ed443637aa588a156652cf4f935fe502190f641c72d12fd89b18a",
    ),
    (
        "data/inference/image/AbandonedAbsoluteGoldfinch_mp4-4_jpg.rf."
        "ac988d297800cd8b322613668200a0ac.jpg"
    ): MediaSpec(
        "fencstats:sabre-szatmari-ferjani",
        "sabre",
        "original_still_frame",
        None,
        28_763,
        "1ff23f6a30a0697309dc8117a5ce65025fa9db8580a5016d11ff3d0627da7ddd",
    ),
    (
        "data/inference/image/AbandonedBackGnu_mp4-3_jpg.rf."
        "ae9d35cc08343aa823415c0f6acec54b.jpg"
    ): MediaSpec(
        "fencstats:sabre-gregorio-emura",
        "sabre",
        "original_still_frame",
        None,
        39_993,
        "29c6edf7958d132dbbf92278dfef02071c6418d87f21eef9fbae8c099c1377e8",
    ),
    (
        "data/inference/image/AbandonedBackGnu_mp4-4_jpg.rf."
        "a7e463e2d8cd32ea18699b6547b2f62d.jpg"
    ): MediaSpec(
        "fencstats:sabre-gregorio-emura",
        "sabre",
        "original_still_frame",
        None,
        37_450,
        "1e1d6997838b1df8428c34348de594bf85bb769a3b21d0292854a3b81e69f1a8",
    ),
    (
        "data/inference/image/BarrenSecondhandAcornwoodpecker_mp4-3_jpg.rf."
        "67b37225576d99878640023e9f72fc27.jpg"
    ): MediaSpec(
        "fencstats:sabre-kim-navarro",
        "sabre",
        "original_still_frame",
        None,
        29_197,
        "2abc883f9abfae0483e184fc578a16db40886fb36d48d7c5c5324c7672f42e85",
    ),
    (
        "data/inference/image/ScalyPleasantAmazonparrot_mp4-0_jpg.rf."
        "965787c98759006e494a540d83283073.jpg"
    ): MediaSpec(
        "fencstats:sabre-deary-oh",
        "sabre",
        "original_still_frame",
        None,
        33_708,
        "36bf15aef3f1426c3958350f5b3ae3f5b4b157397be093da0fb6bd7fe2a270f3",
    ),
    (
        "data/inference/image/ScalyPleasantAmazonparrot_mp4-5_jpg.rf."
        "ce2f9d04ef261cbc42823ea23d9ce1fe.jpg"
    ): MediaSpec(
        "fencstats:sabre-deary-oh",
        "sabre",
        "original_still_frame",
        None,
        33_953,
        "49e3e09e6593113b0275917927321b04be6394499d1be5e8a1e98455cea85847",
    ),
    (
        "data/inference/image/ShadowyLightheartedGuernseycow_mp4-4_jpg.rf."
        "ba610fb6e34b9cedcab084e37ba3b4a1.jpg"
    ): MediaSpec(
        "fencstats:sabre-criscio-skarbonkiewicz",
        "sabre",
        "original_still_frame",
        None,
        37_136,
        "f9c9df4a3975db06c7005bfd9b0e798027b5e2faf855f69049a86f31ba162349",
    ),
    "data/inference/image/image1.jpg": MediaSpec(
        "fencstats:sabre-szatmari-ferjani",
        "sabre",
        "original_still_frame",
        None,
        332_113,
        "edf67f7c70d600023127cf670e22a2dd668c0481472754206bd33f7cb4008d2c",
    ),
}


THEEPHOP_MEDIA: dict[str, MediaSpec] = {
    "uploads/foil_vid1.mp4": MediaSpec(
        "theephop:foil-macchi-itkin",
        "foil",
        "original_broadcast_excerpt",
        None,
        4_590_885,
        "41ea448a3010f4df276d34d0703676216f5f0da33ffda7530f66616669663b7d",
    ),
    "uploads/processed_foil_vid1.mp4": MediaSpec(
        "theephop:foil-macchi-itkin",
        "foil",
        "pose_overlay_derivative",
        "uploads/foil_vid1.mp4",
        669_943,
        "ffb544985e7aeee66c992ac8dd708e7101b12adc47558eba97facdfb6a6c5a95",
    ),
}


TENNIS_VIDEO_MEDIA: dict[str, MediaSpec] = {
    "example_input_videos/BidaReizlin.mp4": MediaSpec(
        "tennis-tool:epee-bida-reizlin",
        "epee",
        "original_full_bout_video",
        None,
        31_780_878,
        "bb1a4b04bebe0bc464e025fa956f15f760a5dc301f869b6430f94a1b754ce0a6",
    ),
    "example_input_videos/BidaYamada.mp4": MediaSpec(
        "tennis-tool:epee-bida-yamada",
        "epee",
        "original_full_bout_video",
        None,
        29_066_051,
        "a0f7f3411b835d8d62e9e90625eeeaf4c3dad8e1d0e43f0dcaa906db8bbd0797",
    ),
    "example_input_videos/SantarelliBorel.mp4": MediaSpec(
        "tennis-tool:epee-santarelli-borel",
        "epee",
        "original_full_bout_video",
        None,
        24_102_467,
        "6c0083d6c0fccd1b8834b68daafbd618609c4a6f181ca038ad7b309580299c42",
    ),
    "example_input_videos/video144p.avi": MediaSpec(
        "tennis-tool:epee-video144p",
        "epee",
        "original_full_bout_video",
        None,
        33_522_498,
        "4277ea6ab741181bb33675c517d0d35863ca5feb09739312aab79c4f3b941e80",
    ),
    "OUTPUTVID.avi": MediaSpec(
        "tennis-tool:epee-video144p",
        "epee",
        "cnn_filtered_video_derivative",
        "example_input_videos/video144p.avi",
        42_942_642,
        "c3bfac0bc95514eac09907ced35aa5d116572ae45c7f3814428fc2523b9f9fe1",
    ),
}

TENNIS_LABELS_SHA256 = "392b374a5ae52df9bcb0217b7ae128672999e58fecb9b258598f1a2fd8f2d33b"
TENNIS_FRAME_COUNT = 313
TENNIS_FRAME_BYTES = 530_438
TENNIS_FRAME_INVENTORY_SHA256 = (
    "e99a63063f9279ab8e3810659fe342716fc95052a470653e5137b5f7961d4127"
)

RIGHTS_BOUNDARY = (
    "Repository publication does not establish redistribution rights for bundled competition "
    "broadcast footage. Keep raw media, extracted frames, row-level inventories, pose arrays, "
    "models, and predictions local. Publish only adapter code, snapshot and output hashes, "
    "aggregate counts, and clearly bounded findings."
)


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
            raise ValueError("GitHub archive does not have one stable root")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) <= 1:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("GitHub archive contains an unsafe member path")
            destination = (resolved_root / relative).resolve()
            if resolved_root not in destination.parents:
                raise ValueError("GitHub archive escapes its extraction root")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("GitHub archive contains a non-regular member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("GitHub archive member could not be opened")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def acquire_source(
    spec: SourceSpec,
    archive_path: Path,
    output_directory: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> Path:
    """Download, verify, and safely extract one pinned source snapshot."""

    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        spec.archive_url,
        archive_path,
        user_agent=user_agent,
        expected_bytes=spec.archive_bytes,
    )
    if sha256_file(archive_path) != spec.archive_sha256:
        raise ValueError(f"{spec.source_id} archive checksum changed")
    if force and output_directory.is_dir():
        shutil.rmtree(output_directory)
    if not (output_directory / "README.md").is_file():
        _safe_extract_stripped(archive_path, output_directory)
    if sha256_file(output_directory / "README.md") != spec.readme_sha256:
        raise ValueError(f"{spec.source_id} README checksum changed")
    if spec.license_relative_path is not None:
        license_path = output_directory / spec.license_relative_path
        if not license_path.is_file() or sha256_file(license_path) != spec.license_sha256:
            raise ValueError(f"{spec.source_id} license checksum changed")
    return output_directory


def _cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as error:  # pragma: no cover - environment-specific guidance
        raise RuntimeError("Install pisteflow-analytics[video] to audit media") from error


def _inspect_video(path: Path) -> dict[str, int | float]:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
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
        raise ValueError(f"Invalid video metadata: {path}")
    return {
        "width": width,
        "height": height,
        "reported_frames": reported_frames,
        "sequentially_readable_frames": readable_frames,
        "reported_minus_readable_frames": reported_frames - readable_frames,
        "fps": fps,
        "duration_seconds": reported_frames / fps,
    }


def _inspect_image(path: Path) -> dict[str, int | float | None]:
    cv2 = _cv2()
    image = cv2.imread(str(path))
    if image is None or len(image.shape) < 2:
        raise ValueError(f"Could not decode image: {path}")
    height, width = image.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "reported_frames": None,
        "sequentially_readable_frames": None,
        "reported_minus_readable_frames": None,
        "fps": None,
        "duration_seconds": None,
    }


def _stable_media_id(source_id: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{source_id}:{relative_path}".encode()).hexdigest()[:20]
    return f"github-fencing-media:{digest}"


def _verify_media(path: Path, spec: MediaSpec, relative_path: str) -> None:
    if path.stat().st_size != spec.expected_bytes or sha256_file(path) != spec.expected_sha256:
        raise ValueError(f"Media checksum changed: {relative_path}")


def _row(
    source: SourceSpec,
    source_directory: Path,
    relative_path: str,
    media_spec: MediaSpec,
) -> dict[str, Any]:
    path = source_directory / relative_path
    _verify_media(path, media_spec, relative_path)
    media_kind = "video" if path.suffix.lower() in {".mp4", ".avi", ".mov"} else "image"
    inspection = _inspect_video(path) if media_kind == "video" else _inspect_image(path)
    return {
        "media_id": _stable_media_id(source.source_id, relative_path),
        "source_id": source.source_id,
        "source_url": source.source_url,
        "snapshot": source.snapshot,
        "source_group_id": media_spec.source_group_id,
        "relative_path": relative_path,
        "media_kind": media_kind,
        "media_role": media_spec.media_role,
        "weapon": media_spec.weapon,
        "derivative_of": media_spec.derivative_of,
        "is_original_media": media_spec.derivative_of is None,
        "label": None,
        "label_source": None,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "license_status": source.license_status,
        **inspection,
    }


def _inventory_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix()
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_tennis_frames(source_directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_path = source_directory / "labels.txt"
    if sha256_file(labels_path) != TENNIS_LABELS_SHA256:
        raise ValueError("Tennis-tool labels checksum changed")
    labels = labels_path.read_text(encoding="utf-8")
    if set(labels) - {"0", "1"} or len(labels) != TENNIS_FRAME_COUNT:
        raise ValueError("Tennis-tool binary labels changed")
    frame_paths = sorted(
        (source_directory / "classified_frames").glob("*.jpg"),
        key=lambda path: int(path.stem),
    )
    if [int(path.stem) for path in frame_paths] != list(range(1, TENNIS_FRAME_COUNT + 1)):
        raise ValueError("Tennis-tool frame numbering changed")
    if sum(path.stat().st_size for path in frame_paths) != TENNIS_FRAME_BYTES:
        raise ValueError("Tennis-tool frame bytes changed")
    if _inventory_digest(frame_paths, source_directory) != TENNIS_FRAME_INVENTORY_SHA256:
        raise ValueError("Tennis-tool frame inventory checksum changed")
    hashes = [sha256_file(path) for path in frame_paths]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Tennis-tool frame inventory contains exact duplicates")

    source = SOURCES["tennis_tool"]
    rows: list[dict[str, Any]] = []
    for path, label in zip(frame_paths, labels, strict=True):
        relative_path = path.relative_to(source_directory).as_posix()
        rows.append(
            {
                "media_id": _stable_media_id(source.source_id, relative_path),
                "source_id": source.source_id,
                "source_url": source.source_url,
                "snapshot": source.snapshot,
                "source_group_id": "tennis-tool:epee-video144p",
                "relative_path": relative_path,
                "media_kind": "image",
                "media_role": "manually_classified_downscaled_frame_derivative",
                "weapon": "epee",
                "derivative_of": "example_input_videos/video144p.avi",
                "is_original_media": False,
                "label": "fencing_visible" if label == "1" else "fencing_not_visible",
                "label_source": "upstream_manual_binary_label",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "license_status": source.license_status,
                **_inspect_image(path),
            }
        )
    return rows, {
        "manually_classified_frame_derivatives": len(rows),
        "fencing_visible_frames": labels.count("1"),
        "fencing_not_visible_frames": labels.count("0"),
        "frame_bytes": sum(path.stat().st_size for path in frame_paths),
        "exact_duplicate_frames": len(hashes) - len(set(hashes)),
        "frame_inventory_sha256": _inventory_digest(frame_paths, source_directory),
    }


def prepare(
    source_directories: dict[str, Path],
    output_path: Path,
    report_path: Path,
    *,
    archive_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Write a local row inventory and a publishable aggregate audit."""

    if set(source_directories) != set(SOURCES):
        raise ValueError("All three GitHub fencing-media sources are required")
    rows: list[dict[str, Any]] = []
    for relative_path, media_spec in FENCSTATS_MEDIA.items():
        rows.append(
            _row(SOURCES["fencstats"], source_directories["fencstats"], relative_path, media_spec)
        )
    for relative_path, media_spec in THEEPHOP_MEDIA.items():
        rows.append(
            _row(SOURCES["theephop"], source_directories["theephop"], relative_path, media_spec)
        )
    for relative_path, media_spec in TENNIS_VIDEO_MEDIA.items():
        rows.append(
            _row(
                SOURCES["tennis_tool"],
                source_directories["tennis_tool"],
                relative_path,
                media_spec,
            )
        )
    tennis_frame_rows, tennis_frame_audit = _prepare_tennis_frames(
        source_directories["tennis_tool"]
    )
    rows.extend(tennis_frame_rows)

    frame = pd.DataFrame(rows).sort_values(["source_id", "source_group_id", "relative_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    videos = frame[frame["media_kind"] == "video"]
    original_videos = videos[videos["is_original_media"]]
    images = frame[frame["media_kind"] == "image"]
    original_images = images[images["is_original_media"]]
    report: dict[str, Any] = {
        "sources": {
            source_id: {
                "source_url": spec.source_url,
                "snapshot": spec.snapshot,
                "archive_bytes": (
                    archive_paths[source_id].stat().st_size
                    if archive_paths is not None
                    else None
                ),
                "archive_sha256": (
                    sha256_file(archive_paths[source_id]) if archive_paths is not None else None
                ),
                "license_status": spec.license_status,
            }
            for source_id, spec in SOURCES.items()
        },
        "rights_boundary": RIGHTS_BOUNDARY,
        "inventory_rows": len(frame),
        "source_groups": int(frame["source_group_id"].nunique()),
        "video_files": len(videos),
        "original_video_files": len(original_videos),
        "derivative_video_files": len(videos) - len(original_videos),
        "original_still_images": len(original_images),
        "derived_frame_images": len(images) - len(original_images),
        "weapon_original_video_counts": {
            str(key): int(value)
            for key, value in original_videos["weapon"].value_counts().sort_index().items()
        },
        "weapon_source_group_counts": {
            str(key): int(value)
            for key, value in (
                frame[["weapon", "source_group_id"]]
                .drop_duplicates()["weapon"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "original_video_duration_seconds": round(
            float(original_videos["duration_seconds"].sum()), 6
        ),
        "original_video_duration_seconds_by_weapon": {
            str(key): round(float(value), 6)
            for key, value in (
                original_videos.groupby("weapon")["duration_seconds"].sum().sort_index().items()
            )
        },
        "sequential_decode": {
            "reported_frames": int(videos["reported_frames"].sum()),
            "sequentially_readable_frames": int(videos["sequentially_readable_frames"].sum()),
            "reported_minus_readable_frames": int(videos["reported_minus_readable_frames"].sum()),
        },
        "tennis_tool_binary_frames": tennis_frame_audit,
        "media_role_counts": dict(sorted(Counter(frame["media_role"]).items())),
        "output": "data/processed/github-fencing-media-inventory.parquet",
        "output_sha256": sha256_file(output_path),
        "claim_boundary": (
            "The ten original videos and nine original stills represent fourteen conservative "
            "visual source groups, not hundreds of independent bouts or athletes. Two videos "
            "and 313 downscaled images are derivatives and inherit their parent bout group. "
            "The frame labels only distinguish whether fencing is visible; they are not action, "
            "touch, priority, or athlete-ground-truth labels. Every evaluation split must hold "
            "out complete source_group_id values."
        ),
    }
    write_json(report_path, report)
    return report


def fetch_and_prepare(
    cache_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire all three pinned snapshots and prepare their joint audit."""

    archive_root = cache_directory / "source-archives"
    extraction_root = cache_directory / "github-fencing-media"
    archive_paths: dict[str, Path] = {}
    source_directories: dict[str, Path] = {}
    for source_id, spec in SOURCES.items():
        archive_path = archive_root / spec.archive_name
        source_directory = extraction_root / source_id
        acquire_source(
            spec,
            archive_path,
            source_directory,
            user_agent=user_agent,
            force=force,
        )
        archive_paths[source_id] = archive_path
        source_directories[source_id] = source_directory
    return prepare(
        source_directories,
        output_path,
        report_path,
        archive_paths=archive_paths,
    )
