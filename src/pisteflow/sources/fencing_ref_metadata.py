"""Audit a legacy scoreboard-delta clip catalog without downloading YouTube media."""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/sholtodouglas/fencing_ref_dataset"
SNAPSHOT = "d8eb8a959b8b1f651b451a7beafc320318450156"
ARCHIVE_URL = f"https://codeload.github.com/sholtodouglas/fencing_ref_dataset/tar.gz/{SNAPSHOT}"
EXPECTED_ARCHIVE_BYTES = 1_430_436
EXPECTED_ARCHIVE_SHA256 = "039f4b0ec105dbce8e50fae5ed6d6c66aedf92ec2cf2ffb84edecb9040246af2"
LICENSE = "No repository or dataset license found; metadata and derivatives remain local"
LABELS = {"left", "right", "together"}


def _youtube_id(url: str) -> str:
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("v", [])
    if parsed.hostname not in {"youtube.com", "www.youtube.com"} or len(values) != 1:
        raise ValueError(f"Unexpected fencing-reference URL: {url}")
    return values[0]


def _group_id(video_id: str) -> str:
    digest = hashlib.sha256(f"pisteflow:fencing-ref:{video_id}".encode()).hexdigest()[:20]
    return f"fencing-ref-video:{digest}"


def _safe_extract(archive_path: Path, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (root / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError("Fencing-reference archive contains an unsafe member path")
            if member.issym() or member.islnk():
                raise ValueError("Fencing-reference archive contains a link member")
        archive.extractall(output_directory, members=members, filter="data")
    candidates = sorted(path for path in output_directory.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError("Fencing-reference archive root changed")
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
        raise ValueError("Fencing-reference archive checksum changed")
    expected_root = output_directory / "fencing_ref_dataset-main"
    if force or not expected_root.is_dir():
        return _safe_extract(archive_path, output_directory)
    return expected_root


def _load_split(source_directory: Path, name: str) -> dict[str, dict[str, Any]]:
    payload = json.loads((source_directory / "dataset" / f"{name}.json").read_text())
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, dict) for key, value in payload.items()
    ):
        raise ValueError(f"Fencing-reference {name} split is malformed")
    return payload


def _fma_video_ids(archive_path: Path | None) -> set[str]:
    if archive_path is None or not archive_path.is_file():
        return set()
    with ZipFile(archive_path) as archive:
        descriptor = pd.read_csv(
            BytesIO(archive.read("Kaggle/ClipDescriptorKaggle.csv")),
            encoding="utf-8-sig",
        )
    return {_youtube_id(str(value)) for value in descriptor["URL"]}


def _metadata_tree_sha256(source_directory: Path) -> str:
    paths = sorted((source_directory / "per_vid_labels").glob("*.json"))
    paths.extend(
        [source_directory / "dataset" / "train.json", source_directory / "dataset" / "valid.json"]
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(source_directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prepare(
    source_directory: Path,
    output_path: Path,
    link_manifest_path: Path,
    report_path: Path,
    *,
    archive_path: Path | None = None,
    fma_archive_path: Path | None = None,
) -> dict[str, Any]:
    train = _load_split(source_directory, "train")
    valid = _load_split(source_directory, "valid")
    duplicate_keys = set(train) & set(valid)
    if duplicate_keys:
        raise ValueError("Fencing-reference train and valid sample keys overlap")

    rows: list[dict[str, Any]] = []
    video_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    split_video_ids: dict[str, set[str]] = {"train": set(), "valid": set()}
    for split_name, split in (("train", train), ("valid", valid)):
        for sample_key, value in split.items():
            annotation = value.get("annotations")
            frame_annotation = value.get("annotations_frame")
            if not isinstance(annotation, dict) or not isinstance(frame_annotation, dict):
                raise ValueError("Fencing-reference annotation structure changed")
            label = str(annotation.get("label"))
            if label not in LABELS or frame_annotation.get("label") != label:
                raise ValueError("Fencing-reference label vocabulary changed")
            seconds = annotation.get("segment")
            frames = frame_annotation.get("segment")
            if not (
                isinstance(seconds, list)
                and len(seconds) == 2
                and isinstance(frames, list)
                and len(frames) == 2
            ):
                raise ValueError("Fencing-reference segment structure changed")
            start_seconds, end_seconds = map(float, seconds)
            start_frame, end_frame = map(int, frames)
            if end_seconds <= start_seconds or end_frame <= start_frame:
                raise ValueError("Fencing-reference segment is reversed")
            video_id = _youtube_id(str(value.get("url")))
            if not sample_key.startswith(f"{video_id}_"):
                raise ValueError("Fencing-reference sample/video crosswalk changed")
            split_video_ids[split_name].add(video_id)
            video_counts[video_id][label] += 1
            video_counts[video_id][f"split_{split_name}"] += 1
            rows.append(
                {
                    "sample_id": (
                        "fencing-ref:"
                        + hashlib.sha256(sample_key.encode()).hexdigest()[:20]
                    ),
                    "group_id": _group_id(video_id),
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "scoreboard_delta_label": label,
                    "upstream_split": split_name,
                }
            )

    output = pd.DataFrame(rows).sort_values(
        ["group_id", "start_frame", "sample_id"], kind="stable"
    )
    duplicate_windows = output.duplicated(
        ["group_id", "start_frame", "end_frame"], keep=False
    )
    if duplicate_windows.any():
        raise ValueError("Fencing-reference contains duplicate source-video windows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    link_rows = []
    for video_id, counts in sorted(video_counts.items()):
        link_rows.append(
            {
                "video_id": video_id,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "annotation_count": sum(counts[label] for label in sorted(LABELS)),
                "label_counts": {label: counts[label] for label in sorted(LABELS)},
                "appears_in_upstream_train": bool(counts["split_train"]),
                "appears_in_upstream_valid": bool(counts["split_valid"]),
                "media_acquired": False,
                "license_status": "unknown",
            }
        )
    link_manifest_sha256 = write_jsonl(link_manifest_path, link_rows)

    all_video_ids = set(video_counts)
    fma_ids = _fma_video_ids(fma_archive_path)
    per_video_files = sorted((source_directory / "per_vid_labels").glob("*.json"))
    label_counts = Counter(output["scoreboard_delta_label"])
    original_split_overlap = split_video_ids["train"] & split_video_ids["valid"]
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        "annotations": len(output),
        "source_videos_with_annotations": len(all_video_ids),
        "per_video_annotation_files": len(per_video_files),
        "files_without_selected_split_annotations": len(per_video_files) - len(all_video_ids),
        "train_annotations": len(train),
        "valid_annotations": len(valid),
        "train_source_videos": len(split_video_ids["train"]),
        "valid_source_videos": len(split_video_ids["valid"]),
        "train_valid_source_video_overlap": len(original_split_overlap),
        "valid_videos_already_seen_in_train_fraction": (
            len(original_split_overlap) / len(split_video_ids["valid"])
        ),
        "label_counts": dict(sorted(label_counts.items())),
        "fencing_matches_ai_source_videos": len(fma_ids) if fma_ids else None,
        "source_video_overlap_with_fencing_matches_ai": len(all_video_ids & fma_ids),
        "source_videos_new_beyond_fencing_matches_ai": len(all_video_ids - fma_ids),
        "source_video_union_with_fencing_matches_ai": len(all_video_ids | fma_ids),
        "metadata_tree_sha256": _metadata_tree_sha256(source_directory),
        "archive_bytes": archive_path.stat().st_size if archive_path is not None else None,
        "archive_sha256": sha256_file(archive_path) if archive_path is not None else None,
        "output_sha256": sha256_file(output_path),
        "output": "data/processed/fencing-reference-scoreboard-deltas.parquet",
        "link_manifest_sha256": link_manifest_sha256,
        "link_manifest": "data/manifests/fencing-reference-video-links.jsonl",
        "claim_boundary": (
            "Automatic four-second double-light clip intervals with left, right, or together "
            "scoreboard-delta labels. These labels are not verified referee decisions, right-of-"
            "way calls, touch correctness, or weapon labels. YouTube media are not downloaded or "
            "redistributed. Every one of the upstream validation videos also appears in its train "
            "split, so the published split is rejected and any future model must hold out complete "
            "source-video groups. The source has no license. Overlapping source videos with the "
            "Fencing Matches AI archive are identified and never counted as independent videos."
        ),
    }
    write_json(report_path, summary)
    return summary
