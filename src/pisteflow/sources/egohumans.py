"""Exact fencing-only inventory for the EgoHumans multi-view benchmark."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pisteflow.provenance import sha256_file, write_json, write_jsonl

DATASET_PAGE = "https://rawalkhirodkar.github.io/egohumans/"
REPOSITORY = "https://github.com/rawalkhirodkar/egohumans"
REPOSITORY_SNAPSHOT = "d7875dd4b3e3cc4e306af386d5c4febf1290a3e9"
ALL_ANGLES_REPOSITORY = "https://huggingface.co/datasets/ch-chenyu/All-Angles-Bench"
ALL_ANGLES_SNAPSHOT = "bb91c32faa9cb23a4302df53e2442c3ac69fe21f"
DATA_FOLDER_ID = "1JD963urzuzV_R_6FOVOtlx8UupwUuknR"
BENCHMARK_FOLDER_ID = "1cz1P-fO5bkZSbGhCqE1sABDIHHA74JWz"
EGO_RGB_FILE_ID = "1Eigfw_bZWzuwA2qboTTWY5Ah6ACS-gZ8"
EGO_RGB_BYTES = 331_828_578
EGO_RGB_SHA256 = "c7bf0cdba39635c867e9d91a955491b41822e25cf62f7ac492cf321a56aef3a8"

RAW_SEQUENCE_FILE_IDS = {
    "001": "1sOa-YsPDJFIPC17z4DDyQA-gAPJa8uzo",
    "002": "195ABHXOlkHVhQwGWv0rqAcoRqsKKOgO0",
    "003": "1o1Lmn08RgusbBUQ0K0pOyqeEqGfcHnXq",
    "004": "1hsUBUjcgghX5PLnuFTPouNNVpeVIMM6Y",
    "005": "1gIvSVP7nfzLhMUiAnl6eYLSukld_e2pK",
    "006": "1JlJ5p2tD0netEk45oYL3iVOIq5f3VXaf",
    "007": "1pig_6ErpxRGXdo4BOFVPIpGKowux7JNF",
    "008": "1UtykV6bzVO5o2f7SLF-j8BH231W2tzbt",
    "009": "1gQHoLw2bwkJACRSyaMhpydAfQp70rbAj",
    "010": "1jmAL1WaO4Dzi9aaJvxUYO6uLHme9yNZd",
    "011": "1QtS_VSZJsMeExhRgTx0XiUblOFI98v-v",
    "012": "1uZOB5WhP8Zwi5-pom_s3QEx7fo-qYWBM",
    "013": "1AX1omSP8Lj3PcXWBsmJoBkpNKQ6aB6FQ",
    "014": "1a5cYkz8uzDVS-dp3u779Glbv2GMGdMoV",
}

EXPECTED_IMAGES = 19_928
EXPECTED_ANNOTATIONS = 29_192
EXPECTED_2D_KEYPOINTS = 493_021
EXPECTED_3D_KEYPOINTS = 496_264
EXPECTED_SEQUENCES = 13
EXPECTED_VIEWS = 35
EXPECTED_CAPTURE_ALIASES = 3

PREVIEW_FILES = {
    "cam03.jpg": (134_815, "21d0da6ec2c99658c9a271a2853788a0ad307ac0b334d9af1f32fd02ebdcf921"),
    "cam06.jpg": (123_794, "00a3004bada329088667a7024bdd849988fb4431da0c98c9072a7d5d6a5b215d"),
    "cam09.jpg": (119_668, "fc16c9500ce782d0275c80b64b7c853f0ecea5aa75874df770fb70e4cc442eb6"),
    "cam12.jpg": (118_881, "d45d0aa461c30fd573ee765518e1b0248bd510180659108257cab44a82fb6cdd"),
    "cam13.jpg": (127_878, "bc01a627c83d151924825026d107307b63902fb7bbea61bc2daa780c693f569b"),
}
OFFICIAL_DEMO_VIDEO_IDS = (
    "Lcdjv3wxoWg",
    "ZE3ZBy_E3Z8",
    "27x17t9I4d8",
    "ji58myAkeVE",
)


def acquire_ego_rgb(destination: Path, *, force: bool = False) -> Path:
    """Download the public benchmark JSON through its stable Drive file ID."""

    if destination.exists() and not force:
        _validate_source(destination)
        return destination
    if force and destination.exists():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from importlib import import_module

        gdown = import_module("gdown")
    except ImportError as exc:  # pragma: no cover - exercised only without collector extras
        raise RuntimeError("Install the collectors extra to download EgoHumans") from exc
    result = gdown.download(
        id=EGO_RGB_FILE_ID,
        output=str(destination),
        quiet=False,
        resume=True,
    )
    if result is None:
        raise RuntimeError("EgoHumans Google Drive download did not return a file")
    _validate_source(destination)
    return destination


def _validate_source(path: Path) -> None:
    if path.stat().st_size != EGO_RGB_BYTES:
        raise ValueError(f"EgoHumans ego-RGB annotation byte size changed: {path.stat().st_size}")
    actual = sha256_file(path)
    if actual != EGO_RGB_SHA256:
        raise ValueError(f"EgoHumans ego-RGB annotation checksum changed: {actual}")


def _load(path: Path) -> dict[str, Any]:
    _validate_source(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("EgoHumans benchmark root is not an object")
    for key in ("images", "annotations", "categories", "licenses"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"EgoHumans benchmark is missing list field: {key}")
    return payload


def _sequence_from_file_name(file_name: str) -> str:
    parts = file_name.split("/")
    if len(parts) < 7 or parts[0] != "03_fencing" or not parts[1].endswith("_fencing"):
        raise ValueError(f"Unexpected EgoHumans fencing image path: {file_name}")
    return parts[1].removesuffix("_fencing")


def image_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    images = [
        image
        for image in payload["images"]
        if isinstance(image, dict)
        and isinstance(image.get("file_name"), str)
        and str(image["file_name"]).startswith("03_fencing/")
    ]
    image_ids = {int(image["id"]) for image in images}
    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        if isinstance(image_id, int) and image_id in image_ids:
            annotations_by_image[image_id].append(annotation)

    rows: list[dict[str, Any]] = []
    for image in images:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        sequence = _sequence_from_file_name(file_name)
        parts = file_name.split("/")
        camera = parts[3]
        annotations = annotations_by_image[image_id]
        timestamps = sorted(
            {
                int(row["time_stamp"])
                for row in annotations
                if isinstance(row.get("time_stamp"), int)
            }
        )
        humans = sorted(
            {
                str(row["human_name"])
                for row in annotations
                if isinstance(row.get("human_name"), str)
            }
        )
        camera_names = sorted(
            {
                str(row["camera_name"])
                for row in annotations
                if isinstance(row.get("camera_name"), str)
            }
        )
        keypoints_2d = sum(
            int(row.get("num_keypoints", 0))
            for row in annotations
            if isinstance(row.get("num_keypoints", 0), int)
        )
        keypoints_3d = sum(
            len(value) // 4
            for row in annotations
            if isinstance((value := row.get("keypoints_3d")), list)
        )
        rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "sequence": sequence,
                "camera": camera,
                "camera_mode": "rgb",
                "video_name": str(image["video_name"]),
                "source_group_id": f"egohumans:03_fencing:{sequence}",
                "view_group_id": f"egohumans:03_fencing:{sequence}:{camera}:rgb",
                "frame_number": int(Path(file_name).stem),
                "annotation_timestamps": timestamps,
                "person_annotations": len(annotations),
                "capture_aliases": humans,
                "annotation_camera_names": camera_names,
                "observed_2d_keypoints": keypoints_2d,
                "available_3d_keypoints": keypoints_3d,
                "weapon": "unknown",
                "activity": "fencing",
                "image_license_id": int(image["license"]),
                "media_downloaded": False,
                "raw_sequence_file_id": RAW_SEQUENCE_FILE_IDS[sequence],
            }
        )
    rows.sort(key=lambda row: int(row["image_id"]))
    return rows


def _group_records(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output: list[dict[str, Any]] = []
    for group_id, members in sorted(groups.items()):
        sequences = sorted({str(row["sequence"]) for row in members})
        aliases = sorted({alias for row in members for alias in row["capture_aliases"]})
        output.append(
            {
                key: group_id,
                "source_group_id": str(members[0]["source_group_id"]),
                "sequences": sequences,
                "views": sorted({str(row["view_group_id"]) for row in members}),
                "images": len(members),
                "person_annotations": sum(int(row["person_annotations"]) for row in members),
                "observed_2d_keypoints": sum(int(row["observed_2d_keypoints"]) for row in members),
                "available_3d_keypoints": sum(
                    int(row["available_3d_keypoints"]) for row in members
                ),
                "capture_aliases": aliases,
                "first_frame_number": min(int(row["frame_number"]) for row in members),
                "last_frame_number": max(int(row["frame_number"]) for row in members),
                "weapon": "unknown",
                "raw_sequence_file_id": str(members[0]["raw_sequence_file_id"]),
                "raw_sequence_url": (
                    "https://drive.google.com/uc?id=" + str(members[0]["raw_sequence_file_id"])
                ),
            }
        )
    return output


def preview_records(preview_dir: Path) -> list[dict[str, Any]]:
    """Verify the small public All-Angles mirror of one synchronized fencing scene."""

    rows: list[dict[str, Any]] = []
    for name, (expected_bytes, expected_sha256) in sorted(PREVIEW_FILES.items()):
        path = preview_dir / name
        if not path.is_file():
            raise ValueError(f"Missing EgoHumans synchronized preview: {path}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(
                f"EgoHumans synchronized preview changed: {name} "
                f"bytes={actual_bytes}, sha256={actual_sha256}"
            )
        camera = path.stem
        rows.append(
            {
                "preview_id": f"egohumans:001_fencing:{camera}",
                "source_group_id": "egohumans:03_fencing:001",
                "sequence": "001",
                "camera": camera,
                "path": str(path),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "media_downloaded": True,
                "weapon": "unknown",
                "activity": "fencing",
                "license": "CC BY-NC-SA 2.0 (upstream embedded image license)",
                "mirror_repository": ALL_ANGLES_REPOSITORY,
                "mirror_snapshot": ALL_ANGLES_SNAPSHOT,
            }
        )
    return rows


def prepare(
    source_path: Path,
    image_manifest_path: Path,
    view_manifest_path: Path,
    sequence_manifest_path: Path,
    report_path: Path,
    *,
    preview_dir: Path | None = None,
    preview_manifest_path: Path | None = None,
) -> dict[str, Any]:
    payload = _load(source_path)
    rows = image_records(payload)
    annotations = sum(int(row["person_annotations"]) for row in rows)
    keypoints_2d = sum(int(row["observed_2d_keypoints"]) for row in rows)
    keypoints_3d = sum(int(row["available_3d_keypoints"]) for row in rows)
    sequences = _group_records(rows, "source_group_id")
    views = _group_records(rows, "view_group_id")
    aliases = {alias for row in rows for alias in row["capture_aliases"]}
    actual = {
        "images": len(rows),
        "annotations": annotations,
        "2d_keypoints": keypoints_2d,
        "3d_keypoints": keypoints_3d,
        "sequences": len(sequences),
        "views": len(views),
        "capture_aliases": len(aliases),
    }
    expected = {
        "images": EXPECTED_IMAGES,
        "annotations": EXPECTED_ANNOTATIONS,
        "2d_keypoints": EXPECTED_2D_KEYPOINTS,
        "3d_keypoints": EXPECTED_3D_KEYPOINTS,
        "sequences": EXPECTED_SEQUENCES,
        "views": EXPECTED_VIEWS,
        "capture_aliases": EXPECTED_CAPTURE_ALIASES,
    }
    if actual != expected:
        raise ValueError(f"EgoHumans fencing benchmark counts changed: {actual} != {expected}")

    image_sha256 = write_jsonl(image_manifest_path, rows)
    view_sha256 = write_jsonl(view_manifest_path, views)
    sequence_sha256 = write_jsonl(sequence_manifest_path, sequences)
    previews: list[dict[str, Any]] = []
    preview_sha256: str | None = None
    if preview_dir is not None:
        if preview_manifest_path is None:
            raise ValueError("preview_manifest_path is required when preview_dir is supplied")
        previews = preview_records(preview_dir)
        preview_sha256 = write_jsonl(preview_manifest_path, previews)
    sequence_ids = sorted({str(row["sequence"]) for row in rows})
    missing = sorted(set(RAW_SEQUENCE_FILE_IDS).difference(sequence_ids))
    image_license = next(
        (
            license_row
            for license_row in payload["licenses"]
            if isinstance(license_row, dict) and license_row.get("id") == 1
        ),
        None,
    )
    summary: dict[str, Any] = {
        "dataset_page": DATASET_PAGE,
        "repository": REPOSITORY,
        "repository_snapshot": REPOSITORY_SNAPSHOT,
        "all_angles_mirror": ALL_ANGLES_REPOSITORY,
        "all_angles_snapshot": ALL_ANGLES_SNAPSHOT,
        "benchmark_folder_id": BENCHMARK_FOLDER_ID,
        "source_file_id": EGO_RGB_FILE_ID,
        "source_bytes": source_path.stat().st_size,
        "source_sha256": sha256_file(source_path),
        **actual,
        "annotated_sequence_ids": sequence_ids,
        "raw_sequence_archives_listed": len(RAW_SEQUENCE_FILE_IDS),
        "raw_sequence_archives_without_ego_rgb_annotations": missing,
        "embedded_coco_image_license": image_license,
        "weapon": "unknown",
        "media_downloaded": False,
        "verified_synchronized_preview_images": len(previews),
        "verified_synchronized_preview_groups": len(
            {str(row["source_group_id"]) for row in previews}
        ),
        "verified_synchronized_preview_bytes": sum(int(row["bytes"]) for row in previews),
        "official_demo_video_links": [
            f"https://www.youtube.com/watch?v={video_id}"
            for video_id in OFFICIAL_DEMO_VIDEO_IDS
        ],
        "grouping_rule": (
            "hold out complete 03_fencing subsequences; views and frames from one capture "
            "sequence must never cross train, validation, or test partitions"
        ),
        "outputs": {
            "images": "data/manifests/egohumans-fencing-ego-rgb-images.jsonl",
            "views": "data/manifests/egohumans-fencing-ego-rgb-views.jsonl",
            "sequences": "data/manifests/egohumans-fencing-sequences.jsonl",
            "synchronized_preview": (
                "data/manifests/egohumans-fencing-synchronized-preview.jsonl"
                if previews
                else None
            ),
        },
        "output_sha256": {
            "images": image_sha256,
            "views": view_sha256,
            "sequences": sequence_sha256,
            "synchronized_preview": preview_sha256,
        },
        "claim_boundary": (
            "The pinned public EgoHumans benchmark contains 19,928 ego-RGB fencing frames, "
            "29,192 person instances, 493,021 observed 2D keypoints, and 496,264 available "
            "3D keypoints across 35 camera views and 13 annotated capture sequences. The "
            "three aria names are capture aliases, not evidence of three globally distinct "
            "athletes. Fourteen raw fencing sequence archives are listed, but sequence 007 "
            "has no ego-RGB rows in this benchmark JSON. The embedded COCO metadata labels "
            "images CC BY-NC-SA 2.0. Weapon, touch, score, and bout labels are absent."
            " Five locally verified synchronized stills are different camera views of one "
            "scene, not five bouts or five independent samples; four official demo videos are "
            "links only and are not counted as downloaded training media."
        ),
    }
    write_json(report_path, summary)
    return summary
