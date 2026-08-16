"""Research-only fencing images and 2D poses from MPII Human Pose."""

from __future__ import annotations

import io
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

DATASET_PAGE = (
    "https://www.mpi-inf.mpg.de/de/departments/computer-vision-and-machine-learning/"
    "software-and-datasets/mpii-human-pose-dataset"
)
ANNOTATION_URL = (
    "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/"
    "mpii_human_pose_v1_u12_2.zip"
)
ANNOTATION_BYTES = 12_340_483
ANNOTATION_SHA256 = "3e8bd6b61d39bc2e2bc5cd33a17b0bd08cd2d06c302f24f552042e53a35bcb76"
MAT_MEMBER = "mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
HF_MIRROR = "Voxel51/MPII_Human_Pose_Dataset"
HF_SNAPSHOT = "27de5937bd42235aef731784068ee2c04808d3e3"
HF_IMAGE_PREFIX = (
    f"https://huggingface.co/datasets/{HF_MIRROR}/resolve/{HF_SNAPSHOT}/data/data_15"
)

EXPECTED_IMAGES = 54
EXPECTED_SOURCE_VIDEOS = 7
EXPECTED_PERSON_ANNOTATIONS = 102
EXPECTED_KEYPOINTS = 1_442

VIDEO_LABELS: dict[str, dict[str, str]] = {
    "a8IdfA5fXJs": {
        "title": "How To Fence: The Basics of Fencing, Taught by Olympians",
        "weapon": "unknown",
        "weapon_evidence": "general_instruction_title_does_not_identify_weapon",
    },
    "Z86tpjRaiK8": {
        "title": "Speed of Fencing long version",
        "weapon": "unknown",
        "weapon_evidence": "source_title_does_not_identify_weapon",
    },
    "nGearEu2PlU": {
        "title": "Aron Szilagyi Wins Fencing Sabre Gold - London 2012 Olympics",
        "weapon": "sabre",
        "weapon_evidence": "source_title_explicit",
    },
    "sF7HsBr0bzY": {
        "title": "Men's Individual Sabre Semi-Finals - London 2012 Olympics",
        "weapon": "sabre",
        "weapon_evidence": "source_title_explicit",
    },
    "voEDmSkhlxI": {
        "title": "Sheng wins Men's Individual Foil Gold - London 2012 Olympics",
        "weapon": "foil",
        "weapon_evidence": "source_title_explicit",
    },
    "Dr9q645buro": {
        "title": "Intro to fencing: what is fencing?",
        "weapon": "unknown",
        "weapon_evidence": "general_instruction_title_does_not_identify_weapon",
    },
    "FNp_JLft_tA": {
        "title": "Fencing CWCH 2010 Mens Epee Gold Medal Match",
        "weapon": "epee",
        "weapon_evidence": "source_title_explicit",
    },
}


def acquire_annotations(destination: Path, *, user_agent: str, force: bool = False) -> Path:
    if force and destination.exists():
        destination.unlink()
    download_resumable(
        ANNOTATION_URL,
        destination,
        user_agent=user_agent,
        expected_bytes=ANNOTATION_BYTES,
    )
    actual = sha256_file(destination)
    if actual != ANNOTATION_SHA256:
        raise ValueError(f"MPII annotation checksum changed: {actual}")
    return destination


def _python_scalar(value: Any) -> int | float | None:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return None


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return []


def records_from_release(release: dict[str, Any]) -> list[dict[str, Any]]:
    annolist = release.get("annolist")
    activities = release.get("act")
    video_list = release.get("video_list")
    image_splits = release.get("img_train")
    if isinstance(video_list, np.ndarray):
        video_list = video_list.tolist()
    if isinstance(image_splits, np.ndarray):
        image_splits = image_splits.tolist()
    if (
        not isinstance(annolist, list)
        or not isinstance(activities, list)
        or not isinstance(video_list, list)
        or not isinstance(image_splits, list)
    ):
        raise ValueError("MPII annotation lists are malformed")
    if len(annolist) != len(activities) or len(annolist) != len(image_splits):
        raise ValueError("MPII annotation vector lengths differ")

    rows: list[dict[str, Any]] = []
    for index, (activity, annotation) in enumerate(zip(activities, annolist, strict=True)):
        activity_name = activity.get("act_name") if isinstance(activity, dict) else None
        if not isinstance(activity_name, str) or activity_name != "fencing":
            continue
        if not isinstance(annotation, dict):
            raise ValueError("MPII fencing image annotation is malformed")
        image = annotation.get("image")
        filename = image.get("name") if isinstance(image, dict) else None
        video_index = _python_scalar(annotation.get("vididx"))
        frame_second = _python_scalar(annotation.get("frame_sec"))
        image_split = _python_scalar(image_splits[index])
        if not isinstance(filename, str) or not isinstance(video_index, int) or video_index <= 0:
            raise ValueError("MPII fencing source mapping is incomplete")
        if not isinstance(image_split, int):
            raise ValueError("MPII fencing image split is malformed")
        youtube_id = str(video_list[video_index - 1])
        source = VIDEO_LABELS.get(
            youtube_id,
            {
                "title": "title unavailable",
                "weapon": "unknown",
                "weapon_evidence": "no_reviewed_source_title",
            },
        )

        people: list[dict[str, Any]] = []
        for person_index, rectangle in enumerate(
            _as_dict_list(annotation.get("annorect")), start=1
        ):
            annotation_points = rectangle.get("annopoints")
            points = (
                _as_dict_list(annotation_points.get("point"))
                if isinstance(annotation_points, dict)
                else []
            )
            keypoints = []
            for point in points:
                point_id = _python_scalar(point.get("id"))
                x = _python_scalar(point.get("x"))
                y = _python_scalar(point.get("y"))
                if not isinstance(point_id, int) or x is None or y is None:
                    raise ValueError("MPII fencing keypoint is malformed")
                keypoints.append(
                    {
                        "id": point_id,
                        "x": x,
                        "y": y,
                        "is_visible": _python_scalar(point.get("is_visible")),
                    }
                )
            keypoints.sort(key=lambda point: int(str(point["id"])))
            objpos = rectangle.get("objpos")
            people.append(
                {
                    "person_index": person_index,
                    "head_bbox_xyxy": [
                        _python_scalar(rectangle.get(key)) for key in ("x1", "y1", "x2", "y2")
                    ],
                    "person_scale": _python_scalar(rectangle.get("scale")),
                    "object_position_xy": (
                        [_python_scalar(objpos.get("x")), _python_scalar(objpos.get("y"))]
                        if isinstance(objpos, dict)
                        else [None, None]
                    ),
                    "keypoints": keypoints,
                }
            )

        rows.append(
            {
                "image_id": f"mpii:{filename.removesuffix('.jpg')}",
                "annotation_image_index": index + 1,
                "filename": filename,
                "activity_label": "fencing",
                "category_label": "sports",
                "split": "training" if image_split == 1 else "testing",
                "youtube_id": youtube_id,
                "source_group_id": f"youtube:{youtube_id}",
                "source_url": f"https://www.youtube.com/watch?v={youtube_id}",
                "source_title_snapshot": source["title"],
                "frame_second": frame_second,
                "weapon": source["weapon"],
                "weapon_label_source": source["weapon_evidence"],
                "people": people,
                "person_annotations": len(people),
                "keypoint_annotations": sum(len(person["keypoints"]) for person in people),
                "annotation_license": "Simplified BSD; research-purpose statement in README",
                "image_rights": "research_only_original_image_copyright_not_owned_by_mpii",
                "media_downloaded": False,
                "mirror_snapshot": HF_SNAPSHOT,
            }
        )
    rows.sort(key=lambda row: str(row["image_id"]))
    return rows


def parse_annotations(archive_path: Path) -> list[dict[str, Any]]:
    if archive_path.stat().st_size != ANNOTATION_BYTES:
        raise ValueError("MPII annotation archive byte size changed")
    actual = sha256_file(archive_path)
    if actual != ANNOTATION_SHA256:
        raise ValueError(f"MPII annotation checksum changed: {actual}")
    with zipfile.ZipFile(archive_path) as archive:
        if MAT_MEMBER not in archive.namelist() or archive.testzip() is not None:
            raise ValueError("MPII annotation archive integrity check failed")
        release = loadmat(io.BytesIO(archive.read(MAT_MEMBER)), simplify_cells=True)["RELEASE"]
    if not isinstance(release, dict):
        raise ValueError("MPII RELEASE structure is malformed")
    rows = records_from_release(release)
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f"MPII fencing image count changed: {len(rows)}")
    if len({str(row["youtube_id"]) for row in rows}) != EXPECTED_SOURCE_VIDEOS:
        raise ValueError("MPII fencing source-video count changed")
    if sum(int(row["person_annotations"]) for row in rows) != EXPECTED_PERSON_ANNOTATIONS:
        raise ValueError("MPII fencing person-annotation count changed")
    if sum(int(row["keypoint_annotations"]) for row in rows) != EXPECTED_KEYPOINTS:
        raise ValueError("MPII fencing keypoint count changed")
    return rows


def _valid_jpeg(path: Path) -> bool:
    if path.stat().st_size < 4:
        return False
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(-2, 2)
        end = handle.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def download_images(
    rows: list[dict[str, Any]],
    image_directory: Path,
    *,
    user_agent: str,
    workers: int = 4,
) -> list[dict[str, Any]]:
    image_directory.mkdir(parents=True, exist_ok=True)

    def acquire(row: dict[str, Any]) -> dict[str, Any]:
        filename = str(row["filename"])
        destination = image_directory / filename
        download_resumable(
            f"{HF_IMAGE_PREFIX}/{filename}",
            destination,
            user_agent=user_agent,
        )
        if not _valid_jpeg(destination):
            raise ValueError(f"MPII mirror image is not a complete JPEG: {filename}")
        return {
            **row,
            "media_downloaded": True,
            "image_bytes": destination.stat().st_size,
            "image_sha256": sha256_file(destination),
            "local_filename": filename,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(acquire, rows))


def prepare(
    annotation_archive: Path,
    image_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    acquire_media: bool = True,
    workers: int = 4,
) -> dict[str, Any]:
    rows = parse_annotations(annotation_archive)
    if acquire_media:
        rows = download_images(rows, image_directory, user_agent=user_agent, workers=workers)
    rows.sort(key=lambda row: str(row["image_id"]))
    manifest_sha256 = write_jsonl(manifest_path, rows)
    by_weapon = Counter(str(row["weapon"]) for row in rows)
    summary: dict[str, Any] = {
        "dataset_page": DATASET_PAGE,
        "annotation_url": ANNOTATION_URL,
        "annotation_bytes": annotation_archive.stat().st_size,
        "annotation_sha256": sha256_file(annotation_archive),
        "image_mirror": HF_MIRROR,
        "image_mirror_snapshot": HF_SNAPSHOT,
        "images": len(rows),
        "source_videos": len({str(row["youtube_id"]) for row in rows}),
        "person_annotations": sum(int(row["person_annotations"]) for row in rows),
        "keypoint_annotations": sum(int(row["keypoint_annotations"]) for row in rows),
        "by_weapon": dict(sorted(by_weapon.items())),
        "media_downloaded": sum(bool(row["media_downloaded"]) for row in rows),
        "downloaded_image_bytes": sum(int(row.get("image_bytes", 0)) for row in rows),
        "grouping_rule": "hold out all frames from the same source_group_id together",
        "output": "data/manifests/mpii-fencing-pose-images.jsonl",
        "manifest_sha256": manifest_sha256,
        "claim_boundary": (
            "Fifty-four real fencing frames from seven YouTube source videos with 102 person "
            "and 1,442 2D keypoint annotations. Explicit source titles identify 4 epee, 3 foil, "
            "and 26 sabre frames; the remaining 21 stay weapon-unknown. MPII permits the "
            "annotations and code for research but states that it does not own the source-image "
            "copyright, so images and image-derived artifacts remain local research inputs and "
            "are not redistributed. Frames are not independent athlete or bout samples."
        ),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    annotation_archive: Path,
    image_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
    acquire_media: bool = True,
    workers: int = 4,
) -> dict[str, Any]:
    acquire_annotations(annotation_archive, user_agent=user_agent, force=force)
    return prepare(
        annotation_archive,
        image_directory,
        manifest_path,
        report_path,
        user_agent=user_agent,
        acquire_media=acquire_media,
        workers=workers,
    )
