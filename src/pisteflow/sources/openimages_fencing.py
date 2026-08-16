"""Filter and acquire human-verified fencing labels from Open Images V7."""

from __future__ import annotations

import base64
import csv
import hashlib
import ssl
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import truststore

from pisteflow.provenance import read_jsonl, sha256_file, write_json, write_jsonl

SOURCE_PAGE = "https://storage.googleapis.com/openimages/web/download_v7.html"
ANNOTATION_LICENSE = "CC BY 4.0"
IMAGE_LICENSE_BOUNDARY = (
    "Open Images documents image-level author and license metadata separately. Images are kept "
    "local until each selected row is joined to that metadata; no image redistribution is "
    "authorized by the annotation license alone."
)
S3_BASE = "https://open-images-dataset.s3.amazonaws.com"

IMAGE_METADATA_FILES: dict[str, dict[str, str | int | None]] = {
    "train": {
        "name": "oidv6-train-images-with-labels-with-rotation.csv",
        "bytes": 2_684_720_962,
        "sha256": "5758734fae55dddf7ed6e4505e25a19c45ea5f6cff7f524eae1721b841b98a79",
        "url": (
            "https://storage.googleapis.com/openimages/v6/"
            "oidv6-train-images-with-labels-with-rotation.csv"
        ),
    },
    "validation": {
        "name": "validation-images-with-rotation.csv",
        "bytes": 15_245_485,
        "sha256": "ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17",
        "url": (
            "https://storage.googleapis.com/openimages/2018_04/validation/"
            "validation-images-with-rotation.csv"
        ),
    },
    "test": {
        "name": "test-images-with-rotation.csv",
        "bytes": 45_227_339,
        "sha256": "de55c6b8cbda32a79f4a20f28572d54e9c49d527a4006725675764074a51a36d",
        "url": (
            "https://storage.googleapis.com/openimages/2018_04/test/"
            "test-images-with-rotation.csv"
        ),
    },
}

IMAGE_METADATA_HEADER = [
    "ImageID",
    "Subset",
    "OriginalURL",
    "OriginalLandingURL",
    "License",
    "AuthorProfileURL",
    "Author",
    "Title",
    "OriginalSize",
    "OriginalMD5",
    "Thumbnail300KURL",
    "Rotation",
]

CC_BY_2_LICENSES = {
    "http://creativecommons.org/licenses/by/2.0",
    "http://creativecommons.org/licenses/by/2.0/",
    "https://creativecommons.org/licenses/by/2.0",
    "https://creativecommons.org/licenses/by/2.0/",
}

MID_TO_LABEL = {
    "/m/02y8z": "fencing",
    "/m/0h8mjgm": "fencing_weapon",
    "/m/026q93": "foil_weapon",
    "/m/06s7q8": "sabre",
    "/m/0bmj3zs": "wheelchair_fencing",
    "/m/032t45": "epee",
}
MID_TO_WEAPON = {
    "/m/026q93": "foil",
    "/m/06s7q8": "sabre",
    "/m/032t45": "epee",
}
EXPECTED_CLASS_NAMES = {
    "/m/02y8z": "Fencing",
    "/m/0h8mjgm": "Fencing weapon",
    "/m/026q93": "Foil (Weapon)",
    "/m/06s7q8": "Sabre (Fencing)",
    "/m/0bmj3zs": "Wheelchair fencing",
    "/m/032t45": "Épée",
}

ANNOTATION_FILES: dict[str, dict[str, str | int]] = {
    "train": {
        "name": "oidv7-train-annotations-human-imagelabels.csv",
        "bytes": 2_735_816_020,
        "sha256": "0237e1e4080291b959ceb99484a8f2d723aeeb7d15b2adf60337f2c027174116",
        "url": (
            "https://storage.googleapis.com/openimages/v7/"
            "oidv7-train-annotations-human-imagelabels.csv"
        ),
    },
    "validation": {
        "name": "validation-annotations-human-imagelabels.csv",
        "bytes": 24_824_095,
        "sha256": "c2bc8b312ee7ed01ac4c58a62c2432c7f6b83b320874cbb0644e2f51aed3faad",
        "url": (
            "https://storage.googleapis.com/openimages/v5/"
            "validation-annotations-human-imagelabels.csv"
        ),
    },
    "test": {
        "name": "test-annotations-human-imagelabels.csv",
        "bytes": 75_042_713,
        "sha256": "e45c80b31dadd6a91eda7c36529d7a85ec1c4771fe7d9e07dab9ec2bcf3a3fca",
        "url": (
            "https://storage.googleapis.com/openimages/v5/"
            "test-annotations-human-imagelabels.csv"
        ),
    },
}
CLASS_DESCRIPTION_FILE: dict[str, str | int] = {
    "name": "oidv7-class-descriptions.csv",
    "bytes": 501_178,
    "sha256": "84a4373a0efb7fd6d93fe19b0e7ceb6c1b855c233d13b9b78a9a33655c9fdce3",
    "url": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv",
}

EXPECTED_TARGET_LABEL_ROWS = 9_528
EXPECTED_POSITIVE_LABEL_ROWS = 1_503
EXPECTED_NEGATIVE_LABEL_ROWS = 8_025
EXPECTED_TARGET_IMAGES = 5_540
EXPECTED_POSITIVE_IMAGES = 646
EXPECTED_NEGATIVE_ONLY_IMAGES = 4_894
EXPECTED_POSITIVE_IMAGES_BY_SPLIT = {"test": 77, "train": 546, "validation": 23}
EXPECTED_TARGET_IMAGES_BY_SPLIT = {"test": 138, "train": 5_352, "validation": 50}
EXPECTED_POSITIVE_LABEL_COUNTS = {
    "epee": 229,
    "fencing": 429,
    "fencing_weapon": 381,
    "foil_weapon": 281,
    "sabre": 181,
    "wheelchair_fencing": 2,
}
EXPECTED_WEAPON_STATUS = {
    "ambiguous_multi_weapon": 183,
    "epee": 93,
    "foil": 104,
    "sabre": 77,
    "unknown": 189,
}


def _validate_input(path: Path, spec: dict[str, str | int]) -> None:
    if path.stat().st_size != int(str(spec["bytes"])):
        raise ValueError(f"Open Images input size changed: {path.name}")
    if sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"Open Images input checksum changed: {path.name}")


def _validate_metadata_input(path: Path, spec: dict[str, str | int | None]) -> str:
    if path.stat().st_size != int(str(spec["bytes"])):
        raise ValueError(f"Open Images metadata input size changed: {path.name}")
    digest = sha256_file(path)
    expected_digest = spec.get("sha256")
    if expected_digest is not None and digest != str(expected_digest):
        raise ValueError(f"Open Images metadata checksum changed: {path.name}")
    return digest


def _class_descriptions(path: Path) -> dict[str, str]:
    _validate_input(path, CLASS_DESCRIPTION_FILE)
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for mid, name in csv.reader(handle):
            if mid in MID_TO_LABEL:
                values[mid] = name
    if values != EXPECTED_CLASS_NAMES:
        raise ValueError(f"Open Images class descriptions changed: {values}")
    return values


def _split_annotations(
    path: Path,
    split: str,
) -> tuple[dict[str, dict[str, tuple[bool, str]]], Counter[str]]:
    spec = ANNOTATION_FILES[split]
    _validate_input(path, spec)
    images: defaultdict[str, dict[str, tuple[bool, str]]] = defaultdict(dict)
    source_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ImageID", "Source", "LabelName", "Confidence"]:
            raise ValueError(f"Open Images annotation header changed: {path.name}")
        for item in reader:
            mid = item["LabelName"]
            if mid not in MID_TO_LABEL:
                continue
            confidence = float(item["Confidence"])
            if confidence not in {0.0, 1.0}:
                raise ValueError(f"Unexpected Open Images confidence: {confidence}")
            image_id = item["ImageID"]
            value = (confidence == 1.0, item["Source"])
            previous = images[image_id].get(mid)
            if previous is not None and previous != value:
                raise ValueError(f"Conflicting Open Images image label: {split}/{image_id}/{mid}")
            images[image_id][mid] = value
            source_counts[item["Source"]] += 1
    return dict(images), source_counts


def _weapon_status(positive_mids: list[str]) -> str:
    weapons = {MID_TO_WEAPON[mid] for mid in positive_mids if mid in MID_TO_WEAPON}
    if len(weapons) == 1:
        return next(iter(weapons))
    if len(weapons) > 1:
        return "ambiguous_multi_weapon"
    return "unknown"


def records_from_annotations(data_directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one multi-label record per target image across official splits."""

    _class_descriptions(data_directory / str(CLASS_DESCRIPTION_FILE["name"]))
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for split, spec in ANNOTATION_FILES.items():
        annotations, split_sources = _split_annotations(
            data_directory / str(spec["name"]), split
        )
        source_counts.update(split_sources)
        for image_id, labels in annotations.items():
            positive_mids = sorted(mid for mid, value in labels.items() if value[0])
            negative_mids = sorted(mid for mid, value in labels.items() if not value[0])
            rows.append(
                {
                    "image_id": image_id,
                    "source_group_id": f"openimages:{image_id}",
                    "upstream_split": split,
                    "positive_mids": positive_mids,
                    "positive_labels": [MID_TO_LABEL[mid] for mid in positive_mids],
                    "negative_mids": negative_mids,
                    "negative_labels": [MID_TO_LABEL[mid] for mid in negative_mids],
                    "annotation_sources": sorted({value[1] for value in labels.values()}),
                    "has_positive_target": bool(positive_mids),
                    "weapon_status": _weapon_status(positive_mids),
                    "wheelchair_fencing_positive": "/m/0bmj3zs" in positive_mids,
                    "download_url": f"{S3_BASE}/{split}/{image_id}.jpg",
                    "media_acquired": False,
                    "image_bytes": None,
                    "image_sha256": None,
                    "width": None,
                    "height": None,
                    "annotation_license": ANNOTATION_LICENSE,
                    "image_license_status": "pending_image_metadata_join_local_only",
                    "image_license": None,
                    "original_url": None,
                    "original_landing_url": None,
                    "author_profile_url": None,
                    "author": None,
                    "title": None,
                    "original_size": None,
                    "original_md5": None,
                    "thumbnail_300k_url": None,
                    "rotation": None,
                    "acquisition_source": None,
                }
            )
    rows.sort(key=lambda row: (str(row["upstream_split"]), str(row["image_id"])))
    summary = _summarize(rows, source_counts)
    return rows, summary


def _summarize(rows: list[dict[str, Any]], source_counts: Counter[str]) -> dict[str, Any]:
    positive_rows = [row for row in rows if bool(row["has_positive_target"])]
    positive_label_counts = Counter(
        label for row in rows for label in row["positive_labels"]
    )
    negative_label_counts = Counter(
        label for row in rows for label in row["negative_labels"]
    )
    weapon_status = Counter(str(row["weapon_status"]) for row in positive_rows)
    target_by_split = Counter(str(row["upstream_split"]) for row in rows)
    positive_by_split = Counter(str(row["upstream_split"]) for row in positive_rows)
    target_label_rows = sum(
        len(row["positive_mids"]) + len(row["negative_mids"]) for row in rows
    )
    positive_label_rows = sum(len(row["positive_mids"]) for row in rows)
    summary: dict[str, Any] = {
        "source_page": SOURCE_PAGE,
        "annotation_license": ANNOTATION_LICENSE,
        "image_license_boundary": IMAGE_LICENSE_BOUNDARY,
        "target_label_rows": target_label_rows,
        "positive_label_rows": positive_label_rows,
        "negative_label_rows": target_label_rows - positive_label_rows,
        "target_images": len(rows),
        "positive_images": len(positive_rows),
        "negative_only_images": len(rows) - len(positive_rows),
        "target_images_by_split": dict(sorted(target_by_split.items())),
        "positive_images_by_split": dict(sorted(positive_by_split.items())),
        "positive_label_counts": dict(sorted(positive_label_counts.items())),
        "negative_label_counts": dict(sorted(negative_label_counts.items())),
        "positive_weapon_status": dict(sorted(weapon_status.items())),
        "annotation_source_counts": dict(sorted(source_counts.items())),
        "positive_images_with_multi_weapon_labels": sum(
            row["weapon_status"] == "ambiguous_multi_weapon" for row in positive_rows
        ),
        "positive_images_with_no_specific_weapon": sum(
            row["weapon_status"] == "unknown" for row in positive_rows
        ),
        "positive_wheelchair_fencing_images": sum(
            bool(row["wheelchair_fencing_positive"]) for row in positive_rows
        ),
    }
    expected = {
        "target_label_rows": EXPECTED_TARGET_LABEL_ROWS,
        "positive_label_rows": EXPECTED_POSITIVE_LABEL_ROWS,
        "negative_label_rows": EXPECTED_NEGATIVE_LABEL_ROWS,
        "target_images": EXPECTED_TARGET_IMAGES,
        "positive_images": EXPECTED_POSITIVE_IMAGES,
        "negative_only_images": EXPECTED_NEGATIVE_ONLY_IMAGES,
        "target_images_by_split": EXPECTED_TARGET_IMAGES_BY_SPLIT,
        "positive_images_by_split": EXPECTED_POSITIVE_IMAGES_BY_SPLIT,
        "positive_label_counts": EXPECTED_POSITIVE_LABEL_COUNTS,
        "positive_weapon_status": EXPECTED_WEAPON_STATUS,
    }
    actual = {key: summary[key] for key in expected}
    if actual != expected:
        raise ValueError(f"Open Images fencing label counts changed: {actual} != {expected}")
    return summary


def prepare(
    data_directory: Path,
    manifest_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    rows, summary = records_from_annotations(data_directory)
    manifest_sha256 = write_jsonl(manifest_path, rows)
    summary.update(
        {
            "manifest": "data/manifests/openimages-fencing-images.jsonl",
            "manifest_sha256": manifest_sha256,
            "input_files": {
                split: {
                    "bytes": int(str(spec["bytes"])),
                    "sha256": str(spec["sha256"]),
                    "url": str(spec["url"]),
                }
                for split, spec in ANNOTATION_FILES.items()
            },
            "claim_boundary": (
                "The six target classes yield 646 human-positive images, not 1,503 independent "
                "images; one image may carry several positive labels. Of those images, 183 have "
                "multiple weapon classes marked positive, so only 274 have exactly one positive "
                "foil, epee, or sabre class and 189 have no specific weapon class. These are "
                "image-level labels rather than athlete identities, actions, bouts, boxes, or "
                "source-video groups. Official validation/test splits are retained, while the "
                "4,894 negative-only images are cataloged as class-specific hard negatives."
            ),
        }
    )
    write_json(summary_path, summary)
    return summary


def _metadata_payload(item: dict[str, str]) -> dict[str, Any]:
    original_size = item["OriginalSize"].strip()
    rotation = item["Rotation"].strip()
    return {
        "image_license": item["License"].strip() or None,
        "original_url": item["OriginalURL"].strip() or None,
        "original_landing_url": item["OriginalLandingURL"].strip() or None,
        "author_profile_url": item["AuthorProfileURL"].strip() or None,
        "author": item["Author"].strip() or None,
        "title": item["Title"].strip() or None,
        "original_size": int(original_size) if original_size else None,
        "original_md5": item["OriginalMD5"].strip() or None,
        "thumbnail_300k_url": item["Thumbnail300KURL"].strip() or None,
        "rotation": float(rotation) if rotation else None,
        "image_license_status": (
            "image_metadata_joined_cc_by_2_local_copy"
            if item["License"].strip() in CC_BY_2_LICENSES
            else "image_metadata_joined_non_cc_by_2_local_only"
        ),
    }


def _selected_metadata(
    path: Path,
    split: str,
    target_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    spec = IMAGE_METADATA_FILES[split]
    digest = _validate_metadata_input(path, spec)
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != IMAGE_METADATA_HEADER:
            raise ValueError(f"Open Images metadata header changed: {path.name}")
        for item in reader:
            image_id = item["ImageID"]
            if image_id not in target_ids:
                continue
            if item["Subset"] != split:
                raise ValueError(
                    f"Open Images metadata split changed: {image_id}/{item['Subset']} != {split}"
                )
            payload = _metadata_payload(item)
            previous = selected.get(image_id)
            if previous is not None and previous != payload:
                raise ValueError(f"Conflicting Open Images metadata: {split}/{image_id}")
            selected[image_id] = payload
    return selected, digest


def enrich_image_metadata(
    manifest_path: Path,
    data_directory: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Join official per-image attribution and license metadata to target rows."""

    rows = read_jsonl(manifest_path)
    by_split: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[str(row["upstream_split"])].add(str(row["image_id"]))

    joined: dict[str, dict[str, Any]] = {}
    input_files: dict[str, dict[str, Any]] = {}
    missing_by_split: dict[str, list[str]] = {}
    for split, spec in IMAGE_METADATA_FILES.items():
        selected, digest = _selected_metadata(
            data_directory / str(spec["name"]),
            split,
            by_split[split],
        )
        joined.update(selected)
        missing_by_split[split] = sorted(by_split[split] - set(selected))
        input_files[split] = {
            "bytes": int(str(spec["bytes"])),
            "sha256": digest,
            "url": str(spec["url"]),
            "target_rows_joined": len(selected),
        }

    updated: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        payload = joined.get(str(row["image_id"]))
        if payload is not None:
            next_row.update(payload)
        if bool(next_row.get("media_acquired")) and not next_row.get("acquisition_source"):
            next_row["acquisition_source"] = "official_s3"
        updated.append(next_row)
    manifest_sha256 = write_jsonl(manifest_path, updated)

    positive = [row for row in updated if bool(row["has_positive_target"])]
    license_counts = Counter(str(row.get("image_license")) for row in updated)
    positive_license_counts = Counter(str(row.get("image_license")) for row in positive)
    report = {
        "target_images": len(updated),
        "target_images_with_metadata": sum(row.get("image_license") is not None for row in updated),
        "positive_images": len(positive),
        "positive_images_with_metadata": sum(
            row.get("image_license") is not None for row in positive
        ),
        "cc_by_2_target_images": sum(
            str(row.get("image_license")) in CC_BY_2_LICENSES for row in updated
        ),
        "cc_by_2_positive_images": sum(
            str(row.get("image_license")) in CC_BY_2_LICENSES for row in positive
        ),
        "license_counts": dict(sorted(license_counts.items())),
        "positive_license_counts": dict(sorted(positive_license_counts.items())),
        "missing_image_ids_by_split": missing_by_split,
        "input_files": input_files,
        "manifest_sha256": manifest_sha256,
        "license_boundary": IMAGE_LICENSE_BOUNDARY,
    }
    write_json(report_path, report)
    return report


def _licensed_for_backup_acquisition(row: dict[str, Any]) -> bool:
    return str(row.get("image_license")) in CC_BY_2_LICENSES


def _acquisition_candidates(row: dict[str, Any]) -> list[tuple[str, str]]:
    values = [
        ("official_s3", str(row.get("download_url") or "")),
        ("original_url", str(row.get("original_url") or "")),
        ("thumbnail_300k", str(row.get("thumbnail_300k_url") or "")),
    ]
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for source, url in values:
        if not url or url in seen:
            continue
        seen.add(url)
        result.append((source, url))
    return result


def _original_metadata_matches(path: Path, row: dict[str, Any]) -> bool:
    expected_size = row.get("original_size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        return False
    expected_md5 = str(row.get("original_md5") or "")
    if not expected_md5:
        return True
    expected = base64.b64decode(expected_md5, validate=True)
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest() == expected


def _download_one(
    client: httpx.Client,
    row: dict[str, Any],
    output_directory: Path,
    *,
    attempts: int = 6,
) -> tuple[str, dict[str, Any] | None, str | None]:
    image_id = str(row["image_id"])
    split = str(row["upstream_split"])
    destination = output_directory / split / f"{image_id}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jpg.part")
    error: Exception | None = None
    existing_source = str(row.get("acquisition_source") or "official_s3")
    candidates = (
        [(existing_source, "")] if destination.exists() else _acquisition_candidates(row)
    )
    if not destination.exists() and not _licensed_for_backup_acquisition(row):
        return image_id, None, "missing CC BY 2.0 image metadata; backup acquisition skipped"
    for acquisition_source, url in candidates:
        for attempt in range(attempts):
            try:
                if not destination.exists():
                    with client.stream("GET", url) as response:
                        if response.status_code == 429 or response.status_code >= 500:
                            error = RuntimeError(f"HTTP {response.status_code}")
                            if attempt + 1 < attempts:
                                time.sleep(min(float(2**attempt), 30.0))
                            continue
                        response.raise_for_status()
                        with temporary.open("wb") as handle:
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                handle.write(chunk)
                    temporary.replace(destination)
                if acquisition_source == "original_url" and not _original_metadata_matches(
                    destination, row
                ):
                    raise ValueError("original image size or MD5 does not match metadata")

                from PIL import Image

                with Image.open(destination) as image:
                    image.verify()
                with Image.open(destination) as image:
                    width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("non-positive image dimensions")
                return (
                    image_id,
                    {
                        "media_acquired": True,
                        "image_bytes": destination.stat().st_size,
                        "image_sha256": sha256_file(destination),
                        "width": width,
                        "height": height,
                        "acquisition_source": acquisition_source,
                    },
                    None,
                )
            except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
                error = exc
                temporary.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                retryable = isinstance(exc, RuntimeError) and str(exc).startswith("HTTP ")
                if retryable and attempt + 1 < attempts:
                    continue
                break
    return image_id, None, str(error)


def download_positive_images(
    manifest_path: Path,
    output_directory: Path,
    report_path: Path,
    *,
    user_agent: str,
    workers: int = 6,
) -> dict[str, Any]:
    if workers < 1 or workers > 12:
        raise ValueError("Open Images workers must be between 1 and 12")
    rows = read_jsonl(manifest_path)
    selected = [
        row
        for row in rows
        if bool(row["has_positive_target"]) and not bool(row.get("media_acquired"))
    ]
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    updates: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with httpx.Client(
        follow_redirects=True,
        trust_env=False,
        verify=ssl_context,
        timeout=httpx.Timeout(90, read=300),
        headers={"User-Agent": user_agent},
    ) as client:

        def acquire(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
            return _download_one(client, row, output_directory)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for image_id, update, error in executor.map(acquire, selected):
                if update is not None:
                    updates[image_id] = update
                if error is not None:
                    source_row = next(
                        row for row in selected if str(row["image_id"]) == image_id
                    )
                    failures.append(
                        {
                            "image_id": image_id,
                            "upstream_split": str(source_row["upstream_split"]),
                            "weapon_status": str(source_row["weapon_status"]),
                            "error": error,
                        }
                    )

    updated: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        next_row.update(updates.get(str(row["image_id"]), {}))
        updated.append(next_row)
    manifest_sha256 = write_jsonl(manifest_path, updated)
    downloaded = [row for row in updated if bool(row["media_acquired"])]
    acquisition_sources = Counter(str(row.get("acquisition_source")) for row in downloaded)
    summary = {
        "attempted_missing_positive_images": len(selected),
        "downloaded_positive_images": len(downloaded),
        "downloaded_bytes": sum(int(row["image_bytes"] or 0) for row in downloaded),
        "exact_unique_image_hashes": len({str(row["image_sha256"]) for row in downloaded}),
        "exact_duplicate_image_files": len(downloaded)
        - len({str(row["image_sha256"]) for row in downloaded}),
        "acquisition_source_counts": dict(sorted(acquisition_sources.items())),
        "failures": failures,
        "manifest_sha256": manifest_sha256,
        "license_boundary": IMAGE_LICENSE_BOUNDARY,
    }
    write_json(report_path, summary)
    return summary
