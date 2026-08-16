"""Acquire and audit three CMU fencing scoreboard/touch detection releases."""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePath
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

TOUCH_REPOSITORY = "mastefan/2025-24679-Project-One-Fencing-Score-Detection"
TOUCH_REVISION = "3922c6e6f41504ddb626151af1dbc7c0893ac6c0"
SIGNAL_REPOSITORY = "emkessle/2025-24679-Project-One-Fencing-Score-Detection-Timeseries-labeled"
SIGNAL_REVISION = "9085266fc133e0fba6b9ab6aa8ea7724481e9bbb"
YOLO_REPOSITORY = "mastefan/fencing-scoreboard-yolov8"
YOLO_REVISION = "9e026c7b7da57f73239e7c75294bc737faba9533"
TOUCH_URL = (
    f"https://huggingface.co/datasets/{TOUCH_REPOSITORY}/resolve/{TOUCH_REVISION}/"
    "data/original-00000-of-00001.parquet?download=true"
)
SIGNAL_URL = (
    f"https://huggingface.co/datasets/{SIGNAL_REPOSITORY}/resolve/{SIGNAL_REVISION}/"
    "data/original-00000-of-00001.parquet?download=true"
)
YOLO_TRAIN_URL = (
    f"https://huggingface.co/datasets/{YOLO_REPOSITORY}/resolve/{YOLO_REVISION}/"
    "data/train-00000-of-00001.parquet?download=true"
)
YOLO_VALID_URL = (
    f"https://huggingface.co/datasets/{YOLO_REPOSITORY}/resolve/{YOLO_REVISION}/"
    "data/valid-00000-of-00001.parquet?download=true"
)
_FILES = {
    "touch": (
        TOUCH_URL,
        28_641_568,
        "695160497d74ed882519a9f79a49a9177e2f90ea960c0f072ddb93aa2a875695",
    ),
    "signal": (
        SIGNAL_URL,
        50_535,
        "86799f88be5558a35f51c311e90dc0bc3d857f54e7283f95c58868c3efb0c3f0",
    ),
    "yolo_train": (
        YOLO_TRAIN_URL,
        46_187,
        "8b418398c7da5bb5e584223f0e7a6647046eef3df57471dc1d8a423bd61dc75f",
    ),
    "yolo_valid": (
        YOLO_VALID_URL,
        5_386,
        "c556357fa595cb5b2211967f62589c2068100c1008ecf50093b6ab7bee3bfc06",
    ),
}


def _canonical_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _group_id(value: str) -> str:
    digest = sha256(f"cmu-score-group:{_canonical_title(value)}".encode()).hexdigest()[:20]
    return f"cmu-score-bout:{digest}"


def _title_and_frame(value: str) -> tuple[str, int | None, str]:
    filename = PurePath(value).name
    base = re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", filename, flags=re.IGNORECASE)
    base = re.sub(r"\.jpg$", "", base, flags=re.IGNORECASE)
    base = base.removesuffix("_jpg").removeprefix("Copy-of-")
    match = re.fullmatch(r"(?P<title>.+)_frame_(?P<frame>\d+)", base)
    if match is None:
        return base, None, base
    return match.group("title"), int(match.group("frame")), base


def normalize_touch_images(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if list(frame.columns) != ["image", "label"]:
        raise ValueError("CMU touch-image schema changed")
    records: list[dict[str, Any]] = []
    for source_row, row in enumerate(frame.itertuples(index=False), start=1):
        image = row.image
        if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
            raise ValueError("CMU touch-image payload changed")
        if not isinstance(image.get("path"), str):
            raise ValueError("CMU touch-image path changed")
        label = int(row.label)
        if label not in {0, 1}:
            raise ValueError("CMU touch-image label changed")
        title, frame_number, _base = _title_and_frame(image["path"])
        if frame_number is None:
            raise ValueError("CMU touch-image frame path changed")
        payload = image["bytes"]
        image_sha256 = sha256(payload).hexdigest()
        records.append(
            {
                "sample_id": f"cmu-touch:{image_sha256[:20]}",
                "source_row": source_row,
                "group_id": _group_id(title),
                "source_frame_number": frame_number,
                "touch_label": label,
                "image_bytes": len(payload),
                "image_sha256": image_sha256,
                "weapon": "foil",
                "media_in_local_parquet": True,
                "source_repository": f"https://huggingface.co/datasets/{TOUCH_REPOSITORY}",
                "license": "No explicit dataset license in structured card metadata",
            }
        )
    if len({str(record["image_sha256"]) for record in records}) != len(records):
        raise ValueError("CMU original touch-image split contains exact pixel duplicates")
    return records


def normalize_signals(frame: pd.DataFrame) -> pd.DataFrame:
    expected = ["video_id", "frame_id", "timestamp", "red_ratio", "green_ratio", "Label"]
    if list(frame.columns) != expected:
        raise ValueError("CMU score-signal schema changed")
    result = pd.DataFrame(
        {
            "group_id": frame["video_id"].astype("string").map(_group_id),
            "frame_id": pd.to_numeric(frame["frame_id"], errors="raise").astype("int32"),
            "timestamp_seconds": pd.to_numeric(frame["timestamp"], errors="raise"),
            "red_ratio": pd.to_numeric(frame["red_ratio"], errors="raise"),
            "green_ratio": pd.to_numeric(frame["green_ratio"], errors="raise"),
            "label_strength": pd.to_numeric(frame["Label"], errors="raise"),
        }
    )
    if result.isna().any().any():
        raise ValueError("CMU score signals contain missing values")
    if not result["red_ratio"].between(0, 1).all() or not result["green_ratio"].between(0, 1).all():
        raise ValueError("CMU score-signal color ratios are outside [0, 1]")
    if not result["label_strength"].between(0, 1).all():
        raise ValueError("CMU score-signal labels are outside [0, 1]")
    if result.duplicated(["group_id", "frame_id"]).any():
        raise ValueError("CMU score signals repeat a group/frame key")
    result["touch_decay_window"] = result["label_strength"].gt(0)
    result["touch_peak"] = result["label_strength"].eq(1)
    result["weapon"] = "foil"
    result["source_license"] = "MIT stated in the dataset README"
    return result.sort_values(["group_id", "frame_id"], kind="stable").reset_index(drop=True)


def _box_count(value: object) -> int:
    text = str(value).strip()
    if not text:
        return 0
    count = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError("CMU YOLO label format changed")
        coordinates = [float(part) for part in parts[1:]]
        if not all(0 <= coordinate <= 1 for coordinate in coordinates):
            raise ValueError("CMU YOLO coordinate is outside [0, 1]")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError("CMU YOLO box has non-positive extent")
        count += 1
    return count


def normalize_yolo(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    if split not in {"train", "valid"}:
        raise ValueError("CMU YOLO split must be train or valid")
    if list(frame.columns) != ["image", "label"]:
        raise ValueError("CMU YOLO metadata schema changed")
    rows: list[dict[str, Any]] = []
    for source_row, row in enumerate(frame.itertuples(index=False), start=1):
        if not isinstance(row.image, str):
            raise ValueError("CMU YOLO image reference changed")
        title, frame_number, base = _title_and_frame(row.image)
        group_id = _group_id(title)
        base_frame_token = f"{group_id}:{frame_number if frame_number is not None else base}"
        rows.append(
            {
                "sample_id": "cmu-yolo-" + sha256(f"{split}:{row.image}".encode()).hexdigest()[:20],
                "source_row": source_row,
                "upstream_split": split,
                "group_id": group_id,
                "source_frame_number": frame_number,
                "base_frame_id": "cmu-yolo-base-"
                + sha256(base_frame_token.encode()).hexdigest()[:20],
                "box_count": _box_count(row.label),
                "media_available_in_release": False,
                "weapon": "foil_or_generic_scoreboard",
                "source_license": (
                    "MIT stated in README; underlying public/personal footage described as fair use"
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare(
    cache_directory: Path,
    touch_manifest_path: Path,
    signal_output_path: Path,
    yolo_output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    paths = {key: cache_directory / f"{key}.parquet" for key in _FILES}
    for key, path in paths.items():
        expected_sha256 = _FILES[key][2]
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"CMU {key} snapshot checksum changed")

    touch_records = normalize_touch_images(pd.read_parquet(paths["touch"]))
    touch_manifest_sha256 = write_jsonl(touch_manifest_path, touch_records)
    signals = normalize_signals(pd.read_parquet(paths["signal"]))
    signal_output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(signal_output_path, index=False)
    yolo = pd.concat(
        [
            normalize_yolo(pd.read_parquet(paths["yolo_train"]), split="train"),
            normalize_yolo(pd.read_parquet(paths["yolo_valid"]), split="valid"),
        ],
        ignore_index=True,
    )
    yolo_output_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.to_parquet(yolo_output_path, index=False)

    train = yolo[yolo["upstream_split"] == "train"]
    valid = yolo[yolo["upstream_split"] == "valid"]
    train_groups = set(train["group_id"])
    valid_groups = set(valid["group_id"])
    train_frames = set(train["base_frame_id"])
    valid_frames = set(valid["base_frame_id"])
    touch_groups = {str(record["group_id"]) for record in touch_records}
    signal_groups = set(signals["group_id"].astype(str))
    summary: dict[str, Any] = {
        "claim_boundary": (
            "The original image split contains 1,116 embedded foil frames from ten source "
            "bouts, but only 59 are labelled touch. The signal table has 2,226 frame rows and "
            "only ten peak events, one per source bout. The YOLO release contains 825 path-and-"
            "box rows but no image pixels; its upstream train/validation split is audited for "
            "source overlap and is not accepted as an independent evaluation split. Direct "
            "bout titles and athlete names are replaced by stable group hashes."
        ),
        "prepared_at": datetime.now(UTC).isoformat(),
        "touch_repository": f"https://huggingface.co/datasets/{TOUCH_REPOSITORY}",
        "signal_repository": f"https://huggingface.co/datasets/{SIGNAL_REPOSITORY}",
        "yolo_repository": f"https://huggingface.co/datasets/{YOLO_REPOSITORY}",
        "snapshot_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "touch_images": len(touch_records),
        "touch_image_groups": len(touch_groups),
        "touch_label_distribution": dict(
            sorted(Counter(int(record["touch_label"]) for record in touch_records).items())
        ),
        "exact_touch_image_hash_duplicates": 0,
        "signal_rows": len(signals),
        "signal_groups": len(signal_groups),
        "signal_decay_window_rows": int(signals["touch_decay_window"].sum()),
        "signal_peak_rows": int(signals["touch_peak"].sum()),
        "touch_signal_group_overlap": len(touch_groups & signal_groups),
        "yolo_metadata_rows": len(yolo),
        "yolo_train_rows": len(train),
        "yolo_valid_rows": len(valid),
        "yolo_rows_without_boxes": int(yolo["box_count"].eq(0).sum()),
        "yolo_total_boxes": int(yolo["box_count"].sum()),
        "yolo_group_overlap_between_upstream_splits": len(train_groups & valid_groups),
        "yolo_base_frame_overlap_between_upstream_splits": len(train_frames & valid_frames),
        "touch_manifest_sha256": touch_manifest_sha256,
        "signal_output_sha256": sha256_file(signal_output_path),
        "yolo_output_sha256": sha256_file(yolo_output_path),
        "touch_manifest": str(touch_manifest_path),
        "signal_output": str(signal_output_path),
        "yolo_output": str(yolo_output_path),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    cache_directory: Path,
    touch_manifest_path: Path,
    signal_output_path: Path,
    yolo_output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    for key, (url, expected_bytes, _expected_sha256) in _FILES.items():
        path = cache_directory / f"{key}.parquet"
        if force:
            path.unlink(missing_ok=True)
        download_resumable(
            url,
            path,
            user_agent=user_agent,
            expected_bytes=expected_bytes,
        )
    return prepare(
        cache_directory,
        touch_manifest_path,
        signal_output_path,
        yolo_output_path,
        report_path,
    )
