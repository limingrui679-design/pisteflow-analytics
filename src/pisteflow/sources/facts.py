"""Audit the public FACTS fine-grained fencing-action annotation release."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

PAPER_URL = "https://arxiv.org/abs/2412.16454"
REPOSITORY_URL = "https://anonymous.4open.science/r/FACTS-B1C5"
ANNOTATION_URL = (
    "https://anonymous.4open.science/api/repo/FACTS-B1C5/file/"
    "filtered_data_800_fencing.csv"
)
MEDIA_FOLDER_URL = (
    "https://www.dropbox.com/scl/fo/l3kgvx2ujdq4gxi6vivl4/"
    "AO3-ttBw5aS-X4RhZM46F-c?rlkey=xds3b4s28tdo8cz7ycq9qqb6u&dl=0"
)
EXPECTED_BYTES = 394_363
EXPECTED_SHA256 = "5baeb5c417bdac5b487fdf89490d0753dbf3c1ce3fc2abf3f098036fe2886662"
OBSERVED_MEDIA_ZIP_BYTES = 32_611_305_921
LICENSE = "No dataset license found in the public repository on 2026-08-14"
_LABELS = {
    "Attack left": "attack_left",
    "Attack right": "attack_right",
    "Counter Attack left": "counter_attack_left",
    "Counter Attack right": "counter_attack_right",
    "Remise left": "remise_left",
    "Remise right": "remise_right",
    "Riposte left": "riposte_left",
    "Riposte right": "riposte_right",
}
_URL_PATTERN = (
    r"^storage/bout/(?P<source_video_id>[^/]+)/"
    r"(?P<clip_set>combined|clips)/(?P<clip_number>\d+)\.mp4$"
)


def _asset_id(value: str) -> str:
    return "facts-asset-" + sha256(f"facts-asset:{value}".encode()).hexdigest()[:20]


def normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate exact oversampling rows and flag contradictory asset labels."""

    if list(frame.columns) != ["id", "video_url", "label"]:
        raise ValueError("FACTS annotation columns changed")
    source = frame.copy()
    source["id"] = pd.to_numeric(source["id"], errors="raise").astype("int32")
    source["video_url"] = source["video_url"].astype("string").str.strip()
    source["label"] = source["label"].astype("string").str.strip()
    if source[["id", "video_url", "label"]].isna().any().any():
        raise ValueError("FACTS annotations contain missing values")
    if not set(source["label"].unique()).issubset(_LABELS):
        raise ValueError("FACTS action labels changed")

    repeated = source.groupby(["id", "video_url", "label"], sort=False).size().rename(
        "annotation_multiplicity"
    )
    result = source.drop_duplicates().merge(
        repeated.reset_index(), on=["id", "video_url", "label"], validate="one_to_one"
    )
    parsed = result["video_url"].str.extract(_URL_PATTERN)
    if parsed.isna().any(axis=None):
        raise ValueError("FACTS video-path format changed")
    result = pd.concat([result, parsed], axis=1)
    result["clip_number"] = pd.to_numeric(result["clip_number"], errors="raise").astype(
        "int32"
    )
    result["action_label"] = result["label"].map(_LABELS).astype("string")
    result["video_asset_id"] = result["video_url"].map(_asset_id)
    result["group_id"] = "facts-source-video:" + result["source_video_id"].astype("string")
    labels_per_asset = result.groupby("video_url")["action_label"].transform("nunique")
    result["conflicting_asset_label"] = labels_per_asset.gt(1)
    result["training_eligible"] = ~result["conflicting_asset_label"]
    result["weapon"] = "unknown"
    result["media_distribution_url"] = MEDIA_FOLDER_URL
    result["source_license"] = LICENSE
    result["annotation_id"] = "facts-annotation:" + result["id"].astype("string")
    result = result.drop(columns=["id", "label"]).sort_values(
        ["group_id", "clip_number", "action_label", "annotation_id"], kind="stable"
    )
    result = result.reset_index(drop=True)

    conflicts = result.loc[result["conflicting_asset_label"], "video_asset_id"]
    stats = {
        "source_rows": len(source),
        "exact_repeated_rows_after_first": int(source.duplicated().sum()),
        "unique_annotation_rows": len(result),
        "unique_video_assets": int(result["video_asset_id"].nunique()),
        "contradictory_video_assets": int(conflicts.nunique()),
        "contradictory_annotation_rows": int(result["conflicting_asset_label"].sum()),
    }
    return result, stats


def prepare(annotation_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if sha256_file(annotation_path) != EXPECTED_SHA256:
        raise ValueError("FACTS annotation checksum changed")
    source = pd.read_csv(annotation_path)
    if set(source["label"].unique()) != set(_LABELS):
        raise ValueError("FACTS release no longer contains all eight action labels")
    result, stats = normalize(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    source_distribution = source["label"].value_counts().sort_index().astype(int).to_dict()
    unique_distribution = (
        result["action_label"].value_counts().sort_index().astype(int).to_dict()
    )
    summary: dict[str, Any] = {
        "claim_boundary": (
            "The public CSV contains 6,400 rows balanced at 800 per label, but 3,971 rows "
            "are exact repeated annotations. Only 2,429 unique annotation rows and 2,419 "
            "unique clip paths are present; one clip path has contradictory labels and is "
            "excluded from training. Repeated rows are not independent clips or verified "
            "horizontal flips. Weapon labels are absent. The media folder is public but has "
            "no verified redistribution license, so clips remain local/link-only."
        ),
        "paper_url": PAPER_URL,
        "repository_url": REPOSITORY_URL,
        "annotation_url": ANNOTATION_URL,
        "media_folder_url": MEDIA_FOLDER_URL,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "annotation_sha256": sha256_file(annotation_path),
        "output_sha256": sha256_file(output_path),
        "source_label_distribution": source_distribution,
        "unique_annotation_distribution": unique_distribution,
        "source_video_groups": int(result["group_id"].nunique()),
        "clip_sets": result["clip_set"].value_counts().sort_index().astype(int).to_dict(),
        "training_eligible_unique_annotations": int(result["training_eligible"].sum()),
        "media_zip_bytes_observed_2026_08_14": OBSERVED_MEDIA_ZIP_BYTES,
        "output": str(output_path),
        **stats,
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    annotation_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        annotation_path.unlink(missing_ok=True)
    download_resumable(
        ANNOTATION_URL,
        annotation_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    return prepare(annotation_path, output_path, report_path)
