"""Metadata-only fencing subsets from official Kinetics 400, 600, and 700 releases."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import write_json, write_jsonl
from pisteflow.sources.net import download_resumable

DATASET_PAGE = "https://github.com/cvdfoundation/kinetics-dataset"
FENCING_LABELS = {"fencing", "fencing (sport)"}
RELEASES = {
    "400": {
        "name": "Kinetics-400",
        "train": "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
        "validation": "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
    },
    "600": {
        "name": "Kinetics-600",
        "train": "https://s3.amazonaws.com/kinetics/600/annotations/train.csv",
        "validation": "https://s3.amazonaws.com/kinetics/600/annotations/val.csv",
    },
    "700": {
        "name": "Kinetics-700-2020",
        "train": "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv",
        "validation": "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv",
    },
}


def fetch_fencing_manifest(
    cache_directory: Path,
    output_path: Path,
    *,
    user_agent: str,
    release: str = "700",
) -> list[dict[str, Any]]:
    """Collect YouTube IDs and 10-second boundaries; do not fetch media."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if release not in RELEASES:
        raise ValueError(f"Unsupported Kinetics release: {release}")
    release_config = RELEASES[release]
    for declared_split in ("train", "validation"):
        url = str(release_config[declared_split])
        path = cache_directory / f"{declared_split}.csv"
        download_resumable(url, path, user_agent=user_agent)
        frame = pd.read_csv(path)
        required = {"label", "youtube_id", "time_start", "time_end", "split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Kinetics annotations missing columns: {sorted(missing)}")
        fencing = frame[frame["label"].astype(str).str.casefold().isin(FENCING_LABELS)]
        for row in fencing.itertuples(index=False):
            identifier = str(row.youtube_id)
            start = int(row.time_start)
            end = int(row.time_end)
            records.append(
                {
                    "clip_id": f"kinetics{release}:{identifier}:{start:06d}:{end:06d}",
                    "clip_key": f"youtube:{identifier}:{start:06d}:{end:06d}",
                    "release": release,
                    "dataset": str(release_config["name"]),
                    "class_label": "fencing",
                    "weapon": "unknown",
                    "youtube_id": identifier,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "split": declared_split,
                    "source_url": f"https://www.youtube.com/watch?v={identifier}",
                    "dataset_page": DATASET_PAGE,
                    "license_status": "link_only_video_license_not_asserted",
                    "media_downloaded": False,
                }
            )
    records.sort(key=lambda row: str(row["clip_id"]))
    write_jsonl(output_path, records)
    return records


def fetch_all_releases(
    cache_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Index all official releases while explicitly counting cross-release overlap."""

    records: list[dict[str, Any]] = []
    for release in RELEASES:
        records.extend(
            fetch_fencing_manifest(
                cache_directory / f"kinetics{release}-annotations",
                cache_directory / f"kinetics{release}-fencing.jsonl",
                user_agent=user_agent,
                release=release,
            )
        )
    records.sort(key=lambda row: (str(row["clip_key"]), str(row["release"])))
    manifest_sha256 = write_jsonl(output_path, records)
    summary = summarize(records)
    unique_by_key = {str(row["clip_key"]): row for row in records}
    by_release = Counter(str(row["release"]) for row in records)
    summary["by_release"] = {release: int(by_release.get(release, 0)) for release in RELEASES}
    summary.update(
        {
            "claim_boundary": (
                "Official annotation metadata only. Each row identifies a YouTube interval "
                "labelled as fencing, not an acquired or reusable video and not a weapon label. "
                "The same clip may be a member of multiple Kinetics releases and is counted once "
                "in unique-clip totals."
            ),
            "dataset_page": DATASET_PAGE,
            "release_memberships": len(records),
            "unique_clip_intervals": len({str(row["clip_key"]) for row in records}),
            "unique_interval_hours": round(
                sum(int(row["duration_seconds"]) for row in unique_by_key.values()) / 3600,
                2,
            ),
            "cross_release_memberships": len(records)
            - len({str(row["clip_key"]) for row in records}),
            "manifest_sha256": manifest_sha256,
        }
    )
    write_json(report_path, summary)
    return records, summary


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "by_release": dict(
            sorted(Counter(str(row.get("release", "700")) for row in records).items())
        ),
        "by_split": dict(sorted(Counter(str(row["split"]) for row in records).items())),
        "unique_youtube_videos": len({str(row["youtube_id"]) for row in records}),
        "hours": round(sum(int(row["duration_seconds"]) for row in records) / 3600, 2),
        "weapon_label": "unknown; dataset labels fencing activity, not weapon",
    }
