"""Prepare the MIT-labelled men's-foil points trajectory snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = "https://www.kaggle.com/datasets/foillee/fencing-points"
ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/foillee/fencing-points"
LICENSE = "MIT (Kaggle dataset metadata)"
EXPECTED_BYTES = 339_824


def _athlete_id(value: object) -> str:
    return "mf-" + sha256(f"fencing-points-mf:{value}".encode()).hexdigest()[:16]


def normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop direct identity fields and exact duplicate athlete-event rows."""

    required = {
        "fie_ID",
        "fencer_name",
        "age",
        "dob",
        "nationality",
        "rank",
        "host_city",
        "event_type",
        "points_earned",
        "date",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fencing-points table missing columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "athlete_id": frame["fie_ID"].map(_athlete_id),
            "age_upstream": pd.to_numeric(frame["age"], errors="coerce"),
            "rank": pd.to_numeric(frame["rank"], errors="coerce"),
            "host_city": frame["host_city"].astype("string").str.strip(),
            "event_type": frame["event_type"].astype("string").str.strip(),
            "points_earned": pd.to_numeric(frame["points_earned"], errors="coerce"),
            "event_date": pd.to_datetime(frame["date"], errors="coerce"),
        }
    )
    if result[["rank", "points_earned", "event_date"]].isna().any().any():
        raise ValueError("Fencing-points table has invalid rank, points, or date values")
    result["gender"] = "men"
    result["weapon"] = "foil"
    result["age_plausible"] = result["age_upstream"].between(13, 70)
    result["year"] = result["event_date"].dt.year.astype("Int16")
    result["source_url"] = SOURCE_PAGE
    result["source_license"] = LICENSE
    duplicate_mask = result.duplicated(
        ["athlete_id", "event_date", "host_city", "event_type", "rank", "points_earned"],
        keep="first",
    )
    duplicates = int(duplicate_mask.sum())
    result = result.loc[~duplicate_mask].sort_values(
        ["event_date", "event_type", "host_city", "rank", "athlete_id"], kind="stable"
    )
    return result.reset_index(drop=True), duplicates


def prepare(archive_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    with ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Fencing-points archive has a corrupt member: {bad}")
        if set(name for name in archive.namelist() if not name.endswith("/")) != {"data.csv"}:
            raise ValueError("Fencing-points archive members changed")
        with archive.open("data.csv") as handle:
            source = pd.read_csv(handle)
    result, duplicate_rows = normalize(source)
    result["athlete_event_id"] = [f"mf-points-{index + 1:05d}" for index in range(len(result))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Historical senior men's foil points-event rows from an MIT-labelled Kaggle "
            "snapshot, not an official current ranking feed. Names, dates of birth, nationality, "
            "and direct FIE IDs are removed. Exact duplicates are counted and excluded; "
            "implausible ages are retained and flagged."
        ),
        "source_url": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "source_rows": len(source),
        "analytic_rows": len(result),
        "exact_duplicate_rows_excluded": duplicate_rows,
        "unique_pseudonymous_athletes": int(result["athlete_id"].nunique()),
        "events": int(result[["event_date", "host_city", "event_type"]].drop_duplicates().shape[0]),
        "year_min": int(result["year"].min()),
        "year_max": int(result["year"].max()),
        "by_event_type": result.groupby("event_type").size().astype(int).to_dict(),
        "rows_with_implausible_upstream_age": int((~result["age_plausible"]).sum()),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    return prepare(archive_path, output_path, report_path)
