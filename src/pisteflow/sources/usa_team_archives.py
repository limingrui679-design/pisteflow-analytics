"""Privacy-minimized USA team ranking histories from a third-party snapshot."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = "https://www.kaggle.com/datasets/ajaxofgreee/usa-fencing-team-archives"
ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "ajaxofgreee/usa-fencing-team-archives?datasetVersionNumber=1"
)
LICENSE = "Unknown (Kaggle dataset metadata)"
EXPECTED_BYTES = 906_089
EXPECTED_SHA256 = "a13d04ac1e7123de91705a25f5a5f71ffa214216fa60070316e24a62ad159872"
_MEMBER_PATTERN = re.compile(
    r"^(?P<sex>[MW])(?P<weapon>[EFS])_(?P<age>Cdt|Jr|Sr)_usfa_results\.json$"
)
_SEX = {"M": "men", "W": "women"}
_WEAPON = {"E": "epee", "F": "foil", "S": "sabre"}
_AGE = {"Cdt": "cadet", "Jr": "junior", "Sr": "senior"}
_EXPECTED_MEMBERS = {
    f"{sex}{weapon}_{age}_usfa_results.json"
    for sex in _SEX
    for weapon in _WEAPON
    for age in _AGE
}


def _series_id(member: str, source_name: str) -> str:
    digest = sha256(f"usa-team-archive:{member}:{source_name}".encode()).hexdigest()[:20]
    return f"uta-series:{digest}"


def normalize_member(member: str, payload: dict[str, Any]) -> pd.DataFrame:
    """Turn one sex/weapon/age JSON member into a name-free ranking panel."""

    match = _MEMBER_PATTERN.fullmatch(member)
    if match is None:
        raise ValueError(f"Unexpected USA team archive member: {member}")
    if not payload:
        raise ValueError(f"USA team archive member is empty: {member}")

    records: list[dict[str, Any]] = []
    source_name_by_id: dict[str, str] = {}
    for source_name, athlete in payload.items():
        if (
            not isinstance(source_name, str)
            or not source_name.strip()
            or not isinstance(athlete, dict)
        ):
            raise ValueError(f"Malformed athlete entry in {member}")
        results = athlete.get("results")
        if not isinstance(results, dict) or not results:
            raise ValueError(f"Missing ranking history in {member}")
        series_id = _series_id(member, source_name)
        previous = source_name_by_id.setdefault(series_id, source_name)
        if previous != source_name:
            raise ValueError("USA team archive pseudonym collision detected")
        region_reported = bool(str(athlete.get("region") or "").strip())
        for date_text, values in results.items():
            if not isinstance(values, dict):
                raise ValueError(f"Malformed ranking snapshot in {member}")
            required = {"rank", "rolling", "domestic", "international"}
            if not required.issubset(values):
                raise ValueError(f"Ranking fields changed in {member}")
            records.append(
                {
                    "athlete_series_id": series_id,
                    "snapshot_date": date_text,
                    "rank": values["rank"],
                    "rolling_points": values["rolling"],
                    "domestic_points": values["domestic"],
                    "international_points": values["international"],
                    "region_reported": region_reported,
                }
            )

    result = pd.DataFrame(records)
    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"], format="%m/%d/%Y", errors="raise"
    )
    result["rank"] = pd.to_numeric(result["rank"], errors="raise").astype("int32")
    if result["rank"].lt(1).any():
        raise ValueError(f"Non-positive ranking found in {member}")
    for column in ["rolling_points", "domestic_points", "international_points"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    result["component_balance_error"] = (
        result["rolling_points"]
        .sub(result["domestic_points"])
        .sub(result["international_points"])
        .abs()
        .gt(1e-6)
    )
    result["negative_points_upstream"] = result[
        ["rolling_points", "domestic_points", "international_points"]
    ].lt(0).any(axis=1)
    result["gender"] = _SEX[match.group("sex")]
    result["weapon"] = _WEAPON[match.group("weapon")]
    result["age_category"] = _AGE[match.group("age")]
    result["source_member"] = member
    return result.sort_values(
        ["snapshot_date", "athlete_series_id"], kind="stable"
    ).reset_index(drop=True)


def prepare(archive_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if archive_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("USA team archive compressed byte size changed")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != EXPECTED_SHA256:
        raise ValueError("USA team archive checksum changed")

    frames: list[pd.DataFrame] = []
    uncompressed_bytes = 0
    with ZipFile(archive_path) as archive:
        members = {item.filename for item in archive.infolist() if not item.is_dir()}
        if members != _EXPECTED_MEMBERS:
            raise ValueError("USA team archive member inventory changed")
        for member in sorted(members):
            payload = json.loads(archive.read(member))
            if not isinstance(payload, dict):
                raise ValueError(f"USA team archive payload changed in {member}")
            frames.append(normalize_member(member, payload))
        uncompressed_bytes = sum(item.file_size for item in archive.infolist())

    result = pd.concat(frames, ignore_index=True)
    duplicate_key = ["source_member", "athlete_series_id", "snapshot_date"]
    result["duplicate_snapshot_key"] = result.duplicated(duplicate_key, keep=False)
    result["training_eligible"] = ~result.duplicated(duplicate_key, keep="first")
    result = result.sort_values(
        ["snapshot_date", "weapon", "gender", "age_category", "athlete_series_id"],
        kind="stable",
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    by_weapon = result.groupby("weapon").size().astype(int).sort_index().to_dict()
    by_gender = result.groupby("gender").size().astype(int).sort_index().to_dict()
    by_age = result.groupby("age_category").size().astype(int).sort_index().to_dict()
    series_by_weapon = (
        result.groupby("weapon")["athlete_series_id"].nunique().astype(int).sort_index().to_dict()
    )
    negative_by_field = {
        column: int(result[column].lt(0).sum())
        for column in ["rolling_points", "domestic_points", "international_points"]
    }
    report: dict[str, Any] = {
        "claim_boundary": (
            "A third-party snapshot of USA team-ranking histories, not a current official feed "
            "or verified athlete-development record. The 7,504 IDs are category-specific "
            "published-name spellings, so the same person may appear in multiple categories and "
            "dirty spellings are not silently merged. Names and region values are removed, but "
            "stable hashes are pseudonymous rather than anonymous. The unknown-license archive, "
            "row table, and any fitted weights remain local-only. Negative source points are "
            "preserved and flagged rather than repaired."
        ),
        "source_url": SOURCE_PAGE,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_uncompressed_bytes": uncompressed_bytes,
        "archive_sha256": archive_sha256,
        "json_members": len(_EXPECTED_MEMBERS),
        "ranking_snapshot_rows": len(result),
        "category_specific_athlete_series": int(result["athlete_series_id"].nunique()),
        "date_min": result["snapshot_date"].min().date().isoformat(),
        "date_max": result["snapshot_date"].max().date().isoformat(),
        "by_weapon": by_weapon,
        "athlete_series_by_weapon": series_by_weapon,
        "by_gender": by_gender,
        "by_age_category": by_age,
        "negative_point_rows": int(result["negative_points_upstream"].sum()),
        "negative_values_by_field": negative_by_field,
        "component_balance_error_rows": int(result["component_balance_error"].sum()),
        "duplicate_snapshot_key_rows": int(result["duplicate_snapshot_key"].sum()),
        "duplicate_snapshot_key_rows_after_first": int((~result["training_eligible"]).sum()),
        "rows_by_member": dict(sorted(Counter(result["source_member"]).items())),
        "direct_names_removed_from_output": True,
        "region_values_removed_from_output": True,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(report_path, report)
    return report


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
