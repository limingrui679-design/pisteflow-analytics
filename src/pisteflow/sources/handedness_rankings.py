"""Three-weapon year-end ranking and handedness panel from a CC BY supplement."""

from __future__ import annotations

import hashlib
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DOI = "https://doi.org/10.6084/m9.figshare.30163016.v1"
ARTICLE_URL = "https://figshare.com/articles/dataset/30163016"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/58088888"
LICENSE = "CC BY 4.0"
EXPECTED_BYTES = 2_587_010
EXPECTED_MD5 = "86647cf5f20fe1a5d2790179611edb5e"
_COLUMNS = ["sport", "gender", "weapon", "year", "rank", "name", "hand"]


def _canonical_published_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _published_name_key(canonical: str) -> str:
    return "FHR-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter fencing rows, normalize labels, and replace names with stable keys."""

    if list(frame.columns) != _COLUMNS:
        raise ValueError("Handedness-ranking workbook columns changed")
    fencing = frame.loc[frame["sport"].eq("fencing")].copy()
    if fencing.empty:
        raise ValueError("Handedness-ranking workbook has no fencing rows")
    if fencing["name"].isna().any():
        raise ValueError("Handedness-ranking fencing rows contain a missing name")
    if not set(fencing["weapon"]).issubset({"épée", "foil", "sabre"}):
        raise ValueError("Handedness-ranking weapon labels changed")
    if not set(fencing["gender"]).issubset({"female", "male"}):
        raise ValueError("Handedness-ranking gender labels changed")
    if not set(fencing["hand"].dropna()).issubset({"L", "R"}):
        raise ValueError("Handedness-ranking hand labels changed")

    canonical_names = fencing["name"].astype(str).map(_canonical_published_name)
    name_keys = canonical_names.map(_published_name_key)
    if canonical_names.groupby(name_keys).nunique().max() != 1:
        raise ValueError("Published-name pseudonym collision detected")
    result = fencing.drop(columns=["name"]).copy()
    result.insert(0, "source_row", fencing.index.to_numpy() + 1)
    result.insert(1, "athlete_name_key", name_keys.to_numpy())
    result["weapon"] = result["weapon"].replace({"épée": "epee"})
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype("int16")
    result["rank"] = pd.to_numeric(result["rank"], errors="raise").astype("int16")
    result["handedness"] = result.pop("hand").map({"L": "left", "R": "right"})
    result["handedness"] = result["handedness"].fillna("unknown").astype("string")
    result["rank_band_100"] = pd.cut(
        result["rank"],
        bins=[0, 100, 200, 300],
        labels=["001-100", "101-200", "201-300"],
    ).astype("string")
    tie_key = ["gender", "weapon", "year", "rank"]
    result["shared_rank_key"] = result.duplicated(tie_key, keep=False)
    result["source_doi"] = DOI
    return result


def normalize_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Tabelle1", header=5)
    if frame.shape != (68_277, 7):
        raise ValueError(f"Handedness-ranking workbook size changed: {frame.shape}")
    raw_fencing = frame.loc[frame["sport"].eq("fencing")]
    if raw_fencing["name"].nunique() != 9_668:
        raise ValueError("Handedness-ranking published-name spelling count changed")
    result = normalize_frame(frame)
    if len(result) != 35_270 or result["athlete_name_key"].nunique() != 9_600:
        raise ValueError("Handedness-ranking fencing snapshot counts changed")
    return result


def fetch_and_prepare(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        raw_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        raw_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    frame = normalize_snapshot(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    observed = frame.loc[frame["handedness"].ne("unknown")]
    counts = (
        observed.groupby(["gender", "weapon", "handedness"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    report: dict[str, Any] = {
        "claim_boundary": (
            "Year-end top-300 ranking memberships, not independent athletes or bout "
            "outcomes. A person can appear in multiple years and possibly multiple weapon "
            "tables. Stable keys are derived from published name spellings and are not proof "
            "of real-world identity uniqueness. Missing handedness remains unknown."
        ),
        "source_url": ARTICLE_URL,
        "dataset_doi": DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "figshare_file_id": 58_088_888,
        "figshare_supplied_md5": EXPECTED_MD5,
        "raw_sha256": sha256_file(raw_path),
        "ranking_rows": len(frame),
        "published_name_spellings": 9_668,
        "normalized_published_name_keys": int(frame["athlete_name_key"].nunique()),
        "year_min": int(frame["year"].min()),
        "year_max": int(frame["year"].max()),
        "weapon_rows": frame["weapon"].value_counts().sort_index().to_dict(),
        "gender_rows": frame["gender"].value_counts().sort_index().to_dict(),
        "handedness_rows": frame["handedness"].value_counts().sort_index().to_dict(),
        "shared_rank_rows": int(frame["shared_rank_key"].sum()),
        "shared_rank_rows_after_first": int(
            frame.duplicated(["gender", "weapon", "year", "rank"]).sum()
        ),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "handedness_by_gender_weapon": counts.to_dict(orient="records"),
        "direct_names_removed_from_output": True,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(report_path, report)
    return report
