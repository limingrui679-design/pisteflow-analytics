"""Privacy-minimized fencing panel from a CC BY Frontiers supplement."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DOI = "https://doi.org/10.3389/fspor.2021.662203"
ARTICLE_URL = (
    "https://www.frontiersin.org/journals/sports-and-active-living/articles/"
    "10.3389/fspor.2021.662203/full"
)
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/27374984"
LICENSE = "CC BY 4.0"
FIGSHARE_FILE_ID = 27_374_984
EXPECTED_BYTES = 4_807_750
EXPECTED_SHA256 = "28c13c019fd92c8f4b572c05bd3c29656437316b35e42c77917485a9d25f43ec"
RANKING_SHEET = "Dataset-1_YearEndWorldRankings"
ATHLETE_SHEET = "Dataset-2_UniqueAthletes"
YEARS = list(range(2007, 2017))
_RANKING_COLUMNS = [
    "sport",
    "age",
    "sex",
    "weapon",
    "season",
    "year",
    "rank",
    "points",
    "athlete_ID",
    "nation_code",
    "nation_full",
    "UK",
    "DOB",
    "hand",
    "CutOff_Jan1st_Month",
    "CutOff_Jan1st_Quartile",
    "CutOff_Mixed_Month",
    "CutOff_Mixed_Quartile",
]
_ATHLETE_COLUMNS = [
    "sport",
    "age",
    "sex",
    "weapon",
    "athlete_ID",
    "nation_code",
    "nation_full",
    "UK",
    "DOB",
    "hand",
    "CutOff_Jan1st_Month",
    "CutOff_Jan1st_Quartile",
    "CutOff_Mixed_Month",
    "CutOff_Mixed_Quartile",
    *YEARS,
    "all_years",
]


def _athlete_key(raw_id: object) -> str:
    numeric = int(float(str(raw_id)))
    digest = hashlib.sha256(f"frontiers-662203:{numeric}".encode()).hexdigest()[:20]
    return f"FRA-{digest}"


def _normalize_common(frame: pd.DataFrame, expected_columns: Sequence[object]) -> pd.DataFrame:
    if list(frame.columns) != list(expected_columns):
        raise ValueError("Frontiers handedness workbook columns changed")
    fencing = frame.loc[frame["sport"].eq("fencing")].copy()
    if fencing.empty:
        raise ValueError("Frontiers handedness workbook has no fencing rows")
    if not set(fencing["age"]).issubset({"junior", "senior"}):
        raise ValueError("Frontiers handedness age labels changed")
    if not set(fencing["sex"]).issubset({"female", "male"}):
        raise ValueError("Frontiers handedness sex labels changed")
    if not set(fencing["weapon"]).issubset({"épée", "foil", "sabre"}):
        raise ValueError("Frontiers handedness weapon labels changed")
    if not set(fencing["hand"].dropna()).issubset({"L", "R"}):
        raise ValueError("Frontiers handedness hand labels changed")
    if fencing["athlete_ID"].isna().any():
        raise ValueError("Frontiers handedness athlete ID is missing")
    result = fencing.copy()
    result.insert(0, "source_row", fencing.index.to_numpy() + 9)
    result.insert(1, "athlete_key", fencing["athlete_ID"].map(_athlete_key))
    result["weapon"] = result["weapon"].replace({"épée": "epee"})
    result["age_level"] = result.pop("age")
    result["country_code"] = result.pop("nation_code")
    result["uk_member_nation"] = result.pop("UK").astype(bool)
    result["handedness"] = result.pop("hand").map({"L": "left", "R": "right"})
    result["handedness"] = result["handedness"].fillna("unknown").astype("string")
    result["birth_month_january_cutoff"] = result.pop("CutOff_Jan1st_Month").astype(
        "int8"
    )
    result["birth_quarter_january_cutoff"] = result.pop(
        "CutOff_Jan1st_Quartile"
    ).astype("int8")
    result["birth_month_mixed_cutoff"] = result.pop("CutOff_Mixed_Month").astype(
        "int8"
    )
    result["birth_quarter_mixed_cutoff"] = result.pop(
        "CutOff_Mixed_Quartile"
    ).astype("int8")
    # Exact dates, source IDs, and full country names are unnecessary for the analyses here.
    result = result.drop(columns=["sport", "athlete_ID", "nation_full", "DOB"])
    result["source_doi"] = DOI
    return result


def normalize_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the direct-ID-free fencing year-end panel."""

    result = _normalize_common(frame, _RANKING_COLUMNS)
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype("int16")
    result["rank"] = pd.to_numeric(result["rank"], errors="raise").astype("int16")
    result["points"] = pd.to_numeric(result["points"], errors="raise").astype(float)
    if not result["year"].between(2007, 2016).all():
        raise ValueError("Frontiers handedness ranking years changed")
    if not result["rank"].between(1, 200).all():
        raise ValueError("Frontiers handedness ranking scope changed")
    result["shared_rank_key"] = result.duplicated(
        ["age_level", "sex", "weapon", "year", "rank"], keep=False
    )
    return result


def normalize_athlete_memberships(frame: pd.DataFrame) -> pd.DataFrame:
    """Return category memberships and annual listing flags without exact birth dates."""

    result = _normalize_common(frame, _ATHLETE_COLUMNS)
    for year in YEARS:
        values = pd.to_numeric(result.pop(year), errors="raise").astype("int8")
        if not values.isin([0, 1]).all():
            raise ValueError(f"Frontiers handedness {year} membership flag changed")
        result[f"listed_{year}"] = values
    result["all_years"] = pd.to_numeric(result["all_years"], errors="raise").astype(
        "int8"
    )
    if not result[[f"listed_{year}" for year in YEARS]].sum(axis=1).eq(
        result["all_years"]
    ).all():
        raise ValueError("Frontiers handedness annual flags disagree with all_years")
    return result


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rankings_raw = pd.read_excel(path, sheet_name=RANKING_SHEET, header=7)
    athletes_raw = pd.read_excel(path, sheet_name=ATHLETE_SHEET, header=7)
    if rankings_raw.shape != (37_754, 18):
        raise ValueError(f"Frontiers ranking sheet size changed: {rankings_raw.shape}")
    if athletes_raw.shape != (14_038, 25):
        raise ValueError(f"Frontiers athlete sheet size changed: {athletes_raw.shape}")
    rankings = normalize_rankings(rankings_raw)
    athletes = normalize_athlete_memberships(athletes_raw)
    if len(rankings) != 24_000 or rankings["athlete_key"].nunique() != 8_125:
        raise ValueError("Frontiers fencing ranking snapshot counts changed")
    if len(athletes) != 9_858 or athletes["athlete_key"].nunique() != 8_125:
        raise ValueError("Frontiers fencing athlete snapshot counts changed")
    if set(rankings["athlete_key"]) != set(athletes["athlete_key"]):
        raise ValueError("Frontiers ranking and athlete ID sets disagree")
    return rankings, athletes


def fetch_and_prepare(
    raw_path: Path,
    ranking_output: Path,
    athlete_output: Path,
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
    if sha256_file(raw_path) != EXPECTED_SHA256:
        raise ValueError("Frontiers handedness workbook checksum changed")
    rankings, athletes = normalize_snapshot(raw_path)
    ranking_output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(ranking_output, index=False)
    athletes.to_parquet(athlete_output, index=False)
    membership_flags = [f"listed_{year}" for year in YEARS]
    report: dict[str, Any] = {
        "claim_boundary": (
            "CC BY year-end top-200 ranking memberships from 2007-2016, not bouts, "
            "touches, training interventions, or proof of athlete identity. The two sheets "
            "overlap by source athlete ID and are never added as independent people. Exact "
            "birth dates, direct source IDs, and full country names are removed. This older "
            "panel overlaps in years and concepts with the separately integrated 2004-2023 "
            "handedness ranking supplement, but the two cannot be entity-resolved safely "
            "without direct names and therefore are not row-counted together."
        ),
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "figshare_file_id": FIGSHARE_FILE_ID,
        "raw_sha256": sha256_file(raw_path),
        "ranking_memberships": len(rankings),
        "athlete_category_memberships": len(athletes),
        "source_athlete_keys": int(rankings["athlete_key"].nunique()),
        "ranking_year_min": int(rankings["year"].min()),
        "ranking_year_max": int(rankings["year"].max()),
        "ranking_rows_by_weapon": rankings["weapon"].value_counts().sort_index().to_dict(),
        "ranking_rows_by_age_level": rankings["age_level"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "ranking_rows_by_sex": rankings["sex"].value_counts().sort_index().to_dict(),
        "athlete_memberships_by_weapon": athletes["weapon"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "athlete_memberships_by_age_level": athletes["age_level"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "left_handed_athlete_category_memberships": int(
            athletes["handedness"].eq("left").sum()
        ),
        "unknown_handedness_athlete_category_memberships": int(
            athletes["handedness"].eq("unknown").sum()
        ),
        "multi_weapon_source_athlete_keys": int(
            rankings.groupby("athlete_key")["weapon"].nunique().gt(1).sum()
        ),
        "junior_and_senior_source_athlete_keys": int(
            rankings.groupby("athlete_key")["age_level"].nunique().gt(1).sum()
        ),
        "shared_rank_rows": int(rankings["shared_rank_key"].sum()),
        "shared_rank_rows_after_first": int(
            rankings.duplicated(["age_level", "sex", "weapon", "year", "rank"]).sum()
        ),
        "ranking_memberships_minus_annual_flags": int(
            len(rankings) - athletes[membership_flags].sum().sum()
        ),
        "source_discrepancy_note": (
            "The ranking sheet has 24,000 fencing memberships while the annual flags in the "
            "athlete-category sheet sum to 23,998; the two-row source discrepancy is retained."
        ),
        "exact_birth_dates_removed": True,
        "direct_source_ids_removed": True,
        "full_country_names_removed": True,
        "ranking_output": str(ranking_output),
        "ranking_output_sha256": sha256_file(ranking_output),
        "athlete_output": str(athlete_output),
        "athlete_output_sha256": sha256_file(athlete_output),
        "repository_policy": "privacy-minimized CC BY row tables local; code and aggregates public",
    }
    write_json(report_path, report)
    return report
