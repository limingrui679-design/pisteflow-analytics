"""Privacy-minimized CC BY relative-age rows for youth fencers."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://figshare.com/articles/dataset/15183846"
SOURCE_DOI = "https://doi.org/10.6084/m9.figshare.15183846.v2"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/34263101"
LICENSE = "CC BY 4.0"
EXPECTED_BYTES = 110_524
EXPECTED_SHA256 = "0cdd0a501cf7c2797fb7aad3bb2c07a0993e78c6e7cff4b6013d4e4f1c4a6807"

SOURCE_COLUMNS = [
    "ID - Participant",
    "Sex (0 = Male; 1 = Female)",
    "Chronological Age",
    "Valuation Date",
    "Birth Date",
    "Birth Quartile",
    "Birth Quartile Semester",
    "Category",
    "Category by Weight",
    "Sport",
    "Test",
    "Position in National Ranking 2019",
]

WEAPON_MAP = {"Epee": "epee", "Foil": "foil", "Saber": "sabre"}
SEX_MAP = {0: "male", 1: "female"}


def _athlete_code(source_id: str) -> str:
    digest = sha256(f"figshare-relative-age-fencing:{source_id}".encode()).hexdigest()[:18]
    return f"raf-{digest}"


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter fencing rows while removing exact dates and direct source identifiers."""

    if list(frame.columns) != SOURCE_COLUMNS:
        raise ValueError("Relative-age combat-sport columns changed")
    fencing = frame.loc[frame["Sport"].eq("Fencing")].copy()
    if fencing.empty:
        raise ValueError("Relative-age combat-sport snapshot contains no fencing rows")
    weapons = set(fencing["Test"].dropna())
    if not weapons or not weapons.issubset(WEAPON_MAP):
        raise ValueError("Relative-age fencing weapon vocabulary changed")
    sex_codes = set(fencing["Sex (0 = Male; 1 = Female)"].dropna())
    if not sex_codes or not sex_codes.issubset(SEX_MAP):
        raise ValueError("Relative-age fencing sex vocabulary changed")
    if not fencing["Category by Weight"].eq("---").all():
        raise ValueError("Relative-age fencing weight-category field changed")

    age = pd.to_numeric(fencing["Chronological Age"], errors="raise")
    result = pd.DataFrame(
        {
            "athlete_code": fencing["ID - Participant"].astype(str).map(_athlete_code),
            "sex_reported": fencing["Sex (0 = Male; 1 = Female)"].map(SEX_MAP),
            "age_years_rounded_0_1": age.round(1),
            "birth_quarter": pd.to_numeric(
                fencing["Birth Quartile"], errors="raise"
            ).astype("int8"),
            "birth_semester": pd.to_numeric(
                fencing["Birth Quartile Semester"], errors="raise"
            ).astype("int8"),
            "age_category": fencing["Category"].astype("string"),
            "weapon": fencing["Test"].map(WEAPON_MAP),
            "national_ranking_position_2019": pd.to_numeric(
                fencing["Position in National Ranking 2019"], errors="raise"
            ).astype("int16"),
            "source_dataset_doi": SOURCE_DOI,
        }
    )
    if result["athlete_code"].duplicated().any():
        raise ValueError("Relative-age fencing source identifiers are duplicated")
    if not result["birth_quarter"].between(1, 4).all():
        raise ValueError("Relative-age fencing birth-quarter values changed")
    if not result["birth_semester"].between(1, 2).all():
        raise ValueError("Relative-age fencing birth-semester values changed")
    if not result["national_ranking_position_2019"].between(1, 20).all():
        raise ValueError("Relative-age fencing ranking-position values changed")
    return result.reset_index(drop=True)


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, int]:
    source = pd.read_csv(path, sep=";", decimal=",")
    if source.shape != (1_486, 12):
        raise ValueError(f"Relative-age combat-sport snapshot changed: {source.shape}")
    result = normalize_frame(source)
    if len(result) != 405:
        raise ValueError(f"Relative-age fencing row count changed: {len(result)}")
    if result["weapon"].value_counts().to_dict() != {
        "epee": 157,
        "foil": 151,
        "sabre": 97,
    }:
        raise ValueError("Relative-age fencing weapon counts changed")
    return result, len(source)


def fetch_and_prepare(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the official CSV and write a privacy-minimized local table."""

    if force:
        raw_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        raw_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    if sha256_file(raw_path) != EXPECTED_SHA256:
        raise ValueError("Relative-age combat-sport CSV checksum changed")
    result, source_rows = normalize_snapshot(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    report: dict[str, Any] = {
        "claim_boundary": (
            "CC BY cross-sectional 2019 ranking rows for 405 youth fencers across all three "
            "weapons. They are not bouts, training sessions, longitudinal outcomes, current "
            "rankings, or evidence that birth timing causes performance. The source contains "
            "minor athletes' exact birth dates and direct IDs; both exact dates and direct IDs "
            "are removed, age is rounded, and the row table remains local."
        ),
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_rows_all_combat_sports": source_rows,
        "fencing_rows": len(result),
        "fencing_rows_by_weapon": (
            result["weapon"].value_counts().sort_index().astype(int).to_dict()
        ),
        "fencing_rows_by_age_category": (
            result["age_category"].value_counts().sort_index().astype(int).to_dict()
        ),
        "fencing_rows_by_sex_reported": (
            result["sex_reported"].value_counts().sort_index().astype(int).to_dict()
        ),
        "fencing_rows_by_birth_quarter": {
            str(key): int(value)
            for key, value in result["birth_quarter"].value_counts().sort_index().items()
        },
        "rank_position_range": [
            int(result["national_ranking_position_2019"].min()),
            int(result["national_ranking_position_2019"].max()),
        ],
        "exact_birth_dates_removed": True,
        "direct_source_ids_removed": True,
        "age_rounded_to_tenth_year": True,
        "raw_sha256": sha256_file(raw_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(report_path, report)
    return report
