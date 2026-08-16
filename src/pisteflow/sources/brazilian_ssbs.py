"""Local-only fencing subset from the Brazilian SSBS athlete workbook."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://www.scidb.cn/en/detail?dataSetId=ce49994b7f6b4546939eb88c2beff2f6"
SOURCE_DOI = "https://doi.org/10.57760/sciencedb.24779"
DOWNLOAD_URL = "https://china.scidb.cn/download?fileId=945686e2ad64ec1371696098bfb831f6"
LICENSE = "CC BY-NC-ND 4.0"
EXPECTED_BYTES = 25_219
EXPECTED_SHA256 = "8335d8ba7c8f9c37a7687549fd86363db7b4b457e91fd30fb423d5502b799b52"

SOURCE_COLUMNS = [
    "SBSS1",
    "SBSS2",
    "SBSS3",
    "SBSS4",
    "SBSS5",
    "SBSS6",
    "Age (years)?",
    "Biological sex",
    "Sports",
    "Practicing in the last... (years)",
    "Porfessional ou amateur athlete?",
    "Currently using dietary supplements?",
    "Who prescribed the dietary supplement?",
]


def _participant_code(source_row: int) -> str:
    digest = sha256(f"sciencedb-ssbs-fencing:{source_row}".encode()).hexdigest()[:18]
    return f"ssbs-{digest}"


def _five_year_band(value: object) -> str | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    start = int(float(numeric) // 5 * 5)
    return f"{start}-{start + 4}"


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain scale responses while coarsening or removing sensitive covariates."""

    frame = frame.dropna(axis=1, how="all")
    if list(frame.columns) != SOURCE_COLUMNS:
        raise ValueError("Brazilian SSBS workbook columns changed")
    sport = frame["Sports"].astype("string").str.strip().str.casefold()
    source_rows = pd.Series(range(2, len(frame) + 2), index=frame.index)
    fencing = frame.loc[sport.eq("esgrima")].copy()
    fencing_source_rows = source_rows.loc[fencing.index]
    if fencing.empty:
        raise ValueError("Brazilian SSBS workbook contains no fencing rows")
    scale_columns = [f"SBSS{number}" for number in range(1, 7)]
    scales = fencing[scale_columns].apply(pd.to_numeric, errors="raise").astype("int8")
    if not scales.apply(lambda column: column.between(1, 6).all()).all():
        raise ValueError("Brazilian SSBS scale response range changed")
    result = scales.copy()
    result.insert(
        0,
        "participant_code",
        fencing_source_rows.map(_participant_code).to_numpy(),
    )
    result["age_band_years"] = fencing["Age (years)?"].map(_five_year_band).to_numpy()
    result["practice_years_band"] = fencing["Practicing in the last... (years)"].map(
        _five_year_band
    ).to_numpy()
    result["athlete_level_source"] = (
        fencing["Porfessional ou amateur athlete?"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .map({"profissional": "professional", "amateur": "amateur"})
        .to_numpy()
    )
    result["currently_uses_dietary_supplements"] = (
        fencing["Currently using dietary supplements?"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .map({"yes": True, "no": False})
        .to_numpy()
    )
    result["weapon"] = "unknown"
    result["source_dataset_doi"] = SOURCE_DOI
    if result["participant_code"].duplicated().any():
        raise ValueError("Brazilian SSBS fencing participant codes are duplicated")
    return result.reset_index(drop=True)


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, int]:
    source = pd.read_excel(path)
    source_without_empty_columns = source.dropna(axis=1, how="all")
    if source_without_empty_columns.shape != (250, 13):
        raise ValueError(
            f"Brazilian SSBS workbook snapshot changed: {source_without_empty_columns.shape}"
        )
    result = normalize_frame(source)
    if len(result) != 5:
        raise ValueError(f"Brazilian SSBS fencing row count changed: {len(result)}")
    return result, len(source_without_empty_columns)


def fetch_and_prepare(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire and prepare the five-row fencing subset for local analysis only."""

    if force:
        raw_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        raw_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    if sha256_file(raw_path) != EXPECTED_SHA256:
        raise ValueError("Brazilian SSBS workbook checksum changed")
    result, source_rows = normalize_snapshot(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Five fencing rows within a 250-athlete Brazilian sports-supplement-belief "
            "workbook. Weapon and performance outcomes are absent, the sample is far too small "
            "for fencing-population claims, and CC BY-NC-ND prevents redistribution of an "
            "adapted row table. The processed subset remains local; biological sex and detailed "
            "supplement-prescriber text are removed, and age and practice duration are binned."
        ),
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_rows_all_sports": source_rows,
        "fencing_rows": len(result),
        "scale_items": 6,
        "fencing_rows_by_athlete_level_source": (
            result["athlete_level_source"].value_counts().sort_index().astype(int).to_dict()
        ),
        "fencing_rows_currently_using_supplements": int(
            result["currently_uses_dietary_supplements"].sum()
        ),
        "weapon_available": False,
        "performance_outcomes_available": False,
        "biological_sex_removed": True,
        "prescriber_detail_removed": True,
        "row_table_policy": "local_only_no_redistribution",
        "raw_sha256": sha256_file(raw_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(report_path, summary)
    return summary
