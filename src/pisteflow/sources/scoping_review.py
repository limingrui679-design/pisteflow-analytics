"""Locally normalized catalog of quantitative fencing and wheelchair-fencing studies."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DOI = "https://doi.org/10.18745/ds.00026640"
PAGE_URL = "https://uhra.herts.ac.uk/id/eprint/26640/"
DATA_URL = "https://uhra.herts.ac.uk/id/eprint/26640/2/Data_Sheet%20%281%29.xlsx"
README_URL = "https://uhra.herts.ac.uk/id/eprint/26640/1/README.txt"
LICENSE = (
    "Conflicting source statements: the official UHRA file page says CC BY-ND 4.0, "
    "while the deposited README says CC-BY"
)
DATA_BYTES = 292_687
README_BYTES = 4_366

_COLUMNS = [
    "APA Reference",
    "Title",
    "Authors",
    "Year",
    "Journal",
    "Summary",
    "Key Findings",
    "Country",
    "Fencing Only",
    "Able Bodied",
    "N fencers",
    "Fencer Level",
    "Weapon",
    "Gender",
    "Domain",
    "Design",
    "Primary Outcome",
]


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize catalog labels while keeping bibliographic records intact."""

    result = frame.copy()
    result.columns = result.columns.astype(str).str.strip()
    if list(result.columns) != _COLUMNS:
        raise ValueError("Fencing scoping-review columns changed")
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].astype("string").str.strip()
    result.insert(0, "study_code", [f"FSR-{index:04d}" for index in range(1, len(result) + 1)])
    result["year_numeric"] = pd.to_numeric(result["Year"], errors="coerce").astype("Int16")
    result["reported_n_fencers_numeric"] = pd.to_numeric(
        result["N fencers"].astype("string").str.replace(",", "", regex=False),
        errors="coerce",
    ).astype("Int32")
    result["weapon_normalized"] = (
        result["Weapon"]
        .replace(
            {
                "Epee": "epee",
                "Foil": "foil",
                "Saber": "sabre",
                "Multiple": "multiple",
                "Not specified": "unknown",
            }
        )
        .astype("string")
    )
    result["adaptive_scope"] = (
        result["Able Bodied"]
        .replace(
            {
                "Able Bodied": "able_bodied",
                "Wheelchair": "wheelchair",
                "Both": "both",
                "Not specified": "unknown",
            }
        )
        .astype("string")
    )
    result["source_doi"] = DOI
    return result


def normalize_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Data")
    if frame.shape != (445, 17):
        raise ValueError(f"Fencing scoping-review snapshot changed: {frame.shape}")
    return normalize_frame(frame)


def fetch_and_prepare(
    raw_path: Path,
    readme_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        raw_path.unlink(missing_ok=True)
        readme_path.unlink(missing_ok=True)
    download_resumable(DATA_URL, raw_path, user_agent=user_agent, expected_bytes=DATA_BYTES)
    download_resumable(
        README_URL,
        readme_path,
        user_agent=user_agent,
        expected_bytes=README_BYTES,
    )
    if "Licenses/restrictions placed on the data: CC-BY" not in readme_path.read_text(
        encoding="utf-8-sig", errors="replace"
    ):
        raise ValueError("Fencing scoping-review license statement changed")
    frame = normalize_snapshot(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    report: dict[str, Any] = {
        "claim_boundary": (
            "A 445-study bibliographic discovery catalog, not 445 athlete samples and not "
            "row-level data from the underlying studies. Reported sample sizes are metadata "
            "and are never added to PisteFlow's acquired-athlete total; their sum can include "
            "overlapping cohorts, controls, repeated observations, and non-fencers. The official "
            "file page says CC BY-ND 4.0 while the deposited README says CC-BY, so the normalized "
            "row table remains local under the more restrictive public-page statement. The "
            "README calls the file CSV, while the deposit supplies an XLSX workbook."
        ),
        "source_url": PAGE_URL,
        "dataset_doi": DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "raw_sha256": sha256_file(raw_path),
        "readme_sha256": sha256_file(readme_path),
        "study_rows": len(frame),
        "study_rows_with_numeric_fencer_count": int(
            frame["reported_n_fencers_numeric"].notna().sum()
        ),
        "sum_of_study_reported_fencer_observations": int(
            frame["reported_n_fencers_numeric"].sum()
        ),
        "apa_reference_rows_with_doi_pattern": int(
            frame["APA Reference"]
            .astype("string")
            .str.contains(r"10\.\d{4,9}/", case=False, regex=True, na=False)
            .sum()
        ),
        "weapon_scope_rows": frame["weapon_normalized"].value_counts().to_dict(),
        "domain_rows": frame["Domain"].value_counts().to_dict(),
        "adaptive_scope_rows": frame["adaptive_scope"].value_counts().to_dict(),
        "year_min": int(frame["year_numeric"].min()),
        "year_max": int(frame["year_numeric"].max()),
        "title_missing_rows": int(frame["Title"].isna().sum()),
        "duplicate_title_rows": int(frame.duplicated("Title", keep=False).sum()),
        "counted_as_training_samples": False,
        "normalized_row_table_publication_allowed": False,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(report_path, report)
    return report
