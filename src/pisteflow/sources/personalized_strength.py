"""Local-only intervention data for fencers with chronic ankle instability."""

# ruff: noqa: RUF001 - exact upstream workbook headers contain full-width punctuation

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.cloudpmc import download_asset
from pisteflow.sources.net import download_resumable

PMCID = "PMC12764454"
DOI = "https://doi.org/10.1038/s41598-025-29291-w"
ARTICLE_URL = f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/"
XML_URL = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML"
ASSET_URL = (
    "https://pmc.ncbi.nlm.nih.gov/articles/instance/12764454/bin/"
    "41598_2025_29291_MOESM2_ESM.xlsx"
)
LICENSE = "CC BY-NC-ND 4.0"
EXPECTED_BYTES = 58_204
_GROUPS = {1: "control", 2: "standardized", 3: "personalized"}


def _disambiguate_headers(values: list[object]) -> tuple[list[str], set[str]]:
    headers: list[str] = []
    duplicates: set[str] = set()
    seen: dict[str, int] = {}
    for index, value in enumerate(values):
        base = "participant_code" if index == 0 and pd.isna(value) else str(value).strip()
        occurrence = seen.get(base, 0) + 1
        seen[base] = occurrence
        if occurrence == 1:
            headers.append(base)
        else:
            duplicates.add(base)
            headers.append(f"{base}__duplicate_{occurrence}")
    return headers, duplicates


def _table_from_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    headers, duplicates = _disambiguate_headers(raw.iloc[0].tolist())
    frame = raw.iloc[2:].copy().reset_index(drop=True)
    frame.columns = headers
    return frame, duplicates


def normalize_profiles(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove exact dates of birth and study row numbers from the local profile table."""

    frame, _ = _table_from_raw(raw)
    required = {
        "participant_code",
        "Name",
        "Group",
        "Birth",
        "Age（year）",
        "Weight（kg）",
        "Height（cm）",
        "Years of sports experience（year）",
        "Sports Level",
        "Injury Side",
    }
    if set(frame) != required:
        raise ValueError("Personalized-strength profile columns changed")
    result = frame.drop(columns=["Name", "Birth"]).rename(
        columns={
            "Group": "group_code",
            "Age（year）": "age_years",
            "Weight（kg）": "weight_kg",
            "Height（cm）": "height_cm",
            "Years of sports experience（year）": "sports_experience_years",
            "Sports Level": "sports_level",
            "Injury Side": "injury_side",
        }
    )
    result["group_code"] = pd.to_numeric(result["group_code"], errors="raise").astype("int8")
    result["group"] = result["group_code"].map(_GROUPS)
    result["age_below_18"] = pd.to_numeric(result["age_years"], errors="raise").lt(18)
    result["weapon"] = "unknown"
    result["source_doi"] = DOI
    return result


def _phase(sheet_name: str, variable: str) -> str:
    value = f"{sheet_name} {variable}".casefold()
    if "before" in value:
        return "before"
    if "after" in value:
        return "after"
    return "profile_or_unspecified"


def _measurement_key(variable: str) -> str:
    value = unicodedata.normalize("NFKC", variable).casefold()
    value = re.sub(r"\b(before|after|intervetion|intervention)\b", " ", value)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def normalize_measurements(raw_sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a loss-aware long table from the ten outcome sheets."""

    rows: list[dict[str, Any]] = []
    for sheet_name, raw in raw_sheets.items():
        frame, duplicate_headers = _table_from_raw(raw)
        required = {"participant_code", "Group", "Name"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Personalized-strength sheet columns changed: {sheet_name}")
        metadata = {"participant_code", "Group", "Name", "Injury Side"}
        for _, participant in frame.iterrows():
            group_code = int(participant["Group"])
            for variable in [column for column in frame if column not in metadata]:
                base_variable = variable.split("__duplicate_", maxsplit=1)[0]
                value = participant[variable]
                normalized = unicodedata.normalize("NFKC", base_variable).casefold()
                side = (
                    "right"
                    if "right" in normalized
                    else "left"
                    if "left" in normalized
                    else "not_applicable"
                )
                rows.append(
                    {
                        "participant_code": participant["participant_code"],
                        "group_code": group_code,
                        "group": _GROUPS[group_code],
                        "source_sheet": sheet_name,
                        "source_variable": variable,
                        "measurement_key": _measurement_key(variable),
                        "phase": _phase(sheet_name, variable),
                        "side": side,
                        "upstream_duplicate_header": base_variable in duplicate_headers,
                        "value_numeric": pd.to_numeric(value, errors="coerce"),
                        "value_text": None if pd.isna(value) else str(value),
                        "weapon": "unknown",
                        "source_doi": DOI,
                    }
                )
    return pd.DataFrame(rows)


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    workbook = pd.ExcelFile(path)
    if len(workbook.sheet_names) != 11 or workbook.sheet_names[0] != "Basic Information":
        raise ValueError("Personalized-strength workbook sheets changed")
    raw_sheets = {
        name: pd.read_excel(path, sheet_name=name, header=None) for name in workbook.sheet_names
    }
    profiles = normalize_profiles(raw_sheets.pop("Basic Information"))
    measurements = normalize_measurements(raw_sheets)
    expected_codes = {
        f"{prefix}{number}"
        for prefix in ("CG", "TG", "IG")
        for number in range(1, 8)
    }
    if set(profiles["participant_code"]) != expected_codes:
        raise ValueError("Personalized-strength participant codes changed")
    if len(measurements) != 1_512 or set(measurements["participant_code"]) != expected_codes:
        raise ValueError("Personalized-strength measurement-cell count changed")
    return profiles, measurements, workbook.sheet_names


def fetch_and_prepare(
    raw_path: Path,
    xml_path: Path,
    profile_path: Path,
    measurement_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        raw_path.unlink(missing_ok=True)
        xml_path.unlink(missing_ok=True)
    download_resumable(XML_URL, xml_path, user_agent=user_agent)
    xml = xml_path.read_text(encoding="utf-8", errors="replace")
    if "CC BY-NC-ND" not in xml:
        raise ValueError("Personalized-strength article license changed")
    download_asset(
        ASSET_URL,
        raw_path,
        expected_bytes=EXPECTED_BYTES,
        user_agent=user_agent,
        force=force,
        expected_magic=b"PK\x03\x04",
    )
    profiles, measurements, sheets = normalize_snapshot(raw_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(profile_path, index=False)
    measurements.to_parquet(measurement_path, index=False)
    report: dict[str, Any] = {
        "claim_boundary": (
            "Randomized six-week intervention data for 21 fencers with chronic ankle "
            "instability, seven per control, standardized, and personalized group. Weapon "
            "labels are not published. The workbook contains exact birth dates and five "
            "minors; dates and row numbers are removed from local processed tables. Because "
            "the source is CC BY-NC-ND 4.0, raw and adapted row-level tables are local-only "
            "and excluded from Git."
        ),
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "pmcid": PMCID,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "article_xml_sha256": sha256_file(xml_path),
        "raw_sha256": sha256_file(raw_path),
        "participants": len(profiles),
        "group_counts": profiles["group"].value_counts().sort_index().to_dict(),
        "participants_below_18": int(profiles["age_below_18"].sum()),
        "measurement_cells": len(measurements),
        "nonmissing_numeric_measurement_cells": int(measurements["value_numeric"].notna().sum()),
        "source_sheets": sheets,
        "upstream_duplicate_header_cells": int(
            measurements["upstream_duplicate_header"].sum()
        ),
        "weapon_label": "unknown; not published",
        "direct_birth_dates_removed": True,
        "repository_distribution": "local_only",
        "outputs": {
            "profiles": str(profile_path),
            "measurements": str(measurement_path),
        },
        "output_sha256": {
            "profiles": sha256_file(profile_path),
            "measurements": sha256_file(measurement_path),
        },
    }
    write_json(report_path, report)
    return report
