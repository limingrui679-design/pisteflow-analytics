"""CC BY supplementary videos and computed variables for the fencing flick study."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6760795/"
ARTICLE_DOI = "https://doi.org/10.1371/journal.pone.0222075"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DOWNLOAD_BASE = (
    "https://journals.plos.org/plosone/article/file?type=supplementary&"
    "id=info:doi/10.1371/journal.pone.0222075"
)
FILES: tuple[dict[str, str | int], ...] = (
    {
        "filename": "pone.0222075.s001.mp4",
        "content_kind": "simulation",
        "athlete_group": "none",
        "expected_bytes": 60_608,
    },
    {
        "filename": "pone.0222075.s002.mp4",
        "content_kind": "real_high_speed_trial",
        "athlete_group": "fencer_a",
        "expected_bytes": 8_857_517,
    },
    {
        "filename": "pone.0222075.s003.mp4",
        "content_kind": "real_high_speed_trial",
        "athlete_group": "fencer_b",
        "expected_bytes": 6_638_422,
    },
    {
        "filename": "pone.0222075.s004.xlsx",
        "content_kind": "computed_trial_variables",
        "athlete_group": "fencers_a_and_b",
        "expected_bytes": 78_437,
    },
)


def _safe_sheet_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    return value or "sheet"


def _safe_column_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return value or "unnamed"


def normalize_data_sheet(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize labels while preserving every numeric cell in the primary trial sheet."""

    frame = pd.read_excel(path, sheet_name="data", header=1)
    original_columns = [str(column) for column in frame.columns]
    renamed = [_safe_column_name(column) for column in original_columns]
    if len(set(renamed)) != len(renamed):
        raise ValueError("PLOS flick normalized column labels are not unique")
    column_map = dict(zip(original_columns, renamed, strict=True))
    frame.columns = renamed
    trial_column = renamed[0]
    trial_numbers = pd.to_numeric(frame[trial_column], errors="coerce")
    frame = frame.loc[trial_numbers.notna()].copy()
    for column in frame.columns:
        raw = frame[column]
        converted = pd.to_numeric(raw, errors="coerce")
        unexpected = raw.notna() & converted.isna()
        if unexpected.any():
            raise ValueError(f"PLOS flick data sheet has non-numeric values in {column}")
        frame[column] = converted
    frame[trial_column] = frame[trial_column].astype("Int16")
    score_column = renamed[2]
    frame[score_column] = frame[score_column].astype("Int8")
    return frame.reset_index(drop=True), column_map


def fetch_and_prepare(
    raw_directory: Path,
    processed_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download all four article supplements and inventory spreadsheet sheets."""

    raw_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    spreadsheet_path: Path | None = None
    for specification in FILES:
        filename = str(specification["filename"])
        suffix = Path(filename).stem.rsplit(".", maxsplit=1)[-1]
        source_url = f"{DOWNLOAD_BASE}.{suffix}"
        destination = raw_directory / filename
        if force:
            destination.unlink(missing_ok=True)
        download_resumable(
            source_url,
            destination,
            user_agent=user_agent,
            expected_bytes=int(specification["expected_bytes"]),
        )
        if filename.endswith(".xlsx"):
            spreadsheet_path = destination
        records.append(
            {
                "asset_id": f"plos-flick:{filename}",
                "filename": filename,
                "content_kind": str(specification["content_kind"]),
                "athlete_group": str(specification["athlete_group"]),
                "weapon": "foil",
                "source_url": source_url,
                "article_url": ARTICLE_URL,
                "article_doi": ARTICLE_DOI,
                "license": LICENSE,
                "license_url": LICENSE_URL,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "claim_boundary": (
                    "One supplementary asset from a two-fencer laboratory study of the foil "
                    "flick; not a competition outcome or a three-weapon benchmark."
                ),
            }
        )

    if spreadsheet_path is None:
        raise RuntimeError("PLOS fencing-flick spreadsheet was not downloaded")
    sheets: list[dict[str, Any]] = []
    workbook = pd.ExcelFile(spreadsheet_path)
    for sheet_name in workbook.sheet_names:
        raw_sheet = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
        sheet_record: dict[str, Any] = {
            "sheet": sheet_name,
            "raw_rows": len(raw_sheet),
            "raw_columns": len(raw_sheet.columns),
        }
        if sheet_name == "data":
            frame, column_map = normalize_data_sheet(spreadsheet_path)
            output = processed_directory / f"{_safe_sheet_name(sheet_name)}.parquet"
            frame.to_parquet(output, index=False)
            sheet_record.update(
                {
                    "analytic_rows": len(frame),
                    "analytic_columns": len(frame.columns),
                    "missing_values": int(frame.isna().sum().sum()),
                    "column_map": column_map,
                    "output": str(output),
                    "sha256": sha256_file(output),
                }
            )
        else:
            sheet_record["status"] = "inventory_only_no_stable_headers"
        sheets.append(sheet_record)

    manifest_sha256 = write_jsonl(manifest_path, records)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Two real 650-fps foil-flick trials, one simulation, and computed regression "
            "variables from a two-fencer study. Useful as a transparent biomechanics case, "
            "not as an independent athlete-level test set."
        ),
        "source_url": ARTICLE_URL,
        "article_doi": ARTICLE_DOI,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "assets": len(records),
        "real_high_speed_videos": sum(
            row["content_kind"] == "real_high_speed_trial" for row in records
        ),
        "bytes": sum(int(row["bytes"]) for row in records),
        "manifest_sha256": manifest_sha256,
        "spreadsheet_sheets": sheets,
        "output": str(manifest_path),
    }
    write_json(report_path, summary)
    return summary
