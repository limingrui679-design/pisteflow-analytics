"""CC BY attendance and aggregate outcomes from the RIPOSTE pilot trial."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://figshare.com/articles/dataset/25584318"
SOURCE_DOI = "https://doi.org/10.3389/fonc.2024.1335442.s001"
ARTICLE_DOI = "https://doi.org/10.3389/fonc.2024.1335442"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/45583761"
LICENSE = "CC BY 4.0"
EXPECTED_BYTES = 32_412
EXPECTED_SHA256 = "dc9d238ceca8fefa8cd8c676c6cf6d31e902877c39e4c676b1199b7afc7f6ba8"

_WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_INSTRUMENTS = {"QLQ-C30", "QLQ-BR23", "DASH", "HADS", "MFI-20"}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _docx_tables(path: Path) -> list[list[list[str]]]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    tables: list[list[list[str]]] = []
    for table in root.findall(".//w:tbl", _WORD_NS):
        rows: list[list[str]] = []
        for row in table.findall("./w:tr", _WORD_NS):
            cells: list[str] = []
            for cell in row.findall("./w:tc", _WORD_NS):
                text = "".join(node.text or "" for node in cell.findall(".//w:t", _WORD_NS))
                cells.append(_clean(text))
            rows.append(cells)
        tables.append(rows)
    return tables


def _participant_code(source_number: int) -> str:
    digest = sha256(f"riposte-adapted-fencing:{source_number}".encode()).hexdigest()[:18]
    return f"rip-{digest}"


def normalize_attendance_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Normalize the 24 deidentified participant-attendance rows."""

    records: list[dict[str, Any]] = []
    for row in rows[1:]:
        if len(row) != 6 or not row[0].isdigit():
            continue
        source_number = int(row[0])
        group = _clean(row[1]).lower()
        if group not in {"early riposte", "delayed riposte"}:
            raise ValueError(f"RIPOSTE group vocabulary changed: {row[1]!r}")
        withdrew = _clean(row[2]).casefold() == "yes"
        session_match = re.search(r"\d+", row[3])
        if not session_match:
            raise ValueError("RIPOSTE participated-session value is missing")
        sessions = int(session_match.group())
        proposed = int(row[4]) if row[4].isdigit() else None
        rate = int(row[5].removesuffix("%")) if row[5].endswith("%") else None
        if withdrew and (proposed is not None or rate is not None):
            raise ValueError("RIPOSTE withdrawal row unexpectedly has a proposed-session total")
        if proposed is not None and rate != round(100 * sessions / proposed):
            raise ValueError("RIPOSTE participation rate no longer matches session counts")
        records.append(
            {
                "participant_code": _participant_code(source_number),
                "allocation_timing": group.removesuffix(" riposte"),
                "lost_to_follow_up": withdrew,
                "participated_sessions": sessions,
                "proposed_sessions": proposed,
                "participation_rate_percent_source": rate,
                "source_dataset_doi": SOURCE_DOI,
            }
        )
    result = pd.DataFrame(records)
    if result.empty or result["participant_code"].duplicated().any():
        raise ValueError("RIPOSTE attendance rows are empty or duplicated")
    return result


def _estimate_and_interval(value: str) -> tuple[float, float, float]:
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
    if len(numbers) != 3:
        raise ValueError(f"RIPOSTE estimate and confidence interval changed: {value!r}")
    return numbers[0], numbers[1], numbers[2]


def normalize_outcome_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Normalize published group-level change estimates without creating patient rows."""

    instrument: str | None = None
    records: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        first = _clean(row[0])
        section = first.lstrip().upper()
        if section in _INSTRUMENTS:
            instrument = section
            continue
        if len(row) < 8 or not row[1].isdigit() or not row[3].isdigit():
            continue
        if instrument is None:
            raise ValueError("RIPOSTE outcome row appeared before an instrument heading")
        early = _estimate_and_interval(row[2])
        delayed = _estimate_and_interval(row[4])
        difference = _estimate_and_interval(row[6])
        records.append(
            {
                "instrument": instrument,
                "outcome": first,
                "early_n": int(row[1]),
                "early_change_mean": early[0],
                "early_change_ci_low": early[1],
                "early_change_ci_high": early[2],
                "delayed_n": int(row[3]),
                "delayed_change_mean": delayed[0],
                "delayed_change_ci_low": delayed[1],
                "delayed_change_ci_high": delayed[2],
                "difference_of_change_mean": difference[0],
                "difference_of_change_ci_low": difference[1],
                "difference_of_change_ci_high": difference[2],
                "p_value": float(row[7]),
                "source_dataset_doi": SOURCE_DOI,
            }
        )
    result = pd.DataFrame(records)
    if result.empty or result[["instrument", "outcome"]].duplicated().any():
        raise ValueError("RIPOSTE aggregate outcome rows are empty or duplicated")
    return result


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = _docx_tables(path)
    if len(tables) != 4:
        raise ValueError(f"RIPOSTE supplementary table count changed: {len(tables)}")
    outcomes = normalize_outcome_rows(tables[0])
    attendance = normalize_attendance_rows(tables[1])
    if len(attendance) != 24 or int(attendance["lost_to_follow_up"].sum()) != 5:
        raise ValueError("RIPOSTE participant attendance snapshot changed")
    if len(outcomes) != 28:
        raise ValueError("RIPOSTE aggregate outcome snapshot changed")
    return attendance, outcomes


def fetch_and_prepare(
    raw_path: Path,
    attendance_output_path: Path,
    outcome_output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the official supplement and prepare attendance and aggregate tables."""

    if force:
        raw_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        raw_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    if sha256_file(raw_path) != EXPECTED_SHA256:
        raise ValueError("RIPOSTE supplementary DOCX checksum changed")
    attendance, outcomes = normalize_snapshot(raw_path)
    attendance_output_path.parent.mkdir(parents=True, exist_ok=True)
    attendance.to_parquet(attendance_output_path, index=False)
    outcomes.to_parquet(outcome_output_path, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "A CC BY pilot of adapted fencing after breast-cancer surgery: 24 deidentified "
            "participant attendance rows and 28 published group-level secondary-outcome "
            "estimates. Participants are patients in an adapted-activity programme, not "
            "competitive fencers; outcome rows are aggregates, not additional people, and the "
            "small pilot does not establish clinical efficacy or general athletic-training "
            "effects. Direct source participant numbers are replaced with stable codes."
        ),
        "source_url": SOURCE_URL,
        "source_dataset_doi": SOURCE_DOI,
        "article_doi": ARTICLE_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "participants": len(attendance),
        "participants_by_allocation_timing": (
            attendance["allocation_timing"].value_counts().sort_index().astype(int).to_dict()
        ),
        "lost_to_follow_up": int(attendance["lost_to_follow_up"].sum()),
        "participated_sessions_total": int(attendance["participated_sessions"].sum()),
        "participants_with_proposed_session_denominator": int(
            attendance["proposed_sessions"].notna().sum()
        ),
        "aggregate_outcome_rows": len(outcomes),
        "aggregate_outcome_rows_by_instrument": (
            outcomes["instrument"].value_counts().sort_index().astype(int).to_dict()
        ),
        "raw_sha256": sha256_file(raw_path),
        "outputs": {
            "attendance": str(attendance_output_path),
            "aggregate_outcomes": str(outcome_output_path),
        },
        "output_sha256": {
            "attendance": sha256_file(attendance_output_path),
            "aggregate_outcomes": sha256_file(outcome_output_path),
        },
    }
    write_json(report_path, summary)
    return summary
