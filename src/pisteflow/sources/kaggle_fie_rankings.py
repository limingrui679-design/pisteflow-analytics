"""Audit a six-event FIE ranking workbook while keeping athlete rows local."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

DATASET_URL = (
    "https://www.kaggle.com/datasets/sethlogan/fie-fencer-rankings-end-of-20212022-season"
)
ARCHIVE_BYTES = 303_498
ARCHIVE_SHA256 = "65ab3b4ce63808467c860875044bd8c7e1a260f2ff6d7f93ccc623376f37c055"
WORKBOOK_BYTES = 323_128
WORKBOOK_SHA256 = "6e34e6fec37cc1284c4735252cb2296ea39c93bc7839ca817037ecb14a69bc75"

SHEETS = {
    "Epee (F)": ("epee", "female"),
    "Foil (F)": ("foil", "female"),
    "Saber (F)": ("sabre", "female"),
    "Epee (M)": ("epee", "male"),
    "Foil (M)": ("foil", "male"),
    "Saber (M)": ("sabre", "male"),
}
EVENT_PATTERN = re.compile(r"^(\d{2})\.(\d{2})\.(\d{2})\s+(.+)\s+\(([^()]+)\)$")

EXPECTED_RANKING_ROWS = 3_410
EXPECTED_IDENTITY_SPELLING_COUNTRY_KEYS = 3_390
EXPECTED_EVENT_POINT_ROWS = 9_978
EXPECTED_COUNTED_POINT_ROWS = 9_453
EXPECTED_EXCLUDED_POINT_ROWS = 525
EXPECTED_EVENT_LABELS = 62

RIGHTS_BOUNDARY = (
    "Kaggle reports an unknown dataset license, and the workbook appears derived from FIE ranking "
    "pages. Raw names, the workbook, row-level Parquet files, and trained artifacts stay local. "
    "Only code, hashes, aggregate counts, and integrity results belong in a public release."
)


def _identity_key(name: str, country: str) -> str:
    normalized = "|".join((" ".join(name.casefold().split()), country.casefold().strip()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_event_label(label: str) -> tuple[date, str, str]:
    match = EVENT_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"Unexpected ranking event label: {label}")
    day, month, year, location, event_type = match.groups()
    return date(2000 + int(year), int(month), int(day)), location, event_type


def _parse_points(value: object) -> tuple[float, bool] | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    excluded = raw.startswith("(") and raw.endswith(")")
    return float(raw.strip("()")), not excluded


def _validate_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ValueError(f"Pinned Kaggle ranking input changed: {path.name}")


def parse_workbook(workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return pseudonymous rankings and long event-point rows from all six sheets."""

    _validate_file(workbook_path, WORKBOOK_BYTES, WORKBOOK_SHA256)
    excel = pd.ExcelFile(workbook_path)
    if excel.sheet_names != list(SHEETS):
        raise ValueError(f"Ranking workbook sheets changed: {excel.sheet_names}")

    ranking_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    total_mismatches: list[dict[str, Any]] = []
    unique_event_labels: set[str] = set()
    for sheet, (weapon, gender) in SHEETS.items():
        frame = pd.read_excel(workbook_path, sheet_name=sheet)
        required = ["Ranking", "Fencer Name", "Country", "Total"]
        if any(column not in frame.columns for column in required):
            raise ValueError(f"Required columns changed in ranking sheet: {sheet}")
        event_columns = [str(column) for column in frame.columns if column not in required]
        event_metadata = {
            column: _parse_event_label(column)
            for column in event_columns
        }
        unique_event_labels.update(event_columns)
        for _, source_row in frame.iterrows():
            name = str(source_row["Fencer Name"]).strip()
            country = str(source_row["Country"]).strip()
            athlete_key = _identity_key(name, country)
            rank = int(source_row["Ranking"])
            total = float(source_row["Total"])
            ranking_rows.append(
                {
                    "athlete_key": athlete_key,
                    "country": country,
                    "weapon": weapon,
                    "gender": gender,
                    "season": "2021-22",
                    "rank": rank,
                    "total_points": total,
                    "source_sheet": sheet,
                }
            )
            recomputed_total = 0.0
            for event_label, (event_date, location, event_type) in event_metadata.items():
                parsed = _parse_points(source_row[event_label])
                if parsed is None:
                    continue
                points, counted_in_total = parsed
                if counted_in_total:
                    recomputed_total += points
                point_rows.append(
                    {
                        "athlete_key": athlete_key,
                        "country": country,
                        "weapon": weapon,
                        "gender": gender,
                        "season": "2021-22",
                        "event_label": event_label,
                        "event_date": event_date,
                        "location": location,
                        "event_type_code": event_type,
                        "points": points,
                        "counted_in_total": counted_in_total,
                        "source_sheet": sheet,
                    }
                )
            if abs(recomputed_total - total) > 0.01:
                total_mismatches.append(
                    {
                        "athlete_key": athlete_key,
                        "sheet": sheet,
                        "reported": total,
                        "recomputed": recomputed_total,
                    }
                )

    rankings = pd.DataFrame(ranking_rows)
    points = pd.DataFrame(point_rows)
    identity_keys = int(rankings["athlete_key"].nunique())
    counted_rows = int(points["counted_in_total"].sum())
    excluded_rows = len(points) - counted_rows
    observed = {
        "ranking_rows": len(rankings),
        "identity_spelling_country_keys": identity_keys,
        "event_point_rows": len(points),
        "counted_point_rows": counted_rows,
        "excluded_parenthesized_point_rows": excluded_rows,
        "unique_event_labels": len(unique_event_labels),
    }
    expected = {
        "ranking_rows": EXPECTED_RANKING_ROWS,
        "identity_spelling_country_keys": EXPECTED_IDENTITY_SPELLING_COUNTRY_KEYS,
        "event_point_rows": EXPECTED_EVENT_POINT_ROWS,
        "counted_point_rows": EXPECTED_COUNTED_POINT_ROWS,
        "excluded_parenthesized_point_rows": EXPECTED_EXCLUDED_POINT_ROWS,
        "unique_event_labels": EXPECTED_EVENT_LABELS,
    }
    if observed != expected:
        raise ValueError(f"Pinned ranking workbook counts changed: {observed}")
    if total_mismatches:
        raise ValueError(f"Ranking totals failed integrity check: {len(total_mismatches)} rows")
    return rankings, points, observed


def prepare(
    archive_path: Path,
    workbook_path: Path,
    rankings_path: Path,
    points_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Validate, pseudonymize, and store the unknown-license workbook locally."""

    _validate_file(archive_path, ARCHIVE_BYTES, ARCHIVE_SHA256)
    rankings, points, observed = parse_workbook(workbook_path)
    rankings_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(rankings_path, index=False)
    points.to_parquet(points_path, index=False)
    summary: dict[str, Any] = {
        **observed,
        "by_weapon_rankings": dict(Counter(rankings["weapon"])),
        "by_gender_rankings": dict(Counter(rankings["gender"])),
        "country_codes": int(rankings["country"].nunique()),
        "sum_of_reported_total_points": float(rankings["total_points"].sum()),
        "sum_of_counted_event_points": float(
            points.loc[points["counted_in_total"], "points"].sum()
        ),
        "ranking_total_mismatches": 0,
        "dataset_url": DATASET_URL,
        "kaggle_license": "unknown",
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "workbook_bytes": WORKBOOK_BYTES,
        "workbook_sha256": WORKBOOK_SHA256,
        "rankings_sha256": sha256_file(rankings_path),
        "points_sha256": sha256_file(points_path),
        "rights_boundary": RIGHTS_BOUNDARY,
        "claim_boundary": (
            "One end-of-season 2021-22 global ranking workbook across six gender-weapon sheets. "
            "Identity keys are normalized spelling-country pairs, not verified unique people, and "
            "event points are descriptive records rather than training exposure or causal outcomes."
        ),
    }
    if summary["sum_of_reported_total_points"] != summary["sum_of_counted_event_points"]:
        raise ValueError("Aggregate ranking points do not reconcile")
    write_json(summary_path, summary)
    return summary
