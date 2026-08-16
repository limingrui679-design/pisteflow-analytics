"""Audit and privacy-minimize the pinned SPWS veterans ranking workbooks."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/Fencer4Life/spws-automated-ranklist"
SNAPSHOT = "b0631c4bd2923e6a43fc8056a1e772db6b184290"
ARCHIVE_URL = (
    f"https://codeload.github.com/Fencer4Life/spws-automated-ranklist/tar.gz/{SNAPSHOT}"
)
EXPECTED_ARCHIVE_BYTES = 60_374_700
EXPECTED_ARCHIVE_SHA256 = "5359c0f62ec82be17ed3a788ba9fa3ec43d87162374f14102e8518f862784b1a"

LICENSE_BOUNDARY = (
    "The README calls this a public repository but provides no software or dataset license. "
    "Public aggregate audit metadata may be published; source workbooks and pseudonymous "
    "athlete-level ranking and event-result rows remain local."
)

EXPECTED_XLSX_FILES = 189
EXPECTED_UNIQUE_XLSX_HASHES = 188
EXPECTED_BACKUP_XLSX_FILES = 30
EXPECTED_CANONICAL_WORKBOOKS = 139
EXPECTED_RANKING_ROWS = 1_075
EXPECTED_RANKING_NORMALIZED_SPELLINGS = 352
EXPECTED_RAW_EVENT_ROWS = 12_197
EXPECTED_EVENT_ROWS = 12_033
EXPECTED_EVENT_NORMALIZED_SPELLINGS = 1_910
EXPECTED_COMBINED_NORMALIZED_SPELLINGS = 1_914
EXPECTED_PARSED_EVENT_SHEETS = 1_681
EXPECTED_NONEMPTY_EVENT_SHEETS = 998
EXPECTED_EMPTY_EVENT_SHEETS = 683
EXPECTED_DUPLICATE_EVENT_SOURCE_ROWS = 164
EXPECTED_EVENT_POINT_CONFLICTS = 1

_WORKBOOK_PATTERN = re.compile(
    r"^(?P<weapon>FLORET|SZABLA|SZPADA)-(?P<women>K?)(?P<category>0|1|2|3|4|OPEN)-"
    r"(?P<season>20\d{2}(?:-20\d{2})?)(?:v\d+)?\.xlsx$",
    re.IGNORECASE,
)
_WEAPONS = {"FLORET": "foil", "SZABLA": "sabre", "SZPADA": "epee"}
_SKIP_EVENT_SHEETS = {"Kadra", "KlasyfikacjaEVF", "RankingPL", "EU"}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def normalize_published_name(value: str) -> str:
    """Normalize exact published spelling without attempting fuzzy identity resolution."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _athlete_series_id(published_name: str) -> str:
    return _stable_id("spws-name", normalize_published_name(published_name))


def _safe_extract_stripped(archive_path: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        top_levels = {
            PurePosixPath(member.name).parts[0]
            for member in members
            if PurePosixPath(member.name).parts
        }
        if len(top_levels) != 1:
            raise ValueError("SPWS archive does not have one stable root")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) <= 1:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("SPWS archive contains an unsafe member path")
            destination = (resolved_root / relative).resolve()
            if resolved_root not in destination.parents:
                raise ValueError("SPWS archive escapes its extraction root")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("SPWS archive contains a non-regular member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("SPWS archive member could not be opened")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def acquire(
    archive_path: Path,
    output_directory: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> Path:
    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_ARCHIVE_BYTES,
    )
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("SPWS archive checksum changed")
    marker = output_directory / "doc/external_files"
    if force or not marker.is_dir():
        _safe_extract_stripped(archive_path, output_directory)
    return output_directory


def parse_workbook_name(path: Path) -> dict[str, str] | None:
    """Parse the weapon, sex, category, and season encoded in a workbook name."""

    match = _WORKBOOK_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    weapon = _WEAPONS[match.group("weapon").upper()]
    season = match.group("season")
    if match.group("women").upper() == "K":
        gender = "women"
    elif season == "2021":
        gender = "unknown"
    else:
        gender = "men"
    return {
        "weapon": weapon,
        "gender": gender,
        "category": match.group("category").upper(),
        "season": season,
    }


def _is_backup(path: Path) -> bool:
    return any("backup" in part.casefold() for part in path.parts)


def canonical_workbooks(source_directory: Path) -> list[tuple[Path, dict[str, str]]]:
    selected: list[tuple[Path, dict[str, str]]] = []
    for path in sorted(source_directory.rglob("*.xlsx")):
        relative = path.relative_to(source_directory)
        metadata = parse_workbook_name(path)
        if metadata is None or _is_backup(relative):
            continue
        is_historical_core = "external_files" in relative.parts
        is_current_partial = relative.as_posix() == "doc/assets/SZPADA-2-2025-2026.xlsx"
        if is_historical_core or is_current_partial:
            selected.append((path, metadata))
    return selected


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return " ".join(str(value).split())


def _integer_place(value: object) -> int | None:
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _ranking_rows(
    worksheet: Any,
    metadata: dict[str, str],
    workbook_id: str,
    workbook_sha256: str,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    rows: list[dict[str, Any]] = []
    published_spellings: set[str] = set()
    normalized_spellings: set[str] = set()
    for values in worksheet.iter_rows(values_only=True):
        padded = list(values) + [None] * 4
        place = _integer_place(padded[0])
        name = _cell_text(padded[1])
        points = padded[3]
        if place is None or not name:
            continue
        if not isinstance(points, (int, float)):
            raise ValueError(f"SPWS ranking points are not numeric in {workbook_id}")
        athlete_id = _athlete_series_id(name)
        published_spellings.add(name)
        normalized_spellings.add(normalize_published_name(name))
        rows.append(
            {
                "ranking_row_id": _stable_id("spws-ranking", f"{workbook_id}:{place}:{athlete_id}"),
                "athlete_series_id": athlete_id,
                **metadata,
                "rank": place,
                "points": float(points),
                "workbook_id": workbook_id,
                "workbook_sha256": workbook_sha256,
                "identity_boundary": "exact_normalized_published_spelling_not_verified_person",
            }
        )
    return rows, published_spellings, normalized_spellings


def _event_header(worksheet: Any) -> tuple[int, str] | None:
    for row_number, values in enumerate(
        worksheet.iter_rows(min_row=1, max_row=min(12, worksheet.max_row), values_only=True),
        start=1,
    ):
        padded = list(values) + [None] * 6
        second = _cell_text(padded[1]).casefold()
        third = _cell_text(padded[2]).casefold()
        has_name = "name" in third or "nazwisko" in third
        has_place = (
            "miejsce" in second
            or second == "place"
            or second.startswith("pos")
            or isinstance(padded[1], (int, float))
        )
        if has_name and has_place:
            layout = "international" if second.startswith("pos") else "domestic"
            return row_number, layout
    return None


def _international_name(surname_value: object, given_value: object) -> str:
    surname = _cell_text(surname_value)
    given = _cell_text(given_value)
    if not surname:
        return given
    if given and normalize_published_name(given) not in normalize_published_name(surname).split():
        return f"{surname} {given}"
    return surname


def _event_metadata(worksheet: Any, header_row: int) -> tuple[str, str, str | None]:
    descriptor_parts: list[str] = []
    source_url: str | None = None
    for values in worksheet.iter_rows(
        min_row=1,
        max_row=max(1, header_row - 1),
        values_only=True,
    ):
        for value in values:
            text = _cell_text(value)
            if not text:
                continue
            if text.casefold().startswith(("http://", "https://")):
                if text.casefold().startswith("https://"):
                    source_url = "https://" + text[8:]
                else:
                    source_url = "http://" + text[7:]
            elif len(descriptor_parts) < 8:
                descriptor_parts.append(text)
    descriptor = " | ".join(descriptor_parts)
    return descriptor, hashlib.sha256(descriptor.encode()).hexdigest(), source_url


def _normalize_country(value: object) -> str | None:
    text = _cell_text(value).replace("\xa0", "").strip().upper()
    return text or None


def _event_source_rows(
    worksheet: Any,
    header_row: int,
    layout: str,
    metadata: dict[str, str],
    workbook_id: str,
    workbook_sha256: str,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    descriptor, descriptor_sha256, source_url = _event_metadata(worksheet, header_row)
    event_group_id = _stable_id("spws-event", f"{workbook_id}:{worksheet.title}")
    rows: list[dict[str, Any]] = []
    published_spellings: set[str] = set()
    normalized_spellings: set[str] = set()
    for values in worksheet.iter_rows(min_row=header_row + 1, values_only=True):
        padded = list(values) + [None] * 6
        points = padded[0]
        place = _integer_place(padded[1])
        if layout == "international":
            name = _international_name(padded[2], padded[3])
            country = _normalize_country(padded[4])
        else:
            name = _cell_text(padded[2])
            country = _normalize_country(padded[5])
        if place is None or not name:
            continue
        if not isinstance(points, (int, float)):
            raise ValueError(
                f"SPWS event points are not numeric in {workbook_id}/{worksheet.title}"
            )
        athlete_id = _athlete_series_id(name)
        published_spellings.add(name)
        normalized_spellings.add(normalize_published_name(name))
        rows.append(
            {
                "event_group_id": event_group_id,
                "athlete_series_id": athlete_id,
                **metadata,
                "event_sheet": str(worksheet.title),
                "event_descriptor": descriptor,
                "event_descriptor_sha256": descriptor_sha256,
                "source_url": source_url,
                "place": place,
                "points": float(points),
                "country": country,
                "layout": layout,
                "workbook_id": workbook_id,
                "workbook_sha256": workbook_sha256,
                "identity_boundary": "exact_normalized_published_spelling_not_verified_person",
            }
        )
    return rows, published_spellings, normalized_spellings


def _collapse_event_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[
            (
                str(row["event_group_id"]),
                str(row["athlete_series_id"]),
                int(row["place"]),
            )
        ].append(row)

    rows: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        representative = members[0]
        point_values = sorted({float(row["points"]) for row in members})
        countries = sorted(
            {str(row["country"]) for row in members if row["country"] is not None}
        )
        rows.append(
            {
                "event_result_id": _stable_id("spws-event-result", json.dumps(key)),
                **{
                    field: representative[field]
                    for field in (
                        "event_group_id",
                        "athlete_series_id",
                        "weapon",
                        "gender",
                        "category",
                        "season",
                        "event_sheet",
                        "event_descriptor",
                        "event_descriptor_sha256",
                        "source_url",
                        "place",
                        "layout",
                        "workbook_id",
                        "workbook_sha256",
                        "identity_boundary",
                    )
                },
                "points_min": min(point_values),
                "points_max": max(point_values),
                "has_points_conflict": len(point_values) > 1,
                "country_codes": countries,
                "source_row_count": len(members),
            }
        )
    return rows


def _validate_summary(summary: dict[str, Any]) -> None:
    expected = {
        "xlsx_files": EXPECTED_XLSX_FILES,
        "unique_xlsx_hashes": EXPECTED_UNIQUE_XLSX_HASHES,
        "backup_xlsx_files": EXPECTED_BACKUP_XLSX_FILES,
        "canonical_workbooks": EXPECTED_CANONICAL_WORKBOOKS,
        "ranking_rows": EXPECTED_RANKING_ROWS,
        "ranking_normalized_spelling_keys": EXPECTED_RANKING_NORMALIZED_SPELLINGS,
        "raw_event_rows": EXPECTED_RAW_EVENT_ROWS,
        "event_rows": EXPECTED_EVENT_ROWS,
        "event_normalized_spelling_keys": EXPECTED_EVENT_NORMALIZED_SPELLINGS,
        "combined_normalized_spelling_keys": EXPECTED_COMBINED_NORMALIZED_SPELLINGS,
        "parsed_event_sheets": EXPECTED_PARSED_EVENT_SHEETS,
        "nonempty_event_sheets": EXPECTED_NONEMPTY_EVENT_SHEETS,
        "empty_event_sheets": EXPECTED_EMPTY_EVENT_SHEETS,
        "duplicate_event_source_rows": EXPECTED_DUPLICATE_EVENT_SOURCE_ROWS,
        "event_point_conflicts": EXPECTED_EVENT_POINT_CONFLICTS,
    }
    actual = {key: summary[key] for key in expected}
    if actual != expected:
        raise ValueError(f"SPWS snapshot counts changed: {actual} != {expected}")


def prepare(
    source_directory: Path,
    workbook_manifest_path: Path,
    ranking_output_path: Path,
    event_output_path: Path,
    summary_path: Path,
    *,
    archive_path: Path | None = None,
    validate_snapshot: bool = True,
) -> dict[str, Any]:
    all_xlsx = sorted(source_directory.rglob("*.xlsx"))
    xlsx_hashes = Counter(sha256_file(path) for path in all_xlsx)
    selected = canonical_workbooks(source_directory)

    ranking_rows: list[dict[str, Any]] = []
    event_source_rows: list[dict[str, Any]] = []
    workbook_rows: list[dict[str, Any]] = []
    ranking_published: set[str] = set()
    ranking_normalized: set[str] = set()
    event_published: set[str] = set()
    event_normalized: set[str] = set()
    parsed_event_sheets = 0
    nonempty_event_sheets = 0
    empty_event_sheets = 0

    for path, metadata in selected:
        relative = path.relative_to(source_directory).as_posix()
        workbook_id = _stable_id("spws-workbook", relative)
        workbook_sha256 = sha256_file(path)
        workbook = load_workbook(path, read_only=True, data_only=True)
        ranking_sheet_name = (
            "Ranking" if "Ranking" in workbook.sheetnames else workbook.sheetnames[0]
        )
        local_rankings, local_published, local_normalized = _ranking_rows(
            workbook[ranking_sheet_name], metadata, workbook_id, workbook_sha256
        )
        ranking_rows.extend(local_rankings)
        ranking_published.update(local_published)
        ranking_normalized.update(local_normalized)

        local_event_source_count = 0
        local_event_output_count = 0
        local_parsed_sheets = 0
        local_nonempty_sheets = 0
        for worksheet in workbook.worksheets:
            if worksheet.title == ranking_sheet_name or worksheet.title in _SKIP_EVENT_SHEETS:
                continue
            header = _event_header(worksheet)
            if header is None:
                raise ValueError(f"Unrecognized SPWS event sheet: {relative}/{worksheet.title}")
            header_row, layout = header
            source_rows, published, normalized = _event_source_rows(
                worksheet,
                header_row,
                layout,
                metadata,
                workbook_id,
                workbook_sha256,
            )
            parsed_event_sheets += 1
            local_parsed_sheets += 1
            if source_rows:
                nonempty_event_sheets += 1
                local_nonempty_sheets += 1
            else:
                empty_event_sheets += 1
            local_event_source_count += len(source_rows)
            event_source_rows.extend(source_rows)
            event_published.update(published)
            event_normalized.update(normalized)

        local_event_output_count = len(
            _collapse_event_rows(
                [row for row in event_source_rows if row["workbook_id"] == workbook_id]
            )
        )
        workbook_rows.append(
            {
                "workbook_id": workbook_id,
                **metadata,
                "bytes": path.stat().st_size,
                "sha256": workbook_sha256,
                "ranking_rows": len(local_rankings),
                "parsed_event_sheets": local_parsed_sheets,
                "nonempty_event_sheets": local_nonempty_sheets,
                "raw_event_rows": local_event_source_count,
                "event_rows_after_duplicate_collapse": local_event_output_count,
                "repository_path_id": _stable_id("spws-path", relative),
            }
        )

    event_rows = _collapse_event_rows(event_source_rows)
    ranking_frame = pd.DataFrame(ranking_rows)
    event_frame = pd.DataFrame(event_rows)
    for path, frame in ((ranking_output_path, ranking_frame), (event_output_path, event_frame)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    workbook_manifest_sha256 = write_jsonl(workbook_manifest_path, workbook_rows)

    ranking_weapons = dict(sorted(Counter(str(row["weapon"]) for row in ranking_rows).items()))
    event_weapons = dict(sorted(Counter(str(row["weapon"]) for row in event_rows).items()))
    ranking_seasons = dict(sorted(Counter(str(row["season"]) for row in ranking_rows).items()))
    event_seasons = dict(sorted(Counter(str(row["season"]) for row in event_rows).items()))
    ranking_genders = dict(sorted(Counter(str(row["gender"]) for row in ranking_rows).items()))
    event_genders = dict(sorted(Counter(str(row["gender"]) for row in event_rows).items()))
    combined_normalized = ranking_normalized | event_normalized

    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": archive_path.stat().st_size if archive_path is not None else None,
        "archive_sha256": sha256_file(archive_path) if archive_path is not None else None,
        "license_boundary": LICENSE_BOUNDARY,
        "xlsx_files": len(all_xlsx),
        "unique_xlsx_hashes": len(xlsx_hashes),
        "exact_duplicate_xlsx_hash_groups": sum(count > 1 for count in xlsx_hashes.values()),
        "exact_duplicate_xlsx_file_copies": sum(count - 1 for count in xlsx_hashes.values()),
        "backup_xlsx_files": sum(
            _is_backup(path.relative_to(source_directory)) for path in all_xlsx
        ),
        "canonical_workbooks": len(workbook_rows),
        "canonical_workbook_bytes": sum(int(row["bytes"]) for row in workbook_rows),
        "ranking_rows": len(ranking_rows),
        "ranking_published_spellings": len(ranking_published),
        "ranking_normalized_spelling_keys": len(ranking_normalized),
        "ranking_weapon_counts": ranking_weapons,
        "ranking_season_counts": ranking_seasons,
        "ranking_gender_counts": ranking_genders,
        "raw_event_rows": len(event_source_rows),
        "event_rows": len(event_rows),
        "duplicate_event_source_rows": len(event_source_rows) - len(event_rows),
        "event_point_conflicts": sum(bool(row["has_points_conflict"]) for row in event_rows),
        "event_published_spellings": len(event_published),
        "event_normalized_spelling_keys": len(event_normalized),
        "combined_normalized_spelling_keys": len(combined_normalized),
        "event_weapon_counts": event_weapons,
        "event_season_counts": event_seasons,
        "event_gender_counts": event_genders,
        "parsed_event_sheets": parsed_event_sheets,
        "nonempty_event_sheets": nonempty_event_sheets,
        "empty_event_sheets": empty_event_sheets,
        "workbook_manifest": "data/manifests/spws-ranking-workbooks.jsonl",
        "workbook_manifest_sha256": workbook_manifest_sha256,
        "ranking_output": "data/processed/spws-ranking-memberships.parquet",
        "ranking_output_sha256": sha256_file(ranking_output_path),
        "event_output": "data/processed/spws-event-results.parquet",
        "event_output_sha256": sha256_file(event_output_path),
        "claim_boundary": (
            "The 139 selected workbooks yield 1,075 final-ranking memberships and 12,033 "
            "duplicate-collapsed athlete-event placements across all three weapons. Stable IDs "
            "join only exact normalized published-name spellings and therefore represent 1,914 "
            "spelling keys, not verified unique people. Thirty backup workbooks, 15 SuperFive "
            "derivatives, audit fixtures, and redundant assets are not counted as independent "
            "athlete records. One event key has conflicting cached point values and remains "
            "flagged. The repository is public but does not provide a dataset license, so all "
            "athlete-level rows remain local."
        ),
    }
    if validate_snapshot:
        _validate_summary(summary)
    write_json(summary_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    source_directory: Path,
    workbook_manifest_path: Path,
    ranking_output_path: Path,
    event_output_path: Path,
    summary_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    source = acquire(
        archive_path,
        source_directory,
        user_agent=user_agent,
        force=force,
    )
    return prepare(
        source,
        workbook_manifest_path,
        ranking_output_path,
        event_output_path,
        summary_path,
        archive_path=archive_path,
    )
