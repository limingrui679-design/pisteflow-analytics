"""Prepare restricted regional tournament ledgers without publishing personal rows."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

REPOSITORY_URL = "https://github.com/aidanKnox/Fencing-Data-Analytics-Project"
SNAPSHOT_COMMIT = "6c86428686ca4b4842d4e0dfeb4e946208832c1d"
SNAPSHOT_FILES = 297
SNAPSHOT_BYTES = 213_343
SNAPSHOT_FINGERPRINT = "7fccf8d3ac0cd7db83cd907b61cf9154e735a1673837b68af55e4d92f4dc2281"

EXPECTED_LEDGER_FILES = 148
EXPECTED_ROSTER_FILES = 148
EXPECTED_SCORED_BOUTS = 3_728
EXPECTED_WALKOVERS = 118
EXPECTED_TIED_SCORE_ANOMALIES = 5
EXPECTED_PLACEHOLDER_LINES = 346
EXPECTED_ROSTER_MEMBERSHIPS = 4_083
EXPECTED_RAW_NAME_SPELLINGS = 884
EXPECTED_SOURCE_IDENTITY_KEYS = 882
EXPECTED_PROCESSED_IDENTITY_KEYS = 882
EXPECTED_DATED_EVENTS = 135

MONTH_PATTERN = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(20\d{2})\b"
)
EVENT_ID_PATTERN = re.compile(r"^(\d{3})\s")

RIGHTS_BOUNDARY = (
    "The repository's Portfolio License permits personal and educational use but prohibits "
    "redistribution without permission and explicitly notes real individuals. Source CSVs, names, "
    "pseudonymous row tables, and trained artifacts stay local. Public outputs are limited to "
    "code, snapshot hashes, aggregate counts, and data-quality findings."
)


def _identity_key(name: str) -> str:
    normalized = " ".join(name.casefold().split())
    return hashlib.sha256(f"regional-ledger:{normalized}".encode()).hexdigest()


def _event_id(path: Path) -> str:
    match = EVENT_ID_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Tournament file lacks a three-digit event ID: {path.name}")
    return match.group(1)


def _event_month(title: str) -> str | None:
    match = MONTH_PATTERN.match(title)
    if match is None:
        return None
    value = datetime.strptime(" ".join(match.groups()), "%b %Y")
    return value.strftime("%Y-%m")


def _gender_scope(title: str) -> str:
    normalized = title.casefold()
    if "women's" in normalized:
        return "female"
    if "men's" in normalized:
        return "male"
    return "mixed_or_unspecified"


def _level_tags(title: str) -> list[str]:
    normalized = title.casefold()
    candidates = {
        "open": "open",
        "div2": "division_2",
        "div3": "division_3",
        "div4": "division_4",
        "unrated": "unrated",
        "veteran": "veteran",
        "practice": "practice",
        "selection": "selection",
        "walk & roll": "walk_and_roll",
    }
    return [value for token, value in candidates.items() if token in normalized]


def parse_ledger_lines(
    lines: list[str],
    *,
    event_id: str,
    source_file: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse scored and walkover rows while counting placeholders and tied anomalies."""

    if not lines:
        raise ValueError(f"Empty tournament ledger: {source_file}")
    title = lines[0].strip()
    rows: list[dict[str, Any]] = []
    audit = Counter[str]()
    for line_number, raw_line in enumerate(lines[1:], start=2):
        if not raw_line.strip() or raw_line.strip() == "/":
            audit["placeholder_lines"] += 1
            continue
        parts = [part.strip() for part in next(csv.reader([raw_line]))]
        if len(parts) != 4:
            audit["malformed_lines"] += 1
            continue
        left_name, right_name, left_token, right_token = parts
        base: dict[str, Any] = {
            "event_id": event_id,
            "event_title": title,
            "event_month": _event_month(title),
            "gender_scope": _gender_scope(title),
            "level_tags": _level_tags(title),
            "weapon": "unknown",
            "left_athlete_key": _identity_key(left_name),
            "right_athlete_key": _identity_key(right_name),
            "source_file": source_file,
            "source_line": line_number,
        }
        if left_token == ">" or right_token == ">":
            if left_token == right_token:
                audit["malformed_lines"] += 1
                continue
            left_won = left_token == ">"
            rows.append(
                {
                    **base,
                    "outcome_kind": "walkover",
                    "left_score": None,
                    "right_score": None,
                    "winner_athlete_key": (
                        base["left_athlete_key"] if left_won else base["right_athlete_key"]
                    ),
                }
            )
            audit["walkovers"] += 1
            continue
        try:
            left_score = int(left_token)
            right_score = int(right_token)
        except ValueError:
            audit["malformed_lines"] += 1
            continue
        if left_score == right_score:
            audit["tied_score_anomalies"] += 1
            continue
        rows.append(
            {
                **base,
                "outcome_kind": "scored",
                "left_score": left_score,
                "right_score": right_score,
                "winner_athlete_key": (
                    base["left_athlete_key"]
                    if left_score > right_score
                    else base["right_athlete_key"]
                ),
            }
        )
        audit["scored_bouts"] += 1
    return rows, dict(audit)


def _parse_roster(path: Path) -> tuple[str, list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"Empty tournament roster: {path.name}")
    header = [part.strip() for part in next(csv.reader([lines[0]]))]
    title = header[0].removesuffix(" Fencers")
    event_id = _event_id(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines[1:], start=2):
        if not raw_line.strip() or raw_line.strip() == ",":
            continue
        parts = [part.strip() for part in next(csv.reader([raw_line]))]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Unexpected roster row: {path.name}:{line_number}")
        name, rating = parts[:2]
        author_score = parts[2] if len(parts) == 3 and parts[2] else None
        rows.append(
            {
                "event_id": event_id,
                "event_title": title,
                "event_month": _event_month(title),
                "gender_scope": _gender_scope(title),
                "level_tags": _level_tags(title),
                "weapon": "unknown",
                "athlete_key": _identity_key(name),
                "rating_raw": rating,
                "author_score_raw": author_score,
                "source_file": path.name,
                "source_line": line_number,
            }
        )
    return title, rows


def _snapshot_fingerprint(repository: Path) -> tuple[int, int, str]:
    selected = sorted(
        [
            repository / "LICENSE.md",
            *repository.joinpath("res/Ledgers").glob("*.csv"),
            *repository.joinpath("res/Rosters").glob("*.csv"),
        ]
    )
    manifest = [
        {
            "path": str(path.relative_to(repository)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in selected
    ]
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return (
        len(selected),
        sum(path.stat().st_size for path in selected),
        hashlib.sha256(payload).hexdigest(),
    )


def prepare(
    repository: Path,
    bouts_path: Path,
    rosters_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Parse the pinned restricted snapshot into local pseudonymous tables."""

    observed_snapshot = _snapshot_fingerprint(repository)
    expected_snapshot = (SNAPSHOT_FILES, SNAPSHOT_BYTES, SNAPSHOT_FINGERPRINT)
    if observed_snapshot != expected_snapshot:
        raise ValueError(f"Regional tournament snapshot changed: {observed_snapshot}")

    ledger_paths = sorted(repository.joinpath("res/Ledgers").glob("*.csv"))
    roster_paths = sorted(repository.joinpath("res/Rosters").glob("*.csv"))
    if len(ledger_paths) != EXPECTED_LEDGER_FILES or len(roster_paths) != EXPECTED_ROSTER_FILES:
        raise ValueError("Regional tournament source-file counts changed")

    bout_rows: list[dict[str, Any]] = []
    audit = Counter[str]()
    event_titles: dict[str, str] = {}
    for path in ledger_paths:
        event_id = _event_id(path)
        rows, next_audit = parse_ledger_lines(
            path.read_text(encoding="utf-8-sig").splitlines(),
            event_id=event_id,
            source_file=path.name,
        )
        if rows:
            event_titles[event_id] = str(rows[0]["event_title"])
        else:
            event_titles[event_id] = path.read_text(encoding="utf-8-sig").splitlines()[0]
        bout_rows.extend(rows)
        audit.update(next_audit)

    roster_rows: list[dict[str, Any]] = []
    for path in roster_paths:
        _title, rows = _parse_roster(path)
        roster_rows.extend(rows)

    processed_identity_keys = {
        str(row["athlete_key"]) for row in roster_rows
    } | {
        str(row[key])
        for row in bout_rows
        for key in ("left_athlete_key", "right_athlete_key")
    }
    source_identity_keys = {str(row["athlete_key"]) for row in roster_rows}
    raw_source_names: set[str] = set()
    for path in roster_paths:
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
            if raw_line.strip() and raw_line.strip() != ",":
                raw_source_names.add(next(csv.reader([raw_line]))[0].strip())
    for path in ledger_paths:
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
            parts = [part.strip() for part in next(csv.reader([raw_line]))]
            if len(parts) == 4:
                raw_source_names.update(parts[:2])
                source_identity_keys.update((_identity_key(parts[0]), _identity_key(parts[1])))
    observed = {
        "ledger_files": len(ledger_paths),
        "roster_files": len(roster_paths),
        "scored_bouts": audit["scored_bouts"],
        "walkovers": audit["walkovers"],
        "tied_score_anomalies": audit["tied_score_anomalies"],
        "placeholder_lines": audit["placeholder_lines"],
        "malformed_lines_other_than_ties": audit["malformed_lines"],
        "roster_memberships": len(roster_rows),
        "raw_name_spellings": len(raw_source_names),
        "source_identity_normalized_keys": len(source_identity_keys),
        "processed_identity_normalized_keys": len(processed_identity_keys),
        "dated_events": sum(_event_month(title) is not None for title in event_titles.values()),
    }
    expected = {
        "ledger_files": EXPECTED_LEDGER_FILES,
        "roster_files": EXPECTED_ROSTER_FILES,
        "scored_bouts": EXPECTED_SCORED_BOUTS,
        "walkovers": EXPECTED_WALKOVERS,
        "tied_score_anomalies": EXPECTED_TIED_SCORE_ANOMALIES,
        "placeholder_lines": EXPECTED_PLACEHOLDER_LINES,
        "malformed_lines_other_than_ties": 0,
        "roster_memberships": EXPECTED_ROSTER_MEMBERSHIPS,
        "raw_name_spellings": EXPECTED_RAW_NAME_SPELLINGS,
        "source_identity_normalized_keys": EXPECTED_SOURCE_IDENTITY_KEYS,
        "processed_identity_normalized_keys": EXPECTED_PROCESSED_IDENTITY_KEYS,
        "dated_events": EXPECTED_DATED_EVENTS,
    }
    if observed != expected:
        raise ValueError(f"Regional tournament parsed counts changed: {observed}")

    bouts = pd.DataFrame(bout_rows)
    rosters = pd.DataFrame(roster_rows)
    bouts_path.parent.mkdir(parents=True, exist_ok=True)
    bouts.to_parquet(bouts_path, index=False)
    rosters.to_parquet(rosters_path, index=False)
    event_months = {
        event_id: month
        for event_id, title in event_titles.items()
        if (month := _event_month(title)) is not None
    }
    summary: dict[str, Any] = {
        **observed,
        "events": len(event_titles),
        "events_by_year": dict(
            sorted(
                Counter(
                    month.split("-", 1)[0]
                    for month in event_months.values()
                ).items()
            )
        ),
        "scored_bouts_by_winning_score": dict(
            sorted(
                Counter(
                    max(int(row["left_score"]), int(row["right_score"]))
                    for row in bout_rows
                    if row["outcome_kind"] == "scored"
                ).items()
            )
        ),
        "weapon_status": "unknown_not_in_source_files",
        "repository_url": REPOSITORY_URL,
        "snapshot_commit": SNAPSHOT_COMMIT,
        "snapshot_files": SNAPSHOT_FILES,
        "snapshot_bytes": SNAPSHOT_BYTES,
        "snapshot_fingerprint": SNAPSHOT_FINGERPRINT,
        "bouts_sha256": sha256_file(bouts_path),
        "rosters_sha256": sha256_file(rosters_path),
        "rights_boundary": RIGHTS_BOUNDARY,
        "claim_boundary": (
            "Restricted regional direct-elimination ledgers from 148 event files, primarily "
            "2024-2026. The repository license says the names are real, but the source does not "
            "state a weapon; every row therefore remains weapon-unknown and is excluded from "
            "three-weapon comparisons. Five tied-score rows are retained only as audit anomalies."
        ),
    }
    write_json(summary_path, summary)
    return summary
