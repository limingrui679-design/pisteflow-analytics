"""Privacy-minimized parser for the official Paris 2024 fencing results book."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

SOURCE_URL = (
    "https://olympics.com/OG2024/pdf/OG2024/FEN/"
    "OG2024_FEN_B99_FEN-------------------------------.pdf"
)
ARCHIVED_SOURCE_URL = (
    "https://web.archive.org/web/20240805040252id_/"
    "https://olympics.com/OG2024/pdf/OG2024/FEN/"
    "OG2024_FEN_B99_FEN-------------------------------.pdf"
)
LICENSE = "No explicit data-reuse license found; row tables remain local"
EXPECTED_PAGES = 365
EXPECTED_INDIVIDUAL_BOUTS = 212
EXPECTED_TEAM_MATCHES = 72
EXPECTED_TEAM_RELAYS = 648

_FOOTER = re.compile(
    r"\b(?P<report>FEN(?P<gender>[WM])(?P<team>TEAM)?"
    r"(?P<weapon>EPEE|FOIL|SABR|SABRE)-+(?P<stage>[A-Z0-9-]+)_73(?P<kind>[IT]))\b"
)
_DATE = re.compile(r"\b(?:MON|TUE|WED|THU|FRI|SAT|SUN) \d{1,2} [A-Z]{3} 2024\b")
_INDIVIDUAL_ROW = re.compile(
    r"^\s*(?P<left_seed>\d+)\s+(?P<left_name>.+?)\s+(?P<left_noc>[A-Z]{3})\s+"
    r"(?P<left_score>\d+)\s+-\s+(?P<right_score>\d+)\s+"
    r"(?P<right_seed>\d+)\s+(?P<right_name>.+?)\s+(?P<right_noc>[A-Z]{3})\s*$"
)
_TEAM_ROW = re.compile(
    r"^\s*(?P<left_noc>[A-Z]{3})\s+-\s+.+?\s+\((?P<left_seed>\d+)\)\s+"
    r"(?P<left_score>\d+)\s+Final\s+(?P<right_score>\d+)\s+"
    r"(?P<right_noc>[A-Z]{3})\s+-\s+.+?\s+\((?P<right_seed>\d+)\)\s*$"
)
_RELAY_ROW = re.compile(
    r"^\s*(?P<left_name>.+?)\s+(?:P-[yrb]\s+)?(?P<left_touches>\d+)\s+"
    r"(?P<left_score>\d+)\s+(?P<separator>--|OT)\s+(?P<right_score>\d+)\s+"
    r"(?P<right_touches>\d+)\s+(?P<right_name>.+?)(?:\s+P-[yrb])?\s*$"
)
_DURATION = re.compile(
    r"(?:Bout|Match) Duration:\s*(?P<duration>\d+:\d+)\s+Overtime:\s*"
    r"(?P<overtime>--|\d+:\d+)"
)


def _seconds(value: str) -> int | None:
    if value == "--":
        return None
    minutes, seconds = (int(part) for part in value.split(":"))
    return minutes * 60 + seconds


def _athlete_id(name: str, noc: str) -> str:
    normalized = " ".join(name.replace("(S)", "").split()).casefold()
    return hashlib.sha256(f"paris2024|{noc}|{normalized}".encode()).hexdigest()[:20]


def _event_fields(text: str, page_number: int) -> dict[str, Any]:
    footer = _FOOTER.search(text)
    if footer is None:
        raise ValueError(f"Results page {page_number} has no recognized report code")
    date_match = _DATE.search(text)
    if date_match is None:
        raise ValueError(f"Results page {page_number} has no competition date")
    weapon_token = footer.group("weapon")
    weapon = {"EPEE": "epee", "FOIL": "foil", "SABR": "sabre", "SABRE": "sabre"}[
        weapon_token
    ]
    gender = {"W": "women", "M": "men"}[footer.group("gender")]
    team = footer.group("team") is not None
    report_code = footer.group("report")
    stage_code = footer.group("stage")
    return {
        "event_id": f"paris2024:{gender}:{weapon}:{'team' if team else 'individual'}",
        "date": datetime.strptime(date_match.group(), "%a %d %b %Y").date().isoformat(),
        "gender": gender,
        "weapon": weapon,
        "format": "team" if team else "individual",
        "stage": _stage(stage_code, team=team),
        "stage_code": stage_code.rstrip("-"),
        "report_code": report_code,
        "source_page": page_number,
    }


def _stage(stage_code: str, *, team: bool) -> str:
    if stage_code.startswith("R64"):
        return "table_of_64"
    if stage_code.startswith("R32"):
        return "table_of_32"
    if stage_code.startswith("8FNL"):
        return "table_of_16"
    if stage_code.startswith("QFNL"):
        return "quarterfinal"
    if stage_code.startswith("SF5"):
        return "classification_5_8_semifinal"
    if stage_code.startswith("SFNL"):
        return "semifinal"
    if stage_code.startswith("FNL"):
        if "000100" in stage_code:
            return "gold_medal"
        if "000200" in stage_code:
            return "bronze_medal"
        if team and "000300" in stage_code:
            return "fifth_place"
        if team and "000400" in stage_code:
            return "seventh_place"
    raise ValueError(f"Unknown Paris 2024 fencing stage code: {stage_code}")


def parse_individual_page(text: str, page_number: int) -> dict[str, Any]:
    fields = _event_fields(text, page_number)
    if fields["format"] != "individual" or "Bout Results" not in text:
        raise ValueError(f"Page {page_number} is not an individual bout result")
    rows = [
        match.groupdict()
        for line in text.splitlines()
        if (match := _INDIVIDUAL_ROW.match(line))
    ]
    if len(rows) != 1:
        raise ValueError(f"Page {page_number} has {len(rows)} individual score rows")
    result = rows[0]
    bout = re.search(r"BOUT:\s*(\d+)\s+PISTE:\s*([^\n]+)", text)
    duration = _DURATION.search(text)
    if bout is None or duration is None:
        raise ValueError(f"Page {page_number} is missing bout metadata")
    left_score = int(result["left_score"])
    right_score = int(result["right_score"])
    if left_score == right_score:
        raise ValueError(f"Page {page_number} has a tied completed individual bout")
    return {
        **fields,
        "bout_number": int(bout.group(1)),
        "piste": bout.group(2).strip(),
        "left_athlete_id": _athlete_id(result["left_name"], result["left_noc"]),
        "left_noc": result["left_noc"],
        "left_seed": int(result["left_seed"]),
        "left_score": left_score,
        "right_athlete_id": _athlete_id(result["right_name"], result["right_noc"]),
        "right_noc": result["right_noc"],
        "right_seed": int(result["right_seed"]),
        "right_score": right_score,
        "winner_side": "left" if left_score > right_score else "right",
        "score_margin": abs(left_score - right_score),
        "duration_seconds": _seconds(duration.group("duration")),
        "overtime_seconds": _seconds(duration.group("overtime")),
    }


def parse_team_page(text: str, page_number: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields = _event_fields(text, page_number)
    if fields["format"] != "team" or "Match Results" not in text:
        raise ValueError(f"Page {page_number} is not a team match result")
    teams = [match.groupdict() for line in text.splitlines() if (match := _TEAM_ROW.match(line))]
    if len(teams) != 1:
        raise ValueError(f"Page {page_number} has {len(teams)} team score rows")
    result = teams[0]
    match_meta = re.search(r"MATCH:\s*(\d+)\s+PISTE:\s*([^\n]+)", text)
    duration = _DURATION.search(text)
    if match_meta is None or duration is None:
        raise ValueError(f"Page {page_number} is missing team-match metadata")
    left_score = int(result["left_score"])
    right_score = int(result["right_score"])
    if left_score == right_score:
        raise ValueError(f"Page {page_number} has a tied completed team match")
    match_row = {
        **fields,
        "match_number": int(match_meta.group(1)),
        "piste": match_meta.group(2).strip(),
        "left_noc": result["left_noc"],
        "left_seed": int(result["left_seed"]),
        "left_score": left_score,
        "right_noc": result["right_noc"],
        "right_seed": int(result["right_seed"]),
        "right_score": right_score,
        "winner_side": "left" if left_score > right_score else "right",
        "score_margin": abs(left_score - right_score),
        "duration_seconds": _seconds(duration.group("duration")),
        "overtime_seconds": _seconds(duration.group("overtime")),
    }
    try:
        relay_block = text.split("Fencer", 1)[1].split("Match Duration:", 1)[0]
    except IndexError as exc:
        raise ValueError(f"Page {page_number} has no relay table") from exc
    relays = [
        match.groupdict()
        for line in relay_block.splitlines()
        if (match := _RELAY_ROW.match(line))
    ]
    if len(relays) != 9:
        raise ValueError(f"Page {page_number} has {len(relays)} team relays")
    relay_rows: list[dict[str, Any]] = []
    for index, relay in enumerate(relays, start=1):
        left_name = re.sub(r"\s+P-[yrb]\s*$", "", relay["left_name"]).strip()
        right_name = re.sub(r"\s+P-[yrb]\s*$", "", relay["right_name"]).strip()
        relay_rows.append(
            {
                "event_id": fields["event_id"],
                "date": fields["date"],
                "gender": fields["gender"],
                "weapon": fields["weapon"],
                "stage": fields["stage"],
                "report_code": fields["report_code"],
                "source_page": page_number,
                "relay_number": index,
                "left_noc": result["left_noc"],
                "left_athlete_id": _athlete_id(left_name, result["left_noc"]),
                "left_substitute": "(S)" in left_name,
                "left_touches": int(relay["left_touches"]),
                "left_cumulative_score": int(relay["left_score"]),
                "right_noc": result["right_noc"],
                "right_athlete_id": _athlete_id(right_name, result["right_noc"]),
                "right_substitute": "(S)" in right_name,
                "right_touches": int(relay["right_touches"]),
                "right_cumulative_score": int(relay["right_score"]),
                "overtime_relay": relay["separator"] == "OT",
            }
        )
    if relay_rows[-1]["left_cumulative_score"] != left_score or relay_rows[-1][
        "right_cumulative_score"
    ] != right_score:
        raise ValueError(f"Page {page_number} relay totals do not match the team score")
    return match_row, relay_rows


def prepare_pdf(
    pdf_path: Path,
    individual_path: Path,
    team_match_path: Path,
    relay_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install pypdf to parse the Paris 2024 results book") from exc

    reader = PdfReader(pdf_path)
    if len(reader.pages) != EXPECTED_PAGES:
        raise ValueError(f"Unexpected Paris 2024 results-book page count: {len(reader.pages)}")
    individual_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    relay_rows: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or ""
        if "Bout Results" in text:
            individual_rows.append(parse_individual_page(text, page_number))
        elif "Match Results" in text:
            team, relays = parse_team_page(text, page_number)
            team_rows.append(team)
            relay_rows.extend(relays)

    observed = (len(individual_rows), len(team_rows), len(relay_rows))
    expected = (EXPECTED_INDIVIDUAL_BOUTS, EXPECTED_TEAM_MATCHES, EXPECTED_TEAM_RELAYS)
    if observed != expected:
        raise ValueError(
            f"Paris 2024 result counts changed: observed={observed}, expected={expected}"
        )
    report_codes = [row["report_code"] for row in individual_rows + team_rows]
    duplicates = [code for code, count in Counter(report_codes).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate Paris 2024 result reports: {duplicates[:3]}")

    for path, rows in (
        (individual_path, individual_rows),
        (team_match_path, team_rows),
        (relay_path, relay_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    event_counts = Counter(row["event_id"] for row in individual_rows + team_rows)
    unique_athletes = {
        row[key]
        for row in individual_rows + relay_rows
        for key in ("left_athlete_id", "right_athlete_id")
    }
    report: dict[str, Any] = {
        "claim_boundary": (
            "These are internally parsed official Paris 2024 result records, not live scoring, "
            "video-derived actions, or coaching outcomes. Athlete names are converted to stable "
            "event-scoped pseudonyms and never written to processed tables. The source PDF has "
            "no explicit reuse license, so row-level Parquet remains ignored/local; the repository "
            "publishes parsing code, source hash, counts, and aggregate validation only."
        ),
        "source_url": SOURCE_URL,
        "archived_source_url": ARCHIVED_SOURCE_URL,
        "license": LICENSE,
        "source_title": "Official Fencing Results Book - V 1.0 - 04-AUG-2024",
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_pdf_sha256": sha256_file(pdf_path),
        "source_pages": len(reader.pages),
        "individual_bouts": len(individual_rows),
        "team_matches": len(team_rows),
        "team_relays": len(relay_rows),
        "competition_observations": len(individual_rows) + len(team_rows) + len(relay_rows),
        "unique_event_scoped_athlete_ids": len(unique_athletes),
        "events": len(event_counts),
        "event_result_counts": dict(sorted(event_counts.items())),
        "weapons": dict(
            sorted(Counter(row["weapon"] for row in individual_rows + team_rows).items())
        ),
        "genders": dict(
            sorted(Counter(row["gender"] for row in individual_rows + team_rows).items())
        ),
        "overtime_individual_bouts": sum(
            row["overtime_seconds"] is not None for row in individual_rows
        ),
        "overtime_team_matches": sum(row["overtime_seconds"] is not None for row in team_rows),
        "overtime_team_relays": sum(row["overtime_relay"] for row in relay_rows),
        "local_outputs": {
            "individual_bouts": str(individual_path),
            "team_matches": str(team_match_path),
            "team_relays": str(relay_path),
        },
        "local_output_sha256": {
            "individual_bouts": sha256_file(individual_path),
            "team_matches": sha256_file(team_match_path),
            "team_relays": sha256_file(relay_path),
        },
    }
    write_json(report_path, report)
    return report
