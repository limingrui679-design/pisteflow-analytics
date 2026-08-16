"""Local-only parsing of the public NCAA fencing results archive."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd
from bs4 import BeautifulSoup, Tag

from pisteflow.models import Weapon
from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json

ARCHIVE_URL = "https://ncaa.escrimeresults.com/"
RIGHTS_STATUS = "all rights reserved; local analysis only; do not redistribute raw pages"
_PAGE_PATTERN = re.compile(r"^(?:\./)?ncaa(\d{4})\.html$", re.IGNORECASE)
_EVENTS = {
    "WS": ("women", Weapon.SABRE),
    "WF": ("women", Weapon.FOIL),
    "WE": ("women", Weapon.EPEE),
    "MS": ("men", Weapon.SABRE),
    "MF": ("men", Weapon.FOIL),
    "ME": ("men", Weapon.EPEE),
}


def _download(
    client: httpx.Client,
    url: str,
    destination: Path,
    *,
    force: bool,
    attempts: int = 7,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else float(2**attempt)
                )
                time.sleep(min(delay, 45.0))
                continue
            response.raise_for_status()
            atomic_write_bytes(destination, response.content)
            return destination
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError(f"NCAA download failed after {attempts} attempts: {url}") from last_error


def discover_pages(root_html: str) -> list[tuple[int, str]]:
    soup = BeautifulSoup(root_html, "html.parser")
    pages: dict[int, str] = {}
    for anchor in soup.find_all("a", href=True):
        match = _PAGE_PATTERN.match(str(anchor["href"]))
        if not match:
            continue
        year = int(match.group(1))
        pages[year] = urljoin(ARCHIVE_URL, str(anchor["href"]))
    return sorted(pages.items())


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _table_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for table_row in table.find_all("tr"):
        cells = table_row.find_all(["td", "th"])
        values = [_clean(cell.get_text(" ", strip=True)) for cell in cells]
        if values:
            rows.append(values)
    return rows


def _event_heading(soup: BeautifulSoup, event_code: str) -> Tag | None:
    anchor = soup.find("a", attrs={"name": event_code})
    if isinstance(anchor, Tag):
        heading = anchor.find_parent("h2")
        if isinstance(heading, Tag):
            return heading
    return None


def _matrix_rows(heading: Tag) -> list[list[str]]:
    for table in heading.find_all_next("table", limit=5):
        rows = _table_rows(table)
        if not rows:
            continue
        header = rows[0]
        if {"School", "Name", "No", "V", "TS", "TR"}.issubset(header):
            # The 2007 export omits row-closing tags, so the built-in parser
            # leaves most cells outside parsed ``tr`` elements. Matrix cells
            # have a fixed width, so restore only those missing row bounds.
            cells = [
                _clean(cell.get_text(" ", strip=True))
                for cell in table.find_all(["td", "th"])
            ]
            if len(cells) > sum(map(len, rows)) and len(cells) % len(header) == 0:
                rows = [
                    cells[offset : offset + len(header)]
                    for offset in range(0, len(cells), len(header))
                ]
            return rows
    return []


def _athlete_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", normalized.casefold()).strip("-")


def _integer(value: str) -> int | None:
    match = re.search(r"[-+]?\d+", value)
    return int(match.group()) if match else None


def _score(value: str) -> tuple[bool | None, int | None]:
    cleaned = value.strip().upper()
    if not cleaned or cleaned in {"-", "X", "D", "A"}:
        return None, None
    victory = cleaned.startswith("V")
    score = _integer(cleaned)
    return victory, score


def parse_event_matrix(
    rows: list[list[str]],
    *,
    year: int,
    event_code: str,
    source_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize an NCAA pool matrix into athlete-season and unique-bout rows."""

    if not rows:
        return [], []
    header = rows[0]
    required = {"School", "Name", "No", "V", "TS", "TR"}
    if not required.issubset(header):
        raise ValueError(f"NCAA {year} {event_code} matrix has an unexpected header")
    gender, weapon = _EVENTS[event_code]
    positions = {name: header.index(name) for name in required}
    opponent_columns = {int(value): index for index, value in enumerate(header) if value.isdigit()}

    athlete_rows: list[dict[str, Any]] = []
    by_number: dict[int, dict[str, Any]] = {}
    raw_by_number: dict[int, list[str]] = {}
    for values in rows[1:]:
        if len(values) < len(header):
            values = [*values, *("" for _ in range(len(header) - len(values)))]
        number = _integer(values[positions["No"]])
        name = values[positions["Name"]]
        if number is None or not name:
            continue
        victories = _integer(values[positions["V"]])
        touches_scored = _integer(values[positions["TS"]])
        touches_received = _integer(values[positions["TR"]])
        athlete = {
            "athlete_season_id": f"ncaa:{year}:{event_code}:{number}",
            "athlete_key": _athlete_key(name),
            "name": name,
            "school": values[positions["School"]],
            "year": year,
            "event_code": event_code,
            "gender": gender,
            "weapon": weapon.value,
            "pool_number": number,
            "victories": victories,
            "pool_bouts": len(opponent_columns) - 1,
            "touches_scored": touches_scored,
            "touches_received": touches_received,
            "indicator": (
                touches_scored - touches_received
                if touches_scored is not None and touches_received is not None
                else None
            ),
            "source_url": source_url,
            "rights_status": RIGHTS_STATUS,
        }
        athlete_rows.append(athlete)
        by_number[number] = athlete
        raw_by_number[number] = values

    bout_rows: list[dict[str, Any]] = []
    for left_number in sorted(by_number):
        for right_number in sorted(by_number):
            if left_number >= right_number:
                continue
            left_values = raw_by_number[left_number]
            right_values = raw_by_number[right_number]
            left_column = opponent_columns.get(right_number)
            right_column = opponent_columns.get(left_number)
            if left_column is None or right_column is None:
                continue
            left_raw = left_values[left_column] if left_column < len(left_values) else ""
            right_raw = right_values[right_column] if right_column < len(right_values) else ""
            left_victory, left_score = _score(left_raw)
            right_victory, right_score = _score(right_raw)
            winner: str | None = None
            conflict = False
            if left_victory is True and right_victory is not True:
                winner = "left"
            elif right_victory is True and left_victory is not True:
                winner = "right"
            elif left_victory is not None or right_victory is not None:
                conflict = True
            bout_rows.append(
                {
                    "bout_id": f"ncaa:{year}:{event_code}:{left_number}:{right_number}",
                    "year": year,
                    "event_code": event_code,
                    "gender": gender,
                    "weapon": weapon.value,
                    "left_athlete_season_id": by_number[left_number]["athlete_season_id"],
                    "left_athlete_key": by_number[left_number]["athlete_key"],
                    "left_name": by_number[left_number]["name"],
                    "left_school": by_number[left_number]["school"],
                    "right_athlete_season_id": by_number[right_number]["athlete_season_id"],
                    "right_athlete_key": by_number[right_number]["athlete_key"],
                    "right_name": by_number[right_number]["name"],
                    "right_school": by_number[right_number]["school"],
                    "left_score": left_score,
                    "right_score": right_score,
                    "winner": winner,
                    "parse_conflict": conflict,
                    "source_url": source_url,
                    "rights_status": RIGHTS_STATUS,
                }
            )
    return athlete_rows, bout_rows


def parse_page(
    path: Path,
    *,
    year: int,
    source_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    athletes: list[dict[str, Any]] = []
    bouts: list[dict[str, Any]] = []
    missing_events: list[str] = []
    for event_code in _EVENTS:
        heading = _event_heading(soup, event_code)
        rows = _matrix_rows(heading) if heading is not None else []
        if not rows:
            missing_events.append(event_code)
            continue
        event_athletes, event_bouts = parse_event_matrix(
            rows,
            year=year,
            event_code=event_code,
            source_url=source_url,
        )
        athletes.extend(event_athletes)
        bouts.extend(event_bouts)
    return athletes, bouts, missing_events


def _counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    materialized = list(rows)
    values = sorted({str(row[key]) for row in materialized})
    return {value: sum(str(row[key]) == value for row in materialized) for value in values}


def fetch_and_prepare(
    cache_directory: Path,
    athlete_output: Path,
    bout_output: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch every listed championship page and build local Parquet tables."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=180),
        headers={"User-Agent": user_agent},
    ) as client:
        root_path = _download(
            client,
            ARCHIVE_URL,
            cache_directory / "index.html",
            force=force,
        )
        pages = discover_pages(root_path.read_text(encoding="utf-8", errors="replace"))
        all_athletes: list[dict[str, Any]] = []
        all_bouts: list[dict[str, Any]] = []
        page_reports: list[dict[str, Any]] = []
        for year, source_url in pages:
            page_path = _download(
                client,
                source_url,
                cache_directory / f"ncaa{year}.html",
                force=force,
            )
            athletes, bouts, missing_events = parse_page(
                page_path,
                year=year,
                source_url=source_url,
            )
            all_athletes.extend(athletes)
            all_bouts.extend(bouts)
            page_reports.append(
                {
                    "year": year,
                    "url": source_url,
                    "raw_sha256": sha256_file(page_path),
                    "athlete_seasons": len(athletes),
                    "pool_bouts": len(bouts),
                    "missing_events": missing_events,
                }
            )

    athlete_frame = pd.DataFrame(all_athletes).sort_values(
        ["year", "event_code", "pool_number"], kind="stable"
    )
    bout_frame = pd.DataFrame(all_bouts).sort_values(
        ["year", "event_code", "bout_id"], kind="stable"
    )
    if athlete_frame.empty or bout_frame.empty:
        raise RuntimeError("No NCAA athlete or bout rows were parsed")
    athlete_output.parent.mkdir(parents=True, exist_ok=True)
    bout_output.parent.mkdir(parents=True, exist_ok=True)
    athlete_frame.to_parquet(athlete_output, index=False)
    bout_frame.to_parquet(bout_output, index=False)

    completed = bout_frame[bout_frame["winner"].notna()]
    summary = {
        "claim_boundary": (
            "Public NCAA championship score matrices parsed for local historical analysis. "
            "The source page states all rights reserved; raw pages and row-level tables are not "
            "redistributed, and predictive associations are not causal athlete evaluations."
        ),
        "source_url": ARCHIVE_URL,
        "rights_status": RIGHTS_STATUS,
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "seasons": len(page_reports),
        "year_min": min(report["year"] for report in page_reports),
        "year_max": max(report["year"] for report in page_reports),
        "athlete_seasons": len(athlete_frame),
        "unique_normalized_athletes": int(athlete_frame["athlete_key"].nunique()),
        "pool_bouts": len(bout_frame),
        "completed_pool_bouts": len(completed),
        "parse_conflicts": int(bout_frame["parse_conflict"].sum()),
        "by_weapon_athlete_seasons": _counts(all_athletes, "weapon"),
        "by_weapon_pool_bouts": _counts(all_bouts, "weapon"),
        "athlete_table_sha256": sha256_file(athlete_output),
        "bout_table_sha256": sha256_file(bout_output),
        "pages": page_reports,
        "outputs": {
            "athletes": str(athlete_output),
            "bouts": str(bout_output),
        },
    }
    write_json(report_path, summary)
    return summary
