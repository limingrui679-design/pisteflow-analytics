"""Link-only catalog of European Fencing Confederation event result pages."""

from __future__ import annotations

import re
import ssl
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import truststore

from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json, write_jsonl

SOURCE_URL = "https://www.fencing-efc.eu/results"
LICENSE = "No explicit reuse license found; factual link catalog only"
_LOCATION = re.compile(r"window\.location\s*=\s*'([^']+)'")
_DATE = re.compile(r"^(\d{2})/(\d{2})(?:\s*-\s*(\d{2})/(\d{2}))?/(\d{4})$")


def _text(node: Any) -> str:
    return " ".join(node.stripped_strings)


def _parse_dates(value: str) -> tuple[date, date]:
    compact = re.sub(r"\s+", " ", value.strip())
    match = _DATE.fullmatch(compact)
    if match is None:
        raise ValueError(f"Unexpected EFC date range: {value!r}")
    start_day, start_month, end_day, end_month, end_year = match.groups()
    if end_day is None or end_month is None:
        end_day, end_month = start_day, start_month
    year = int(end_year)
    start_year = year - 1 if int(start_month) > int(end_month) else year
    return (
        date(start_year, int(start_month), int(start_day)),
        date(year, int(end_month), int(end_day)),
    )


def parse_page(payload: bytes, page_number: int) -> list[dict[str, Any]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - optional collector dependency
        raise RuntimeError("Install pisteflow-analytics[collectors] to parse EFC pages") from exc

    soup = BeautifulSoup(payload, "html.parser")
    rows: list[dict[str, Any]] = []
    for row in soup.select(".tr.tr-link"):
        onclick = row.get("onclick")
        if not isinstance(onclick, str):
            raise ValueError("EFC result row is missing its navigation handler")
        location = _LOCATION.search(onclick)
        if location is None:
            raise ValueError("EFC result row is missing its event landing URL")
        landing_url = location.group(1)
        cells = []
        for child in row.find_all(recursive=False):
            classes = child.get("class")
            if isinstance(classes, list) and "td" in classes:
                cells.append(child)
        if len(cells) != 8:
            raise ValueError(f"EFC result row cell count changed: {len(cells)}")
        date_node = cells[2].find(class_="text-nowrap")
        if date_node is None:
            raise ValueError("EFC result row is missing its date range")
        start_date, end_date = _parse_dates(_text(date_node))
        weapon_raw = _text(cells[3])
        discipline_labels = [
            item.strip().casefold() for item in weapon_raw.split(",") if item.strip()
        ]
        accepted_labels = {"epee", "foil", "sabre", "other"}
        if not set(discipline_labels).issubset(accepted_labels):
            raise ValueError(f"Unknown EFC discipline labels: {discipline_labels}")
        weapons = [
            label for label in discipline_labels if label in {"epee", "foil", "sabre"}
        ]
        auxiliary_disciplines = [
            label for label in discipline_labels if label not in {"epee", "foil", "sabre"}
        ]
        download_value = cells[7].get("href")
        download = download_value if isinstance(download_value, str) else None
        rows.append(
            {
                "event_id": f"efc:{Path(urlparse(landing_url).path).name}",
                "name": _text(cells[0]),
                "city": _text(cells[1]) or None,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "weapons": weapons,
                "auxiliary_disciplines": auxiliary_disciplines,
                "category": _text(cells[5]) or None,
                "gender_icon_count": len(cells[4].find_all("svg")),
                "event_type_icon_count": len(cells[6].find_all("svg")),
                "landing_url": landing_url,
                "download_url": download,
                "catalog_page": page_number,
                "license_status": "link_only_no_explicit_reuse_license",
            }
        )
    if not rows:
        raise ValueError(f"EFC result catalog page {page_number} has no event rows")
    return rows


def _page_count(payload: bytes) -> int:
    pages = [int(value) for value in re.findall(rb"[?&]page=(\d+)", payload)]
    return max(pages, default=1)


def _fetch_page(client: httpx.Client, page: int, path: Path, *, force: bool) -> bytes:
    if path.exists() and not force:
        return path.read_bytes()
    url = SOURCE_URL if page == 1 else f"{SOURCE_URL}?page={page}"
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = client.get(url)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else float(2**attempt)
                time.sleep(min(delay, 30.0))
                continue
            response.raise_for_status()
            if b"class=\"tr tr-link" not in response.content:
                raise RuntimeError("EFC response no longer contains result rows")
            atomic_write_bytes(path, response.content)
            return response.content
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < 6:
                time.sleep(min(float(2**attempt), 30.0))
    raise RuntimeError(f"Could not fetch EFC result page {page}") from last_error


def fetch_and_prepare(
    cache_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(
        follow_redirects=True,
        trust_env=False,
        verify=ssl_context,
        timeout=httpx.Timeout(60, read=120),
        headers={"User-Agent": user_agent},
    ) as client:
        first_path = cache_directory / "page-0001.html"
        first = _fetch_page(client, 1, first_path, force=force)
        pages = _page_count(first)
        if not 60 <= pages <= 200:
            raise ValueError(f"Unexpected EFC result page count: {pages}")
        rows = parse_page(first, 1)
        page_paths = [first_path]
        for page in range(2, pages + 1):
            page_path = cache_directory / f"page-{page:04d}.html"
            payload = _fetch_page(client, page, page_path, force=force)
            rows.extend(parse_page(payload, page))
            page_paths.append(page_path)

    event_ids = [row["event_id"] for row in rows]
    if len(event_ids) != len(set(event_ids)):
        duplicates = [key for key, count in Counter(event_ids).items() if count > 1]
        raise ValueError(f"EFC event IDs overlap across pages: {duplicates[:3]}")
    rows.sort(key=lambda row: (row["start_date"], row["event_id"]))
    today = date.today().isoformat()
    weapon_memberships = Counter(weapon for row in rows for weapon in row["weapons"])
    report: dict[str, Any] = {
        "claim_boundary": (
            "A link-only catalog of public European Fencing Confederation event pages. The site "
            "offers result downloads but states no explicit data-reuse license, so PisteFlow "
            "publishes event-level factual links and aggregates only. It does not count events "
            "as bouts, athletes, or row-level results, and future calendar entries are separated "
            "from completed events. Gender and individual/team SVG icons are counted but not "
            "semantically decoded because the HTML provides no accessible labels."
        ),
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "catalog_pages": pages,
        "events": len(rows),
        "completed_or_started_events": sum(row["start_date"] <= today for row in rows),
        "future_events": sum(row["start_date"] > today for row in rows),
        "event_weapon_memberships": dict(sorted(weapon_memberships.items())),
        "events_with_auxiliary_disciplines": sum(
            bool(row["auxiliary_disciplines"]) for row in rows
        ),
        "auxiliary_discipline_memberships": dict(
            sorted(
                Counter(
                    discipline
                    for row in rows
                    for discipline in row["auxiliary_disciplines"]
                ).items()
            )
        ),
        "events_with_result_download_links": sum(row["download_url"] is not None for row in rows),
        "date_min": min(row["start_date"] for row in rows),
        "date_max": max(row["end_date"] for row in rows),
        "categories": dict(sorted(Counter(row["category"] or "unknown" for row in rows).items())),
        "page_sha256": {path.name: sha256_file(path) for path in page_paths},
        "manifest": str(manifest_path),
    }
    report["manifest_sha256"] = write_jsonl(manifest_path, rows)
    write_json(report_path, report)
    return report
