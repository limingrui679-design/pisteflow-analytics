"""Public-catalog, link-only collector for FencingTV."""

from __future__ import annotations

import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import HttpUrl

from pisteflow.models import LicenseStatus, SourceKind, VideoRecord
from pisteflow.provenance import write_jsonl
from pisteflow.sources.classification import (
    classify_content,
    infer_weapon_candidates,
    primary_weapon,
)

BASE_URL = "https://fencingtv.com"
CATALOG_URL = f"{BASE_URL}/api/video/videos"


def _get_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, int],
    attempts: int = 6,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 45.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("FencingTV response is not an object")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 45.0))
    message = f"FencingTV catalog request failed after {attempts} attempts"
    raise RuntimeError(message) from last_error


def fetch_public_catalog(
    output_path: Path,
    *,
    user_agent: str,
    page_size: int = 1_000,
) -> list[VideoRecord]:
    """Fetch public, active catalog rows; never request playback URLs."""

    discovered_at = datetime.now(UTC)
    raw_rows: list[dict[str, Any]] = []
    offset = 0
    query_count: int | None = None
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=120),
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    ) as client:
        while query_count is None or offset < query_count:
            payload = _get_json(
                client,
                CATALOG_URL,
                params={"limit": page_size, "offset": offset},
            )
            page = payload.get("data", [])
            metadata = payload.get("metadata", {})
            if not isinstance(page, list):
                raise ValueError("FencingTV response data is not a list")
            if isinstance(metadata, dict) and isinstance(metadata.get("queryCount"), int):
                query_count = int(metadata["queryCount"])
            raw_rows.extend(row for row in page if isinstance(row, dict))
            if not page:
                break
            offset += len(page)
            if len(page) < page_size and (query_count is None or offset >= query_count):
                break

    records: list[VideoRecord] = []
    seen: set[str] = set()
    for row in raw_rows:
        identifier = str(row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        if (
            not identifier
            or not title
            or identifier in seen
            or row.get("visibility") != "public"
            or row.get("status") != "active"
        ):
            continue
        seen.add(identifier)
        tags = row.get("tags", [])
        tags = tags if isinstance(tags, list) else []
        athletes = [tag for tag in tags if isinstance(tag, dict) and tag.get("type") == "athlete"]
        competitions = [
            tag for tag in tags if isinstance(tag, dict) and tag.get("type") == "competition"
        ]
        competition = str(competitions[0].get("title")) if competitions else None
        description = str(row.get("description") or "").strip() or None
        candidates = infer_weapon_candidates(title, description, competition)
        source_url = f"{BASE_URL}/videos/{identifier}"
        category_ids = [str(value) for value in row.get("categories", [])]
        source_created_at = _parse_datetime(row.get("createdAt"))
        records.append(
            VideoRecord(
                video_id=f"fencingtv:{identifier}",
                title=title,
                weapon=primary_weapon(candidates),
                weapon_candidates=list(candidates),
                source_kind=SourceKind.FENCING_TV,
                source_url=HttpUrl(source_url),
                landing_page_url=HttpUrl(source_url),
                license_status=LicenseStatus.LINK_ONLY,
                athlete_names=[str(tag.get("title")) for tag in athletes if tag.get("title")],
                athlete_ids=[
                    str(tag.get("externalId")) for tag in athletes if tag.get("externalId")
                ],
                competition=competition,
                discovered_at=discovered_at,
                reviewed=False,
                group_id=(
                    f"fencingtv-competition:{competitions[0].get('externalId')}"
                    if competitions
                    else f"fencingtv:{identifier}"
                ),
                content_kind=classify_content(
                    title,
                    description=description,
                    athlete_count=len(athletes),
                ),
                source_created_at=source_created_at,
                thumbnail_url=row.get("thumbnailUrl"),
                metadata={
                    "catalog_status": row.get("status"),
                    "visibility": row.get("visibility"),
                    "playback_policy": row.get("playbackPolicy"),
                    "category_ids": category_ids,
                    "competition_id": competitions[0].get("externalId") if competitions else None,
                    "automatic_weapon_label": True,
                },
                notes=(
                    "metadata-only; playback endpoint not requested; "
                    f"weapon_candidates={','.join(weapon.value for weapon in candidates) or 'none'}"
                ),
            )
        )
    records.sort(key=lambda record: record.video_id)
    write_jsonl(output_path, [record.model_dump(mode="json") for record in records])
    return records


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def summarize(records: list[VideoRecord]) -> dict[str, Any]:
    return {
        "records": len(records),
        "by_weapon": dict(sorted(Counter(row.weapon.value for row in records).items())),
        "by_content_kind": dict(sorted(Counter(row.content_kind for row in records).items())),
        "athlete_linked_videos": sum(len(row.athlete_ids) >= 1 for row in records),
        "two_athlete_videos": sum(len(row.athlete_ids) >= 2 for row in records),
        "unique_athletes": len({athlete for row in records for athlete in row.athlete_ids}),
        "unique_competitions": len({row.competition for row in records if row.competition}),
        "mixed_weapon": sum(len(row.weapon_candidates) > 1 for row in records),
    }
