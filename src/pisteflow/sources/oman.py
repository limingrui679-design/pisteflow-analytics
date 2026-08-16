"""Collector for the Oman Fencing Committee's documented public API."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import HttpUrl

from pisteflow.models import CompetitionResult, DatasetSnapshot, LicenseStatus, Weapon
from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json

BASE_URL = "https://fencing.om"
ENDPOINTS = {
    "athletes": "/api/public/athletes",
    "events": "/api/public/events",
    "results": "/api/public/results",
    "rankings": "/api/public/rankings.json",
    "clubs": "/api/public/stats/clubs",
    "weapons": "/api/public/stats/weapons",
    "birthyears": "/api/public/stats/birthyears",
    "stats": "/api/public/stats",
    "medals": "/api/public/medals.json",
    "disciplines": "/api/public/disciplines",
    "referees": "/api/public/referees.json",
    "coaches": "/api/public/coaches",
    "leaderboard": "/api/public/leaderboard",
    "glossary": "/api/public/glossary.json",
}

_LIST_KEYS = {
    "athletes": "athletes",
    "events": "events",
    "results": "results",
    "rankings": "rankings",
    "clubs": "clubs",
    "weapons": "weapons",
    "disciplines": "disciplines",
    "referees": "referees",
    "coaches": "coaches",
    "leaderboard": "leaderboard",
    "glossary": "terms",
}


def _canonical_payload(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _record_count(name: str, payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    key = _LIST_KEYS.get(name)
    rows = payload.get(key) if key is not None else None
    if isinstance(rows, list):
        return len(rows)
    count = payload.get("count")
    if isinstance(count, int) and count >= 0:
        return count
    if name == "birthyears":
        total = payload.get("total")
        return int(total) if isinstance(total, int) and total >= 0 else 0
    if name in {"stats", "medals"}:
        return 1
    return 0


def fetch_all(output_directory: Path, user_agent: str) -> list[DatasetSnapshot]:
    output_directory.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(UTC)
    snapshots: list[DatasetSnapshot] = []
    with httpx.Client(
        base_url=BASE_URL,
        follow_redirects=True,
        timeout=60,
        headers={"User-Agent": user_agent},
    ) as client:
        for name, endpoint in ENDPOINTS.items():
            response = client.get(endpoint)
            response.raise_for_status()
            payload = response.json()
            raw = _canonical_payload(payload) + b"\n"
            atomic_write_bytes(output_directory / f"{name}.json", raw)
            count = _record_count(name, payload)
            snapshots.append(
                DatasetSnapshot(
                    snapshot_id=f"oman:{name}:{fetched_at.date().isoformat()}",
                    source_name=f"Oman Fencing Committee {name}",
                    source_url=HttpUrl(f"{BASE_URL}{endpoint}"),
                    fetched_at=fetched_at,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    record_count=count,
                    schema_version="oman-public-api-v1",
                    license_status=LicenseStatus.UNKNOWN,
                    metadata={"access": "public_no_auth_cors", "redistribution": "not_asserted"},
                )
            )
    write_json(
        output_directory / "snapshot-manifest.json",
        [snapshot.model_dump(mode="json") for snapshot in snapshots],
    )
    return snapshots


def summarize(snapshots: list[DatasetSnapshot], output_path: Path) -> dict[str, Any]:
    by_endpoint = {
        snapshot.source_name.removeprefix("Oman Fencing Committee "): snapshot.record_count
        for snapshot in snapshots
    }
    summary: dict[str, Any] = {
        "source_url": "https://fencing.om/developers",
        "official_body": "Oman Fencing Committee",
        "access": "official public API; unauthenticated; CORS-enabled",
        "license": "No data-reuse license asserted",
        "endpoint_snapshots": len(snapshots),
        "records_by_endpoint_native_unit": dict(sorted(by_endpoint.items())),
        "snapshot_manifest_sha256": sha256_file(output_path / "snapshot-manifest.json"),
        "claim_boundary": (
            "Current official public-API snapshots from the Oman Fencing Committee. Counts are "
            "reported separately by endpoint because athletes, results, clubs, referees, terms, "
            "and aggregates are incompatible units. Public access is not treated as a reuse "
            "license; row-level payloads remain in the ignored cache."
        ),
    }
    return summary


def normalize_results(raw_path: Path) -> list[CompetitionResult]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    fetched_at = datetime.fromtimestamp(raw_path.stat().st_mtime, UTC)
    rows: list[CompetitionResult] = []
    for item in payload.get("results", []):
        raw_weapon = str(item.get("weapon", "unknown")).lower()
        try:
            weapon = Weapon(raw_weapon)
        except ValueError:
            weapon = Weapon.UNKNOWN
        rows.append(
            CompetitionResult(
                result_id=str(item["id"]),
                athlete_id=str(item["athleteId"]),
                athlete_name=item.get("athleteNameEn"),
                tournament=str(item["tournamentEn"]),
                event_date=item["date"],
                city=item.get("city"),
                weapon=weapon,
                medal=item.get("medal", "none"),
                level=item.get("level"),
                notes=item.get("notesEn"),
                source_url=HttpUrl(f"{BASE_URL}/api/public/results"),
                fetched_at=fetched_at,
            )
        )
    return rows
