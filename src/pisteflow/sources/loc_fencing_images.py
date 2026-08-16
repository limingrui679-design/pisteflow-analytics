"""Acquire rights-screened historical fencing images from the Library of Congress."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json, write_jsonl
from pisteflow.sources.commons import DEFAULT_USER_AGENT
from pisteflow.sources.net import download_resumable

SEARCH_URL = "https://www.loc.gov/photos/"
SOURCE_PAGE = "https://www.loc.gov/apis/"
QUERY = "fencing"
_OPEN_RIGHTS = re.compile(r"no known restrictions|public domain", re.IGNORECASE)
_WEAPONS = {
    "epee": re.compile(r"\b(?:epee|épée)\b", re.IGNORECASE),
    "foil": re.compile(r"\b(?:foil|foils|foilist)\b", re.IGNORECASE),
    "sabre": re.compile(r"\b(?:sabre|saber)\b", re.IGNORECASE),
}
CLAIM_BOUNDARY = (
    "Library of Congress rows are historical fencing-context images selected by an exact subject "
    "heading and item-level rights advisory. They are not modern three-weapon, athlete identity, "
    "action, bout, touch, score, or skill ground truth. Dates, titles, and descriptions can be "
    "uncertain catalog metadata. Use only for historical or broad visual pretraining, with entire "
    "collection groups held out together."
)


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str] | None = None,
    attempts: int = 7,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else float(2**attempt)
                )
                last_error = RuntimeError(f"LOC API returned HTTP {response.status_code}")
                time.sleep(min(delay, 45.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("LOC API returned a non-object payload")
            return payload
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError(f"LOC request exhausted its retry budget: {url}") from last_error


def _subject_is_fencing(result: dict[str, Any]) -> bool:
    subjects = result.get("subject")
    return isinstance(subjects, list) and any(
        str(subject).strip().casefold() == "fencing" for subject in subjects
    )


def _item_identifier(result: dict[str, Any]) -> str:
    raw = str(result.get("id") or "").rstrip("/")
    identifier = raw.rsplit("/", 1)[-1]
    if not identifier:
        raise ValueError("LOC result has no item identifier")
    return identifier


def _rights_advisory(record: dict[str, Any]) -> str | None:
    nested = record.get("item")
    if not isinstance(nested, dict):
        return None
    value = nested.get("rights_advisory") or nested.get("rights_information")
    return str(value).strip() if value else None


def _best_image_url(record: dict[str, Any]) -> str | None:
    urls = record.get("image_url")
    if not isinstance(urls, list):
        return None
    candidates: list[tuple[int, str]] = []
    for raw in urls:
        url = str(raw)
        width_match = re.search(r"(?:#|&)h=\d+&w=(\d+)", url)
        width = int(width_match.group(1)) if width_match else 0
        if width <= 1600:
            candidates.append((width, url.split("#", 1)[0]))
    if not candidates:
        return None
    return max(candidates, key=lambda value: value[0])[1]


def _weapon_candidates(record: dict[str, Any]) -> list[str]:
    nested = record.get("item")
    details = nested if isinstance(nested, dict) else {}
    text = " ".join(
        str(value)
        for value in (
            record.get("title"),
            " ".join(str(item) for item in record.get("description", [])),
            details.get("summary"),
        )
        if value
    )
    return [weapon for weapon, pattern in _WEAPONS.items() if pattern.search(text)]


def _collection_group(record: dict[str, Any]) -> str:
    collections = record.get("raw_collections")
    values = [str(value) for value in collections] if isinstance(collections, list) else []
    basis = "|".join(sorted(values)) or str(record.get("partof") or "unknown")
    return "loc-collection:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def fetch_and_prepare(
    cache_directory: Path,
    output_directory: Path,
    manifest_path: Path,
    summary_path: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    force: bool = False,
) -> dict[str, Any]:
    """Search once, inspect every exact-subject item, and acquire open-rights images."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    search_cache = cache_directory / "photos-fencing-search.json"
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=120),
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        if search_cache.exists() and not force:
            search_payload = json.loads(search_cache.read_text(encoding="utf-8"))
        else:
            search_payload = _request_json(
                client,
                SEARCH_URL,
                params={"q": QUERY, "fo": "json", "c": "100"},
            )
            atomic_write_bytes(
                search_cache,
                (json.dumps(search_payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
        if not isinstance(search_payload, dict):
            raise TypeError("LOC cached search payload is not an object")
        raw_results = search_payload.get("results", [])
        results = [
            result
            for result in raw_results
            if isinstance(result, dict) and _subject_is_fencing(result)
        ] if isinstance(raw_results, list) else []

        rows: list[dict[str, Any]] = []
        excluded_rights: list[dict[str, str | None]] = []
        failures: list[dict[str, str]] = []
        for result in results:
            identifier = _item_identifier(result)
            item_cache = cache_directory / "items" / f"{identifier}.json"
            try:
                if item_cache.exists() and not force:
                    payload = json.loads(item_cache.read_text(encoding="utf-8"))
                else:
                    payload = _request_json(
                        client,
                        f"https://www.loc.gov/item/{identifier}/",
                        params={"fo": "json"},
                    )
                    item_cache.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_bytes(
                        item_cache,
                        (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                            "utf-8"
                        ),
                    )
                    time.sleep(1.0)
                record = payload.get("item") if isinstance(payload, dict) else None
                if not isinstance(record, dict):
                    raise TypeError("LOC item payload has no item object")
                rights = _rights_advisory(record)
                if not rights or not _OPEN_RIGHTS.search(rights):
                    excluded_rights.append({"item_id": identifier, "rights_advisory": rights})
                    continue
                image_url = _best_image_url(record)
                if image_url is None:
                    raise ValueError("LOC item has no bounded image URL")
                destination = output_directory / f"loc-{identifier}.jpg"
                if force:
                    destination.unlink(missing_ok=True)
                download_resumable(image_url, destination, user_agent=user_agent)
                try:
                    from PIL import Image
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError("Pillow is required for LOC image validation") from exc
                with Image.open(destination) as image:
                    image.verify()
                with Image.open(destination) as image:
                    width, height, image_format = int(image.width), int(image.height), image.format
                if image_format != "JPEG":
                    raise ValueError(f"Unexpected LOC image format: {image_format}")
                weapons = _weapon_candidates(record)
                nested = record.get("item")
                details = nested if isinstance(nested, dict) else {}
                rows.append(
                    {
                        "image_id": f"loc:{identifier}",
                        "loc_item_id": identifier,
                        "title": str(record.get("title") or details.get("title") or identifier),
                        "date": str(record.get("date")) if record.get("date") else None,
                        "description": [
                            str(value) for value in record.get("description", [])
                        ],
                        "subjects": [str(value) for value in record.get("subject", [])],
                        "weapon_status": weapons[0] if len(weapons) == 1 else (
                            "ambiguous_multi_weapon" if weapons else "unknown"
                        ),
                        "weapon_candidates": weapons,
                        "label_scope": "historical_fencing_context_not_modern_action_truth",
                        "source_group_id": _collection_group(record),
                        "source_url": f"https://www.loc.gov/item/{identifier}/",
                        "download_url": image_url,
                        "rights_advisory": rights,
                        "rights_status": "no_known_restrictions_or_public_domain_statement",
                        "width": width,
                        "height": height,
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                        "local_path": str(destination),
                        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                    }
                )
            except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append({"item_id": identifier, "error": str(exc)})

    rows.sort(key=lambda row: str(row["image_id"]))
    first_by_hash: dict[str, str] = {}
    for row in rows:
        digest = str(row["sha256"])
        row["exact_duplicate_of"] = first_by_hash.get(digest)
        first_by_hash.setdefault(digest, str(row["image_id"]))
        row["eligible_exact_unique_historical_context"] = row["exact_duplicate_of"] is None
    manifest_sha256 = write_jsonl(manifest_path, rows)
    summary: dict[str, Any] = {
        "source_page": SOURCE_PAGE,
        "query": QUERY,
        "search_results": len(raw_results) if isinstance(raw_results, list) else 0,
        "exact_fencing_subject_results": len(results),
        "downloaded_open_rights_records": len(rows),
        "unique_downloaded_sha256": len({str(row["sha256"]) for row in rows}),
        "downloaded_bytes": sum(int(row["bytes"]) for row in rows),
        "collection_groups": len({str(row["source_group_id"]) for row in rows}),
        "by_weapon_status": dict(Counter(str(row["weapon_status"]) for row in rows)),
        "excluded_without_recognized_open_rights": excluded_rights,
        "failures": failures,
        "manifest_sha256": manifest_sha256,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(summary_path, summary)
    return summary
