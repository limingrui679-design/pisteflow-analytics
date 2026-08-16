"""Discover license-aware modern fencing still images from Wikimedia Commons."""

from __future__ import annotations

import hashlib
import html
import re
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from pisteflow.provenance import read_jsonl, sha256_file, write_json, write_jsonl
from pisteflow.sources.commons import (
    COMMONS_API,
    DEFAULT_USER_AGENT,
    _download_with_retry,
    _is_open_license,
)

ROOT_CATEGORIES = {
    "epee": "Category:Fencing (épée)",
    "foil": "Category:Fencing (foil)",
    "sabre": "Category:Fencing (sabre)",
}

# These first-level branches are people directories, equipment, historical art, or team rosters.
# They can be useful context, but inheriting a modern action label from them would be misleading.
BLOCKED_FIRST_LEVEL_PATTERNS = (
    re.compile(r"\bfencers?$", re.IGNORECASE),
    re.compile(r"\bfoils?$", re.IGNORECASE),
    re.compile(r"\bépées?$", re.IGNORECASE),
    re.compile(r"\bnational .+ teams?$", re.IGNORECASE),
    re.compile(r"\bbell-guard test$", re.IGNORECASE),
    re.compile(r"\b1899 world fencing championships$", re.IGNORECASE),
)

ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}

LABEL_BOUNDARY = (
    "Weapon candidates come from Commons category paths, not pixel review. A single-weapon path "
    "is suitable only as a weak contextual label; multi-root files remain ambiguous. None of these "
    "rows is a frame-level action, touch, identity, or skill label."
)


def _plain_text(value: str | None) -> str | None:
    if not value:
        return None
    without_tags = re.sub(r"<[^>]+>", " ", value)
    result = re.sub(r"\s+", " ", html.unescape(without_tags)).strip()
    return result or None


def _metadata_value(metadata: dict[str, object], key: str) -> str | None:
    raw = metadata.get(key)
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    return str(value) if value is not None else None


def _blocked_first_level(category: str) -> bool:
    name = category.removeprefix("Category:")
    return any(pattern.search(name) for pattern in BLOCKED_FIRST_LEVEL_PATTERNS)


def _cache_path(cache_directory: Path, category: str) -> Path:
    digest = hashlib.sha256(category.encode("utf-8")).hexdigest()
    return cache_directory / f"{digest}.json"


def _request_json(
    client: httpx.Client,
    params: dict[str, str],
    *,
    attempts: int = 7,
    minimum_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(COMMONS_API, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else float(2**attempt)
                )
                last_error = RuntimeError(f"HTTP {response.status_code} from Commons API")
                time.sleep(min(delay, 45.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("Commons API returned a non-object payload")
            if "error" in payload:
                raise RuntimeError(f"Commons API error: {payload['error']}")
            time.sleep(minimum_interval_seconds)
            return payload
        except (httpx.HTTPError, RuntimeError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError("Commons API request exhausted its retry budget") from last_error


def _category_members(
    client: httpx.Client,
    category: str,
    cache_directory: Path,
    *,
    refresh: bool,
) -> list[dict[str, Any]]:
    cache = _cache_path(cache_directory, category)
    if cache.exists() and not refresh:
        cached_payload = read_jsonl(cache)
        if len(cached_payload) == 1 and isinstance(cached_payload[0].get("members"), list):
            return [
                item
                for item in cached_payload[0]["members"]
                if isinstance(item, dict)
            ]

    members: list[dict[str, Any]] = []
    continuation: dict[str, str] = {}
    while True:
        api_payload = _request_json(
            client,
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": "file|subcat",
                "cmlimit": "500",
                "maxlag": "5",
            }
            | continuation,
        )
        query = api_payload.get("query", {})
        if not isinstance(query, dict):
            raise TypeError(f"Commons category query is malformed: {category}")
        page = query.get("categorymembers", [])
        if not isinstance(page, list):
            raise TypeError(f"Commons category members are malformed: {category}")
        members.extend(item for item in page if isinstance(item, dict))
        raw_continue = api_payload.get("continue")
        if not isinstance(raw_continue, dict):
            break
        continuation = {str(key): str(value) for key, value in raw_continue.items()}

    cache.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache, [{"category": category, "members": members}])
    return members


def category_paths(
    client: httpx.Client,
    cache_directory: Path,
    *,
    max_depth: int = 2,
    refresh: bool = False,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    """Return file-to-weapon/path mappings from controlled modern-sport branches."""

    if max_depth < 0 or max_depth > 3:
        raise ValueError("Commons image category depth must be between zero and three")

    file_weapons: defaultdict[str, set[str]] = defaultdict(set)
    file_paths: defaultdict[str, set[str]] = defaultdict(set)
    categories_by_weapon: Counter[str] = Counter()
    for weapon, root in ROOT_CATEGORIES.items():
        queue: deque[tuple[str, int, tuple[str, ...]]] = deque([(root, 0, (root,))])
        visited: set[str] = set()
        while queue:
            category, depth, path = queue.popleft()
            if category in visited:
                continue
            visited.add(category)
            categories_by_weapon[weapon] += 1
            for member in _category_members(
                client,
                category,
                cache_directory,
                refresh=refresh,
            ):
                namespace = int(member.get("ns", -1))
                title = str(member.get("title", ""))
                if namespace == 6 and title.startswith("File:"):
                    file_weapons[title].add(weapon)
                    file_paths[title].add(" > ".join(path))
                elif namespace == 14 and depth < max_depth:
                    if depth == 0 and _blocked_first_level(title):
                        continue
                    queue.append((title, depth + 1, (*path, title)))
    return dict(file_weapons), dict(file_paths), dict(categories_by_weapon)


def _fetch_metadata_batch(client: httpx.Client, titles: list[str]) -> list[dict[str, Any]]:
    payload = _request_json(
        client,
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata",
            "maxlag": "5",
        },
    )
    query = payload.get("query", {})
    if not isinstance(query, dict):
        return []
    pages = query.get("pages", [])
    if not isinstance(pages, list):
        return []
    return [page for page in pages if isinstance(page, dict)]


def _source_group(paths: Iterable[str]) -> str:
    first_children: set[str] = set()
    for path in paths:
        parts = path.split(" > ")
        first_children.add(parts[1] if len(parts) > 1 else parts[0])
    if len(first_children) == 1:
        value = next(iter(first_children))
    else:
        value = "multi_branch:" + "|".join(sorted(first_children))
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _thumbnail_url(original_url: str, mime_type: str, width: int) -> str:
    """Build MediaWiki's deterministic raster-thumbnail URL.

    Commons currently rate-limits many original-file paths for ten minutes while
    serving the corresponding cached thumbnail paths normally.  A bounded
    thumbnail is also a more appropriate acquisition for this image-classification
    inventory than downloading 9.2 GB of camera originals and archival TIFFs.
    """

    if width < 320 or width > 2560:
        raise ValueError("Commons thumbnail width must be between 320 and 2560")
    parsed = urlsplit(original_url)
    marker = "/wikipedia/commons/"
    if parsed.netloc != "upload.wikimedia.org" or marker not in parsed.path:
        raise ValueError(f"Unexpected Commons original URL: {original_url!r}")
    prefix, relative = parsed.path.split(marker, 1)
    filename = relative.rsplit("/", 1)[-1]
    if mime_type == "image/tiff":
        thumbnail_name = f"{width}px-{filename}.jpg"
    elif mime_type in {"image/jpeg", "image/png"}:
        thumbnail_name = f"{width}px-{filename}"
    else:
        raise ValueError(f"Unsupported Commons thumbnail MIME type: {mime_type!r}")
    path = f"{prefix}{marker}thumb/{relative}/{thumbnail_name}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def discover(
    output_path: Path,
    summary_path: Path,
    cache_directory: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    max_depth: int = 2,
    refresh: bool = False,
) -> dict[str, Any]:
    """Discover open and closed rows while retaining per-file licensing evidence."""

    with httpx.Client(
        follow_redirects=True,
        timeout=90,
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        weapons, paths, category_counts = category_paths(
            client,
            cache_directory,
            max_depth=max_depth,
            refresh=refresh,
        )
        rows: list[dict[str, Any]] = []
        titles = sorted(weapons)
        for start in range(0, len(titles), 20):
            for page in _fetch_metadata_batch(client, titles[start : start + 20]):
                title = str(page.get("title", ""))
                image_info_rows = page.get("imageinfo")
                if (
                    title not in weapons
                    or not isinstance(image_info_rows, list)
                    or not image_info_rows
                ):
                    continue
                image_info = image_info_rows[0]
                if not isinstance(image_info, dict):
                    continue
                mime = str(image_info.get("mime", ""))
                if mime not in ALLOWED_IMAGE_MIME_TYPES:
                    continue
                metadata = image_info.get("extmetadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                license_code = _metadata_value(metadata, "License")
                license_short_name = _metadata_value(metadata, "LicenseShortName")
                open_license = _is_open_license(license_code, license_short_name)
                candidates = sorted(weapons[title])
                weapon_status = candidates[0] if len(candidates) == 1 else "ambiguous_multi_weapon"
                landing_page = (
                    "https://commons.wikimedia.org/wiki/"
                    + quote(title.replace(" ", "_"), safe="():,'-._~")
                )
                source_paths = sorted(paths[title])
                rows.append(
                    {
                        "image_id": f"commons:{page.get('pageid', title)}",
                        "title": title.removeprefix("File:"),
                        "weapon_status": weapon_status,
                        "weapon_candidates": candidates,
                        "weapon_label_source": "commons_category_path_weak_context_label",
                        "label_scope": "context_not_action",
                        "category_paths": source_paths,
                        "source_group": _source_group(source_paths),
                        "landing_page_url": landing_page,
                        "direct_media_url": str(image_info.get("url")) if open_license else None,
                        "mime_type": mime,
                        "width": int(image_info["width"]) if image_info.get("width") else None,
                        "height": int(image_info["height"]) if image_info.get("height") else None,
                        "bytes": int(image_info["size"]) if image_info.get("size") else None,
                        "commons_sha1": str(image_info.get("sha1")) or None,
                        "license_status": "open" if open_license else "unknown",
                        "license_code": license_code or license_short_name,
                        "license_short_name": license_short_name,
                        "license_url": _metadata_value(metadata, "LicenseUrl"),
                        "attribution": _plain_text(_metadata_value(metadata, "Artist")),
                        "description": _plain_text(
                            _metadata_value(metadata, "ImageDescription")
                        ),
                        "credit": _plain_text(_metadata_value(metadata, "Credit")),
                        "discovered_at": datetime.now(UTC).isoformat(),
                        "sha256": None,
                        "local_path": None,
                    }
                )

    rows.sort(key=lambda row: (row["weapon_status"], str(row["title"]).lower()))
    manifest_sha256 = write_jsonl(output_path, rows)
    commons_sha1_counts = Counter(
        str(row["commons_sha1"]) for row in rows if row["commons_sha1"] is not None
    )
    summary: dict[str, Any] = {
        "records": len(rows),
        "open_records": sum(row["license_status"] == "open" for row in rows),
        "unknown_license_records": sum(row["license_status"] != "open" for row in rows),
        "by_weapon_status": dict(Counter(str(row["weapon_status"]) for row in rows)),
        "category_counts_by_root_weapon": category_counts,
        "source_groups": len({str(row["source_group"]) for row in rows}),
        "unique_commons_sha1": len(commons_sha1_counts),
        "exact_duplicate_rows": sum(count - 1 for count in commons_sha1_counts.values()),
        "declared_bytes": sum(int(row["bytes"] or 0) for row in rows),
        "max_category_depth": max_depth,
        "manifest_sha256": manifest_sha256,
        "label_boundary": LABEL_BOUNDARY,
        "claim_boundary": (
            "License-aware still-image discovery from controlled epee, foil, and sabre Commons "
            "category branches. Counts are source files, not athletes or independent actions."
        ),
    }
    write_json(summary_path, summary)
    return summary


def download(
    manifest_path: Path,
    output_directory: Path,
    summary_path: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    workers: int = 1,
    thumbnail_width: int = 1280,
) -> dict[str, Any]:
    """Download bounded license-verified Commons thumbnails with checksums."""

    if workers < 1 or workers > 2:
        raise ValueError("Commons still-image workers must be one or two")
    if thumbnail_width < 320 or thumbnail_width > 2560:
        raise ValueError("Commons thumbnail width must be between 320 and 2560")
    rows = read_jsonl(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    downloaded = 0
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30, read=120),
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        for row in rows:
            direct_url = row.get("direct_media_url")
            if row.get("license_status") != "open" or not direct_url:
                continue
            image_id = str(row["image_id"])
            mime_type = str(row.get("mime_type", ""))
            download_url = _thumbnail_url(str(direct_url), mime_type, thumbnail_width)
            extension = ".jpg" if mime_type in {"image/jpeg", "image/tiff"} else ".png"
            destination = (
                output_directory
                / str(row["weapon_status"])
                / f"{image_id.replace(':', '_')}-w{thumbnail_width}{extension}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            already_downloaded = destination.exists()
            try:
                if not already_downloaded:
                    _download_with_retry(
                        client,
                        download_url,
                        destination,
                        expected_bytes=None,
                        attempts=3,
                    )
                row["sha256"] = sha256_file(destination)
                row["local_path"] = str(destination.relative_to(output_directory.parent.parent))
                row["download_variant"] = f"commons_thumbnail_w{thumbnail_width}"
                row["download_url"] = download_url
                row["downloaded_bytes"] = destination.stat().st_size
                downloaded += 1
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                failures.append({"image_id": image_id, "error": str(exc)})
            # The current downloader is intentionally sequential.  Pace only network
            # requests (not checksum-only resume rows) so reruns do not trigger the
            # upload host's temporary rate limit after a long successful prefix.
            if not already_downloaded:
                time.sleep(0.6 / workers)
    manifest_sha256 = write_jsonl(manifest_path, rows)
    summary: dict[str, Any] = {
        "records": len(rows),
        "open_records": sum(row.get("license_status") == "open" for row in rows),
        "downloaded_records": downloaded,
        "unique_downloaded_sha256": len(
            {str(row["sha256"]) for row in rows if row.get("sha256")}
        ),
        "download_variant": f"commons_thumbnail_w{thumbnail_width}",
        "downloaded_bytes": sum(int(row.get("downloaded_bytes") or 0) for row in rows),
        "declared_original_bytes": sum(int(row.get("bytes") or 0) for row in rows),
        "failures": failures,
        "manifest_sha256": manifest_sha256,
        "label_boundary": LABEL_BOUNDARY,
    }
    write_json(summary_path, summary)
    return summary
