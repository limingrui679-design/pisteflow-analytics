"""Official YouTube-8M fencing-label catalogue with live link checks.

The Google release licenses its labels and precomputed features under CC BY
4.0.  That license does not transfer to the linked YouTube media, so this
collector deliberately stores metadata and viewing links only.
"""

from __future__ import annotations

import json
import re
import ssl
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import truststore

from pisteflow.provenance import atomic_write_bytes, read_jsonl, write_json, write_jsonl
from pisteflow.sources.classification import (
    classify_content,
    infer_weapon_candidates,
    primary_weapon,
)

DATASET_PAGE = "https://research.google.com/youtube8m/download.html"
EXPLORER_PAGE = "https://research.google.com/youtube8m/explore.html"
HISTOGRAM_URL = "https://research.google.com/youtube8m/csv/2/train-histogram-min.csv"
JSONP_BASE_URL = "https://storage.googleapis.com/data.yt8m.org/2/j"
AWS_REGISTRY_PAGE = "https://registry.opendata.aws/yt8m/"
FENCING_MID = "/m/02y8z"
FENCING_LABEL = "Fencing"
EXPECTED_MEMBERS = 346

_LABEL_JSONP = re.compile(r'^p\("(?P<mid>[^"\\]+)",(?P<ids>\[.*\])\);\s*$', re.DOTALL)
_ID_JSONP = re.compile(
    r'^i\("(?P<anonymous>[^"\\]+)","(?P<youtube>[^"\\]+)"\);\s*$',
    re.DOTALL,
)
_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_histogram(payload: str) -> tuple[str, int]:
    """Return the Fencing MID and row count from the official histogram."""

    matches: list[tuple[str, int]] = []
    for line in payload.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and parts[2] == FENCING_LABEL:
            matches.append((parts[1], int(parts[0])))
    if matches != [(FENCING_MID, EXPECTED_MEMBERS)]:
        raise ValueError(f"Unexpected YouTube-8M Fencing histogram rows: {matches}")
    return matches[0]


def parse_label_jsonp(payload: str) -> list[str]:
    """Parse the official entity-to-anonymous-video lookup without eval."""

    match = _LABEL_JSONP.fullmatch(payload.strip())
    if not match or f"/m/{match.group('mid')}" != FENCING_MID:
        raise ValueError("Unexpected YouTube-8M label JSONP payload")
    identifiers = json.loads(match.group("ids"))
    if not isinstance(identifiers, list) or not all(
        isinstance(value, str) and len(value) == 4 for value in identifiers
    ):
        raise ValueError("YouTube-8M label JSONP contains invalid anonymous IDs")
    if len(identifiers) != EXPECTED_MEMBERS or len(set(identifiers)) != EXPECTED_MEMBERS:
        raise ValueError(
            "YouTube-8M Fencing member count changed: "
            f"expected {EXPECTED_MEMBERS}, got {len(identifiers)}"
        )
    return identifiers


def parse_id_jsonp(payload: str, anonymous_id: str) -> str:
    """Parse one official anonymous-ID to YouTube-ID lookup."""

    match = _ID_JSONP.fullmatch(payload.strip())
    if not match or match.group("anonymous") != anonymous_id:
        raise ValueError(f"Unexpected YouTube-8M ID JSONP for {anonymous_id}")
    youtube_id = match.group("youtube")
    if not _YOUTUBE_ID.fullmatch(youtube_id):
        raise ValueError(f"Invalid YouTube ID mapped from {anonymous_id}: {youtube_id}")
    return youtube_id


def _client(user_agent: str) -> httpx.Client:
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(
        follow_redirects=True,
        trust_env=False,
        verify=ssl_context,
        timeout=httpx.Timeout(30, read=60),
        headers={"User-Agent": user_agent},
    )


def _get_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def _mapping_url(anonymous_id: str) -> str:
    return f"{JSONP_BASE_URL}/i/{anonymous_id[:2]}/{anonymous_id}.js"


def _load_or_fetch_mapping(
    anonymous_id: str,
    cache_directory: Path,
    *,
    user_agent: str,
) -> tuple[str, str | None]:
    path = cache_directory / "id-mappings" / anonymous_id[:2] / f"{anonymous_id}.js"
    try:
        if path.exists():
            payload = path.read_text(encoding="utf-8")
        else:
            with _client(user_agent) as client:
                payload = _get_text(client, _mapping_url(anonymous_id))
            atomic_write_bytes(path, payload.encode("utf-8"))
        return parse_id_jsonp(payload, anonymous_id), None
    except (httpx.HTTPError, OSError, UnicodeError, ValueError) as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _load_or_fetch_oembed(
    youtube_id: str,
    cache_directory: Path,
    *,
    user_agent: str,
    refresh: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    path = cache_directory / "oembed" / f"{youtube_id}.json"
    if path.exists() and not refresh:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None, None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return None, f"cached_{type(exc).__name__}: {exc}"
    url = (
        "https://www.youtube.com/oembed?format=json&url="
        f"https://www.youtube.com/watch?v={youtube_id}"
    )
    try:
        with _client(user_agent) as client:
            response = client.get(url)
            if response.status_code in {401, 403, 404}:
                return None, f"http_{response.status_code}"
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("title"), str):
            raise ValueError("YouTube oEmbed response lacks a title")
        write_json(path, payload)
        return payload, None
    except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _known_youtube_ids(manifest_directory: Path) -> dict[str, set[str]]:
    """Index existing provenance manifests by explicit 11-character IDs."""

    by_id: dict[str, set[str]] = {}
    for path in sorted(manifest_directory.glob("*.jsonl")):
        if path.name == "youtube8m-fencing.jsonl":
            continue
        for row in read_jsonl(path):
            values: list[str] = []
            for key in ("youtube_id", "video_id", "url", "source_url", "video_url"):
                value = row.get(key)
                if isinstance(value, str):
                    values.append(value)
            for value in values:
                candidates = re.findall(
                    r"(?:youtu\.be/|[?&]v=|youtube:)?([A-Za-z0-9_-]{11})(?:\b|$)",
                    value,
                )
                for candidate in candidates:
                    if _YOUTUBE_ID.fullmatch(candidate):
                        by_id.setdefault(candidate, set()).add(path.name)
    return by_id


def fetch_catalog(
    cache_directory: Path,
    output_path: Path,
    summary_path: Path,
    manifest_directory: Path,
    *,
    user_agent: str,
    workers: int = 8,
    refresh_oembed: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch all official Fencing memberships, mappings, and public link metadata."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    histogram_path = cache_directory / "train-histogram-min.csv"
    label_path = cache_directory / "fencing-members.js"
    with _client(user_agent) as client:
        if not histogram_path.exists():
            atomic_write_bytes(histogram_path, _get_text(client, HISTOGRAM_URL).encode("utf-8"))
        if not label_path.exists():
            label_url = f"{JSONP_BASE_URL}/v/{FENCING_MID.removeprefix('/m/')}.js"
            atomic_write_bytes(label_path, _get_text(client, label_url).encode("utf-8"))

    _mid, histogram_count = parse_histogram(histogram_path.read_text(encoding="utf-8"))
    anonymous_ids = parse_label_jsonp(label_path.read_text(encoding="utf-8"))

    mappings: dict[str, str] = {}
    mapping_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        mapping_futures = {
            executor.submit(
                _load_or_fetch_mapping,
                anonymous_id,
                cache_directory,
                user_agent=user_agent,
            ): anonymous_id
            for anonymous_id in anonymous_ids
        }
        for mapping_future in as_completed(mapping_futures):
            anonymous_id = mapping_futures[mapping_future]
            youtube_id, error = mapping_future.result()
            if error:
                mapping_failures[anonymous_id] = error
            else:
                mappings[anonymous_id] = youtube_id

    oembed: dict[str, dict[str, Any]] = {}
    oembed_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        oembed_futures = {
            executor.submit(
                _load_or_fetch_oembed,
                youtube_id,
                cache_directory,
                user_agent=user_agent,
                refresh=refresh_oembed,
            ): youtube_id
            for youtube_id in mappings.values()
        }
        for oembed_future in as_completed(oembed_futures):
            current_youtube_id = oembed_futures[oembed_future]
            oembed_payload, oembed_error = oembed_future.result()
            if oembed_payload is not None:
                oembed[current_youtube_id] = oembed_payload
            elif oembed_error:
                oembed_failures[current_youtube_id] = oembed_error

    existing = _known_youtube_ids(manifest_directory)
    rows: list[dict[str, Any]] = []
    for anonymous_id in anonymous_ids:
        mapped_youtube_id = mappings.get(anonymous_id)
        metadata = oembed.get(mapped_youtube_id or "", {})
        title = str(metadata.get("title") or "")
        candidates = infer_weapon_candidates(title)
        overlaps = sorted(existing.get(mapped_youtube_id or "", set()))
        rows.append(
            {
                "dataset": "YouTube-8M May 2018",
                "anonymous_video_id": anonymous_id,
                "youtube_id": mapped_youtube_id,
                "video_url": (
                    f"https://www.youtube.com/watch?v={mapped_youtube_id}"
                    if mapped_youtube_id
                    else None
                ),
                "title": title or None,
                "author_name": metadata.get("author_name"),
                "author_url": metadata.get("author_url"),
                "oembed_available": bool(metadata),
                "oembed_error": oembed_failures.get(mapped_youtube_id or ""),
                "label_mid": FENCING_MID,
                "label": FENCING_LABEL,
                "label_granularity": "machine_generated_video_level",
                "weapon": primary_weapon(candidates).value,
                "weapon_candidates": [weapon.value for weapon in candidates],
                "weapon_label_source": "current_oembed_title_automatic_hint_only",
                "content_kind": classify_content(title) if title else "unknown",
                "group_id": f"youtube8m:{anonymous_id}",
                "overlap_manifests": overlaps,
                "dataset_page": DATASET_PAGE,
                "explorer_page": EXPLORER_PAGE,
                "dataset_license": "CC BY 4.0 for YouTube-8M labels and features",
                "media_rights": "underlying YouTube media rights are not transferred",
                "redistribution": "metadata_and_viewing_link_only",
                "mapping_error": mapping_failures.get(anonymous_id),
                "claim_boundary": (
                    "One machine-generated broad Fencing video-level membership. It is not a "
                    "weapon, athlete, bout, touch, action, or frame-level ground-truth label."
                ),
            }
        )
    rows.sort(key=lambda row: str(row["anonymous_video_id"]))
    manifest_sha256 = write_jsonl(output_path, rows)
    summary: dict[str, Any] = {
        "dataset_page": DATASET_PAGE,
        "explorer_page": EXPLORER_PAGE,
        "aws_registry_page": AWS_REGISTRY_PAGE,
        "label": FENCING_LABEL,
        "label_mid": FENCING_MID,
        "histogram_members": histogram_count,
        "records": len(rows),
        "mapped_youtube_ids": sum(row["youtube_id"] is not None for row in rows),
        "unique_youtube_ids": len({row["youtube_id"] for row in rows if row["youtube_id"]}),
        "mapping_failures": mapping_failures,
        "oembed_available": sum(bool(row["oembed_available"]) for row in rows),
        "oembed_unavailable_or_nonembeddable": sum(
            not bool(row["oembed_available"]) for row in rows
        ),
        "by_weapon_title_hint": dict(
            sorted(Counter(str(row["weapon"]) for row in rows).items())
        ),
        "overlap_rows": sum(bool(row["overlap_manifests"]) for row in rows),
        "overlap_by_manifest": dict(
            sorted(
                Counter(
                    manifest
                    for row in rows
                    for manifest in row["overlap_manifests"]
                ).items()
            )
        ),
        "manifest_sha256": manifest_sha256,
        "output": str(output_path),
        "dataset_license": "CC BY 4.0 for YouTube-8M labels and precomputed features",
        "media_policy": "link_only; no raw YouTube media downloaded or redistributed",
        "claim_boundary": (
            "The official explorer lists 346 broad Fencing video-level memberships. Current "
            "oEmbed success is only a public-link availability signal. These rows do not provide "
            "three-weapon, athlete, bout, touch, action, or frame-level ground truth."
        ),
    }
    write_json(summary_path, summary)
    return rows, summary
