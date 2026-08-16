"""License-aware Commons images linked from the CC0 Wikidata fencer registry."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any
from urllib.parse import quote

import httpx

from pisteflow.provenance import read_jsonl, sha256_file, write_json, write_jsonl
from pisteflow.sources.commons import DEFAULT_USER_AGENT, _download_with_retry, _is_open_license
from pisteflow.sources.commons_images import (
    _fetch_metadata_batch,
    _metadata_value,
    _plain_text,
    _thumbnail_url,
)

ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/tiff"}
LABEL_BOUNDARY = (
    "Wikidata P18 links an image to one or more fencer entities, but PisteFlow has not pixel-"
    "verified the depicted identity. A portrait or event photograph is not a fencing-action, "
    "weapon, touch, bout, or skill label. Weapon hints are inherited from the athlete registry."
)


def _file_title(value: object) -> str | None:
    if not value:
        return None
    title = str(value).strip().replace("_", " ")
    if not title:
        return None
    return title if title.startswith("File:") else f"File:{title}"


def image_memberships(athletes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Collapse P18 links by Commons title while preserving athlete memberships."""

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for athlete in athletes:
        title = _file_title(athlete.get("image_file"))
        if title is None:
            continue
        grouped[title].append(
            {
                "athlete_id": str(athlete["athlete_id"]),
                "wikidata_id": str(athlete["wikidata_id"]),
                "name": str(athlete["name"]),
                "weapons": sorted({str(item) for item in athlete.get("weapons", [])}),
                "gender": str(athlete.get("gender", "unknown")),
            }
        )
    for memberships in grouped.values():
        memberships.sort(key=lambda item: item["wikidata_id"])
    return dict(grouped)


def _weapon_status(memberships: list[dict[str, Any]]) -> tuple[str, list[str]]:
    candidates = sorted(
        {
            str(weapon)
            for membership in memberships
            for weapon in membership.get("weapons", [])
            if weapon in {"epee", "foil", "sabre"}
        }
    )
    if not candidates:
        return "unknown", []
    if len(candidates) == 1:
        return candidates[0], candidates
    return "ambiguous_multi_weapon", candidates


def discover(
    athlete_manifest_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    """Resolve P18 files to Commons metadata and retain per-file license evidence."""

    athletes = read_jsonl(athlete_manifest_path)
    memberships_by_title = image_memberships(athletes)
    discovered_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    rows: list[dict[str, Any]] = []
    titles = sorted(memberships_by_title)
    with httpx.Client(
        follow_redirects=True,
        timeout=90,
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        for start in range(0, len(titles), 20):
            for page in _fetch_metadata_batch(client, titles[start : start + 20]):
                title = str(page.get("title", ""))
                memberships = memberships_by_title.get(title)
                image_info_rows = page.get("imageinfo")
                if not memberships or not isinstance(image_info_rows, list) or not image_info_rows:
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
                weapon_status, weapon_candidates = _weapon_status(memberships)
                landing_page = "https://commons.wikimedia.org/wiki/" + quote(
                    title.replace(" ", "_"), safe="():,'-._~"
                )
                rows.append(
                    {
                        "image_id": f"commons:{page.get('pageid', title)}",
                        "title": title.removeprefix("File:"),
                        "athlete_memberships": memberships,
                        "athlete_membership_count": len(memberships),
                        "weapon_status": weapon_status,
                        "weapon_candidates": weapon_candidates,
                        "identity_label_source": "wikidata_P18_unverified_by_pisteflow",
                        "label_scope": "athlete_associated_image_not_fencing_action",
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
                        "discovered_at": discovered_at,
                        "sha256": None,
                        "local_path": None,
                    }
                )

    rows.sort(key=lambda row: (str(row["weapon_status"]), str(row["title"]).lower()))
    manifest_sha256 = write_jsonl(output_path, rows)
    summary: dict[str, Any] = {
        "athlete_registry_records": len(athletes),
        "athletes_with_P18": sum(bool(row.get("image_file")) for row in athletes),
        "unique_P18_titles": len(memberships_by_title),
        "resolved_image_records": len(rows),
        "open_records": sum(row["license_status"] == "open" for row in rows),
        "unknown_license_records": sum(row["license_status"] != "open" for row in rows),
        "athlete_image_memberships": sum(row["athlete_membership_count"] for row in rows),
        "multi_athlete_image_records": sum(row["athlete_membership_count"] > 1 for row in rows),
        "by_weapon_status": dict(Counter(str(row["weapon_status"]) for row in rows)),
        "unique_commons_sha1": len(
            {str(row["commons_sha1"]) for row in rows if row["commons_sha1"] is not None}
        ),
        "manifest_sha256": manifest_sha256,
        "label_boundary": LABEL_BOUNDARY,
        "claim_boundary": (
            "License-resolved Commons files linked from Wikidata fencer entities. Counts are "
            "unique source images and image-to-athlete memberships, not independent athletes, "
            "fencing actions, bouts, or verified face-identification labels."
        ),
    }
    write_json(summary_path, summary)
    return summary


def _download_selection(
    row: dict[str, Any], requested_width: int
) -> tuple[str, str, str]:
    """Avoid invalid Commons upscaling URLs by using small originals directly."""

    direct_url = str(row["direct_media_url"])
    mime_type = str(row.get("mime_type", ""))
    source_width = row.get("width")
    if isinstance(source_width, int) and source_width <= requested_width:
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/tiff": ".tif",
        }[mime_type]
        return direct_url, "commons_original_no_upscale", extension
    extension = ".jpg" if mime_type in {"image/jpeg", "image/tiff"} else ".png"
    return (
        _thumbnail_url(direct_url, mime_type, requested_width),
        f"commons_thumbnail_w{requested_width}",
        extension,
    )


def download(
    manifest_path: Path,
    output_directory: Path,
    summary_path: Path,
    *,
    reuse_manifest_path: Path | None = None,
    network: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    thumbnail_width: int = 1024,
    workers: int = 6,
) -> dict[str, Any]:
    """Download bounded open-license thumbnails and checksum every acquired file."""

    if thumbnail_width < 320 or thumbnail_width > 2560:
        raise ValueError("Wikidata athlete-image width must be between 320 and 2560")
    if workers < 1 or workers > 12:
        raise ValueError("Wikidata athlete-image workers must be between 1 and 12")
    rows = read_jsonl(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    downloaded = 0
    reused = 0
    reusable_by_commons_sha1: dict[str, dict[str, Any]] = {}
    if reuse_manifest_path is not None:
        for reuse_candidate in read_jsonl(reuse_manifest_path):
            commons_sha1 = str(reuse_candidate.get("commons_sha1") or "")
            if (
                commons_sha1
                and reuse_candidate.get("sha256")
                and reuse_candidate.get("local_path")
            ):
                reusable_by_commons_sha1[commons_sha1] = reuse_candidate
    rate_limit_event = Event()
    network_rows: list[dict[str, Any]] = []
    for row in rows:
        direct_url = row.get("direct_media_url")
        if row.get("license_status") != "open" or not direct_url:
            continue
        reusable: dict[str, Any] | None = reusable_by_commons_sha1.get(
            str(row.get("commons_sha1") or "")
        )
        if reusable is not None:
            reusable_path = Path(str(reusable["local_path"]))
            if not reusable_path.is_absolute():
                reusable_path = output_directory.parent.parent / reusable_path
            if reusable_path.exists() and sha256_file(reusable_path) == reusable["sha256"]:
                row["sha256"] = reusable["sha256"]
                row["local_path"] = reusable["local_path"]
                row["download_variant"] = "reused_commons_category_copy"
                row["download_url"] = reusable.get("download_url")
                row["downloaded_bytes"] = reusable.get("downloaded_bytes")
                reused += 1
                continue
        if not network:
            failures.append(
                {
                    "image_id": str(row["image_id"]),
                    "error": "deferred_network_disabled",
                }
            )
            continue
        network_rows.append(row)

    def acquire(row: dict[str, Any], client: httpx.Client) -> tuple[bool, str | None]:
        if rate_limit_event.is_set():
            return False, "deferred_after_commons_rate_limit"
        download_url, download_variant, extension = _download_selection(row, thumbnail_width)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(row["image_id"]))
        destination = (
            output_directory
            / str(row["weapon_status"])
            / f"{safe_id}-w{thumbnail_width}{extension}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not destination.exists():
                _download_with_retry(
                    client,
                    download_url,
                    destination,
                    expected_bytes=None,
                    attempts=2,
                )
            row["sha256"] = sha256_file(destination)
            row["local_path"] = str(destination.relative_to(output_directory.parent.parent))
            row["download_variant"] = download_variant
            row["download_url"] = download_url
            row["downloaded_bytes"] = destination.stat().st_size
            return True, None
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            error = str(exc)
            if "HTTP 429" in error and "cooldown" in error:
                rate_limit_event.set()
            return False, error

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30, connect=15, read=45, write=30, pool=30),
        limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(acquire, row, client): row for row in network_rows}
        for future in as_completed(futures):
            row = futures[future]
            succeeded, error = future.result()
            if succeeded:
                downloaded += 1
            else:
                failures.append(
                    {"image_id": str(row["image_id"]), "error": str(error)}
                )

    manifest_sha256 = write_jsonl(manifest_path, rows)
    checksums = [str(row["sha256"]) for row in rows if row.get("sha256")]
    failure_counts = Counter(
        (
            str(failure["error"])
            if str(failure["error"]).startswith("deferred_")
            else "download_error"
        )
        for failure in failures
    )
    summary: dict[str, Any] = {
        "records": len(rows),
        "open_records": sum(row.get("license_status") == "open" for row in rows),
        "downloaded_records": downloaded,
        "reused_existing_commons_records": reused,
        "unique_downloaded_sha256": len(set(checksums)),
        "exact_duplicate_downloads": len(checksums) - len(set(checksums)),
        "requested_thumbnail_width": thumbnail_width,
        "workers": workers,
        "by_download_variant": dict(
            Counter(str(row.get("download_variant")) for row in rows if row.get("sha256"))
        ),
        "downloaded_bytes": sum(int(row.get("downloaded_bytes") or 0) for row in rows),
        "failure_count": len(failures),
        "failures_by_reason": dict(sorted(failure_counts.items())),
        "failure_examples": failures[:20],
        "rate_limited": rate_limit_event.is_set(),
        "network_enabled": network,
        "manifest_sha256": manifest_sha256,
        "label_boundary": LABEL_BOUNDARY,
    }
    write_json(summary_path, summary)
    return summary
