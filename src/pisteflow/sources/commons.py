"""Discover and download verified-open fencing videos from Wikimedia Commons."""

from __future__ import annotations

import html
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
import yaml
from pydantic import HttpUrl

from pisteflow.models import ContentKind, LicenseStatus, SourceKind, VideoRecord, VideoSplit, Weapon
from pisteflow.provenance import read_jsonl, sha256_file, write_json, write_jsonl

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_USER_AGENT = "PisteFlowAnalytics/0.1 (open fencing research; contact in repository)"

_OPEN_LICENSE_PREFIXES = (
    "cc0",
    "cc-by-",
    "cc-by-sa-",
    "public domain",
    "pdm",
)


def _plain_text(value: str | None) -> str | None:
    if not value:
        return None
    without_tags = re.sub(r"<[^>]+>", " ", value)
    cleaned = re.sub(r"\s+", " ", html.unescape(without_tags)).strip()
    return cleaned or None


def _metadata_value(metadata: dict[str, object], key: str) -> str | None:
    raw = metadata.get(key)
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    return str(value) if value is not None else None


def _is_open_license(code: str | None, short_name: str | None) -> bool:
    normalized = " ".join(part for part in (code, short_name) if part).lower()
    return any(token in normalized for token in _OPEN_LICENSE_PREFIXES)


def _source_family(title: str) -> str:
    normalized = title.lower()
    if "evd-" in normalized:
        return "evd_visual_dictionary"
    if "coupe du monde juniors dourdan" in normalized:
        return "dourdan_junior_world_cup"
    if "yakimenko vs. limbach" in normalized:
        return "leipzig_2010"
    if "aspromonte vs. baldini" in normalized:
        return "aspromonte_baldini"
    if "floretkampioenschap" in normalized:
        return "dutch_foil_archive"
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _training_segments(row: dict[str, object]) -> list[list[float]]:
    raw_segments = row.get("training_segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("Commons training_segments must be a list")
    segments: list[list[float]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, list) or len(raw_segment) != 2:
            raise ValueError("Commons training segment must contain start and end seconds")
        start, end = (float(raw_segment[0]), float(raw_segment[1]))
        if start < 0 or end <= start:
            raise ValueError("Commons training segment bounds are invalid")
        segments.append([start, end])
    for previous, current in pairwise(segments):
        if current[0] < previous[1]:
            raise ValueError("Commons training segments must be sorted and non-overlapping")
    return segments


def load_curated_titles(path: Path) -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload.get("videos", []) if isinstance(payload, dict) else []
    return {
        str(row["title"]): {
            "weapon": Weapon(str(row["weapon"])).value,
            "split": str(row.get("split", "train")),
            "group_id": str(row.get("group_id", _source_family(str(row["title"])))),
            "weapon_evidence": str(row.get("weapon_evidence", "curator_reviewed_source_label")),
            "content_kind": str(row.get("content_kind", "bout")),
            "training_segments": _training_segments(row),
        }
        for row in rows
    }


def _fetch_title_batch(
    client: httpx.Client,
    titles: list[str],
    *,
    attempts: int = 7,
) -> list[dict[str, object]]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(
                COMMONS_API,
                params={
                    "action": "query",
                    "titles": "|".join(titles),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                    "maxlag": "5",
                    "format": "json",
                    "formatversion": "2",
                },
            )
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
            return list(payload.get("query", {}).get("pages", []))
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError("Commons metadata request exhausted its retry budget") from last_error


def discover_curated(
    curated_path: Path,
    output_path: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[VideoRecord]:
    """Resolve reviewed Commons titles into a license-aware machine manifest."""

    curated = load_curated_titles(curated_path)
    records: list[VideoRecord] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=60,
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        titles = list(curated)
        for start in range(0, len(titles), 20):
            for page in _fetch_title_batch(client, titles[start : start + 20]):
                title = str(page.get("title", ""))
                image_info_rows = page.get("imageinfo")
                if not isinstance(image_info_rows, list) or not image_info_rows:
                    continue
                image_info = image_info_rows[0]
                if not isinstance(image_info, dict) or title not in curated:
                    continue
                metadata = image_info.get("extmetadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                license_code = _metadata_value(metadata, "License")
                short_name = _metadata_value(metadata, "LicenseShortName")
                is_open = _is_open_license(license_code, short_name)
                direct_url = HttpUrl(str(image_info["url"])) if is_open else None
                source_url = f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}"
                description = _plain_text(_metadata_value(metadata, "ImageDescription"))
                attribution = _plain_text(_metadata_value(metadata, "Artist"))
                usage_terms = _plain_text(_metadata_value(metadata, "UsageTerms"))
                notes = "; ".join(
                    part
                    for part in (
                        f"family={curated[title]['group_id']}",
                        description,
                        f"usage={usage_terms}" if usage_terms else None,
                    )
                    if part
                )
                records.append(
                    VideoRecord(
                        video_id=f"commons:{page.get('pageid', title)}",
                        title=title.removeprefix("File:"),
                        weapon=Weapon(str(curated[title]["weapon"])),
                        source_kind=SourceKind.WIKIMEDIA_COMMONS,
                        source_url=HttpUrl(source_url),
                        direct_media_url=direct_url,
                        landing_page_url=HttpUrl(source_url),
                        license_status=(LicenseStatus.OPEN if is_open else LicenseStatus.UNKNOWN),
                        license_code=license_code or short_name,
                        attribution=attribution,
                        bytes=int(image_info["size"]) if image_info.get("size") else None,
                        width=int(image_info["width"]) if image_info.get("width") else None,
                        height=int(image_info["height"]) if image_info.get("height") else None,
                        discovered_at=datetime.now(UTC),
                        reviewed=True,
                        split=cast(VideoSplit, curated[title]["split"]),
                        group_id=str(curated[title]["group_id"]),
                        content_kind=cast(ContentKind, curated[title]["content_kind"]),
                        metadata={
                            "weapon_label_source": curated[title]["weapon_evidence"],
                            "training_segments": curated[title]["training_segments"],
                        },
                        notes=notes,
                    )
                )
    records.sort(key=lambda item: (item.weapon.value, item.title.lower()))
    write_jsonl(output_path, [record.model_dump(mode="json") for record in records])
    return records


def load_manifest(path: Path) -> list[VideoRecord]:
    return [VideoRecord.model_validate(row) for row in read_jsonl(path)]


def select_training_derivative(
    derivatives: list[dict[str, Any]],
    *,
    maximum_height: int = 720,
) -> dict[str, Any] | None:
    """Choose the largest bounded WebM transcode, never the original-file alias."""

    candidates = [
        derivative
        for derivative in derivatives
        if isinstance(derivative.get("src"), str)
        and isinstance(derivative.get("transcodekey"), str)
        and str(derivative.get("type", "")).startswith("video/webm")
        and isinstance(derivative.get("height"), int)
        and 0 < int(derivative["height"]) <= maximum_height
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            int(row.get("height", 0)),
            int(row.get("width", 0)),
            int(row.get("bandwidth", 0)),
        ),
    )


def _fetch_training_derivative(
    client: httpx.Client,
    title: str,
    *,
    attempts: int = 6,
) -> dict[str, Any] | None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(
                COMMONS_API,
                params={
                    "action": "query",
                    "titles": f"File:{title}",
                    "prop": "videoinfo",
                    "viprop": "derivatives",
                    "maxlag": "5",
                    "format": "json",
                    "formatversion": "2",
                },
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"HTTP {response.status_code} from Commons API")
                time.sleep(min(float(2**attempt), 30.0))
                continue
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", [])
            if not isinstance(pages, list) or not pages:
                return None
            video_info = pages[0].get("videoinfo", [])
            if not isinstance(video_info, list) or not video_info:
                return None
            derivatives = video_info[0].get("derivatives", [])
            if not isinstance(derivatives, list):
                return None
            return select_training_derivative(
                [row for row in derivatives if isinstance(row, dict)]
            )
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 30.0))
    if last_error is not None:
        raise RuntimeError(f"Could not resolve Commons transcode for {title}") from last_error
    return None


def download_open_media(
    manifest_path: Path,
    output_directory: Path,
    *,
    max_total_bytes: int | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    report_path: Path | None = None,
    workers: int = 1,
) -> tuple[list[VideoRecord], list[dict[str, str]]]:
    """Download only rows whose Commons license is already verified as open."""

    records = load_manifest(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    if workers < 1 or workers > 6:
        raise ValueError("Commons download workers must be between 1 and 6")
    selected: list[tuple[int, VideoRecord]] = []
    reserved_bytes = 0
    for index, record in enumerate(records):
        if record.license_status is not LicenseStatus.OPEN or record.direct_media_url is None:
            continue
        if (
            max_total_bytes is not None
            and record.bytes is not None
            and reserved_bytes + record.bytes > max_total_bytes
        ):
            continue
        reserved_bytes += record.bytes or 0
        selected.append((index, record))
    selected.sort(key=lambda item: (item[1].bytes or 2**63, item[0]))

    updated_by_index: dict[int, VideoRecord] = {}
    failures: list[dict[str, str]] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(120, read=600),
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:

        def acquire(
            item: tuple[int, VideoRecord],
        ) -> tuple[int, VideoRecord, dict[str, str] | None]:
            index, record = item
            assert record.direct_media_url is not None
            download_record = record
            expected_bytes = record.bytes
            if record.sha256 is None:
                try:
                    derivative = _fetch_training_derivative(client, record.title)
                except RuntimeError:
                    derivative = None
                if derivative is not None:
                    derivative_url = str(derivative["src"])
                    metadata = dict(record.metadata)
                    metadata.update(
                        {
                            "original_media_url": str(record.direct_media_url),
                            "original_media_bytes": record.bytes,
                            "local_media_variant": str(derivative["transcodekey"]),
                            "local_media_width": derivative.get("width"),
                            "local_media_height": derivative.get("height"),
                            "local_media_bandwidth": derivative.get("bandwidth"),
                        }
                    )
                    download_record = record.model_copy(
                        update={
                            "direct_media_url": HttpUrl(derivative_url),
                            "metadata": metadata,
                        }
                    )
                    expected_bytes = None
            assert download_record.direct_media_url is not None
            extension = (
                Path(str(download_record.direct_media_url).split("?", 1)[0]).suffix or ".video"
            )
            safe_id = record.video_id.replace(":", "_")
            destination = output_directory / record.weapon.value / f"{safe_id}{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not destination.exists():
                    _download_with_retry(
                        client,
                        str(download_record.direct_media_url),
                        destination,
                        expected_bytes=expected_bytes,
                    )
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                return index, record, {"video_id": record.video_id, "error": str(exc)}
            file_size = destination.stat().st_size
            return (
                index,
                download_record.model_copy(
                    update={"sha256": sha256_file(destination), "bytes": file_size}
                ),
                None,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, next_record, failure in executor.map(acquire, selected):
                updated_by_index[index] = next_record
                if failure is not None:
                    failures.append(failure)
    updated = [updated_by_index.get(index, record) for index, record in enumerate(records)]
    manifest_sha256 = write_jsonl(
        manifest_path,
        [record.model_dump(mode="json") for record in updated],
    )
    if report_path is not None:
        write_json(
            report_path,
            {
                "attempted_open_records": sum(
                    record.license_status is LicenseStatus.OPEN for record in records
                ),
                "downloaded_records": sum(record.sha256 is not None for record in updated),
                "failures": failures,
                "manifest_sha256": manifest_sha256,
            },
        )
    return updated, failures


def _clean_download_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.netloc == "upload.wikimedia.org":
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _download_with_retry(
    client: httpx.Client,
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None,
    attempts: int = 6,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    cleaned_url = _clean_download_url(url)
    last_error: Exception | None = None
    for attempt in range(attempts):
        existing = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with client.stream("GET", cleaned_url, headers=headers) as response:
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else float(2**attempt)
                    )
                    last_error = RuntimeError(f"HTTP {response.status_code} for {cleaned_url}")
                    if response.status_code == 429 and delay > 60:
                        last_error = RuntimeError(
                            f"HTTP 429 for {cleaned_url}; server requested a {delay:.0f}s cooldown"
                        )
                        break
                    time.sleep(min(delay, 45.0))
                    continue
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                with temporary.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        handle.write(chunk)
            actual_bytes = temporary.stat().st_size
            if expected_bytes is not None and actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"size mismatch for {destination.name}: "
                    f"expected {expected_bytes}, got {actual_bytes}"
                )
            temporary.replace(destination)
            return
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError(f"download failed after {attempts} attempts: {cleaned_url}") from last_error


def summarize(records: Iterable[VideoRecord]) -> dict[str, object]:
    rows = list(records)
    by_weapon = {
        weapon.value: sum(row.weapon is weapon for row in rows)
        for weapon in (Weapon.EPEE, Weapon.FOIL, Weapon.SABRE)
    }
    return {
        "records": len(rows),
        "open": sum(row.license_status is LicenseStatus.OPEN for row in rows),
        "with_checksum": sum(row.sha256 is not None for row in rows),
        "bytes": sum(row.bytes or 0 for row in rows),
        "by_weapon": by_weapon,
    }
