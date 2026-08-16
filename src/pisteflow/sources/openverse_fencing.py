"""Discover and acquire non-Wikimedia openly licensed fencing still images."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from pisteflow.provenance import (
    atomic_write_bytes,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from pisteflow.sources.commons import DEFAULT_USER_AGENT

API_URL = "https://api.openverse.org/v1/images/"
SOURCE_PAGE = "https://openverse.org/"
LICENSE_FILTER = "by,by-sa,cc0,pdm"
EXCLUDED_SOURCE = "wikimedia"
PAGE_SIZE = 20
MAX_ANONYMOUS_RESULTS_PER_QUERY = 240
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Quoted English phrases sharply reduce the physical-fence ambiguity. Multilingual
# queries recover openly licensed images whose metadata never contains English terms.
QUERY_SPECS: tuple[tuple[str, str, str | None, bool], ...] = (
    ('"foil fencing"', "foil_fencing", "foil", True),
    ('"epee fencing"', "epee_fencing", "epee", True),
    ('"épée fencing"', "epee_fencing_accented", "epee", True),
    ('"sabre fencing"', "sabre_fencing", "sabre", True),
    ('"saber fencing"', "saber_fencing", "sabre", True),
    ('"Olympic fencing"', "olympic_fencing", None, True),
    ('"Paralympic fencing"', "paralympic_fencing", None, True),
    ('"wheelchair fencing"', "wheelchair_fencing", None, True),
    ('"fencing competition"', "fencing_competition", None, True),
    ('"fencing tournament"', "fencing_tournament", None, True),
    ('"fencing championship"', "fencing_championship", None, True),
    ('"fencing world cup"', "fencing_world_cup", None, True),
    ('"fencing grand prix"', "fencing_grand_prix", None, True),
    ("foil fencer", "foil_fencer", "foil", True),
    ("epee fencer", "epee_fencer", "epee", True),
    ("sabre fencer", "sabre_fencer", "sabre", True),
    ("saber fencer", "saber_fencer", "sabre", True),
    ("fencer", "fencer_broad", None, False),
    ('"escrime sportive"', "french_sport_fencing", None, True),
    ('"esgrima deportiva"', "spanish_sport_fencing", None, True),
    ('"scherma sportiva"', "italian_sport_fencing", None, True),
    ('"Sportfechten"', "german_sport_fencing", None, True),
    ('"szermierka sportowa"', "polish_sport_fencing", None, True),
    ('"击剑比赛"', "chinese_fencing_competition", None, True),
    ('"フェンシング競技"', "japanese_fencing_competition", None, True),
    ('"펜싱 경기"', "korean_fencing_competition", None, True),
    ('"fencing bout"', "fencing_bout", None, True),
    ('"fencing match"', "fencing_match", None, True),
    ('"fencing athlete"', "fencing_athlete", None, True),
    ('"fencing team"', "fencing_team", None, True),
    ('"fencing lunge"', "fencing_lunge", None, True),
    ('"en garde" fencing', "en_garde_fencing", None, True),
    ('"foil competition"', "foil_competition", "foil", True),
    ('"epee competition"', "epee_competition", "epee", True),
    ('"épée competition"', "epee_competition_accented", "epee", True),
    ('"sabre competition"', "sabre_competition", "sabre", True),
    ('"saber competition"', "saber_competition", "sabre", True),
    ('"wheelchair fencer"', "wheelchair_fencer", None, True),
    ('"para fencing"', "para_fencing", None, True),
    ('"escrime fleuret"', "french_foil", "foil", True),
    ('"escrime épée"', "french_epee", "epee", True),
    ('"escrime sabre"', "french_sabre", "sabre", True),
    ('"compétition escrime"', "french_competition", None, True),
    ('"esgrima florete"', "spanish_foil", "foil", True),
    ('"esgrima espada"', "spanish_epee", "epee", True),
    ('"esgrima sable"', "spanish_sabre", "sabre", True),
    ('"competición esgrima"', "spanish_competition", None, True),
    ('"scherma fioretto"', "italian_foil", "foil", True),
    ('"scherma spada"', "italian_epee", "epee", True),
    ('"scherma sciabola"', "italian_sabre", "sabre", True),
    ('"Fechten Florett"', "german_foil", "foil", True),
    ('"Fechten Degen"', "german_epee", "epee", True),
    ('"Fechten Säbel"', "german_sabre", "sabre", True),
    ('"szermierka floret"', "polish_foil", "foil", True),
    ('"szermierka szpada"', "polish_epee", "epee", True),
    ('"szermierka szabla"', "polish_sabre", "sabre", True),
    ('"фехтование рапира"', "russian_foil", "foil", True),
    ('"фехтование шпага"', "russian_epee", "epee", True),
    ('"фехтование сабля"', "russian_sabre", "sabre", True),
    ('"vívás verseny"', "hungarian_fencing", None, True),
    ('"eskrim müsabaka"', "turkish_fencing", None, True),
    ('"フェンシング エペ"', "japanese_epee", "epee", True),
    ('"フェンシング フルーレ"', "japanese_foil", "foil", True),
    ('"フェンシング サーブル"', "japanese_sabre", "sabre", True),
    ('"펜싱 에페"', "korean_epee", "epee", True),
    ('"펜싱 플뢰레"', "korean_foil", "foil", True),
    ('"펜싱 사브르"', "korean_sabre", "sabre", True),
    ('"击剑 重剑"', "chinese_epee", "epee", True),
    ('"击剑 花剑"', "chinese_foil", "foil", True),
    ('"击剑 佩剑"', "chinese_sabre", "sabre", True),
)

_SPORT_CONTEXT = re.compile(
    r"\b(fenc(?:e|er|ers|ing)|foil(?:ist)?|epee|épée|sabre|saber|olympi(?:c|an)|"
    r"paralympi(?:c|an)|tournament|championship|competition|bout|piste|en garde|"
    r"pentathlon|athlete)\b|escrime|esgrima|scherma|fechten|szermier|击剑|"
    r"フェンシング|펜싱|фехтован|vívás|eskrim",
    re.IGNORECASE,
)
_BROAD_FENCER_SUPPORT = re.compile(
    r"\b(fencing|foil|epee|épée|sabre|saber|olympi(?:c|an)|paralympi(?:c|an)|"
    r"tournament|championship|competition|bout|piste|mask|athlete|en garde)\b|"
    r"escrime|esgrima|scherma|fechten|szermier|击剑|フェンシング|펜싱|"
    r"фехтован|vívás|eskrim",
    re.IGNORECASE,
)
_NEGATIVE_CONTEXT = re.compile(
    r"\b(sukhoi|su-24|aircraft|airplane|aeroplane|fighter jet|hms gay fencer|"
    r"garden fence|security fence|sports fence|weldmesh|ballstop|muga|fence supplier|"
    r"fence installation|fence contractor|fencer sculpture|ice fencer|lego|cosplay|"
    r"video game|star wars|animal crossing)\b",
    re.IGNORECASE,
)
_WEAPON_PATTERNS = {
    "epee": re.compile(r"\b(?:epee|épée|épéiste)\b", re.IGNORECASE),
    "foil": re.compile(r"\b(?:foil|foilist|fleuret)\b", re.IGNORECASE),
    "sabre": re.compile(r"\b(?:sabre|saber|sabreurs?)\b", re.IGNORECASE),
}
_FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}

CLAIM_BOUNDARY = (
    "Openverse results are strong-query context candidates, not pixel-reviewed weapon, action, "
    "athlete, bout, touch, or skill labels. Openverse itself warns that indexed license metadata "
    "may be inaccurate, so every row retains its upstream landing page, creator, attribution, and "
    "license statement. Wikimedia is excluded because the controlled Commons source is audited "
    "separately. Exact duplicate bytes are eligible only once, and creator-level groups must stay "
    "intact in evaluation."
)


def _request_json(
    client: httpx.Client,
    params: dict[str, str],
    *,
    attempts: int = 8,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(API_URL, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else float(2**attempt)
                )
                last_error = RuntimeError(f"Openverse API returned HTTP {response.status_code}")
                time.sleep(min(delay, 60.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("Openverse API returned a non-object payload")
            return payload
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 60.0))
    raise RuntimeError("Openverse API request exhausted its retry budget") from last_error


def _cache_path(cache_directory: Path, query_family: str, page: int) -> Path:
    return cache_directory / query_family / f"page-{page:02d}.json"


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _result_text(item: dict[str, Any]) -> str:
    tags = item.get("tags")
    tag_rows = tags if isinstance(tags, list) else []
    tag_names = [
        str(tag.get("name", ""))
        for tag in tag_rows
        if isinstance(tag, dict)
    ]
    return " ".join(
        part
        for part in (
            str(item.get("title") or ""),
            str(item.get("description") or ""),
            " ".join(tag_names),
        )
        if part
    )


def _is_candidate(item: dict[str, Any], *, strong_query: bool) -> bool:
    text = _result_text(item)
    if _NEGATIVE_CONTEXT.search(text):
        return False
    if not _SPORT_CONTEXT.search(text):
        return False
    return strong_query or bool(_BROAD_FENCER_SUPPORT.search(text))


def _weapon_candidates(text: str, hints: set[str]) -> list[str]:
    values = set(hints)
    values.update(weapon for weapon, pattern in _WEAPON_PATTERNS.items() if pattern.search(text))
    return [weapon for weapon in ("epee", "foil", "sabre") if weapon in values]


def _creator_group(item: dict[str, Any]) -> str:
    source = str(item.get("source") or item.get("provider") or "unknown").lower()
    creator = str(item.get("creator") or "").strip().lower()
    if creator:
        basis = f"{source}|creator|{creator}"
    else:
        landing = str(item.get("foreign_landing_url") or item.get("url") or "")
        parsed = urlsplit(landing)
        basis = f"{source}|host|{parsed.netloc.lower()}"
    return "openverse-group:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _normalized_row(
    item: dict[str, Any],
    matches: list[tuple[str, str | None]],
) -> dict[str, Any]:
    text = _result_text(item)
    hints = {hint for _, hint in matches if hint is not None}
    weapons = _weapon_candidates(text, hints)
    weapon_status = weapons[0] if len(weapons) == 1 else (
        "ambiguous_multi_weapon" if weapons else "unknown"
    )
    tags = item.get("tags")
    tag_rows = tags if isinstance(tags, list) else []
    return {
        "image_id": f"openverse:{item['id']}",
        "openverse_id": str(item["id"]),
        "title": str(item.get("title")) if item.get("title") else None,
        "creator": str(item.get("creator")) if item.get("creator") else None,
        "creator_url": str(item.get("creator_url")) if item.get("creator_url") else None,
        "provider": str(item.get("provider")) if item.get("provider") else None,
        "source": str(item.get("source")) if item.get("source") else None,
        "source_group_id": _creator_group(item),
        "query_families": sorted({family for family, _ in matches}),
        "weapon_status": weapon_status,
        "weapon_candidates": weapons,
        "weapon_label_source": "query_and_upstream_text_weak_context",
        "semantic_review_status": "strong_query_candidate_pixels_not_reviewed",
        "tags": [
            str(tag.get("name"))
            for tag in tag_rows
            if isinstance(tag, dict) and tag.get("name")
        ],
        "fields_matched": [str(value) for value in item.get("fields_matched", [])],
        "landing_page_url": str(item.get("foreign_landing_url") or "") or None,
        "direct_media_url": str(item.get("url") or "") or None,
        "thumbnail_url": str(item.get("thumbnail") or "") or None,
        "license": str(item.get("license") or "").lower() or None,
        "license_version": (
            str(item.get("license_version")) if item.get("license_version") else None
        ),
        "license_url": str(item.get("license_url")) if item.get("license_url") else None,
        "attribution": str(item.get("attribution")) if item.get("attribution") else None,
        "declared_width": int(item["width"]) if item.get("width") else None,
        "declared_height": int(item["height"]) if item.get("height") else None,
        "declared_filesize": int(item["filesize"]) if item.get("filesize") else None,
        "discovered_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "local_path": None,
        "download_url_used": None,
        "image_format": None,
        "width": None,
        "height": None,
        "downloaded_bytes": None,
        "sha256": None,
        "exact_duplicate_of": None,
        "eligible_exact_unique_weak_context": False,
    }


def discover(
    manifest_path: Path,
    summary_path: Path,
    cache_directory: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    refresh: bool = False,
) -> dict[str, Any]:
    """Collect the anonymous API's bounded top results for every precise query."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    matches_by_id: dict[str, list[tuple[str, str | None]]] = {}
    items_by_id: dict[str, dict[str, Any]] = {}
    raw_by_query: Counter[str] = Counter()
    rejected_by_query: Counter[str] = Counter()
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=120),
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        for query, family, weapon_hint, strong_query in QUERY_SPECS:
            page = 1
            page_count = 1
            while page <= page_count:
                cache_path = _cache_path(cache_directory, family, page)
                if cache_path.exists() and not refresh:
                    payload = _load_payload(cache_path)
                else:
                    payload = _request_json(
                        client,
                        {
                            "q": query,
                            "page": str(page),
                            "page_size": str(PAGE_SIZE),
                            "license": LICENSE_FILTER,
                            "excluded_source": EXCLUDED_SOURCE,
                            "filter_dead": "true",
                            "mature": "false",
                        },
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_bytes(
                        cache_path,
                        (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                            "utf-8"
                        ),
                    )
                    time.sleep(1.25)
                raw_page_count = int(payload.get("page_count") or 0)
                page_count = min(raw_page_count, MAX_ANONYMOUS_RESULTS_PER_QUERY // PAGE_SIZE)
                results = payload.get("results", [])
                if not isinstance(results, list):
                    raise TypeError(f"Openverse results were malformed for {family} page {page}")
                for item in results:
                    if not isinstance(item, dict) or not item.get("id"):
                        continue
                    raw_by_query[family] += 1
                    if not _is_candidate(item, strong_query=strong_query):
                        rejected_by_query[family] += 1
                        continue
                    identifier = str(item["id"])
                    items_by_id.setdefault(identifier, item)
                    matches_by_id.setdefault(identifier, []).append((family, weapon_hint))
                page += 1

    rows = [
        _normalized_row(items_by_id[identifier], matches_by_id[identifier])
        for identifier in sorted(items_by_id)
    ]
    manifest_sha256 = write_jsonl(manifest_path, rows)
    summary: dict[str, Any] = {
        "source_page": SOURCE_PAGE,
        "api_url": API_URL,
        "license_filter": LICENSE_FILTER,
        "excluded_source": EXCLUDED_SOURCE,
        "queries": len(QUERY_SPECS),
        "raw_query_memberships": sum(raw_by_query.values()),
        "raw_by_query": dict(sorted(raw_by_query.items())),
        "semantic_rejections": sum(rejected_by_query.values()),
        "semantic_rejections_by_query": dict(sorted(rejected_by_query.items())),
        "candidate_records": len(rows),
        "candidate_query_memberships": sum(len(row["query_families"]) for row in rows),
        "by_weapon_status": dict(Counter(str(row["weapon_status"]) for row in rows)),
        "by_source": dict(Counter(str(row["source"]) for row in rows)),
        "creator_groups": len({str(row["source_group_id"]) for row in rows}),
        "manifest_sha256": manifest_sha256,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(summary_path, summary)
    return summary


def _download_image(
    client: httpx.Client,
    urls: list[str],
    destination_prefix: Path,
) -> tuple[Path, str, int, int, str]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for Openverse image acquisition") from exc

    errors: list[str] = []
    for url in urls:
        temporary = destination_prefix.with_suffix(".part")
        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                declared = int(response.headers.get("Content-Length", "0") or 0)
                if declared > MAX_IMAGE_BYTES:
                    raise ValueError(f"declared image exceeds {MAX_IMAGE_BYTES} bytes")
                size = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_IMAGE_BYTES:
                            raise ValueError(f"downloaded image exceeds {MAX_IMAGE_BYTES} bytes")
                        handle.write(chunk)
            with Image.open(temporary) as image:
                image.verify()
            with Image.open(temporary) as image:
                image_format = str(image.format)
                width, height = int(image.width), int(image.height)
            extension = _FORMAT_EXTENSIONS.get(image_format)
            if extension is None:
                raise ValueError(f"unsupported image format: {image_format}")
            destination = destination_prefix.with_suffix(extension)
            temporary.replace(destination)
            return destination, image_format, width, height, url
        except (httpx.HTTPError, OSError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def download(
    manifest_path: Path,
    output_directory: Path,
    summary_path: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    """Acquire bounded previews and update the manifest with byte-level provenance."""

    rows = read_jsonl(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(90, read=180),
        trust_env=False,
        headers={"User-Agent": user_agent},
    ) as client:
        for row in rows:
            identifier = str(row["openverse_id"])
            prefix = output_directory / str(row["weapon_status"]) / f"openverse-{identifier}"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            existing = next((path for path in prefix.parent.glob(prefix.name + ".*")), None)
            try:
                if existing is None:
                    urls = [
                        str(url)
                        for url in (row.get("thumbnail_url"), row.get("direct_media_url"))
                        if url
                    ]
                    if not urls:
                        raise RuntimeError("record has no media URL")
                    path, image_format, width, height, used_url = _download_image(
                        client, urls, prefix
                    )
                    time.sleep(0.35)
                else:
                    try:
                        from PIL import Image
                    except ImportError as exc:  # pragma: no cover
                        raise RuntimeError(
                            "Pillow is required for Openverse image acquisition"
                        ) from exc
                    with Image.open(existing) as image:
                        image.verify()
                    with Image.open(existing) as image:
                        path = existing
                        image_format = str(image.format)
                        width, height = int(image.width), int(image.height)
                    used_url = str(row.get("download_url_used") or "resumed_existing_file")
                row["local_path"] = str(path)
                row["download_url_used"] = used_url
                row["image_format"] = image_format
                row["width"] = width
                row["height"] = height
                row["downloaded_bytes"] = path.stat().st_size
                row["sha256"] = sha256_file(path)
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append({"image_id": str(row["image_id"]), "error": str(exc)})

    first_by_hash: dict[str, str] = {}
    for row in rows:
        digest = str(row.get("sha256") or "")
        if not digest:
            row["exact_duplicate_of"] = None
            row["eligible_exact_unique_weak_context"] = False
            continue
        row["exact_duplicate_of"] = first_by_hash.get(digest)
        first_by_hash.setdefault(digest, str(row["image_id"]))
        row["eligible_exact_unique_weak_context"] = row["exact_duplicate_of"] is None

    manifest_sha256 = write_jsonl(manifest_path, rows)
    acquired = [row for row in rows if row.get("sha256")]
    summary: dict[str, Any] = {
        "source_page": SOURCE_PAGE,
        "candidate_records": len(rows),
        "downloaded_records": len(acquired),
        "unique_downloaded_sha256": len({str(row["sha256"]) for row in acquired}),
        "exact_duplicate_rows": len(acquired)
        - len({str(row["sha256"]) for row in acquired}),
        "downloaded_bytes": sum(int(row.get("downloaded_bytes") or 0) for row in acquired),
        "downloaded_by_weapon_status": dict(
            Counter(str(row["weapon_status"]) for row in acquired)
        ),
        "downloaded_by_source": dict(Counter(str(row["source"]) for row in acquired)),
        "creator_groups": len({str(row["source_group_id"]) for row in acquired}),
        "failures": failures,
        "manifest_sha256": manifest_sha256,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(summary_path, summary)
    return summary
