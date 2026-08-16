"""Metadata-only import for the official FIE YouTube channel."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import HttpUrl

from pisteflow.models import LicenseStatus, SourceKind, VideoRecord, Weapon
from pisteflow.provenance import read_jsonl, write_json, write_jsonl
from pisteflow.sources.classification import (
    classify_content,
    infer_weapon_candidates,
    primary_weapon,
)

FIE_CHANNEL_URL = "https://www.youtube.com/@FIEvideo"


def import_flat_playlist(input_path: Path, output_path: Path) -> list[VideoRecord]:
    """Convert a yt-dlp flat-playlist snapshot without downloading any media."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("yt-dlp payload does not contain an entries list")
    epoch = payload.get("epoch")
    discovered_at = (
        datetime.fromtimestamp(float(epoch), UTC)
        if isinstance(epoch, int | float)
        else datetime.fromtimestamp(input_path.stat().st_mtime, UTC)
    )

    records: list[VideoRecord] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        identifier = str(raw.get("id") or "").strip()
        title = str(raw.get("title") or "").strip()
        if not identifier or not title or identifier in seen:
            continue
        seen.add(identifier)
        source_url = str(raw.get("url") or f"https://www.youtube.com/watch?v={identifier}")
        candidates = infer_weapon_candidates(title)
        duration = _optional_float(raw.get("duration"))
        records.append(
            VideoRecord(
                video_id=f"youtube:{identifier}",
                title=title,
                weapon=primary_weapon(candidates),
                weapon_candidates=list(candidates),
                source_kind=SourceKind.YOUTUBE,
                source_url=HttpUrl(source_url),
                landing_page_url=HttpUrl(source_url),
                license_status=LicenseStatus.LINK_ONLY,
                duration_seconds=duration,
                discovered_at=discovered_at,
                reviewed=False,
                group_id=f"youtube:{identifier}",
                content_kind=classify_content(title, duration_seconds=duration),
                view_count=_optional_int(raw.get("view_count")),
                metadata={
                    "channel": "International Fencing Federation",
                    "channel_url": FIE_CHANNEL_URL,
                    "catalog_method": "yt_dlp_flat_playlist",
                    "automatic_weapon_label": True,
                },
                notes=(
                    "metadata-only; media not downloaded; "
                    f"weapon_candidates={','.join(weapon.value for weapon in candidates) or 'none'}"
                ),
            )
        )
    records.sort(key=lambda record: record.video_id)
    write_jsonl(output_path, [record.model_dump(mode="json") for record in records])
    return records


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    return None


def summarize(records: list[VideoRecord]) -> dict[str, Any]:
    by_weapon = Counter(record.weapon.value for record in records)
    by_kind = Counter(record.content_kind for record in records)
    return {
        "records": len(records),
        "by_weapon": dict(sorted(by_weapon.items())),
        "by_content_kind": dict(sorted(by_kind.items())),
        "mixed_weapon": sum(len(record.weapon_candidates) > 1 for record in records),
        "explicit_single_weapon": sum(len(record.weapon_candidates) == 1 for record in records),
        "hours_with_duration": round(
            sum(record.duration_seconds or 0 for record in records) / 3600,
            2,
        ),
    }


def fetch_playlist_members(
    index_path: Path,
    cache_directory: Path,
    output_path: Path,
    *,
    workers: int = 2,
) -> dict[str, Any]:
    """Enumerate official playlist membership without retrieving any media."""

    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the collectors extra: pip install -e '.[collectors]'") from exc

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    playlists = [row for row in payload.get("entries", []) if isinstance(row, dict)]
    cache_directory.mkdir(parents=True, exist_ok=True)

    def load_playlist(row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        identifier = str(row.get("id") or "").strip()
        url = str(row.get("url") or "").strip()
        cache_path = cache_directory / f"{identifier}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), None
        options = {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=False)
            if not isinstance(info, dict):
                raise ValueError("empty playlist payload")
            write_json(cache_path, info)
            return info, None
        except Exception as exc:  # yt-dlp has extractor-specific exception classes
            return row, str(exc)

    loaded: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_rows = {executor.submit(load_playlist, row): row for row in playlists}
        for future in as_completed(future_rows):
            original = future_rows[future]
            info, error = future.result()
            if error:
                failures.append({"playlist_id": str(original.get("id")), "error": error})
            else:
                loaded.append(info)

    rows: list[dict[str, Any]] = []
    for playlist in loaded:
        playlist_id = str(playlist.get("id") or "")
        playlist_title = str(playlist.get("title") or "")
        entries = [row for row in playlist.get("entries", []) if isinstance(row, dict)]
        title_candidates = infer_weapon_candidates(playlist_title)
        member_candidate_sets = [
            infer_weapon_candidates(str(entry.get("title") or "")) for entry in entries
        ]
        observed_single = [values[0] for values in member_candidate_sets if len(values) == 1]
        unanimous_member_weapon = (
            observed_single[0]
            if len(observed_single) >= 2 and len(set(observed_single)) == 1
            else None
        )
        playlist_candidates = (
            title_candidates
            if title_candidates
            else ((unanimous_member_weapon,) if unanimous_member_weapon else ())
        )
        evidence = "playlist_title" if title_candidates else "member_title_consensus"
        for position, entry in enumerate(entries, start=1):
            video_id = str(entry.get("id") or "").strip()
            if not video_id:
                continue
            rows.append(
                {
                    "playlist_id": playlist_id,
                    "playlist_title": playlist_title,
                    "playlist_url": f"https://www.youtube.com/playlist?list={playlist_id}",
                    "position": position,
                    "video_id": video_id,
                    "video_title": str(entry.get("title") or ""),
                    "video_url": f"https://www.youtube.com/watch?v={video_id}",
                    "playlist_weapon_candidates": [weapon.value for weapon in playlist_candidates],
                    "playlist_weapon_evidence": evidence if playlist_candidates else "none",
                }
            )
    rows.sort(key=lambda row: (str(row["playlist_id"]), int(row["position"])))
    write_jsonl(output_path, rows)
    return {
        "playlists_in_index": len(playlists),
        "playlists_loaded": len(loaded),
        "failures": failures,
        "membership_rows": len(rows),
        "unique_videos": len({str(row["video_id"]) for row in rows}),
        "weapon_evidence_rows": sum(bool(row["playlist_weapon_candidates"]) for row in rows),
        "output": str(output_path),
    }


def enrich_from_playlists(
    video_manifest_path: Path,
    membership_path: Path,
) -> list[VideoRecord]:
    """Add playlist-backed labels only when all evidence agrees on one weapon."""

    records = [VideoRecord.model_validate(row) for row in read_jsonl(video_manifest_path)]
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(membership_path):
        by_video.setdefault(str(row["video_id"]), []).append(row)

    enriched: list[VideoRecord] = []
    for record in records:
        identifier = record.video_id.removeprefix("youtube:")
        memberships = by_video.get(identifier, [])
        candidates = set(record.weapon_candidates)
        for membership in memberships:
            for value in membership.get("playlist_weapon_candidates", []):
                candidates.add(Weapon(str(value)))
        ordered = tuple(sorted(candidates, key=lambda weapon: weapon.value))
        metadata = dict(record.metadata)
        metadata["playlists"] = [
            {
                "id": row["playlist_id"],
                "title": row["playlist_title"],
                "weapon_evidence": row["playlist_weapon_evidence"],
            }
            for row in memberships
        ]
        metadata["weapon_evidence_sources"] = [
            "video_title",
            *("playlist" for _ in memberships),
        ]
        enriched.append(
            record.model_copy(
                update={
                    "weapon": primary_weapon(ordered),
                    "weapon_candidates": list(ordered),
                    "metadata": metadata,
                    "notes": (
                        "metadata-only; media not downloaded; "
                        f"weapon_candidates={','.join(value.value for value in ordered) or 'none'}"
                    ),
                }
            )
        )
    write_jsonl(
        video_manifest_path,
        [record.model_dump(mode="json") for record in enriched],
    )
    return enriched
