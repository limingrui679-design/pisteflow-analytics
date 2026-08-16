"""Verified-license fencing videos hosted by the Internet Archive."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

_BELL_TOWER_2016_IDS = [
    "Flickr-29467365844",
    "Flickr-29468024623",
    "Flickr-29468269233",
    "Flickr-29468656993",
    "Flickr-29469541574",
    "Flickr-29470411433",
    "Flickr-29801369170",
    "Flickr-29801844430",
    "Flickr-29802379110",
    "Flickr-29982410562",
    "Flickr-30014104321",
    "Flickr-30015851951",
    "Flickr-30061650416",
    "Flickr-30096913085",
]
_BELL_TOWER_2012_IDS = [
    "Flickr-7976244921",
    "Flickr-7975299509",
    "Flickr-7975412769",
    "Flickr-7975153907",
    "Flickr-7975535280",
    "Flickr-7976362166",
    "Flickr-7976238703",
    "Flickr-7975015099",
    "Flickr-7975471901",
    "Flickr-7974879962",
    "Flickr-7974897017",
    "Flickr-7975637822",
    "Flickr-7976257138",
    "Flickr-7975337980",
    "Flickr-7975257677",
    "Flickr-7976233955",
    "Flickr-7975269074",
    "Flickr-7975174700",
    "Flickr-7975344903",
    "Flickr-7974844242",
    "Flickr-7975388680",
    "Flickr-7975438672",
    "Flickr-7976329222",
    "Flickr-7975240350",
    "Flickr-7975290257",
    "Flickr-7975485273",
    "Flickr-7975405278",
    "Flickr-7975420614",
    "Flickr-7975460229",
    "Flickr-7975467094",
    "Flickr-7975177601",
]
_BELL_TOWER_IDS = [*_BELL_TOWER_2012_IDS, *_BELL_TOWER_2016_IDS]
_BELL_TOWER_INTERVIEW_ID = "Flickr-29802379110"
_SPORTS_ATTACK_ID = "mactvmiv-Sports_Attack_Season_4_Episode_5"
IDENTIFIERS = [*_BELL_TOWER_IDS, "teamfoil042405", _SPORTS_ATTACK_ID]
_EXPECTED_LICENSE = {
    **{identifier: "by-nc/2.0" for identifier in _BELL_TOWER_IDS},
    "teamfoil042405": "by-nc-nd/2.0",
    _SPORTS_ATTACK_ID: "by-nc/3.0",
}


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metadata(identifier: str, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("metadata", {}).get("identifier") not in {None, identifier}:
        raise ValueError(f"Internet Archive identifier mismatch for {identifier}")
    metadata = payload.get("metadata")
    files = payload.get("files")
    if not isinstance(metadata, dict) or not isinstance(files, list):
        raise ValueError(f"Internet Archive metadata schema changed for {identifier}")
    license_url = str(metadata.get("licenseurl") or "")
    if _EXPECTED_LICENSE[identifier] not in license_url:
        raise ValueError(f"Internet Archive license changed for {identifier}: {license_url}")
    mp4_files = [
        item
        for item in files
        if isinstance(item, dict) and str(item.get("name", "")).casefold().endswith(".mp4")
    ]
    if len(mp4_files) != 1:
        raise ValueError(f"Expected one MP4 for {identifier}, found {len(mp4_files)}")
    media = mp4_files[0]
    required = ["name", "size", "md5", "sha1", "length"]
    if any(not media.get(field) for field in required):
        raise ValueError(f"Internet Archive media metadata incomplete for {identifier}")
    filename = str(media["name"])
    is_team_foil = identifier == "teamfoil042405"
    is_interview = identifier == _BELL_TOWER_INTERVIEW_ID
    is_sports_attack = identifier == _SPORTS_ATTACK_ID
    if is_sports_attack:
        source_event = "Sports Attack fencing instruction segment (2015)"
        group_id = "ia:sports-attack-s4e5"
        weapon = "unknown"
        weapon_label_source = "multi_weapon_instructional_segment_manual_review_2026_08_15"
        content_type = "real_multi_weapon_instructional_demonstration"
        training_eligible = True
        segment_start_seconds = 586.0
        segment_end_seconds = 839.0
        attribution = "MACTV via Internet Archive item mactvmiv-Sports_Attack_Season_4_Episode_5"
        derivative_policy = "noncommercial_attributed_derivatives_only"
    else:
        is_bell_tower_2012 = identifier in _BELL_TOWER_2012_IDS
        source_event = (
            "Team Foil 2005"
            if is_team_foil
            else ("Bell Tower Classic 2012" if is_bell_tower_2012 else "Bell Tower Classic 2016")
        )
        group_id = (
            "ia:team-foil-2005"
            if is_team_foil
            else (
                "ia:bell-tower-classic-2012"
                if is_bell_tower_2012
                else "ia:bell-tower-classic-2016"
            )
        )
        weapon = "unknown" if is_interview or is_bell_tower_2012 else "foil"
        weapon_label_source = (
            "internet_archive_title"
            if is_team_foil
            else (
                "not_applicable_interview"
                if is_interview
                else (
                    "pending_multi_frame_visual_review"
                    if is_bell_tower_2012
                    else "manual_visual_review_of_multiple_frames_2026_08_14"
                )
            )
        )
        content_type = "event_interview" if is_interview else "real_fencing_action_footage"
        training_eligible = not is_interview
        segment_start_seconds = 0.0
        segment_end_seconds = float(media["length"])
        attribution = (
            "Internet Archive item and original Flickr uploader Ron Miller"
            if not is_team_foil
            else "Internet Archive item teamfoil042405"
        )
        derivative_policy = (
            "local_analysis_only_no_published_derivatives"
            if is_team_foil
            else "noncommercial_attributed_derivatives_only"
        )
    return {
        "video_id": f"internet-archive:{identifier}",
        "archive_identifier": identifier,
        "filename": filename,
        "download_url": f"https://archive.org/download/{identifier}/{quote(filename)}",
        "source_page": f"https://archive.org/details/{identifier}",
        "source_event": source_event,
        "group_id": group_id,
        "weapon": weapon,
        "weapon_label_source": weapon_label_source,
        "content_type": content_type,
        "training_eligible": training_eligible,
        "segment_start_seconds": segment_start_seconds,
        "segment_end_seconds": segment_end_seconds,
        "training_duration_seconds": (
            segment_end_seconds - segment_start_seconds if training_eligible else 0.0
        ),
        "duration_seconds": float(media["length"]),
        "bytes": int(media["size"]),
        "width": int(media["width"]) if media.get("width") else None,
        "height": int(media["height"]) if media.get("height") else None,
        "source_md5": str(media["md5"]),
        "source_sha1": str(media["sha1"]),
        "license_url": license_url,
        "attribution": attribution,
        "derivative_policy": derivative_policy,
        "claim_boundary": (
            "One public real-fencing video asset; not an annotated bout, athlete record, "
            "independent touch sample, or weapon label unless explicitly stated."
        ),
    }


def fetch_and_prepare(
    cache_directory: Path,
    media_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    media_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for identifier in IDENTIFIERS:
        try:
            metadata_path = cache_directory / f"{identifier}.json"
            if force:
                metadata_path.unlink(missing_ok=True)
            download_resumable(
                f"https://archive.org/metadata/{identifier}",
                metadata_path,
                user_agent=user_agent,
            )
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            record = parse_metadata(identifier, payload)
            destination = media_directory / str(record["filename"])
            if force:
                destination.unlink(missing_ok=True)
            download_resumable(
                str(record["download_url"]),
                destination,
                user_agent=user_agent,
                expected_bytes=int(record["bytes"]),
            )
            if _md5_file(destination) != record["source_md5"]:
                raise ValueError(f"Internet Archive MP4 checksum mismatch for {identifier}")
            record["local_sha256"] = sha256_file(destination)
            record["downloaded"] = True
            records.append(record)
        except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
            failures.append({"archive_identifier": identifier, "error": str(exc)})
    records.sort(key=lambda row: str(row["video_id"]))
    manifest_sha256 = write_jsonl(manifest_path, records)
    summary: dict[str, Any] = {
        "claim_boundary": (
            f"{len(records)} of {len(IDENTIFIERS)} discovered real videos currently pass "
            "item-level Creative Commons, download, size, and checksum validation. The 31 short "
            "Bell Tower Classic 2012 clips are one source-event group and remain weapon-unknown "
            "until multi-frame review. Manual review identifies thirteen 2016 Bell Tower action "
            "clips as foil; one 2016 asset is an interview and is excluded from action training. "
            "The 46-minute Team Foil video is marked foil by its source title, but its "
            "NoDerivatives license keeps all analysis derivatives local. The Sports Attack "
            "episode contributes only its manually reviewed 586-839 second multi-weapon "
            "instructional segment; it is not a weapon-labelled bout."
        ),
        "prepared_at": datetime.now(UTC).isoformat(),
        "videos": len(records),
        "action_training_eligible_videos": sum(bool(row["training_eligible"]) for row in records),
        "source_event_groups": len({str(row["group_id"]) for row in records}),
        "total_bytes": sum(int(row["bytes"]) for row in records),
        "total_duration_seconds": sum(float(row["duration_seconds"]) for row in records),
        "training_duration_seconds": sum(
            float(row["training_duration_seconds"]) for row in records
        ),
        "by_weapon": dict(sorted(Counter(str(row["weapon"]) for row in records).items())),
        "by_derivative_policy": dict(
            sorted(Counter(str(row["derivative_policy"]) for row in records).items())
        ),
        "expected_discovered_videos": len(IDENTIFIERS),
        "failures": failures,
        "manifest_sha256": manifest_sha256,
        "output": str(manifest_path),
    }
    write_json(report_path, summary)
    return summary
