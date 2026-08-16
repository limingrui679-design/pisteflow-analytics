"""Licensed instructional fencing clips from MIT OpenCourseWare."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

IDENTIFIER = "MITPE.740S06"
DATASET_PAGE = "https://ocw.mit.edu/courses/pe-740-fencing-spring-2007/pages/video/"
METADATA_URL = f"https://archive.org/metadata/{IDENTIFIER}"
MEDIA_BASE = f"https://archive.org/download/{IDENTIFIER}"
LICENSE = "CC BY-NC-SA 3.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/"


def _license_identity(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    return parsed.netloc.casefold(), parsed.path.rstrip("/").casefold()


def infer_action(filename: str) -> str:
    stem = filename.removesuffix("-220k_512kb.mp4")
    if stem == "parry_4_repost":
        return "parry_4_riposte"
    if stem.startswith("simple_thrust_with_"):
        return stem
    if stem == "simple_thrusts_group":
        return "simple_thrust"
    if "jump_forward_jump_back" in stem:
        return "jump_forward_back"
    if stem in {"jumpe_lunge", "jump_lunge_group", "jump_lunge"}:
        return "jump_lunge"
    if stem.startswith("advance_lunge"):
        return "advance_lunge"
    if stem == "lunge":
        return "lunge"
    if "step_forward" in stem or "steps_forward" in stem:
        return "advance"
    if "step_back" in stem or "steps_back" in stem:
        return "retreat"
    if stem.startswith("group_footwork"):
        return "mixed_footwork"
    return "other"


def fetch_and_download(
    output_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
) -> dict[str, Any]:
    with httpx.Client(
        follow_redirects=True,
        timeout=60,
        headers={"User-Agent": user_agent},
    ) as client:
        response = client.get(METADATA_URL)
        response.raise_for_status()
        metadata = response.json()
    item = metadata.get("metadata", {})
    if _license_identity(str(item.get("licenseurl", ""))) != _license_identity(LICENSE_URL):
        raise ValueError("MIT OCW archive license changed; review before downloading")
    files = [
        row for row in metadata.get("files", []) if str(row.get("name", "")).endswith("_512kb.mp4")
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in sorted(files, key=lambda value: str(value["name"])):
        filename = str(row["name"])
        destination = output_directory / filename
        download_resumable(
            f"{MEDIA_BASE}/{filename}",
            destination,
            user_agent=user_agent,
            expected_bytes=int(row["size"]),
        )
        md5 = hashlib.md5(destination.read_bytes(), usedforsecurity=False).hexdigest()
        if md5 != row.get("md5"):
            raise ValueError(f"MIT OCW checksum mismatch for {filename}")
        records.append(
            {
                "clip_id": f"mit-ocw:{filename.removesuffix('.mp4')}",
                "filename": filename,
                "action_label": infer_action(filename),
                "weapon": "unknown",
                "content_kind": "instructional",
                "group_id": IDENTIFIER,
                "source_url": f"{MEDIA_BASE}/{filename}",
                "dataset_page": DATASET_PAGE,
                "license": LICENSE,
                "license_url": LICENSE_URL,
                "attribution": "MIT OpenCourseWare, PE.740 Fencing, Spring 2007",
                "bytes": destination.stat().st_size,
                "upstream_md5": md5,
                "sha256": sha256_file(destination),
            }
        )
    manifest_sha256 = write_jsonl(manifest_path, records)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Licensed instructional demonstrations from one MIT course/capture context. "
            "They are action examples, not competition outcomes or weapon labels."
        ),
        "source_url": DATASET_PAGE,
        "archive_identifier": IDENTIFIER,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "clips": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "by_action": dict(sorted(Counter(str(row["action_label"]) for row in records).items())),
        "manifest_sha256": manifest_sha256,
        "output": str(manifest_path),
    }
    write_json(report_path, summary)
    return summary
