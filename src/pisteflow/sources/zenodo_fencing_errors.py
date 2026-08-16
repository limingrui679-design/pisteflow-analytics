"""Audited inventory for the CC BY fencing-movement and error-video dataset."""

from __future__ import annotations

import hashlib
import ssl
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import httpx
import pandas as pd
import truststore

from pisteflow.provenance import (
    atomic_write_bytes,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)

SOURCE_URL = "https://zenodo.org/records/7643646"
SOURCE_DOI = "10.5281/zenodo.7643646"
PAPER_DOI = "10.1109/SDS57534.2023.00025"
LICENSE = "CC BY 4.0"
TAIL_BYTES = 512 * 1024


@dataclass(frozen=True)
class ArchiveSpec:
    action: str
    remote_key: str
    local_name: str
    expected_bytes: int
    expected_md5: str
    expected_originals: int
    labels_name: str | None = None
    labels_md5: str | None = None
    labels_rows: int = 0

    @property
    def url(self) -> str:
        encoded = httpx.URL("https://zenodo.org").copy_with(
            path=f"/api/records/7643646/files/{self.remote_key}/content"
        )
        return str(encoded)


SPECS = (
    ArchiveSpec(
        "counterattack",
        "counterattack.zip",
        "counterattack.zip",
        9_576_409_433,
        "34f4babe64c66bac9447eba008c87e4c",
        298,
        "task3_counterattack_multi_labels.csv",
        "0d16a111b3b96c8330d8b2ab55ea30a9",
        118,
    ),
    ArchiveSpec(
        "fleche",
        "fle\u0300che.zip",
        "fleche.zip",
        8_714_970_070,
        "541e4aa7d7e98147b6f10315b3e9c737",
        358,
        "task3_fleche_multi_labels.csv",
        "c95af2255bfbe4d1239ccf95d7da2bff",
        179,
    ),
    ArchiveSpec(
        "lunge",
        "lunge.zip",
        "lunge.zip",
        16_999_360_989,
        "fb900d585ce54082276af6ce346d41fe",
        513,
        "task3_lunge_multi_labels.csv",
        "a88029c7c84246da68388698e7f1f83a",
        197,
    ),
    ArchiveSpec(
        "parry",
        "parry.zip",
        "parry.zip",
        4_493_229_247,
        "ce779f5b4d36870e0e8f7a610c19b3d7",
        120,
    ),
)

_LABEL_MAP = {
    "correct": "correct",
    "arm_not_streched": "arm_not_stretched",
    "arm_to_late": "arm_too_late",
    "unbalanced": "unbalanced",
    "both_legs_moved_together": "both_legs_moved_together",
    "shoulder_drop": "shoulder_drop",
    "stood_up": "stood_up",
    "front_knee_not_over_front_foot": "front_knee_not_over_front_foot",
    "rear_leg_not_streched": "rear_leg_not_stretched",
    "upper_body_not_straight": "upper_body_not_straight",
}


@dataclass(frozen=True)
class CentralEntry:
    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int


@dataclass(frozen=True)
class MultiLabelAnnotation:
    active_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _zip64_directory(payload: bytes) -> tuple[int, int, int] | None:
    position = payload.rfind(b"PK\x06\x06")
    if position < 0:
        return None
    fields = struct.unpack_from("<4sQ2H2L4Q", payload, position)
    return int(fields[7]), int(fields[8]), int(fields[9])


def parse_central_directory_tail(payload: bytes, total_bytes: int) -> list[CentralEntry]:
    """Parse a complete ZIP central directory held in a byte-range tail."""

    eocd_position = payload.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise ValueError("ZIP end-of-central-directory record is absent from tail")
    standard = struct.unpack_from("<4s4H2LH", payload, eocd_position)
    count, directory_bytes, directory_offset = int(standard[4]), int(standard[5]), int(standard[6])
    zip64 = _zip64_directory(payload)
    if zip64 is not None:
        count, directory_bytes, directory_offset = zip64
    tail_offset = total_bytes - len(payload)
    position = directory_offset - tail_offset
    if position < 0 or position + directory_bytes > len(payload):
        raise ValueError("ZIP tail does not contain the complete central directory")

    entries: list[CentralEntry] = []
    directory_end = position + directory_bytes
    while position < directory_end:
        if payload[position : position + 4] != b"PK\x01\x02":
            raise ValueError("ZIP central-directory entry signature changed")
        fields = struct.unpack_from("<4s6H3L5H2L", payload, position)
        flags = int(fields[3])
        compressed = int(fields[8])
        uncompressed = int(fields[9])
        name_bytes = int(fields[10])
        extra_bytes = int(fields[11])
        comment_bytes = int(fields[12])
        if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF:
            raise ValueError("Unexpected ZIP64-sized individual video member")
        name_start = position + 46
        raw_name = payload[name_start : name_start + name_bytes]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        entries.append(
            CentralEntry(
                name=raw_name.decode(encoding),
                compressed_bytes=compressed,
                uncompressed_bytes=uncompressed,
                crc32=int(fields[7]),
            )
        )
        position = name_start + name_bytes + extra_bytes + comment_bytes
    if len(entries) != count or position != directory_end:
        raise ValueError("ZIP central-directory count or byte extent changed")
    return entries


def _entries_from_complete_archive(path: Path) -> list[CentralEntry]:
    with ZipFile(path) as archive:
        return [
            CentralEntry(item.filename, item.compress_size, item.file_size, item.CRC)
            for item in archive.infolist()
        ]


def _fetch_tail(
    spec: ArchiveSpec,
    path: Path,
    *,
    user_agent: str,
    force: bool,
    attempts: int = 7,
) -> bytes:
    expected = min(TAIL_BYTES, spec.expected_bytes)
    if path.exists() and path.stat().st_size == expected and not force:
        return path.read_bytes()
    start = spec.expected_bytes - expected
    range_value = f"bytes={start}-{spec.expected_bytes - 1}"
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    last_error: Exception | None = None
    with httpx.Client(
        follow_redirects=True,
        trust_env=False,
        verify=ssl_context,
        timeout=httpx.Timeout(60, read=300),
        headers={"User-Agent": user_agent},
    ) as client:
        for attempt in range(attempts):
            try:
                response = client.get(spec.url, headers={"Range": range_value})
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else float(2**attempt)
                    time.sleep(min(delay, 45.0))
                    continue
                response.raise_for_status()
                content_range = response.headers.get("Content-Range")
                expected_range = f"bytes {start}-{spec.expected_bytes - 1}/{spec.expected_bytes}"
                if response.status_code != 206 or content_range != expected_range:
                    raise RuntimeError("Zenodo byte-range response did not match the pinned file")
                if len(response.content) != expected:
                    raise RuntimeError("Zenodo ZIP tail length changed")
                atomic_write_bytes(path, response.content)
                return response.content
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError(f"Could not fetch ZIP inventory tail for {spec.action}") from last_error


def _label_rows(
    spec: ArchiveSpec, archive_directory: Path
) -> dict[str, MultiLabelAnnotation]:
    if spec.labels_name is None or spec.labels_md5 is None:
        return {}
    path = archive_directory / spec.labels_name
    if not path.exists() or _md5_file(path) != spec.labels_md5:
        raise ValueError(f"Missing or changed multi-label CSV for {spec.action}")
    frame = pd.read_csv(path)
    if len(frame) != spec.labels_rows or "video_name" not in frame:
        raise ValueError(f"Multi-label row count or schema changed for {spec.action}")
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]
    if frame["video_name"].isna().any() or frame["video_name"].duplicated().any():
        raise ValueError(f"Multi-label video IDs are missing or duplicated for {spec.action}")
    label_columns = [column for column in frame if column != "video_name"]
    unknown = {column.casefold() for column in label_columns}.difference(_LABEL_MAP)
    if unknown:
        raise ValueError(f"Unknown error labels for {spec.action}: {sorted(unknown)}")
    numeric = frame[label_columns].apply(pd.to_numeric, errors="raise")
    invalid = numeric.notna() & ~numeric.isin([0, 1])
    if invalid.any().any():
        raise ValueError(f"Non-binary multi-label values for {spec.action}")
    labels: dict[str, MultiLabelAnnotation] = {}
    for row_index, video_name in frame["video_name"].items():
        active = [
            _LABEL_MAP[column.casefold()]
            for column in label_columns
            if numeric.loc[row_index, column] == 1
        ]
        missing = [
            _LABEL_MAP[column.casefold()]
            for column in label_columns
            if pd.isna(numeric.loc[row_index, column])
        ]
        labels[str(video_name)] = MultiLabelAnnotation(tuple(active), tuple(missing))
    return labels


def build_action_rows(
    spec: ArchiveSpec,
    entries: list[CentralEntry],
    multi_labels: dict[str, MultiLabelAnnotation],
    *,
    archive_complete_local: bool,
) -> list[dict[str, Any]]:
    media = [
        entry
        for entry in entries
        if entry.name.casefold().endswith((".mov", ".mp4"))
        and not entry.name.startswith("__MACOSX/")
        and not PurePosixPath(entry.name).name.startswith("._")
    ]
    grouped: dict[str, list[CentralEntry]] = defaultdict(list)
    for entry in media:
        stem = PurePosixPath(entry.name).stem
        grouped[stem.removesuffix("_mirrored")].append(entry)
    if len(grouped) != spec.expected_originals:
        raise ValueError(f"Independent-video count changed for {spec.action}")

    rows: list[dict[str, Any]] = []
    for base_name, members in sorted(grouped.items()):
        originals = [
            item
            for item in members
            if not PurePosixPath(item.name).stem.endswith("_mirrored")
        ]
        mirrors = [item for item in members if PurePosixPath(item.name).stem.endswith("_mirrored")]
        if len(originals) != 1 or len(mirrors) != 1:
            raise ValueError(f"Original/mirror pair changed for {spec.action}/{base_name}")
        original, mirror = originals[0], mirrors[0]
        original_parts = PurePosixPath(original.name).parts
        mirror_parts = PurePosixPath(mirror.name).parts
        if len(original_parts) < 3 or original_parts[1] != mirror_parts[1]:
            raise ValueError(f"Source class mismatch for {spec.action}/{base_name}")
        source_label = original_parts[1].casefold()
        if spec.action == "parry":
            canonical_label = source_label
            error_pattern = None
            movement_variant = canonical_label
        else:
            if source_label not in _LABEL_MAP:
                raise ValueError(f"Unknown source folder label for {spec.action}: {source_label}")
            canonical_label = _LABEL_MAP[source_label]
            error_pattern = canonical_label
            movement_variant = None
        multi_label_row_available = base_name in multi_labels
        annotation = multi_labels.get(base_name)
        active_labels = list(annotation.active_labels) if annotation is not None else []
        missing_labels = list(annotation.missing_labels) if annotation is not None else []
        group_digest = hashlib.sha256(f"{spec.action}:{base_name}".encode()).hexdigest()[:20]
        rows.append(
            {
                "clip_group_id": f"zfe:{group_digest}",
                "action": spec.action,
                "weapon": "unknown",
                "source_single_label": source_label,
                "canonical_single_label": canonical_label,
                "movement_variant": movement_variant,
                "error_pattern": error_pattern,
                "multi_label_row_available": multi_label_row_available,
                "multi_label_has_active_label": bool(active_labels),
                "multi_labels": active_labels,
                "multi_label_missing_fields": missing_labels,
                "single_label_present_in_multi_labels": (
                    canonical_label in active_labels if multi_label_row_available else None
                ),
                "source_video_name": base_name,
                "original_member": original.name,
                "mirror_member": mirror.name,
                "original_uncompressed_bytes": original.uncompressed_bytes,
                "mirror_uncompressed_bytes": mirror.uncompressed_bytes,
                "original_crc32": f"{original.crc32:08x}",
                "mirror_crc32": f"{mirror.crc32:08x}",
                "mirror_is_augmentation_not_independent_sample": True,
                "archive": spec.local_name,
                "archive_complete_local": archive_complete_local,
                "athlete_id_available": False,
                "source_url": SOURCE_URL,
                "license": LICENSE,
            }
        )
    orphan_label_ids = set(multi_labels).difference(grouped)
    if orphan_label_ids:
        examples = sorted(orphan_label_ids)[:3]
        raise ValueError(f"Multi-label IDs absent from {spec.action} archive: {examples}")
    return rows


def prepare(
    archive_directory: Path,
    tail_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force_tails: bool = False,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    archive_reports: dict[str, Any] = {}
    tail_directory.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        archive_path = archive_directory / spec.local_name
        local_bytes = archive_path.stat().st_size if archive_path.exists() else 0
        complete = local_bytes == spec.expected_bytes
        if complete:
            observed_md5 = _md5_file(archive_path)
            if observed_md5 != spec.expected_md5:
                raise ValueError(f"Complete archive checksum changed for {spec.action}")
            entries = _entries_from_complete_archive(archive_path)
            inventory_method = "complete_local_archive"
            inventory_sha256 = sha256_file(archive_path)
        else:
            tail_path = tail_directory / f"{spec.action}.zip.tail"
            payload = _fetch_tail(
                spec,
                tail_path,
                user_agent=user_agent,
                force=force_tails,
            )
            entries = parse_central_directory_tail(payload, spec.expected_bytes)
            observed_md5 = None
            inventory_method = "remote_http_range_central_directory"
            inventory_sha256 = sha256_bytes(payload)
        multi_labels = _label_rows(spec, archive_directory)
        rows = build_action_rows(
            spec,
            entries,
            multi_labels,
            archive_complete_local=complete,
        )
        all_rows.extend(rows)
        mismatches = sum(
            row["single_label_present_in_multi_labels"] is False
            for row in rows
            if row["multi_label_row_available"]
        )
        archive_reports[spec.action] = {
            "archive": spec.local_name,
            "expected_bytes": spec.expected_bytes,
            "expected_md5": spec.expected_md5,
            "local_bytes": local_bytes,
            "local_complete": complete,
            "local_md5": observed_md5,
            "inventory_method": inventory_method,
            "inventory_sha256": inventory_sha256,
            "central_directory_entries": len(entries),
            "original_videos": len(rows),
            "mirrored_augmentations": len(rows),
            "multi_label_rows": len(multi_labels),
            "single_vs_multi_label_mismatches": mismatches,
            "source_single_labels": dict(
                sorted(Counter(row["source_single_label"] for row in rows).items())
            ),
        }

    original_count = len(all_rows)
    if original_count != 1_289:
        raise ValueError("Pinned Zenodo independent-video total changed")
    local_rows = [row for row in all_rows if row["archive_complete_local"]]
    report: dict[str, Any] = {
        "claim_boundary": (
            "The CC BY 4.0 deposit contains 1,289 original controlled-training clips and one "
            "horizontal mirror for each original. Mirrors are augmentations, never independent "
            "samples. ZIP central directories and official byte counts establish the full source "
            "inventory; only clips whose archive is complete and checksum-matched locally are "
            "locally trainable. The deposit does not publish athlete identities or weapon labels, "
            "so clip-group splitting prevents mirror leakage but cannot demonstrate "
            "athlete-independent or weapon-specific generalization."
        ),
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "paper_doi": PAPER_DOI,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_archive_bytes": sum(spec.expected_bytes for spec in SPECS),
        "source_original_videos": original_count,
        "source_mirrored_augmentations": original_count,
        "source_media_files": original_count * 2,
        "source_originals_by_action": dict(
            sorted(Counter(row["action"] for row in all_rows).items())
        ),
        "multi_label_rows": sum(row["multi_label_row_available"] for row in all_rows),
        "multi_label_rows_without_active_label": sum(
            row["multi_label_row_available"] and not row["multi_label_has_active_label"]
            for row in all_rows
        ),
        "multi_label_rows_with_multiple_active_labels": sum(
            len(row["multi_labels"]) > 1 for row in all_rows
        ),
        "multi_label_rows_with_missing_fields": sum(
            bool(row["multi_label_missing_fields"]) for row in all_rows
        ),
        "local_complete_archives": sum(item["local_complete"] for item in archive_reports.values()),
        "local_complete_original_videos": len(local_rows),
        "local_complete_media_files": len(local_rows) * 2,
        "local_archive_bytes_acquired": sum(
            item["local_bytes"] for item in archive_reports.values()
        ),
        "weapon_labels_available": 0,
        "athlete_ids_available": 0,
        "archives": archive_reports,
        "manifest": str(manifest_path),
    }
    report["manifest_sha256"] = write_jsonl(manifest_path, all_rows)
    write_json(report_path, report)
    return report
