"""Selective local acquisition of UCF101 fencing and contrast clips."""

from __future__ import annotations

import ssl
import struct
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import httpx
import truststore

from pisteflow.provenance import atomic_write_bytes, read_jsonl, sha256_file, write_jsonl
from pisteflow.sources.net import download_resumable

DATASET_PAGE = "https://www.crcv.ucf.edu/research/data-sets/ucf101/"
SPLITS_ORIGIN_URL = (
    "https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip"
)
SPLITS_URL = (
    "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/"
    "UCF101TrainTestSplits-RecognitionTask.zip"
)
SPLITS_SHA256 = "5c0d1a53b8ed364a2ac830a73f405e51bece7d98ce1254fd19ed4a36b224bd27"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/"
    "UCF-101.zip?download=true"
)
ARCHIVE_BYTES = 6_957_373_664
ARCHIVE_SHA256 = "eb77e54dfafd9c77e7b086f5e5ef7738cabd16a701b3f5c301a1f6c81fc4756e"
CENTRAL_DIRECTORY_OFFSET = 6_955_776_487
CENTRAL_DIRECTORY_BYTES = 1_597_079
CENTRAL_DIRECTORY_SHA256 = "1f572545589c0b55e4031e6dd23657d578e804b37c348d8db97780715582636d"
DEFAULT_CLASSES = ("Fencing",)


def build_manifest(
    split_archive: Path,
    output_path: Path,
    *,
    user_agent: str,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> list[dict[str, Any]]:
    """Use the official split lists rather than guessing clip filenames."""

    download_resumable(
        SPLITS_URL,
        split_archive,
        user_agent=user_agent,
        expected_bytes=113_943,
    )
    actual_sha256 = sha256_file(split_archive)
    if actual_sha256 != SPLITS_SHA256:
        raise ValueError(
            f"UCF101 split archive checksum mismatch: expected {SPLITS_SHA256}, got {actual_sha256}"
        )
    wanted = set(classes)
    with ZipFile(split_archive) as archive:
        train_rows = archive.read("ucfTrainTestlist/trainlist01.txt").decode().splitlines()
        test_rows = archive.read("ucfTrainTestlist/testlist01.txt").decode().splitlines()

    split_by_path: dict[str, str] = {}
    for line in train_rows:
        relative = line.split()[0]
        if PurePosixPath(relative).parts[0] in wanted:
            split_by_path[relative] = "train"
    for line in test_rows:
        relative = line.strip()
        if relative and PurePosixPath(relative).parts[0] in wanted:
            split_by_path[relative] = "test"

    records: list[dict[str, Any]] = []
    for relative, split in sorted(split_by_path.items()):
        path = PurePosixPath(relative)
        group = path.stem.split("_g", 1)[-1].split("_c", 1)[0]
        records.append(
            {
                "clip_id": path.stem,
                "dataset": "UCF101",
                "class_label": path.parts[0],
                "task_label": "fencing" if path.parts[0] == "Fencing" else "contrast",
                "group_id": f"ucf101:{path.parts[0]}:g{group}",
                "official_split_1": split,
                "source_url": DATASET_PAGE,
                "archive_url": ARCHIVE_URL,
                "archive_sha256": ARCHIVE_SHA256,
                "archive_member": f"UCF-101/{relative}",
                "dataset_page": DATASET_PAGE,
                "split_origin_url": SPLITS_ORIGIN_URL,
                "split_archive_sha256": SPLITS_SHA256,
                "license_status": "research_dataset_terms_unverified",
                "redistribution": "raw_media_not_committed",
                "sha256": None,
                "bytes": None,
            }
        )
    write_jsonl(output_path, records)
    return records


def parse_central_directory(payload: bytes) -> dict[str, dict[str, int]]:
    """Read the pinned ZIP central directory without acquiring the 6.96 GB archive."""

    members: dict[str, dict[str, int]] = {}
    position = 0
    header = struct.Struct("<4s6H3I5H2I")
    while position < len(payload):
        if position + header.size > len(payload):
            raise ValueError("UCF101 central directory ends inside a header")
        fields = header.unpack_from(payload, position)
        if fields[0] != b"PK\x01\x02":
            raise ValueError(f"Invalid UCF101 central-directory signature at {position}")
        flags = fields[3]
        compression = fields[4]
        crc32 = fields[7]
        compressed_bytes = fields[8]
        uncompressed_bytes = fields[9]
        filename_bytes = fields[10]
        extra_bytes = fields[11]
        comment_bytes = fields[12]
        local_header_offset = fields[16]
        end = position + header.size + filename_bytes + extra_bytes + comment_bytes
        if end > len(payload):
            raise ValueError("UCF101 central-directory member exceeds the payload")
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = payload[
            position + header.size : position + header.size + filename_bytes
        ].decode(encoding)
        extra = payload[
            position + header.size + filename_bytes :
            position + header.size + filename_bytes + extra_bytes
        ]
        if max(compressed_bytes, uncompressed_bytes, local_header_offset) == 0xFFFFFFFF:
            zip64 = _zip64_extra(extra, name)
            zip64_position = 0
            if uncompressed_bytes == 0xFFFFFFFF:
                uncompressed_bytes = struct.unpack_from("<Q", zip64, zip64_position)[0]
                zip64_position += 8
            if compressed_bytes == 0xFFFFFFFF:
                compressed_bytes = struct.unpack_from("<Q", zip64, zip64_position)[0]
                zip64_position += 8
            if local_header_offset == 0xFFFFFFFF:
                local_header_offset = struct.unpack_from("<Q", zip64, zip64_position)[0]
        members[name] = {
            "flags": flags,
            "compression": compression,
            "crc32": crc32,
            "compressed_bytes": compressed_bytes,
            "bytes": uncompressed_bytes,
            "filename_bytes": filename_bytes,
            "central_extra_bytes": extra_bytes,
            "local_header_offset": local_header_offset,
        }
        position = end
    return members


def _zip64_extra(extra: bytes, name: str) -> bytes:
    position = 0
    while position + 4 <= len(extra):
        field_id, field_bytes = struct.unpack_from("<HH", extra, position)
        start = position + 4
        end = start + field_bytes
        if end > len(extra):
            break
        if field_id == 0x0001:
            return extra[start:end]
        position = end
    raise ValueError(f"ZIP64 metadata is missing for {name}")


def _fetch_range(
    client: httpx.Client,
    start: int,
    end: int,
    *,
    attempts: int = 8,
) -> bytes:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"})
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(float(2**attempt), 30.0))
                continue
            if response.status_code != 206:
                raise RuntimeError(
                    f"UCF101 mirror ignored byte range {start}-{end}: {response.status_code}"
                )
            if len(response.content) != expected:
                raise RuntimeError(
                    f"UCF101 byte-range length mismatch: expected {expected}, "
                    f"got {len(response.content)}"
                )
            return response.content
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 30.0))
    raise RuntimeError(f"Could not fetch UCF101 byte range {start}-{end}") from last_error


def _central_directory(
    client: httpx.Client,
    cache_path: Path,
) -> dict[str, dict[str, int]]:
    if not cache_path.exists() or sha256_file(cache_path) != CENTRAL_DIRECTORY_SHA256:
        payload = _fetch_range(
            client,
            CENTRAL_DIRECTORY_OFFSET,
            CENTRAL_DIRECTORY_OFFSET + CENTRAL_DIRECTORY_BYTES - 1,
        )
        atomic_write_bytes(cache_path, payload)
    payload = cache_path.read_bytes()
    if len(payload) != CENTRAL_DIRECTORY_BYTES:
        raise ValueError("Cached UCF101 central-directory size mismatch")
    if sha256_file(cache_path) != CENTRAL_DIRECTORY_SHA256:
        raise ValueError("Cached UCF101 central-directory checksum mismatch")
    return parse_central_directory(payload)


def _extract_remote_member(
    client: httpx.Client,
    member: dict[str, int],
) -> bytes:
    offset = member["local_header_offset"]
    request_bytes = (
        30
        + member["filename_bytes"]
        + member["central_extra_bytes"]
        + member["compressed_bytes"]
        + 2_048
    )
    end = min(offset + request_bytes - 1, CENTRAL_DIRECTORY_OFFSET - 1)
    blob = _fetch_range(client, offset, end)
    local = struct.Struct("<4s5H3I2H").unpack_from(blob)
    if local[0] != b"PK\x03\x04":
        raise ValueError("UCF101 member has an invalid local-file signature")
    if local[2] & 0x1:
        raise ValueError("Encrypted UCF101 members are unsupported")
    if local[3] != member["compression"]:
        raise ValueError("UCF101 local and central compression methods disagree")
    data_start = 30 + local[9] + local[10]
    data_end = data_start + member["compressed_bytes"]
    if data_end > len(blob):
        raise ValueError("UCF101 member byte range is incomplete")
    compressed = blob[data_start:data_end]
    if member["compression"] == 0:
        payload = compressed
    elif member["compression"] == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported UCF101 ZIP method: {member['compression']}")
    if len(payload) != member["bytes"]:
        raise ValueError("UCF101 member uncompressed-size mismatch")
    if zlib.crc32(payload) & 0xFFFFFFFF != member["crc32"]:
        raise ValueError("UCF101 member CRC-32 mismatch")
    return payload


def _valid_avi(path: Path, member: dict[str, int]) -> bool:
    if not path.exists() or path.stat().st_size != member["bytes"]:
        return False
    payload = path.read_bytes()
    return (
        payload[:4] == b"RIFF"
        and payload[8:12] == b"AVI "
        and zlib.crc32(payload) & 0xFFFFFFFF == member["crc32"]
    )


def download_manifest(
    manifest_path: Path,
    output_directory: Path,
    central_directory_path: Path,
    *,
    user_agent: str,
    workers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Selectively range-fetch verified members; the full 6.96 GB ZIP is never acquired."""

    records = read_jsonl(manifest_path)
    updated: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(
        follow_redirects=True,
        trust_env=False,
        verify=ssl_context,
        timeout=httpx.Timeout(60, read=300),
        headers={"User-Agent": user_agent},
    ) as client:
        members = _central_directory(client, central_directory_path)
        fencing_members = {
            name for name, member in members.items() if "/Fencing/" in name and member["bytes"]
        }
        expected_members = {str(record["archive_member"]) for record in records}
        if fencing_members != expected_members:
            raise ValueError(
                "UCF101 manifest and remote archive disagree: "
                f"remote={len(fencing_members)}, manifest={len(expected_members)}"
            )
        def acquire(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str] | None]:
            class_label = str(record["class_label"])
            clip_id = str(record["clip_id"])
            destination = output_directory / class_label / f"{clip_id}.avi"
            member = members[str(record["archive_member"])]
            try:
                if not _valid_avi(destination, member):
                    payload = _extract_remote_member(client, member)
                    if payload[:4] != b"RIFF" or payload[8:12] != b"AVI ":
                        raise ValueError("UCF101 member is not an AVI RIFF payload")
                    atomic_write_bytes(destination, payload)
                next_record = dict(record)
                next_record["sha256"] = sha256_file(destination)
                next_record["bytes"] = destination.stat().st_size
                next_record["crc32"] = f"{member['crc32']:08x}"
                next_record["download_validation"] = "zip_size_crc32_and_avi_signature"
                return next_record, None
            except (OSError, RuntimeError, ValueError, zlib.error) as exc:
                return record, {"clip_id": clip_id, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for next_record, failure in executor.map(acquire, records):
                updated.append(next_record)
                if failure is not None:
                    failures.append(failure)
    write_jsonl(manifest_path, updated)
    return updated, failures


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "downloaded": sum(bool(record.get("sha256")) for record in records),
        "bytes": sum(int(record.get("bytes") or 0) for record in records),
        "by_class": dict(sorted(Counter(str(row["class_label"]) for row in records).items())),
        "by_split": dict(sorted(Counter(str(row["official_split_1"]) for row in records).items())),
        "source_groups": len({str(row["group_id"]) for row in records}),
    }
