"""Cross-source exact-identity and weak perceptual audit for fencing still images."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

from pisteflow.provenance import read_jsonl, sha256_file, write_json, write_jsonl


@dataclass(frozen=True)
class SourceSpec:
    name: str
    manifest_name: str
    id_field: str
    hash_field: str
    path_resolver: Callable[[dict[str, Any], Path], Path | None]


def _manifest_local_path(row: dict[str, Any], repository: Path) -> Path | None:
    value = row.get("local_path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repository / "data" / path


def _openimages_path(row: dict[str, Any], repository: Path) -> Path | None:
    if not row.get("media_acquired"):
        return None
    return (
        repository
        / "data/raw/openimages-v7"
        / str(row["upstream_split"])
        / f"{row['image_id']}.jpg"
    )


def _sports_images_path(row: dict[str, Any], repository: Path) -> Path | None:
    return repository / "data/raw/sports-images/fencing" / str(row["local_filename"])


def _mpii_path(row: dict[str, Any], repository: Path) -> Path | None:
    if not row.get("media_downloaded"):
        return None
    return repository / "data/cache/mpii-human-pose/fencing-images" / str(
        row["local_filename"]
    )


def _fencevision_path(row: dict[str, Any], repository: Path) -> Path | None:
    return (
        repository
        / "data/cache/external-repositories/fencevision/fencing/data/blade_dataset"
        / str(row["relative_image_path"])
    )


def _no_file(_row: dict[str, Any], _repository: Path) -> Path | None:
    return None


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "commons_categories",
        "commons-fencing-images.jsonl",
        "image_id",
        "sha256",
        _manifest_local_path,
    ),
    SourceSpec(
        "wikidata_p18",
        "wikidata-fencer-images.jsonl",
        "image_id",
        "sha256",
        _manifest_local_path,
    ),
    SourceSpec(
        "openimages",
        "openimages-fencing-images.jsonl",
        "image_id",
        "image_sha256",
        _openimages_path,
    ),
    SourceSpec(
        "openverse",
        "openverse-fencing-images.jsonl",
        "image_id",
        "sha256",
        _manifest_local_path,
    ),
    SourceSpec(
        "library_of_congress",
        "loc-fencing-images.jsonl",
        "image_id",
        "sha256",
        _manifest_local_path,
    ),
    SourceSpec(
        "womensports",
        "womensports-fencing-images.jsonl",
        "sample_id",
        "sha256",
        _manifest_local_path,
    ),
    SourceSpec(
        "sports_images",
        "sports-images-fencing.jsonl",
        "image_id",
        "sha256",
        _sports_images_path,
    ),
    SourceSpec(
        "mpii_pose",
        "mpii-fencing-pose-images.jsonl",
        "image_id",
        "image_sha256",
        _mpii_path,
    ),
    SourceSpec(
        "fencevision_blade",
        "fencevision-blade-images.jsonl",
        "sample_id",
        "image_sha256",
        _fencevision_path,
    ),
    SourceSpec(
        "cmu_touch_frames",
        "cmu-touch-images.jsonl",
        "sample_id",
        "image_sha256",
        _no_file,
    ),
    SourceSpec(
        "rt_fencing_touch_crops",
        "rt-fencing-images.jsonl",
        "sample_id",
        "sha256",
        _no_file,
    ),
)

FLICKR_PATTERN = re.compile(r"/(?:photos/[^/]+|photo\.gne\?id=)/(\d{5,})")


def _canonical_landing_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return None
    split = urlsplit(value)
    host = split.netloc.casefold().removeprefix("www.")
    path = split.path.rstrip("/")
    if not host or not path:
        return None
    return f"url:{host}{path}"


def upstream_identities(row: dict[str, Any]) -> list[str]:
    """Return only item-level identities that can connect rows across sources."""

    identities: set[str] = set()
    commons_sha1 = str(row.get("commons_sha1") or "").casefold()
    if re.fullmatch(r"[0-9a-f]{40}", commons_sha1):
        identities.add(f"commons-sha1:{commons_sha1}")
    for field in ["landing_page_url", "original_landing_url"]:
        value = row.get(field)
        canonical = _canonical_landing_url(value)
        if canonical:
            identities.add(canonical)
        if isinstance(value, str):
            match = FLICKR_PATTERN.search(value)
            if match:
                identities.add(f"flickr-photo:{match.group(1)}")
    source_url = row.get("source_url")
    if isinstance(source_url, str) and "loc.gov/item/" in source_url:
        canonical = _canonical_landing_url(source_url)
        if canonical:
            identities.add(canonical)
    return sorted(identities)


def _dhash64(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(
            image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata()
        )
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def audit(
    repository: Path,
    output_path: Path,
    report_path: Path,
    *,
    sources: tuple[SourceSpec, ...] = SOURCES,
) -> dict[str, Any]:
    """Build conservative exact/source-identity groups and weak dHash candidates."""

    manifest_directory = repository / "data/manifests"
    rows: list[dict[str, Any]] = []
    source_manifest_rows: Counter[str] = Counter()
    for specification in sources:
        path = manifest_directory / specification.manifest_name
        if not path.exists():
            continue
        for raw in read_jsonl(path):
            source_manifest_rows[specification.name] += 1
            digest = str(raw.get(specification.hash_field) or "").casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                continue
            local_path = specification.path_resolver(raw, repository)
            file_status = "not_individually_available"
            perceptual_hash: str | None = None
            if local_path is not None:
                if local_path.exists():
                    actual = sha256_file(local_path)
                    if actual != digest:
                        raise ValueError(
                            f"image hash mismatch for {specification.name}: {local_path}"
                        )
                    file_status = "verified"
                    perceptual_hash = _dhash64(local_path)
                else:
                    file_status = "missing"
            identities = upstream_identities(raw)
            rows.append(
                {
                    "source": specification.name,
                    "image_id": str(raw.get(specification.id_field) or ""),
                    "sha256": digest,
                    "upstream_identities": identities,
                    "local_file_status": file_status,
                    "dhash64": perceptual_hash,
                    "weapon": str(
                        raw.get("weapon_status") or raw.get("weapon") or "unknown"
                    ),
                    "source_group_id": str(
                        raw.get("source_group_id")
                        or raw.get("source_group")
                        or raw.get("group_id")
                        or ""
                    ),
                }
            )
    if not rows:
        raise ValueError("image union audit found no checksummed image rows")

    union_find = _UnionFind(len(rows))
    token_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        tokens = [f"sha256:{row['sha256']}", *row["upstream_identities"]]
        for token in tokens:
            previous = token_owner.setdefault(str(token), index)
            union_find.union(index, previous)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union_find.find(index)].append(index)
    for members in components.values():
        signature = "|".join(
            sorted(f"{rows[index]['source']}:{rows[index]['image_id']}" for index in members)
        )
        component_id = f"image-union:{hashlib.sha256(signature.encode()).hexdigest()[:20]}"
        for index in members:
            rows[index]["union_group_id"] = component_id
            rows[index]["union_group_size"] = len(members)
            rows[index]["union_group_sources"] = sorted(
                {str(rows[member]["source"]) for member in members}
            )

    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sha[str(row["sha256"])].append(row)
        for identity in row["upstream_identities"]:
            by_identity[str(identity)].append(row)
    exact_groups = [members for members in by_sha.values() if len(members) > 1]
    cross_source_exact = [
        members for members in exact_groups if len({row["source"] for row in members}) > 1
    ]
    cross_source_identity = [
        members
        for members in by_identity.values()
        if len(members) > 1 and len({row["source"] for row in members}) > 1
    ]
    multirow_components = [members for members in components.values() if len(members) > 1]
    cross_source_components = [
        members
        for members in multirow_components
        if len({rows[index]["source"] for index in members}) > 1
    ]

    verified_dhash = [row for row in rows if isinstance(row["dhash64"], str)]
    dhash_candidate_pairs = 0
    dhash_cross_source_pairs = 0
    for left_index, left in enumerate(verified_dhash):
        left_value = int(str(left["dhash64"]), 16)
        for right in verified_dhash[left_index + 1 :]:
            if left["sha256"] == right["sha256"]:
                continue
            distance = (left_value ^ int(str(right["dhash64"]), 16)).bit_count()
            if distance <= 4:
                dhash_candidate_pairs += 1
                if left["source"] != right["source"]:
                    dhash_cross_source_pairs += 1

    rows.sort(key=lambda row: (str(row["source"]), str(row["image_id"])))
    manifest_sha256 = write_jsonl(output_path, rows)

    summary: dict[str, Any] = {
        "source_manifest_rows": dict(sorted(source_manifest_rows.items())),
        "rows_with_sha256": len(rows),
        "unique_sha256": len(by_sha),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_extra_rows": sum(len(members) - 1 for members in exact_groups),
        "cross_source_exact_duplicate_groups": len(cross_source_exact),
        "cross_source_upstream_identity_groups": len(cross_source_identity),
        "conservative_union_groups": len(components),
        "multirow_union_groups": len(multirow_components),
        "cross_source_union_groups": len(cross_source_components),
        "union_duplicate_extra_rows": len(rows) - len(components),
        "local_files_verified": sum(row["local_file_status"] == "verified" for row in rows),
        "local_files_missing": sum(row["local_file_status"] == "missing" for row in rows),
        "rows_not_individually_available": sum(
            row["local_file_status"] == "not_individually_available" for row in rows
        ),
        "dhash_rows": len(verified_dhash),
        "dhash_distance_le_4_candidate_pairs": dhash_candidate_pairs,
        "dhash_distance_le_4_cross_source_candidate_pairs": dhash_cross_source_pairs,
        "manifest_sha256": manifest_sha256,
        "claim_boundary": (
            "Conservative union groups merge exact downloaded-byte hashes and authoritative "
            "item identities such as Commons SHA1 or Flickr photo IDs. They do not merge dHash "
            "near-neighbors: 64-bit dHash candidates can include visually similar but distinct "
            "fencing frames and require review. Export augmentations, repeated frames, athlete "
            "memberships, and resized copies are not independent athletes or actions. Sources "
            "without per-image hashes are outside this exact audit."
        ),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
