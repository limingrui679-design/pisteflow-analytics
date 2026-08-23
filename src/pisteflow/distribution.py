"""Fail closed when Python distributions contain missing or oversized payloads."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

MAX_DISTRIBUTION_BYTES = 5 * 1024 * 1024
BANNED_SDIST_SEGMENTS = (
    "data/cache/",
    "data/models/",
    "data/processed/",
    "data/raw/",
)
REQUIRED_SDIST_PATHS = {
    "data/registry/sources.yml",
    "docs/showcase/index.html",
    "reports/model-cards/france-spatial-access.json",
    "scripts/audit_distribution.py",
    "scripts/sync_showcase.py",
    "src/pisteflow/distribution.py",
    "src/pisteflow/resources/showcase_cases.json",
}
REQUIRED_WHEEL_PATHS = {
    "pisteflow/commands/showcase.py",
    "pisteflow/distribution.py",
    "pisteflow/entrypoint.py",
    "pisteflow/py.typed",
    "pisteflow/resources/showcase_cases.json",
}


def _safe_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _single(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(candidates)}")
    return candidates[0]


def audit_sdist(path: Path) -> dict[str, int | str]:
    """Validate source-distribution size, paths, payload scope, and review surfaces."""

    if path.stat().st_size > MAX_DISTRIBUTION_BYTES:
        raise ValueError(f"sdist exceeds {MAX_DISTRIBUTION_BYTES} bytes: {path.stat().st_size}")
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
    if any(not _safe_path(member.name) for member in members):
        raise ValueError("sdist contains an absolute or parent-traversing path")
    if any(member.issym() or member.islnk() for member in members):
        raise ValueError("sdist contains a symbolic or hard link")

    relative_names = {
        PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
        for member in members
        if len(PurePosixPath(member.name).parts) > 1
    }
    banned = sorted(
        name
        for name in relative_names
        if name.endswith(".jsonl")
        or any(name.startswith(prefix) for prefix in BANNED_SDIST_SEGMENTS)
    )
    if banned:
        raise ValueError(f"sdist contains excluded payloads: {banned[:5]}")
    missing = sorted(REQUIRED_SDIST_PATHS - relative_names)
    if missing:
        raise ValueError(f"sdist is missing required review surfaces: {missing}")
    return {"path": path.name, "bytes": path.stat().st_size, "members": len(members)}


def audit_wheel(path: Path) -> dict[str, int | str]:
    """Validate wheel size, archive paths, and required runtime package data."""

    if path.stat().st_size > MAX_DISTRIBUTION_BYTES:
        raise ValueError(f"wheel exceeds {MAX_DISTRIBUTION_BYTES} bytes: {path.stat().st_size}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    if any(not _safe_path(name) for name in names):
        raise ValueError("wheel contains an absolute or parent-traversing path")
    missing = sorted(REQUIRED_WHEEL_PATHS - names)
    if missing:
        raise ValueError(f"wheel is missing required runtime files: {missing}")
    return {"path": path.name, "bytes": path.stat().st_size, "members": len(names)}


def main() -> int:
    """Audit one wheel and one source distribution in a directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing one wheel and one sdist")
    arguments = parser.parse_args()
    directory = arguments.directory.expanduser().resolve()
    summary = {
        "sdist": audit_sdist(_single(directory, "*.tar.gz")),
        "wheel": audit_wheel(_single(directory, "*.whl")),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
