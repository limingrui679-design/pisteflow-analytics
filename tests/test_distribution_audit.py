from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from pisteflow.distribution import (
    REQUIRED_SDIST_PATHS,
    REQUIRED_WHEEL_PATHS,
    audit_sdist,
    audit_wheel,
)


def _write_sdist(path: Path, extra_paths: set[str] | None = None) -> None:
    paths = REQUIRED_SDIST_PATHS | (extra_paths or set())
    with tarfile.open(path, mode="w:gz") as archive:
        for relative in sorted(paths):
            payload = relative.encode("utf-8")
            member = tarfile.TarInfo(f"pisteflow-0.2.0/{relative}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def _write_wheel(path: Path, omitted: str | None = None) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for relative in sorted(REQUIRED_WHEEL_PATHS - ({omitted} if omitted else set())):
            archive.writestr(relative, relative)


def test_distribution_audit_accepts_minimal_safe_archives(tmp_path: Path) -> None:
    sdist = tmp_path / "pisteflow.tar.gz"
    wheel = tmp_path / "pisteflow.whl"
    _write_sdist(sdist)
    _write_wheel(wheel)

    assert audit_sdist(sdist)["members"] == len(REQUIRED_SDIST_PATHS)
    assert audit_wheel(wheel)["members"] == len(REQUIRED_WHEEL_PATHS)


def test_distribution_audit_rejects_excluded_payloads(tmp_path: Path) -> None:
    sdist = tmp_path / "pisteflow.tar.gz"
    _write_sdist(sdist, {"data/raw/private-athlete-row.json"})

    with pytest.raises(ValueError, match="excluded payloads"):
        audit_sdist(sdist)


def test_distribution_audit_rejects_missing_runtime_files(tmp_path: Path) -> None:
    wheel = tmp_path / "pisteflow.whl"
    missing = "pisteflow/resources/showcase_cases.json"
    _write_wheel(wheel, omitted=missing)

    with pytest.raises(ValueError, match="missing required runtime files"):
        audit_wheel(wheel)
