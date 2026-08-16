"""Link-only index of the public Log45 sabre touch-clip folder."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pisteflow.provenance import write_json, write_jsonl

DATASET_PAGE = "https://github.com/Log45/Fencing-Judge-AI"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1VmFhBmm88Z9LU0zVJsLevOETzgLU8qim"


class DriveFile(Protocol):
    id: str
    path: str


def records_from_files(files: list[DriveFile]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in files:
        path = PurePosixPath(item.path)
        if path.suffix.casefold() != ".mov":
            continue
        source_group = path.parts[0]
        records.append(
            {
                "clip_id": f"log45:{item.id}",
                "dataset_path": item.path,
                "weapon": "sabre",
                "content_label": "touch_length_clip",
                "source_group": source_group,
                "source_url": f"https://drive.google.com/file/d/{item.id}/view",
                "dataset_page": DATASET_PAGE,
                "download_page": DRIVE_FOLDER,
                "license_status": "unknown; link-only; no redistribution",
                "media_downloaded": False,
                "claim_boundary": (
                    "A real sabre touch-length clip, not an action/referee-call label or an "
                    "independent match because multiple clips share the same source video."
                ),
            }
        )
    return sorted(records, key=lambda row: str(row["dataset_path"]))


def build_manifest(output_path: Path, report_path: Path) -> dict[str, Any]:
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - optional collector
        raise RuntimeError("Install collectors: pip install -e '.[collectors]'") from exc
    files = gdown.download_folder(  # type: ignore[attr-defined]
        url=DRIVE_FOLDER,
        output="log45-sabre-clips",
        quiet=True,
        skip_download=True,
    )
    if files is None:
        raise RuntimeError("Log45 Google Drive index did not return any file records")
    records = records_from_files(files)  # type: ignore[arg-type]
    manifest_sha256 = write_jsonl(output_path, records)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Public link index of real sabre touch-length clips. Raw media and labels are not "
            "redistributed, and clip count is not represented as independent-match count."
        ),
        "source_url": DATASET_PAGE,
        "download_url": DRIVE_FOLDER,
        "indexed_at": datetime.now(UTC).isoformat(),
        "license_status": "unknown; link-only; no redistribution",
        "clips": len(records),
        "source_groups": len({str(row["source_group"]) for row in records}),
        "by_source_group": dict(
            sorted(Counter(str(row["source_group"]) for row in records).items())
        ),
        "manifest_sha256": manifest_sha256,
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
