"""Acquire the small CMU swordplay motion-capture subset as out-of-domain data."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://mocap.cs.cmu.edu/search.php?subjectnumber=&motion=swordplay"
FAQ_URL = "https://mocap.cs.cmu.edu/faqs.php"
BASE_URL = "https://mocap.cs.cmu.edu/subjects/02"
FPS = 120
FILES = {
    "02.asf": (7_268, "c9f5ff45b4437b279f58b95dacf017afd3135373096274df69436a9354d796cf"),
    "02_07.amc": (1_791_825, "6171216b0bd691843096798431a5f201b9d1340503cf89ce20180b6143597cd8"),
    "02_08.amc": (1_195_070, "cbf08bb1aeba0487330b0096adb634170f6ddc617c19349b8954a99904a19634"),
    "02_09.amc": (825_503, "a217c49c4d3e834b9a7da1ccaffb12f0b8aeed9b7c90c114b365e5f3d265a6df"),
}


def parse_amc_frames(payload: bytes) -> int:
    text = payload.decode("ascii").replace("\r\n", "\n")
    if ":FULLY-SPECIFIED" not in text or ":DEGREES" not in text:
        raise ValueError("CMU AMC header is missing expected directives")
    frames = [
        int(match.group())
        for line in text.splitlines()
        if (match := re.fullmatch(r"\d+", line))
    ]
    if not frames or frames != list(range(1, len(frames) + 1)):
        raise ValueError("CMU AMC frame sequence is not contiguous from one")
    return len(frames)


def prepare(
    raw_directory: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    user_agent: str,
) -> dict[str, Any]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    for filename, (expected_bytes, expected_sha256) in FILES.items():
        path = raw_directory / filename
        download_resumable(
            f"{BASE_URL}/{filename}",
            path,
            user_agent=user_agent,
            expected_bytes=expected_bytes,
        )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"CMU motion file checksum mismatch for {filename}: {actual_sha256}"
            )
    skeleton_path = raw_directory / "02.asf"
    rows: list[dict[str, Any]] = []
    for trial in (7, 8, 9):
        motion_path = raw_directory / f"02_{trial:02d}.amc"
        frames = parse_amc_frames(motion_path.read_bytes())
        rows.append(
            {
                "clip_id": f"cmu-mocap:02:{trial:02d}",
                "subject_id": "cmu-mocap:02",
                "source_label": "swordplay",
                "sport_fencing_verified": False,
                "weapon": None,
                "role": "out_of_domain_motion_sanity_check",
                "frames": frames,
                "fps": FPS,
                "duration_seconds": round(frames / FPS, 6),
                "motion_sha256": sha256_file(motion_path),
                "skeleton_sha256": sha256_file(skeleton_path),
                "source_url": f"{BASE_URL}/02_{trial:02d}.amc",
                "license_statement_url": FAQ_URL,
                "license_statement": "may be copied, modified, or redistributed without permission",
            }
        )
    report: dict[str, Any] = {
        "claim_boundary": (
            "CMU labels these files 'swordplay', not Olympic sport fencing. The three trials "
            "come from one subject and cannot be mapped to epee, foil, or sabre. They are retained "
            "only as a permissively reusable out-of-domain motion-format sanity check and are "
            "excluded from fencing model training and athlete counts."
        ),
        "source_url": SOURCE_URL,
        "license_statement_url": FAQ_URL,
        "prepared_at": datetime.now(UTC).isoformat(),
        "subjects": 1,
        "trials": len(rows),
        "frames": sum(row["frames"] for row in rows),
        "duration_seconds": round(sum(row["duration_seconds"] for row in rows), 6),
        "fps": FPS,
        "role": "out_of_domain_only",
        "manifest": str(manifest_path),
    }
    report["manifest_sha256"] = write_jsonl(manifest_path, rows)
    write_json(report_path, report)
    return report
