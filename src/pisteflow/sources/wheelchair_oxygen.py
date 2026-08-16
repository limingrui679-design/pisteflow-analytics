"""CC BY oxygen-uptake measurements from elite wheelchair epee fencers."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DOI = "https://doi.org/10.18745/ds.00027179"
PAGE_URL = "https://uhra.herts.ac.uk/id/eprint/27179/"
ARCHIVE_URL = (
    "https://uhra.herts.ac.uk/id/eprint/27179/2/"
    "OxygenUptakeinWheelchairFencing.zip"
)
README_URL = "https://uhra.herts.ac.uk/id/eprint/27179/4/README.txt"
LICENSE = "CC BY 4.0"
ARCHIVE_BYTES = 122_710
README_BYTES = 11_437

_EXPECTED_MEMBERS = {
    "01_test_and_bouts.csv",
    "02_competition.csv",
    "03_competition_F7_tidy.csv",
    "04_bland_altman_equivalence.R",
    "05_competition_figure.R",
}


def _snake_case(value: str) -> str:
    interim = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^A-Za-z0-9]+", "_", interim).strip("_").lower()


def normalize_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the one-row-per-fencer validity table."""

    if frame.shape != (11, 36) or "Fencer" not in frame:
        raise ValueError(f"Wheelchair-epee summary snapshot changed: {frame.shape}")
    result = frame.rename(columns={column: _snake_case(column) for column in frame}).copy()
    result = result.rename(columns={"fencer": "participant_code"})
    if result["participant_code"].nunique() != 11:
        raise ValueError("Wheelchair-epee summary no longer has 11 participant codes")
    result.insert(0, "source_row", range(1, len(result) + 1))
    result["weapon"] = "epee"
    result["adaptive_format"] = "wheelchair"
    result["source_doi"] = DOI
    return result


def normalize_f3_competition(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the second-by-second F3 competition export."""

    required = {
        "Fencer",
        "Crono_hms",
        "Clock_time",
        "HR_bpm",
        "VO2est_mL_kg_min",
        "Bout",
        "Phase",
    }
    if set(frame) != required:
        raise ValueError("Wheelchair-epee F3 competition columns changed")
    labels = set(frame["Bout"].dropna())
    if not labels or not labels.issubset({"Yes", "No"}):
        raise ValueError("Wheelchair-epee F3 bout labels changed")
    result = frame.rename(
        columns={
            "Fencer": "participant_code",
            "Crono_hms": "elapsed_time",
            "Clock_time": "clock_time",
            "HR_bpm": "heart_rate_bpm",
            "VO2est_mL_kg_min": "vo2_estimated_ml_kg_min",
            "Phase": "competition_phase",
        }
    ).copy()
    result.insert(0, "source_row", range(1, len(result) + 1))
    result["in_bout"] = result.pop("Bout").eq("Yes")
    result["weapon"] = "epee"
    result["adaptive_format"] = "wheelchair"
    result["source_doi"] = DOI
    return result


def normalize_f7_competition(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the F7 time series while retaining its more limited schema."""

    if set(frame) != {"ID", "sec", "vo2", "bout"}:
        raise ValueError("Wheelchair-epee F7 competition columns changed")
    labels = set(frame["bout"].dropna())
    if not labels or not labels.issubset({"Yes", "No"}):
        raise ValueError("Wheelchair-epee F7 bout labels changed")
    result = frame.rename(
        columns={
            "ID": "participant_code",
            "sec": "elapsed_seconds",
            "vo2": "vo2_estimated_ml_kg_min",
        }
    ).copy()
    result.insert(0, "source_row", range(1, len(result) + 1))
    result["in_bout"] = result.pop("bout").eq("Yes")
    result["weapon"] = "epee"
    result["adaptive_format"] = "wheelchair"
    result["source_doi"] = DOI
    return result


def _read_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if set(members) != _EXPECTED_MEMBERS:
            raise ValueError(f"Wheelchair-epee archive members changed: {members}")
        payloads = {
            name: archive.read(name)
            for name in (
                "01_test_and_bouts.csv",
                "02_competition.csv",
                "03_competition_F7_tidy.csv",
            )
        }
    summary = normalize_summary(pd.read_csv(io.BytesIO(payloads["01_test_and_bouts.csv"])))
    f3 = normalize_f3_competition(pd.read_csv(io.BytesIO(payloads["02_competition.csv"])))
    f7 = normalize_f7_competition(
        pd.read_csv(io.BytesIO(payloads["03_competition_F7_tidy.csv"]))
    )
    if len(f3) != 12_116 or set(f3["participant_code"]) != {"F3"}:
        raise ValueError("Wheelchair-epee F3 snapshot size changed")
    if len(f7) != 11_401 or set(f7["participant_code"]) != {"F7"}:
        raise ValueError("Wheelchair-epee F7 snapshot size changed")
    return summary, f3, f7, members


def fetch_and_prepare(
    archive_path: Path,
    readme_path: Path,
    processed_dir: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        archive_path.unlink(missing_ok=True)
        readme_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=ARCHIVE_BYTES,
    )
    download_resumable(
        README_URL,
        readme_path,
        user_agent=user_agent,
        expected_bytes=README_BYTES,
    )
    readme = readme_path.read_text(encoding="utf-8-sig", errors="replace")
    if "Licence:    CC BY 4.0" not in readme:
        raise ValueError("Wheelchair-epee license statement changed")
    summary_frame, f3, f7, members = _read_archive(archive_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "participant_summary": processed_dir / "participant-summary.parquet",
        "competition_f3": processed_dir / "competition-f3-second.parquet",
        "competition_f7": processed_dir / "competition-f7-second.parquet",
    }
    summary_frame.to_parquet(outputs["participant_summary"], index=False)
    f3.to_parquet(outputs["competition_f3"], index=False)
    f7.to_parquet(outputs["competition_f7"], index=False)
    report: dict[str, Any] = {
        "claim_boundary": (
            "Elite wheelchair-epee physiology and competition measurements. The deposited "
            "ZIP contains 11 participant summary rows and second-by-second exports for only "
            "F3 and F7. Its README describes 372 WFST rows, 502 monitored-bout rows, and seven "
            "competition sheets, but those sheets are absent from the deposited CSV archive. "
            "Nothing missing is reconstructed. Bout-state modelling is therefore a two-person "
            "demonstration, not a population-level performance or health model."
        ),
        "source_url": PAGE_URL,
        "dataset_doi": DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "readme_sha256": sha256_file(readme_path),
        "archive_members": members,
        "weapon": "epee",
        "adaptive_format": "wheelchair",
        "study_reported_participants": 11,
        "participant_summary_rows": len(summary_frame),
        "second_level_participant_codes": ["F3", "F7"],
        "f3_second_rows": len(f3),
        "f3_bout_seconds": int(f3["in_bout"].sum()),
        "f7_second_rows": len(f7),
        "f7_bout_seconds": int(f7["in_bout"].sum()),
        "acquired_second_rows": len(f3) + len(f7),
        "readme_claimed_but_absent": {
            "wfst_raw_rows": 372,
            "monitored_bout_raw_rows": 502,
            "competition_participant_sheets": 7,
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "output_sha256": {key: sha256_file(path) for key, path in outputs.items()},
    }
    write_json(report_path, report)
    return report
