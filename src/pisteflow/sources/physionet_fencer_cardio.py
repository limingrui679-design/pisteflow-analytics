"""Privacy-minimized fencing subset of the open ACTES cardiorespiratory dataset."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/"
SOURCE_DOI = "https://doi.org/10.13026/2qs3-kh43"
LICENSE = "Open Data Commons Attribution License 1.0"
BASE_URL = "https://physionet.org/files/actes-cycloergometer-exercise/1.0.0"
FILES = {
    "LICENSE.txt": (
        20_402,
        "86c0ad300f6d380591298bc03652e30b81f65954bbf22435812bcfd46812ab06",
    ),
    "SHA256SUMS.txt": (
        342,
        "894d0d195fe5a869759f701ba51c2d16894dcd2682ce4029fe35fc730d01db0c",
    ),
    "subject-info.csv": (
        590,
        "5ac7cdd41d9112c48121998f19927318bd6dc2241ead97a6baa5e9ae8b60c203",
    ),
    "test_measure.csv": (
        1_588_131,
        "14a9df027caf00ee1dc39f1493e708c27a7da88794ff69a59442326ef7aa0516",
    ),
}
EXPECTED_SUBJECTS = 18
EXPECTED_FENCERS = 10
EXPECTED_MEASUREMENTS = 52_062
EXPECTED_FENCING_MEASUREMENTS = 27_603


def _participant_key(value: object) -> str:
    numeric = int(float(str(value)))
    digest = hashlib.sha256(f"physionet-actes:{numeric}".encode()).hexdigest()[:20]
    return f"ACTES-{digest}"


def _age_band(value: object) -> str:
    age = int(float(str(value)))
    if age < 14:
        return "under_14"
    if age <= 15:
        return "14_15"
    if age <= 17:
        return "16_17"
    return "18_plus"


def normalize(
    subject_info: pd.DataFrame,
    measurements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return direct-ID-free fencer profiles and participant-grouped time series."""

    expected_subject_columns = ["ID", "age", "Weight", "Height", "P_vt1", "P_vt2", "sport"]
    expected_measure_columns = ["ID", "time", "RR", "VO2", "power"]
    if list(subject_info.columns) != expected_subject_columns:
        raise ValueError("ACTES subject-info columns changed")
    if list(measurements.columns) != expected_measure_columns:
        raise ValueError("ACTES measurement columns changed")
    if subject_info["ID"].isna().any() or subject_info["ID"].duplicated().any():
        raise ValueError("ACTES participant IDs are missing or duplicated")
    if measurements[["ID", "time", "VO2", "power"]].isna().any().any():
        raise ValueError("ACTES required time-series values are missing")
    if measurements.duplicated(["ID", "time"]).any():
        raise ValueError("ACTES participant timestamps are duplicated")
    if set(measurements["ID"]) != set(subject_info["ID"]):
        raise ValueError("ACTES subject and measurement ID sets disagree")

    fencers = subject_info.loc[subject_info["sport"].str.lower().eq("fencing")].copy()
    if fencers.empty:
        raise ValueError("ACTES snapshot contains no fencing participants")
    keys = {int(value): _participant_key(value) for value in fencers["ID"]}
    profiles = pd.DataFrame(
        {
            "participant_key": fencers["ID"].map(keys),
            "age_band": fencers["age"].map(_age_band),
            "weight_kg": pd.to_numeric(fencers["Weight"], errors="raise").astype(float),
            "height_cm": pd.to_numeric(fencers["Height"], errors="raise").astype(float),
            "power_vt1_w": pd.to_numeric(fencers["P_vt1"], errors="raise").astype(float),
            "power_vt2_w": pd.to_numeric(fencers["P_vt2"], errors="raise").astype(float),
            "sport": "fencing",
            "weapon": "unknown",
            "source_doi": SOURCE_DOI,
        }
    )
    profiles["bmi"] = profiles["weight_kg"] / (profiles["height_cm"] / 100.0) ** 2

    series = measurements.loc[measurements["ID"].isin(keys)].copy()
    series.insert(0, "participant_key", series["ID"].map(keys))
    series["stage_derived"] = "recovery"
    series.loc[series["time"].lt(0), "stage_derived"] = "pre_exercise"
    series.loc[series["time"].ge(0) & series["power"].gt(0), "stage_derived"] = "effort"
    series = series.rename(
        columns={"time": "time_s", "RR": "rr_ms", "VO2": "vo2_source_units", "power": "power_w"}
    ).drop(columns="ID")
    series["rr_extreme_gt_3000ms"] = series["rr_ms"].gt(3000).fillna(False)
    series["source_doi"] = SOURCE_DOI
    if any(
        not group["time_s"].is_monotonic_increasing
        for _, group in series.groupby("participant_key")
    ):
        raise ValueError("ACTES fencer time series is not monotonic within participant")
    if set(series["participant_key"]) != set(profiles["participant_key"]):
        raise ValueError("ACTES fencer profile and time-series keys disagree")
    if "ID" in profiles or "age" in profiles or "ID" in series:
        raise ValueError("ACTES direct ID or exact age leaked into normalized output")
    return profiles.reset_index(drop=True), series.reset_index(drop=True)


def fetch_and_prepare(
    cache_directory: Path,
    profile_output: Path,
    series_output: Path,
    summary_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    for filename, (expected_bytes, expected_sha256) in FILES.items():
        destination = cache_directory / filename
        if force:
            destination.unlink(missing_ok=True)
        download_resumable(
            f"{BASE_URL}/{filename}",
            destination,
            user_agent=user_agent,
            expected_bytes=expected_bytes,
        )
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"ACTES checksum changed: {filename}")

    upstream_sums = (cache_directory / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for filename in ("LICENSE.txt", "subject-info.csv", "test_measure.csv"):
        if f"{FILES[filename][1]} {filename}" not in upstream_sums:
            raise ValueError(f"ACTES upstream checksum manifest changed: {filename}")

    subjects = pd.read_csv(cache_directory / "subject-info.csv")
    measurements = pd.read_csv(cache_directory / "test_measure.csv")
    if len(subjects) != EXPECTED_SUBJECTS or len(measurements) != EXPECTED_MEASUREMENTS:
        raise ValueError("ACTES snapshot row counts changed")
    profiles, series = normalize(subjects, measurements)
    if len(profiles) != EXPECTED_FENCERS or len(series) != EXPECTED_FENCING_MEASUREMENTS:
        raise ValueError("ACTES fencing subset row counts changed")

    profile_output.parent.mkdir(parents=True, exist_ok=True)
    series_output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(profile_output, index=False)
    series.to_parquet(series_output, index=False)
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "dataset_subjects": len(subjects),
        "dataset_measurement_rows": len(measurements),
        "dataset_subjects_by_sport": subjects["sport"].value_counts().sort_index().to_dict(),
        "fencing_participants": len(profiles),
        "fencing_measurement_rows": len(series),
        "fencing_rows_per_participant": {
            "minimum": int(series.groupby("participant_key").size().min()),
            "median": float(series.groupby("participant_key").size().median()),
            "maximum": int(series.groupby("participant_key").size().max()),
        },
        "fencing_rows_by_derived_stage": dict(Counter(series["stage_derived"])),
        "fencing_missing_rr_rows": int(series["rr_ms"].isna().sum()),
        "fencing_rr_extreme_gt_3000ms_rows": int(series["rr_extreme_gt_3000ms"].sum()),
        "age_bands": profiles["age_band"].value_counts().sort_index().to_dict(),
        "weapon": "unknown",
        "exact_age_removed": True,
        "source_participant_id_removed": True,
        "vo2_unit_note": (
            "The source documentation says mL/min, while deposited values are approximately "
            "0.07-3.46 and may represent L/min; the field is retained as vo2_source_units without "
            "silently converting it."
        ),
        "line_count_note": (
            "The PhysioNet page describes 52,063 lines; the CSV contains 52,062 data rows plus "
            "one header line."
        ),
        "profile_output": str(profile_output),
        "profile_output_sha256": sha256_file(profile_output),
        "series_output": str(series_output),
        "series_output_sha256": sha256_file(series_output),
        "claim_boundary": (
            "Ten national-division teenage fencers completed graded cycling tests. These are "
            "participant-grouped physiological observations, not fencing bouts, weapon labels, "
            "clinical diagnoses, training effects, or a representative athlete population. Exact "
            "age and source IDs are removed; sensitive row tables remain local."
        ),
        "repository_policy": "privacy-minimized row tables local; code and aggregates public",
    }
    write_json(summary_path, summary)
    return summary
