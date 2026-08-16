"""Three-weapon tibial-impact data from real competition-site lunge testing."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DATASET_DOI = "https://doi.org/10.18745/ds.28796"
ARTICLE_DOI = "https://doi.org/10.1371/journal.pone.0323557"
PAGE_URL = "https://uhra.herts.ac.uk/id/eprint/15815/"
DATA_URL = (
    "https://uhra.herts.ac.uk/id/eprint/15815/2/"
    "Surface%20Data%20Collection%20Sheet%20%282%29.xlsx"
)
README_URL = "https://uhra.herts.ac.uk/id/eprint/15815/1/README1.txt"
LICENSE = "CC BY; deposited README does not state a version"
DATA_BYTES = 50_598
README_BYTES = 5_288
_DEMOGRAPHIC_COLUMNS = [
    "Gender",
    "Age",
    "Height (m)",
    "Weight (kg)",
    "BMI",
    "Weapon",
    "Handedness",
    "Years Fencing",
    "Hours a week",
]
_ACCELERATION_COLUMNS = [
    f"{surface}{trial}"
    for surface in "ABCDEF"
    for trial in (1, 2, 3)
] + [f"{surface}_Av" for surface in "ABCDEF"]


def _weapon_memberships(value: object) -> tuple[str, bool, bool, bool]:
    tokens = {
        token
        for token in re.split(r"[,/&]+", str(value).upper().replace(" ", ""))
        if token
    }
    if not tokens or not tokens.issubset({"E", "F", "S"}):
        raise ValueError(f"Unexpected tibial-impact weapon label: {value!r}")
    epee = "E" in tokens
    foil = "F" in tokens
    sabre = "S" in tokens
    label = "+".join(
        weapon for weapon, present in (("epee", epee), ("foil", foil), ("sabre", sabre)) if present
    )
    return label, epee, foil, sabre


def normalize_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create participant, retained-trial, and per-surface mean tables."""

    expected = _DEMOGRAPHIC_COLUMNS + [
        item
        for surface in "ABCDEF"
        for item in (f"{surface}1", f"{surface}2", f"{surface}3", f"{surface}_Av")
    ]
    if list(frame.columns) != expected:
        raise ValueError("Tibial-impact workbook columns changed")
    participant_codes = [f"TIL-{index:03d}" for index in range(1, len(frame) + 1)]
    participants = frame[_DEMOGRAPHIC_COLUMNS].copy()
    participants.insert(0, "participant_code", participant_codes)
    participants = participants.rename(
        columns={
            "Gender": "gender",
            "Age": "age_years",
            "Height (m)": "height_m",
            "Weight (kg)": "weight_kg",
            "BMI": "bmi",
            "Weapon": "weapon_raw",
            "Handedness": "handedness",
            "Years Fencing": "fencing_experience_years",
            "Hours a week": "weekly_training_hours",
        }
    )
    participants["gender"] = participants["gender"].astype(str).str.upper().map(
        {"F": "female", "M": "male"}
    )
    participants["handedness"] = participants["handedness"].map(
        {"L": "left", "R": "right"}
    )
    memberships = participants["weapon_raw"].map(_weapon_memberships)
    participants[["weapon", "epee", "foil", "sabre"]] = pd.DataFrame(
        memberships.tolist(), index=participants.index
    )
    participants["multi_weapon"] = participants[["epee", "foil", "sabre"]].sum(axis=1).gt(1)
    participants["age_below_18"] = pd.to_numeric(
        participants["age_years"], errors="raise"
    ).lt(18)
    participants["source_dataset_doi"] = DATASET_DOI

    trial_rows: list[dict[str, Any]] = []
    average_rows: list[dict[str, Any]] = []
    for row_index, participant_code in enumerate(participant_codes):
        for surface in "ABCDEF":
            values = [frame.loc[row_index, f"{surface}{trial}"] for trial in (1, 2, 3)]
            reported = frame.loc[row_index, f"{surface}_Av"]
            available = [float(value) for value in values if pd.notna(value)]
            recomputed = float(np.mean(available)) if available else None
            for trial_number, value in enumerate(values, start=1):
                trial_rows.append(
                    {
                        "participant_code": participant_code,
                        "surface": surface,
                        "retained_trial_number": trial_number,
                        "peak_axial_acceleration_g": value,
                        "missing_value": pd.isna(value),
                        "source_dataset_doi": DATASET_DOI,
                    }
                )
            average_rows.append(
                {
                    "participant_code": participant_code,
                    "surface": surface,
                    "retained_trial_count": len(available),
                    "reported_mean_acceleration_g": reported,
                    "recomputed_mean_acceleration_g": recomputed,
                    "reported_recomputed_difference_g": (
                        None
                        if pd.isna(reported) or recomputed is None
                        else float(reported) - recomputed
                    ),
                    "source_dataset_doi": DATASET_DOI,
                }
            )
    return participants, pd.DataFrame(trial_rows), pd.DataFrame(average_rows)


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    if frame.shape != (69, 33):
        raise ValueError(f"Tibial-impact workbook snapshot changed: {frame.shape}")
    participants, trials, averages = normalize_frame(frame)
    if len(trials) != 1_242 or len(averages) != 414:
        raise ValueError("Tibial-impact normalized row counts changed")
    if int(trials["peak_axial_acceleration_g"].notna().sum()) != 1_235:
        raise ValueError("Tibial-impact nonmissing trial count changed")
    return participants, trials, averages


def fetch_and_prepare(
    raw_path: Path,
    readme_path: Path,
    processed_dir: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        raw_path.unlink(missing_ok=True)
        readme_path.unlink(missing_ok=True)
    download_resumable(DATA_URL, raw_path, user_agent=user_agent, expected_bytes=DATA_BYTES)
    download_resumable(
        README_URL,
        readme_path,
        user_agent=user_agent,
        expected_bytes=README_BYTES,
    )
    if "Licenses/restrictions placed on the data: CC-BY" not in readme_path.read_text(
        encoding="utf-8-sig", errors="replace"
    ):
        raise ValueError("Tibial-impact dataset license statement changed")
    participants, trials, averages = normalize_snapshot(raw_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "participants": processed_dir / "participants.parquet",
        "retained_trials": processed_dir / "retained-lunge-trials.parquet",
        "surface_means": processed_dir / "participant-surface-means.parquet",
    }
    participants.to_parquet(outputs["participants"], index=False)
    trials.to_parquet(outputs["retained_trials"], index=False)
    averages.to_parquet(outputs["surface_means"], index=False)
    nonmissing_trials = trials.loc[~trials["missing_value"]]
    mean_differences = averages["reported_recomputed_difference_g"].abs()
    report: dict[str, Any] = {
        "claim_boundary": (
            "Anonymous repeated-measures lunge testing at the 2024 US Senior National "
            "Championships across epee, foil, sabre, and multi-weapon participants. Sixty-nine "
            "analyzed rows are not 70 volunteers. Each participant performed five lunges per "
            "surface, but the deposit retains only three values after excluding the highest "
            "and lowest; therefore 1,235 nonmissing retained values are acquired, not all "
            "2,070 performed lunges. Tibial acceleration is an impact marker, not an observed "
            "injury outcome."
        ),
        "source_url": PAGE_URL,
        "dataset_doi": DATASET_DOI,
        "article_doi": ARTICLE_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "raw_sha256": sha256_file(raw_path),
        "readme_sha256": sha256_file(readme_path),
        "participants": len(participants),
        "study_reported_volunteers": 70,
        "participants_below_18": int(participants["age_below_18"].sum()),
        "gender_counts": participants["gender"].value_counts().sort_index().to_dict(),
        "weapon_memberships": {
            weapon: int(participants[weapon].sum()) for weapon in ("epee", "foil", "sabre")
        },
        "multi_weapon_participants": int(participants["multi_weapon"].sum()),
        "retained_trial_slots": len(trials),
        "nonmissing_retained_trial_values": len(nonmissing_trials),
        "missing_retained_trial_values": int(trials["missing_value"].sum()),
        "participant_surface_mean_rows": len(averages),
        "nonmissing_participant_surface_means": int(
            averages["reported_mean_acceleration_g"].notna().sum()
        ),
        "reported_mean_recompute_max_abs_difference_g": float(mean_differences.max()),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "output_sha256": {key: sha256_file(path) for key, path in outputs.items()},
    }
    write_json(report_path, report)
    return report
