"""Privacy-minimized CC BY measurements for 23 U15 fencing athletes."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

ARTICLE_DOI = "https://doi.org/10.3389/fphys.2019.00405"
DATASET_DOI = "https://doi.org/10.3389/fphys.2019.00405.s001"
SOURCE_URL = "https://figshare.com/articles/dataset/8047721"
API_URL = "https://api.figshare.com/v2/articles/8047721"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/14994896"
LICENSE = "CC BY 4.0"
EXPECTED_BYTES = 14_336
EXPECTED_SHA256 = "cb0384568a52a7c37979c13ea098000c8cc0c537b99bd0d21e12916f744c80a0"

_SOURCE_COLUMNS = [
    "Sport",
    "Sport_Number",
    "Sex",
    "Test_age_T1",
    "Body_height_cm_T1",
    "Sitting_height_cm_T1",
    "Arm_span_cm_T1",
    "Arm_length_cm_T1",
    "Leg_length_cm_T1",
    "Chest_girth_cm_T1",
    "Shoulder_width_cm_T1",
    "Crista_width_cm_T1",
    "Calf_girth_cm_T1",
    "Thigh_circumference_cm_T1",
    "Waist_girth_cm_T1",
    "LowerLeg_length_cm_T1",
    "Ankle_circumference_cm_T1",
    "Subscapular_angle_mm_T1",
    "Abdomen_mm_T1",
    "Upper_arm_mm_T1",
    "Tendon_length_cm_T1",
    "Body_weight_kg_T1",
    "Eye_hand_reaction_time",
    "Back_strength_kg_T1",
    "Heart_rate_rest_bpm_T1",
    "Vital_capacity_ml_T1",
    "Hemoglobin_mass_g_l_T1",
    "Blood_pressure_mmHg_T1_sys",
    "Blood_pressure_mmHg_T1_dia",
    "Urine_protein_T1",
    "Menarche_yr_mth_T1",
    "Breast_T1",
    "Pubes_Girls_T1",
    "Testis_T1",
    "Pubes_Boys_T1",
]

_ANALYTIC_COLUMNS = {
    "Test_age_T1": "age_months",
    "Body_height_cm_T1": "body_height_cm",
    "Sitting_height_cm_T1": "sitting_height_cm",
    "Arm_span_cm_T1": "arm_span_cm",
    "Arm_length_cm_T1": "arm_length_cm",
    "Leg_length_cm_T1": "leg_length_cm",
    "Chest_girth_cm_T1": "chest_girth_cm",
    "Shoulder_width_cm_T1": "shoulder_width_cm",
    "Crista_width_cm_T1": "crista_width_cm",
    "Calf_girth_cm_T1": "calf_girth_cm",
    "Thigh_circumference_cm_T1": "thigh_circumference_cm",
    "Waist_girth_cm_T1": "waist_girth_cm",
    "LowerLeg_length_cm_T1": "lower_leg_length_cm",
    "Ankle_circumference_cm_T1": "ankle_circumference_cm",
    "Subscapular_angle_mm_T1": "subscapular_skinfold_mm",
    "Abdomen_mm_T1": "abdomen_skinfold_mm",
    "Upper_arm_mm_T1": "upper_arm_skinfold_mm",
    "Tendon_length_cm_T1": "achilles_tendon_length_cm",
    "Body_weight_kg_T1": "body_weight_kg",
    "Eye_hand_reaction_time": "eye_hand_reaction_time_ms",
    "Back_strength_kg_T1": "back_strength_kg",
    "Heart_rate_rest_bpm_T1": "resting_heart_rate_bpm",
    "Vital_capacity_ml_T1": "vital_capacity_ml",
    "Hemoglobin_mass_g_l_T1": "hemoglobin_g_l",
    "Blood_pressure_mmHg_T1_sys": "systolic_blood_pressure_mmhg",
    "Blood_pressure_mmHg_T1_dia": "diastolic_blood_pressure_mmhg",
}


def _participant_code(source_label: str) -> str:
    digest = sha256(f"figshare-youth-athlete:{source_label}".encode()).hexdigest()[:16]
    return f"yaf-{digest}"


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter fencing rows and remove sex, maturation, and source row identifiers."""

    if list(frame.columns) != _SOURCE_COLUMNS:
        raise ValueError("Youth-athlete CSV columns changed")
    source_labels = frame["Sport"].astype("string")
    fencing = frame.loc[source_labels.str.fullmatch(r"fencing-\d+", na=False)].copy()
    if fencing.empty:
        raise ValueError("Youth-athlete CSV contains no fencing rows")
    result = fencing[list(_ANALYTIC_COLUMNS)].rename(columns=_ANALYTIC_COLUMNS)
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.insert(
        0,
        "participant_code",
        fencing["Sport"].astype(str).map(_participant_code).to_numpy(),
    )
    result["source_dataset_doi"] = DATASET_DOI
    result["population"] = "u15_elite_sport_school_fencers"
    return result.reset_index(drop=True)


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, int]:
    source = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(source) != 97:
        raise ValueError(f"Youth-athlete CSV row count changed: {len(source)}")
    result = normalize_frame(source)
    if len(result) != 23 or result["participant_code"].duplicated().any():
        raise ValueError("Youth-athlete fencing sample layout changed")
    return result, len(source)


def fetch_and_prepare(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the official Figshare file and prepare a local-only analytic table."""

    if force:
        raw_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        raw_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    if sha256_file(raw_path) != EXPECTED_SHA256:
        raise ValueError("Youth-athlete CSV checksum changed")
    result, source_rows = normalize_snapshot(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    measurement_columns = [
        column
        for column in result
        if column not in {"participant_code", "source_dataset_doi", "population"}
    ]
    report: dict[str, Any] = {
        "claim_boundary": (
            "CC BY cross-sectional measurements for 23 U15 fencing athletes within a "
            "97-athlete, six-sport study. Weapon, training exposure, bout outcomes, and "
            "longitudinal follow-up are absent. The row table remains local; source row labels, "
            "sex code, urine-protein code, and puberty or maturation fields are excluded. "
            "These data support descriptive athlete profiling, not causal talent selection or "
            "training-effect claims."
        ),
        "source_url": SOURCE_URL,
        "api_url": API_URL,
        "article_doi": ARTICLE_DOI,
        "dataset_doi": DATASET_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_rows_all_sports": source_rows,
        "fencing_participants": len(result),
        "measurement_columns": len(measurement_columns),
        "missing_measurement_cells": int(result[measurement_columns].isna().sum().sum()),
        "weapon_available": False,
        "longitudinal_training_available": False,
        "direct_source_labels_removed": True,
        "sex_code_removed": True,
        "maturation_fields_removed": True,
        "raw_sha256": sha256_file(raw_path),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
    }
    write_json(report_path, report)
    return report
