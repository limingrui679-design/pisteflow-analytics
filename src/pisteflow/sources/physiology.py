"""Open physiological measurements from an elite foil training-camp study."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DATASET_PAGE = "https://uhra.herts.ac.uk/id/eprint/15797/"
DATASET_DOI = "https://doi.org/10.18745/DS.26019"
ARTICLE_DOI = "https://doi.org/10.1371/journal.pone.0281600"
ARCHIVE_URL = (
    "https://uhra.herts.ac.uk/id/eprint/15797/1/"
    "Physiological%20data%20of%20elite%20foil%20fencing.zip"
)
README_URL = "https://uhra.herts.ac.uk/id/eprint/15797/2/README.txt"
LICENSE = "CC BY (version not stated in deposited README)"

_EXPECTED_FILES = {
    "Bloods.csv",
    "Core Temperature.csv",
    "Gas Analysis.csv",
    "Heart Rate.csv",
    "Motion Analysis.csv",
    "Participant Info.csv",
    "README.txt",
    "RPE.csv",
}

_PARTICIPANT_COLUMNS = {
    "ID": "participant_id",
    "Gender": "gender",
    "Age": "age_years",
    "Height": "height_cm",
    "Weight": "weight_kg",
    "BMI": "bmi_kg_m2",
    "Wingspan": "wingspan_cm",
    "Fat mass %": "fat_mass_pct",
    "Muscle mass %": "muscle_mass_pct",
    "Lean mass D arm %": "lean_mass_dominant_arm_pct",
    "Lean mass ND arm": "lean_mass_non_dominant_arm_pct",
    "Lean mass D leg": "lean_mass_dominant_leg_pct",
    "Lean mass ND leg": "lean_mass_non_dominant_leg_pct",
}

_CORE_COLUMNS = [
    "participant_id",
    "core_temperature_baseline_c",
    "core_temperature_pre_pool_c",
    "core_temperature_post_pool_c",
    "core_temperature_pool_difference_c",
    "core_temperature_pre_de_c",
    "core_temperature_post_de_c",
    "core_temperature_de_difference_c",
    "core_temperature_final_c",
]

_SIMPLE_COLUMNS: dict[str, dict[str, str]] = {
    "RPE.csv": {
        "Participant": "participant_id",
        "Poule": "rpe_pool",
        "DE": "rpe_de",
        "Final": "rpe_final",
    },
    "Heart Rate.csv": {
        "Participant": "participant_id",
        "P5_HRAV": "heart_rate_pool_average_bpm",
        "P5_HRMAX": "heart_rate_pool_max_bpm",
        "P5_HR%": "heart_rate_pool_predicted_max_pct",
        "DE_HRAV": "heart_rate_de_average_bpm",
        "DE_HRMAX": "heart_rate_de_max_bpm",
        "DE_HR%": "heart_rate_de_predicted_max_pct",
    },
    "Motion Analysis.csv": {
        "Participant": "participant_id",
        "P total FT": "pool_total_fight_time_s",
        "DE total FT": "de_total_fight_time_s",
        "P Active FT": "pool_active_fight_time_s",
        "DE Active FT": "de_active_fight_time_s",
        "P Total RT": "pool_total_rest_time_s",
        "DE Total RT": "de_total_rest_time_s",
        "P RT during F": "pool_rest_during_fight_s",
        "DE RT during F": "de_rest_during_fight_s",
        "P RT bw bouts": "pool_rest_between_bouts_s",
        "DE RT bw bouts": "de_rest_between_bouts_s",
        "Work:Rest (P)": "pool_work_rest_ratio",
        "Work:rest (DE)": "de_work_rest_ratio",
    },
    "Gas Analysis.csv": {
        "Participant": "participant_id",
        "P_VO2 abs": "pool_vo2_absolute_l_min",
        "P_VO2 rel": "pool_vo2_relative_ml_kg_min",
        "P_VCO2": "pool_vco2_l_min",
        "P_RER": "pool_respiratory_exchange_ratio",
        "DE_VO2 abs": "de_vo2_absolute_l_min",
        "DE_VO2 rel": "de_vo2_relative_ml_kg_min",
        "DE_VCO2": "de_vco2_l_min",
        "DE_RER": "de_respiratory_exchange_ratio",
        "EE (kcal.min)": "energy_expenditure_upstream_kcal_min",
    },
}


def _read_csv(archive: ZipFile, filename: str, **kwargs: Any) -> pd.DataFrame:
    with archive.open(filename) as handle:
        return pd.read_csv(handle, **kwargs)


def _rename_exact(
    frame: pd.DataFrame,
    mapping: dict[str, str],
    filename: str,
    *,
    allowed_footer_labels: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    missing = set(mapping).difference(frame.columns)
    if missing:
        raise ValueError(f"{filename} missing columns: {sorted(missing)}")
    result = frame[list(mapping)].rename(columns=mapping)
    raw_ids = result["participant_id"].astype("string").str.strip()
    ids = pd.to_numeric(raw_ids, errors="coerce")
    footer_labels = set(raw_ids[ids.isna() & raw_ids.notna() & raw_ids.ne("")])
    if footer_labels.difference(allowed_footer_labels):
        raise ValueError(f"{filename} has unexpected non-participant rows: {sorted(footer_labels)}")
    result = result.loc[ids.notna()].copy()
    result["participant_id"] = ids.loc[ids.notna()].astype("Int16")
    return result


def normalize_archive(archive_path: Path) -> pd.DataFrame:
    """Create one deidentified, participant-level analytic table without altering values."""

    with ZipFile(archive_path) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        if names != _EXPECTED_FILES:
            raise ValueError(
                "Physiological foil archive members changed: "
                f"missing={sorted(_EXPECTED_FILES - names)}, "
                f"extra={sorted(names - _EXPECTED_FILES)}"
            )

        participants = _rename_exact(
            _read_csv(archive, "Participant Info.csv"),
            _PARTICIPANT_COLUMNS,
            "Participant Info.csv",
        )

        core = _read_csv(archive, "Core Temperature.csv")
        if len(core.columns) != len(_CORE_COLUMNS):
            raise ValueError("Core Temperature.csv column count changed")
        core.columns = _CORE_COLUMNS
        core_ids = pd.to_numeric(core["participant_id"], errors="coerce")
        unexpected_core_ids = core.loc[
            core_ids.isna() & core["participant_id"].notna(), "participant_id"
        ].astype("string")
        if unexpected_core_ids.str.strip().ne("").any():
            raise ValueError("Core Temperature.csv has unexpected non-participant rows")
        core = core.loc[core_ids.notna()].copy()
        core["participant_id"] = core_ids.loc[core_ids.notna()].astype("Int16")

        blood_raw = _read_csv(archive, "Bloods.csv", skiprows=1)
        if len(blood_raw.columns) != 14:
            raise ValueError("Bloods.csv layout changed")
        left_ids = pd.to_numeric(blood_raw.iloc[:, 0], errors="raise").astype("Int16")
        right_ids = pd.to_numeric(blood_raw.iloc[:, 11], errors="raise").astype("Int16")
        if not left_ids.equals(right_ids):
            raise ValueError("Bloods.csv lactate and glucose participant IDs disagree")
        blood = blood_raw.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 12, 13]].copy()
        blood.columns = [
            "participant_id",
            "lactate_baseline_1_mmol_l",
            "lactate_baseline_2_mmol_l",
            "lactate_baseline_average_mmol_l",
            "lactate_pool_1_mmol_l",
            "lactate_pool_3_mmol_l",
            "lactate_de_1_mmol_l",
            "lactate_de_3_mmol_l",
            "glucose_baseline_mmol_l",
            "glucose_end_mmol_l",
        ]
        blood["participant_id"] = left_ids

        components = [participants, core, blood]
        for filename, mapping in _SIMPLE_COLUMNS.items():
            allowed = frozenset({"W:R"}) if filename == "Motion Analysis.csv" else frozenset()
            components.append(
                _rename_exact(
                    _read_csv(archive, filename),
                    mapping,
                    filename,
                    allowed_footer_labels=allowed,
                )
            )

    result = components[0]
    expected_ids = set(result["participant_id"].astype(int))
    if result["participant_id"].duplicated().any():
        raise ValueError("Participant Info.csv contains duplicate IDs")
    for component in components[1:]:
        component_ids = set(component["participant_id"].astype(int))
        if component_ids != expected_ids or component["participant_id"].duplicated().any():
            raise ValueError(
                "Physiological measurement tables do not share one-to-one participants"
            )
        result = result.merge(component, on="participant_id", how="left", validate="one_to_one")

    result["weapon"] = "foil"
    result["study_context"] = "Spanish national-team training camp, June 2018"
    result["source_doi"] = DATASET_DOI
    ordered = [
        "participant_id",
        "weapon",
        "study_context",
        "source_doi",
        *[column for column in result.columns if column not in {"participant_id"}],
    ]
    # ``dict.fromkeys`` removes the already-inserted context columns while retaining order.
    return result[list(dict.fromkeys(ordered))].sort_values("participant_id").reset_index(drop=True)


def fetch_and_prepare(
    archive_path: Path,
    readme_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download the CC-licensed snapshot and create a value-preserving analytic table."""

    if force:
        archive_path.unlink(missing_ok=True)
        readme_path.unlink(missing_ok=True)
    download_resumable(ARCHIVE_URL, archive_path, user_agent=user_agent, expected_bytes=7_131)
    download_resumable(README_URL, readme_path, user_agent=user_agent)
    readme = readme_path.read_text(encoding="cp1252", errors="replace")
    if "Licenses/restrictions placed on the data: CC-BY" not in readme:
        raise ValueError("Physiological foil data license statement changed")

    frame = normalize_archive(archive_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    numeric = frame.select_dtypes(include="number").drop(columns=["participant_id"])
    negatives = {
        column: int((numeric[column] < 0).sum())
        for column in numeric.columns
        if bool((numeric[column] < 0).any())
    }
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Deidentified measurements from 18 experienced foil fencers in one 2018 Spanish "
            "national-team training-camp study. Values are preserved exactly, including any "
            "unexpected signs; this is not a population norm or evidence for epee or sabre."
        ),
        "source_url": DATASET_PAGE,
        "dataset_doi": DATASET_DOI,
        "article_doi": ARTICLE_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "readme_sha256": sha256_file(readme_path),
        "processed_sha256": sha256_file(output_path),
        "participants": len(frame),
        "by_gender": frame["gender"].value_counts().sort_index().astype(int).to_dict(),
        "analytic_columns": len(frame.columns),
        "numeric_measurements": int(numeric.notna().sum().sum()),
        "missing_numeric_values": int(numeric.isna().sum().sum()),
        "negative_value_counts_preserved": negatives,
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
