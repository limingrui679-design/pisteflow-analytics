"""Privacy-minimized local adapter for a public fencer psychology SPSS file."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://osf.io/pa7fc/"
DOWNLOAD_URL = "https://osf.io/download/b3245/"
EXPECTED_BYTES = 30_255
EXPECTED_SHA256 = "bbb519af9102774bbc3e74e79414a7bdd515461336a704fc3f133a6482e51238"
WEAPONS = {1.0: "foil", 2.0: "epee", 3.0: "sabre"}
SEX = {1.0: "woman", 2.0: "man"}
STUDIES = {1.0: "basic", 2.0: "secondary", 3.0: "university"}

MIPS_ITEMS = [f"MIPS{index}" for index in range(1, 11)]
AAQ_ITEMS = [f"AAQ{index}" for index in range(1, 8)]
ISCCS_ITEMS = [
    "ISCCS1",
    "ISCCS3",
    "ISCCS4",
    "ISCCS5",
    "ISCCS6",
    "ISCCS8",
    "ISCCS9",
    "ISCCS10",
    "ISCCS11",
    "ISCCS12",
    "ISCCS13",
    "ISCCS15",
    "ISCCS16",
    "ISCCS18",
    "ISCCS19",
    "ISCCS20",
    "ISCCS21",
    "ISCCS22",
    "ISCCS23",
    "ISCCS25",
    "ISCCS26",
    "ISCCS27",
    "ISCCS29",
    "ISCCS30",
    "ISCCS31",
    "ISCCS32",
    "ISCCS33",
    "ISCCS35",
    "ISCCS36",
    "ISCCS37",
    "ISCCS38",
]
DERIVED_SCALES = [
    "AAQTOTAL",
    "PERFECTOTAL",
    "MIPSF1perfectionisteffort",
    "MIPSF2negativereaction",
    "ISCCSLOGICALANALYSIS",
    "ISCCSSEEKINGSUPPORT",
    "ISCCSRELAXATION",
    "ISCCSIMAGINERY",
    "TOCISCCS",
    "ISCCSRESIGNATION",
    "ISCCSEMOTIONALOUTPUT",
    "EOCISCCS",
    "ISCCSDISTRACTION",
    "ISCCSDISTANCING",
    "DOCISCCS",
]


def age_band(value: object) -> str:
    age = float(str(value))
    if age < 18:
        return "under_18"
    if age < 21:
        return "18_20"
    if age < 30:
        return "21_29"
    if age < 40:
        return "30_39"
    return "40_plus"


def duration_band(value: object) -> str:
    years = float(str(value))
    if years < 5:
        return "under_5"
    if years < 10:
        return "5_9"
    if years < 20:
        return "10_19"
    return "20_plus"


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact dates/ages and derived cluster labels from the local table."""

    if frame.shape != (78, 107):
        raise ValueError(f"OSF fencer psychology snapshot changed: {frame.shape}")
    if frame.duplicated().any() or frame.isna().any().any():
        raise ValueError("OSF fencer psychology duplicate or missing-value status changed")
    required = {
        "Sex",
        "Yearbirth",
        "Studies",
        "Yearstraining",
        "Nautonómico",
        "Añosentrenoarma",
        "Armaidentificas",
        "Coincidecarma",
        "Trainingsessionsweek",
        "Numbertraininghoursweek",
        "AGE",
        "YEARSTRAINSFENCING",
        "CLUSTERPSYCHOLINFLEX",
        "QCL_2",
        *MIPS_ITEMS,
        *AAQ_ITEMS,
        *ISCCS_ITEMS,
        *DERIVED_SCALES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OSF fencer psychology columns changed: {sorted(missing)}")
    if not set(frame["Armaidentificas"]).issubset(WEAPONS):
        raise ValueError("OSF fencer psychology weapon codes changed")

    result = pd.DataFrame(
        {
            "participant_key": [f"OFP-{index:03d}" for index in range(1, len(frame) + 1)],
            "weapon": frame["Armaidentificas"].map(WEAPONS),
            "sex": frame["Sex"].map(SEX),
            "education_level": frame["Studies"].map(STUDIES),
            "age_band": frame["AGE"].map(age_band),
            "fencing_experience_band": frame["YEARSTRAINSFENCING"].map(duration_band),
            "current_weapon_experience_band": frame["Añosentrenoarma"].map(duration_band),
            "weapon_code_disagreement": frame["Armaidentificas"].ne(frame["Nautonómico"]),
            "current_weapon_years_exceeds_fencing_years": frame["Añosentrenoarma"].gt(
                frame["YEARSTRAINSFENCING"]
            ),
            "training_sessions_per_week": frame["Trainingsessionsweek"],
            "training_hours_per_week": frame["Numbertraininghoursweek"],
            "team_member_last_five_seasons_count": frame["TEAMMEMBERLAST5SEASONS"],
            "national_team_call_band": frame["TIMESCALLEDTEAMRECODIF2"].map(
                {1.0: "under_3", 2.0: "3_plus"}
            ),
            "absolute_podium": frame["ABSOLUTEPODIUMYESNO"].astype("int8"),
            "high_performance_center_recent": frame[
                "Ultimas5temporadascentroaltorendimiento"
            ].map({1.0: True, 2.0: False}),
            "technical_team_code": frame["Equipotécnico"].astype("int8"),
            "technical_team_includes_psychologist": frame["EQUIPOTECNICOPSICOL"].map(
                {1.0: True, 2.0: False}
            ),
        }
    )
    competition_columns = [
        "Nautonómico_A",
        "Nivelnacional_A",
        "Ninternacional_A",
        "Nautonómico_B",
        "Nivelnacional_B",
        "Ninternacional_B",
        "M20_A",
        "Sub23_A",
        "Absoluto_A",
        "M20_B",
        "Sub23_B",
        "Absoluto_B",
        "M20_C",
        "Sub23_C",
        "Absoluto_C",
        "M20_D",
        "Sub23_D",
        "Absoluto_D",
    ]
    for column in [*competition_columns, *MIPS_ITEMS, *ISCCS_ITEMS, *AAQ_ITEMS, *DERIVED_SCALES]:
        result[column] = frame[column].to_numpy()
    result["source_url"] = SOURCE_URL
    return result


def fetch_and_prepare(
    cache_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download the pinned SPSS file and write a local-only analytic table."""

    if force:
        cache_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        cache_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    if sha256_file(cache_path) != EXPECTED_SHA256:
        raise ValueError("OSF fencer psychology checksum changed")
    try:
        pyreadstat: Any = importlib.import_module("pyreadstat")
    except ImportError as error:  # pragma: no cover - environment-specific guidance
        raise RuntimeError("Install pisteflow-analytics[collectors] to read SPSS files") from error
    frame, metadata = pyreadstat.read_sav(cache_path, apply_value_formats=False)
    processed = normalize(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, index=False)

    age_distribution = processed["age_band"].value_counts().sort_index().to_dict()
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "download_url": DOWNLOAD_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "access": "public_download",
        "license": "not_declared_on_osf_node_or_deposited_file",
        "repository_policy": "raw_and_processed_rows_local_only",
        "raw_sha256": EXPECTED_SHA256,
        "raw_bytes": cache_path.stat().st_size,
        "participants": len(processed),
        "by_weapon": processed["weapon"].value_counts().sort_index().to_dict(),
        "by_sex": processed["sex"].value_counts().sort_index().to_dict(),
        "age_bands": age_distribution,
        "weapon_code_disagreement_rows": int(processed["weapon_code_disagreement"].sum()),
        "current_weapon_duration_anomaly_rows": int(
            processed["current_weapon_years_exceeds_fencing_years"].sum()
        ),
        "questionnaire_items": {
            "mips_perfectionism": len(MIPS_ITEMS),
            "isccs_coping": len(ISCCS_ITEMS),
            "aaq_psychological_inflexibility": len(AAQ_ITEMS),
            "deposited_derived_scales_retained": len(DERIVED_SCALES),
            "deposited_cluster_label_and_distance_excluded": 2,
        },
        "variable_labels_available": len(metadata.column_names_to_labels),
        "claim_boundary": (
            "Anonymous cross-sectional self-report data from 78 fencers, not diagnoses, "
            "longitudinal training responses, match outcomes, or causal evidence. Exact birth "
            "year, exact age, training-start year, exact duration variables, deposited cluster "
            "labels, and cluster distances are excluded from the processed table. The identified "
            "weapon field gives 4 foil, 68 epee, and 6 sabre rows; two rows disagree with another "
            "weapon-code field, and four duration values are internally inconsistent. Because no "
            "reuse "
            "license is declared and responses are psychologically sensitive, row data remain "
            "local-only and analyses must be grouped at participant level."
        ),
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "columns": len(processed.columns),
        },
    }
    write_json(report_path, summary)
    return summary
