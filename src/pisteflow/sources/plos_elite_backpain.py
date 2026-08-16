"""Privacy-minimized fencer subset of the CC BY elite-sport back-pain survey."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = (
    "https://plos.figshare.com/articles/dataset/"
    "Back_pain_in_elite_sports_A_cross-sectional_study_on_1114_athletes/5158693"
)
ARTICLE_DOI = "https://doi.org/10.1371/journal.pone.0180130"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/8795344"
EXPECTED_BYTES = 69_901
EXPECTED_SHA256 = "7fe80630c7d59dc9ed1e4402eddc787f3d708e15028ceac3c0ac6837b523dcd8"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

PAIN_BINARY_COLUMNS = [
    "LT_Nacken",
    "LT_BWS",
    "LT_LWS",
    "l12m_Nacken",
    "l12m_BWS",
    "l12m_LWS",
    "l3m_Nacken",
    "l3m_BWS",
    "l3m_LWS",
    "l7d_Nacken",
    "l7d_BWS",
    "l7d_LWS",
    "LT_BP",
    "l12m_BP",
    "l3m_BP",
    "l7d_BP",
]
PAIN_DURATION_COLUMNS = [
    "HWS_12m_Schmerzdauer",
    "HWS_3m_Schmerzdauer",
    "LWS_12m_Schmerzdauer",
    "LWS_3m_Schmerzdauer",
    "BWS_12m_Schmerzdauer",
    "BWS_3m_Schmerzdauer",
]
PAIN_INTENSITY_COLUMNS = [
    "Korff_Schmerz_Augenblick",
    "Korff_stärksterSchmerz_3m",
    "Korff_durchsSchmerz_3m",
]


def _yes_no(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if float(str(value)) == 1:
        return False
    if float(str(value)) == 2:
        return True
    raise ValueError(f"Unexpected back-pain yes/no code: {value}")


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the 23 fencers and coarsen direct biometric quasi-identifiers."""

    if frame.shape != (1_280, 36):
        raise ValueError(f"PLOS back-pain snapshot changed: {frame.shape}")
    fencers = frame.loc[frame["Sportart"].eq(12)].copy()
    if len(fencers) != 23 or fencers.duplicated().any():
        raise ValueError("PLOS back-pain fencer subset membership changed")
    bmi = fencers["Gewicht"] / (fencers["Größe"] / 100) ** 2
    result = pd.DataFrame(
        {
            "participant_key": [f"PEB-{index:02d}" for index in range(1, 24)],
            "sex": fencers["Geschlecht"].map({1.0: "woman", 2.0: "man"}),
            "age_band": pd.cut(
                fencers["Alter_gesamt"],
                bins=[0, 17, 20, 24, 29, float("inf")],
                labels=["under_18", "18_20", "21_24", "25_29", "30_plus"],
            ).astype("string"),
            "bmi": bmi.round(1),
            "weekly_training_hours": fencers["StundenSport_Woche"],
            "sport_experience_band": pd.cut(
                fencers["Jahre_Sport"],
                bins=[0, 4, 9, 14, 19, float("inf")],
                labels=["under_5", "5_9", "10_14", "15_19", "20_plus"],
            ).astype("string"),
            "competition_cadre": fencers["Wettkampfniveau1"].map(
                {1.0: "A", 2.0: "B", 3.0: "C"}
            ),
        }
    )
    for column in PAIN_BINARY_COLUMNS:
        result[column] = fencers[column].map(_yes_no).astype("boolean").to_numpy()
    for column in [*PAIN_DURATION_COLUMNS, *PAIN_INTENSITY_COLUMNS]:
        result[column] = fencers[column].to_numpy()
    result["weapon"] = "unknown"
    result["source_url"] = ARTICLE_DOI
    return result


def fetch_and_prepare(
    cache_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the survey and write the 23-row fencer subset locally."""

    if force:
        cache_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        cache_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    if sha256_file(cache_path) != EXPECTED_SHA256:
        raise ValueError("PLOS elite back-pain checksum changed")
    try:
        pyreadstat: Any = importlib.import_module("pyreadstat")
    except ImportError as error:  # pragma: no cover - environment-specific guidance
        raise RuntimeError("Install pisteflow-analytics[collectors] to read SPSS files") from error
    frame, _metadata = pyreadstat.read_sav(cache_path, apply_value_formats=False)
    processed = normalize(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, index=False)

    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "article_doi": ARTICLE_DOI,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "raw_sha256": EXPECTED_SHA256,
        "raw_rows_all_sports_and_controls": len(frame),
        "fencer_rows": len(processed),
        "weapon": "not_deposited",
        "by_sex": processed["sex"].value_counts().sort_index().to_dict(),
        "age_bands": processed["age_band"].value_counts().sort_index().to_dict(),
        "cadres": processed["competition_cadre"].value_counts().sort_index().to_dict(),
        "self_reported_any_back_pain": {
            "lifetime": int(processed["LT_BP"].sum()),
            "last_12_months": int(processed["l12m_BP"].sum()),
            "last_3_months": int(processed["l3m_BP"].sum()),
            "last_7_days": int(processed["l7d_BP"].sum()),
        },
        "claim_boundary": (
            "Twenty-three elite fencers extracted from a 1,280-row German sport/control survey. "
            "These are anonymous cross-sectional self-reports, not clinical diagnoses or injury "
            "onset labels. All 23 report lifetime back pain, so this subset cannot train an "
            "ever-injured versus never-injured classifier. Weapon is absent. Exact age, height, "
            "weight, and sport-experience years are removed or coarsened; BMI, training volume, "
            "cadre, regional pain periods, and intensity remain for local grouped analysis."
        ),
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "columns": len(processed.columns),
        },
    }
    write_json(report_path, summary)
    return summary
