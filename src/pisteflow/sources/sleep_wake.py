"""Longitudinal wearable sleep records from professional fencing athletes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.cloudpmc import download_asset
from pisteflow.sources.net import download_resumable

PMCID = "PMC11740734"
DOI = "https://doi.org/10.7717/peerj.18812"
ARTICLE_URL = f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/"
XML_URL = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML"
ASSET_URL = (
    "https://pmc.ncbi.nlm.nih.gov/articles/instance/11740734/bin/"
    "peerj-13-18812-s001.xlsx"
)
LICENSE = "CC BY 4.0"
EXPECTED_BYTES = 211_848
_COLUMNS = {
    "ID": "participant_code",
    "date time": "observation_date",
    "Date of sleep": "sleep_date",
    "sleep onset time": "sleep_onset_time",
    "Date of wake-up": "wake_date",
    "wake-up time": "wake_time",
    "deep sleep duration(min)": "deep_sleep_minutes",
    " light sleep duration(min)": "light_sleep_minutes",
    "REM duration(min)": "rem_sleep_minutes",
    "awake duration(min)": "awake_minutes",
    "nap duration": "nap_minutes",
    "total sleep duration(min)": "total_sleep_minutes",
}


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and attach non-destructive quality flags."""

    if list(frame.columns) != list(_COLUMNS):
        raise ValueError("PeerJ sleep-data columns changed")
    exact_duplicate = frame.duplicated(keep=False)
    date_collision = frame.duplicated(["ID", "date time"], keep=False)
    result = frame.rename(columns=_COLUMNS).copy()
    result.insert(0, "source_row", range(1, len(result) + 1))
    result["participant_code"] = result["participant_code"].astype("string")
    for column in ["observation_date", "sleep_date", "wake_date"]:
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    for column in ["sleep_onset_time", "wake_time"]:
        result[column] = result[column].map(lambda value: str(value))
    numeric_columns = [
        "deep_sleep_minutes",
        "light_sleep_minutes",
        "rem_sleep_minutes",
        "awake_minutes",
        "nap_minutes",
        "total_sleep_minutes",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if bool((result[numeric_columns] < 0).any().any()):
        raise ValueError("PeerJ sleep data contains a negative duration")
    series_rows = result.groupby("participant_code")["source_row"].transform("size")
    result["exact_duplicate_group"] = exact_duplicate.to_numpy()
    result["participant_date_collision"] = date_collision.to_numpy()
    result["short_series_under_50_rows"] = series_rows.lt(50)
    result["component_sum_minutes"] = result[
        [
            "deep_sleep_minutes",
            "light_sleep_minutes",
            "rem_sleep_minutes",
            "awake_minutes",
            "nap_minutes",
        ]
    ].sum(axis=1)
    result["component_total_difference_minutes"] = (
        result["component_sum_minutes"] - result["total_sleep_minutes"]
    )
    result["weapon"] = "unknown"
    result["source_doi"] = DOI
    return result


def normalize_snapshot(path: Path) -> pd.DataFrame:
    frame = normalize_frame(pd.read_excel(path))
    if len(frame) != 2_607 or frame["participant_code"].nunique() != 25:
        raise ValueError("PeerJ sleep-data snapshot size changed")
    return frame


def fetch_and_prepare(
    raw_path: Path,
    xml_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        xml_path.unlink(missing_ok=True)
    download_resumable(XML_URL, xml_path, user_agent=user_agent)
    download_asset(
        ASSET_URL,
        raw_path,
        expected_bytes=EXPECTED_BYTES,
        user_agent=user_agent,
        force=force,
        expected_magic=b"PK\x03\x04",
    )
    frame = normalize_snapshot(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Longitudinal wristband sleep measurements from professional fencing-athlete "
            "study codes. The supplement has 2,607 rows and 25 codes, while the article "
            "states 2,459 valid days from 23 athletes; the raw file also dates observations "
            "in 2022 although the methods text says 2023. No unpublished inclusion rule is "
            "reconstructed, no code is assumed to be a particular weapon, and duplicates "
            "and date collisions are flagged rather than silently removed."
        ),
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "pmcid": PMCID,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "article_xml_sha256": sha256_file(xml_path),
        "raw_sha256": sha256_file(raw_path),
        "raw_rows": len(frame),
        "raw_participant_codes": int(frame["participant_code"].nunique()),
        "article_reported_participants": 23,
        "article_reported_valid_days": 2_459,
        "date_min": str(frame["observation_date"].min().date()),
        "date_max": str(frame["observation_date"].max().date()),
        "exact_duplicate_rows": int(frame["exact_duplicate_group"].sum()),
        "participant_date_collision_rows": int(frame["participant_date_collision"].sum()),
        "short_series_participant_codes": int(
            frame.loc[frame["short_series_under_50_rows"], "participant_code"].nunique()
        ),
        "component_total_mismatch_rows": int(
            frame["component_total_difference_minutes"].ne(0).sum()
        ),
        "weapon_label": "unknown; not published in the supplement",
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
