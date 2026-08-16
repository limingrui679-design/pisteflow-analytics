"""CC BY three-weapon elite-fencer strength and lunge-velocity measurements."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.cloudpmc import download_asset
from pisteflow.sources.net import download_resumable

PMCID = "PMC12951885"
DOI = "https://doi.org/10.7717/peerj.20879"
ARTICLE_URL = f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/"
XML_URL = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML"
LICENSE = "CC BY 4.0"
_ASSET_ROOT = "https://pmc.ncbi.nlm.nih.gov/articles/instance/12951885/bin"
PRIMARY_FILENAME = "peerj-14-20879-s001.xlsx"
RELIABILITY_FILENAME = "peerj-14-20879-s002.xlsx"
_EXPECTED_BYTES = {PRIMARY_FILENAME: 22_610, RELIABILITY_FILENAME: 21_643}

_WEAPONS = {"épée": "epee", "foil": "foil", "saber": "sabre"}
_PRIMARY_COLUMNS = {
    "Name": "participant_code",
    "Weapon": "weapon",
    "Age(years)": "age_years",
    "Hight(cm)": "height_cm",
    "Weight(kg)": "weight_kg",
    "Years of training(years)": "training_years",
    "Static Lunge Velocity(m/s)": "static_lunge_velocity_m_s",
    "Advance Lunge Velocity (m/s)": "advance_lunge_velocity_m_s",
    "SJ-Relative Peak Force (N·kg-1)": "sj_relative_peak_force_n_kg",
    "SJ-Relative Peak Power (W·kg-1)": "sj_relative_peak_power_w_kg",
    "SJ-Jump Height (m)": "sj_jump_height_m",
    "SJ-Concentric Peak Velocity (m·s-1)": "sj_concentric_peak_velocity_m_s",
    "DJ-Relative Peak Force (N·kg-1)": "dj_relative_peak_force_n_kg",
    "DJ-Relative Peak Power (W·kg-1)": "dj_relative_peak_power_w_kg",
    "DJ-Contact Time（s）": "dj_contact_time_s",  # noqa: RUF001 - exact source header
    "DJ-Jump Height (m)": "dj_jump_height_m",
    "DJ-RSImod(m·s-1)": "dj_rsi_modified_m_s",
    "CMJ-Relative Peak Force (N·kg-1)": "cmj_relative_peak_force_n_kg",
    "CMJ-Relative Peak Power (W·kg-1)": "cmj_relative_peak_power_w_kg",
    "CMJ-Concentric Peak Velocity (m·s-1)": "cmj_concentric_peak_velocity_m_s",
    "CMJ-Jump Height (m)": "cmj_jump_height_m",
    "CMJ-Contact Time（s）": "cmj_contact_time_s",  # noqa: RUF001 - exact source header
    "CMJ-RSImod(m·s-1)": "cmj_rsi_modified_m_s",
    "IMTP-Relative Peak Force (N·kg-1)": "imtp_relative_peak_force_n_kg",
    "IMTP-50ms RFD(N·s-1)": "imtp_rfd_50ms_n_s",
    "IMTP-100ms RFD(N·s-1)": "imtp_rfd_100ms_n_s",
    "IMTP-150ms RFD(N·s-1)": "imtp_rfd_150ms_n_s",
    "IMTP-200ms RFD(N·s-1)": "imtp_rfd_200ms_n_s",
    "IMTP-250ms RFD(N·s-1)": "imtp_rfd_250ms_n_s",
    "IMTP-300ms RFD(N·s-1)": "imtp_rfd_300ms_n_s",
    "EUR": "eccentric_utilization_ratio",
    "DSI": "dynamic_strength_index",
}
_RELIABILITY_COLUMNS = {
    "Name": "participant_code",
    "Weapon": "weapon",
    "Number of tests": "test_number",
    "imtp-Relative Peak Force (N·kg-1)": "imtp_relative_peak_force_n_kg",
    "imtp-50ms RFD": "imtp_rfd_50ms_n_s",
    "imtp-100ms RFD": "imtp_rfd_100ms_n_s",
    "imtp-150ms RFD": "imtp_rfd_150ms_n_s",
    "imtp-200ms RFD": "imtp_rfd_200ms_n_s",
    "imtp-250ms RFD": "imtp_rfd_250ms_n_s",
    "imtp-300ms RFD": "imtp_rfd_300ms_n_s",
    "DJ-Relative Peak Force (N·kg-1)": "dj_relative_peak_force_n_kg",
    "DJ-Relative Peak Power (W·kg-1)": "dj_relative_peak_power_w_kg",
    "DJ-Jup Height (m)": "dj_jump_height_m",
    "DJ-Contact Time（s）": "dj_contact_time_s",  # noqa: RUF001 - exact source header
    "DJ-RSImod(m·s-1)": "dj_rsi_modified_m_s",
}


def _download_cloudpmc_asset(
    filename: str,
    destination: Path,
    *,
    user_agent: str,
    force: bool,
) -> Path:
    return download_asset(
        f"{_ASSET_ROOT}/{filename}",
        destination,
        expected_bytes=_EXPECTED_BYTES[filename],
        user_agent=user_agent,
        force=force,
        expected_magic=b"PK\x03\x04",
    )


def _normalize_weapon(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    observed = set(frame["weapon"].astype("string"))
    if observed != set(_WEAPONS):
        raise ValueError(f"{filename} weapon labels changed: {sorted(observed)}")
    frame["weapon"] = frame["weapon"].map(_WEAPONS).astype("string")
    return frame


def normalize_primary(path: Path) -> pd.DataFrame:
    """Normalize the 45-row participant table without altering measurements."""

    frame = pd.read_excel(path)
    if list(frame.columns) != list(_PRIMARY_COLUMNS):
        raise ValueError("PeerJ primary-data columns changed")
    result = frame.rename(columns=_PRIMARY_COLUMNS)
    if len(result) != 45 or result["participant_code"].duplicated().any():
        raise ValueError("PeerJ primary-data participant layout changed")
    result = _normalize_weapon(result, PRIMARY_FILENAME)
    expected = {"epee": 17, "foil": 12, "sabre": 16}
    if result["weapon"].value_counts().to_dict() != expected:
        raise ValueError("PeerJ primary-data three-weapon sample counts changed")
    result["source_doi"] = DOI
    result["population"] = "elite_female_fencers"
    return result


def normalize_reliability(path: Path) -> pd.DataFrame:
    """Normalize the repeated-test table while keeping it separate from primary IDs."""

    frame = pd.read_excel(path)
    if list(frame.columns) != list(_RELIABILITY_COLUMNS):
        raise ValueError("PeerJ reliability-data columns changed")
    result = frame.rename(columns=_RELIABILITY_COLUMNS)
    if len(result) != 120:
        raise ValueError("PeerJ reliability-data row count changed")
    result = _normalize_weapon(result, RELIABILITY_FILENAME)
    group_sizes = result.groupby(["weapon", "participant_code"], dropna=False).size()
    if len(group_sizes) != 40 or not group_sizes.eq(3).all():
        raise ValueError("PeerJ reliability-data repeated-test structure changed")
    if set(result["test_number"].astype(int)) != {1, 2, 3}:
        raise ValueError("PeerJ reliability-data test numbering changed")
    result["source_doi"] = DOI
    result["population"] = "elite_female_fencers_reliability_subset"
    return result


def fetch_and_prepare(
    raw_dir: Path,
    xml_path: Path,
    primary_output: Path,
    reliability_output: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch both official supplements and publish traceable analytic tables."""

    if force:
        xml_path.unlink(missing_ok=True)
    download_resumable(XML_URL, xml_path, user_agent=user_agent)
    primary_raw = _download_cloudpmc_asset(
        PRIMARY_FILENAME, raw_dir / PRIMARY_FILENAME, user_agent=user_agent, force=force
    )
    reliability_raw = _download_cloudpmc_asset(
        RELIABILITY_FILENAME,
        raw_dir / RELIABILITY_FILENAME,
        user_agent=user_agent,
        force=force,
    )
    primary = normalize_primary(primary_raw)
    reliability = normalize_reliability(reliability_raw)
    primary_output.parent.mkdir(parents=True, exist_ok=True)
    primary.to_parquet(primary_output, index=False)
    reliability.to_parquet(reliability_output, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "CC BY measurements from 45 elite female Chinese fencers across all three weapons. "
            "The 120-row reliability table represents three repeated tests for 40 study codes; "
            "it is not 120 additional athletes and its letter codes are not joined to the "
            "primary table because the article publishes no crosswalk. Cross-sectional "
            "associations are not causal training effects or population norms."
        ),
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "pmcid": PMCID,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "article_xml_sha256": sha256_file(xml_path),
        "primary_raw_sha256": sha256_file(primary_raw),
        "reliability_raw_sha256": sha256_file(reliability_raw),
        "participants": len(primary),
        "by_weapon_participants": (
            primary["weapon"].value_counts().sort_index().astype(int).to_dict()
        ),
        "participant_measurement_columns": len(_PRIMARY_COLUMNS) - 2,
        "reliability_rows": len(reliability),
        "reliability_participant_codes": int(reliability["participant_code"].nunique()),
        "by_weapon_reliability_codes": (
            reliability.groupby("weapon")["participant_code"]
            .nunique()
            .sort_index()
            .astype(int)
            .to_dict()
        ),
        "primary_output_sha256": sha256_file(primary_output),
        "reliability_output_sha256": sha256_file(reliability_output),
        "primary_output": str(primary_output),
        "reliability_output": str(reliability_output),
    }
    write_json(report_path, summary)
    return summary
