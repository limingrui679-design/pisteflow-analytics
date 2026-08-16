"""CC BY temporal observations for 44 women's wheelchair-foil bouts."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://figshare.com/articles/dataset/30598058"
SOURCE_DOI = "https://doi.org/10.6084/m9.figshare.30598058.v1"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/59495930"
LICENSE = "CC BY 4.0"
FIGSHARE_FILE_ID = 59_495_930
EXPECTED_BYTES = 22_833
EXPECTED_SHA256 = "ea6b0553138e88e01d5f0dd968a30eece9ca6f883c1a54b7e90d3a1367d49199"
EXPECTED_ROWS = 44
EXPECTED_COLUMNS = [
    "Orden",
    "Cat_Discapa",
    "Mano",
    "Fase",
    "MO_FIN_ASAL_PERIODO",
    "MO_FIN_ASAL_PER_MINUT",
    "Consume_tpo_asalto",
    "TTASAL",
    "TTACC",
    "TTPAU",
    "TTDES",
    "TTPA_TV",
    "TTPA_TBNV",
    "TTPA_TNV",
    "TTPA_NORM1",
    "TTPA_NORM2",
    "TTPA_SILLA",
    "TTPA_VID",
    "TTPA_MATERIAL",
    "TTPA_MEDICA",
    "TTPA_AJUMAT",
    "TTPA_AJUSUJ",
    "TTPA_ASEO",
    "TTPA_CFLO",
    "TTPA_RUIDO",
    "TTPA_IN_SILLA",
    "TTPA_IN_VID",
    "TTPA_IN_MATERIAL",
    "TTPA_IN_AJUMAT",
    "TTPA_IN_AJUSUJ",
    "TTPA_IN_CFLO",
    "TTPA_IN_NORM2",
    "TTPA_IN_ASEO",
    "TTPA_IN_MEDICA",
    "TTPA_IN_RUIDO",
    "TTDES_IN_SILLA",
    "TTDES_IN_AJUMAT",
    "TTDES_IN_AJUSUJ",
    "TTDES_IN_T8",
    "TTDES_IN_ASEO",
    "TTDES_FIN1",
    "STACC",
    "STPAU",
    "STDES",
    "STPA_TV",
    "STPA_TBNV",
    "STPA_TNV",
    "STPA_NORM1",
    "STPA_NORM2",
    "STPA_SILLA",
    "STPA_VID",
    "STPA_MATERIAL",
    "STPA_MEDICA",
    "STPA_AJUMAT",
    "STPA_AJUSUJ",
    "STPA_ASEO",
    "STPA_CFLO",
    "STPA_RUIDO",
    "STPA_IN_SILLA",
    "STPA_IN_VID",
    "STPA_IN_MATERIAL",
    "STPA_IN_AJUMAT",
    "STPA_IN_AJUSUJ",
    "STPA_IN_CFLO",
    "STPA_IN_NORM2",
    "STPA_IN_ASEO",
    "STPA_IN_MEDICA",
    "STPA_IN_RUIDO",
    "STDES_IN_SILLA",
    "STDES_IN_AJUMAT",
    "STDES_IN_AJUSUJ",
    "STDES_IN_T8",
    "STDES_IN_ASEO",
    "STDES_FIN1",
    "TSACC",
    "TSPAU",
    "TSDES",
    "TSPA_TV",
    "TSPA_TBNV",
    "TSPA_TNV",
    "TSPA_NORM1",
    "TSPA_NORM2",
    "TSPA_SILLA",
    "TSPA_VID",
    "TSPA_MATERIAL",
    "TSPA_MEDICA",
    "TSPA_AJUMAT",
    "TSPA_AJUSUJ",
    "TSPA_ASEO",
    "TSPA_CFLO",
    "TSPA_RUIDO",
    "TSPA_IN_SILLA",
    "TSPA_IN_VID",
    "TSPA_IN_MATERIAL",
    "TSPA_IN_AJUMAT",
    "TSPA_IN_AJUSUJ",
    "TSPA_IN_CFLO",
    "TSPA_IN_NORM2",
    "TSPA_IN_ASEO",
    "TSPA_IN_MEDICA",
    "TSPA_IN_RUIDO",
    "TSDES_IN_SILLA",
    "TSDES_IN_AJUMAT",
    "TSDES_IN_AJUSUJ",
    "TSDES_IN_T8",
    "TSDES_IN_ASEO",
    "TSDES_FIN1",
]
_HANDEDNESS = {
    "DD": "right_vs_right",
    "ZD": "left_vs_right",
    "ZZ": "left_vs_left",
    "DZ": "right_vs_left",
}


def _bout_key(order: object) -> str:
    numeric = int(float(str(order)))
    digest = hashlib.sha256(f"wheelchair-foil-temporal:{numeric}".encode()).hexdigest()[:20]
    return f"WFT-{digest}"


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source codes without inventing meanings for undocumented variables."""

    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Wheelchair-foil temporal SAV columns changed")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Wheelchair-foil temporal row count changed: {len(frame)}")
    if frame["Orden"].tolist() != [float(value) for value in range(1, EXPECTED_ROWS + 1)]:
        raise ValueError("Wheelchair-foil temporal bout order changed")
    expected_categories = {
        "Cat_Discapa": {"A", "B"},
        "Mano": set(_HANDEDNESS),
        "MO_FIN_ASAL_PERIODO": {"PER2"},
        "MO_FIN_ASAL_PER_MINUT": {"PER2M0", "PER2M1", "PER2M2"},
        "Consume_tpo_asalto": {"NO"},
    }
    for column, allowed in expected_categories.items():
        values = {str(value) for value in frame[column].dropna()}
        if values != allowed:
            raise ValueError(f"Wheelchair-foil temporal {column} vocabulary changed: {values}")
    if frame.select_dtypes(include="number").drop(columns="Orden").lt(0).any().any():
        raise ValueError("Wheelchair-foil temporal numeric values became negative")
    if frame.drop(columns="Orden").duplicated().any():
        raise ValueError("Wheelchair-foil temporal source contains duplicate bout rows")

    result = frame.copy()
    result.insert(0, "bout_key", result["Orden"].map(_bout_key))
    result["Cat_Discapa"] = result["Cat_Discapa"].astype(str)
    result["Mano"] = result["Mano"].astype(str).map(_HANDEDNESS)
    result["Fase"] = result["Fase"].astype(str)
    result["MO_FIN_ASAL_PERIODO"] = result["MO_FIN_ASAL_PERIODO"].astype(str)
    result["MO_FIN_ASAL_PER_MINUT"] = result["MO_FIN_ASAL_PER_MINUT"].astype(str)
    result["Consume_tpo_asalto"] = False
    result = result.rename(
        columns={
            "Cat_Discapa": "wheelchair_category",
            "Mano": "handedness_matchup",
            "Fase": "competition_phase_source_code",
            "MO_FIN_ASAL_PERIODO": "ending_period_source_code",
            "MO_FIN_ASAL_PER_MINUT": "ending_minute_source_code",
            "Consume_tpo_asalto": "consumed_full_bout_time",
        }
    ).drop(columns="Orden")
    result = result.rename(
        columns={column: column.lower() for column in result.columns if column.isupper()}
    )
    result["weapon"] = "foil"
    result["event_type"] = "women_wheelchair_fencing"
    result["source_doi"] = SOURCE_DOI
    result.attrs = {}
    if result["bout_key"].duplicated().any():
        raise ValueError("Wheelchair-foil temporal bout keys are duplicated")
    return result


def fetch_and_prepare(
    raw_path: Path,
    output_path: Path,
    codebook_path: Path,
    summary_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        raw_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        raw_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    if sha256_file(raw_path) != EXPECTED_SHA256:
        raise ValueError("Wheelchair-foil temporal SAV checksum changed")
    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "The wheelchair-foil collector requires the 'collectors' extra."
        ) from exc
    source, metadata = pyreadstat.read_sav(
        raw_path,
        apply_value_formats=True,
        formats_as_category=False,
    )
    output = normalize(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    codebook = {
        "column_labels": metadata.column_names_to_labels,
        "value_labels": {
            column: {str(code): label for code, label in values.items()}
            for column, values in metadata.variable_value_labels.items()
        },
        "source_columns": list(source.columns),
    }
    write_json(codebook_path, codebook)
    numeric = output.select_dtypes(include="number")
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "license": LICENSE,
        "figshare_file_id": FIGSHARE_FILE_ID,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": sha256_file(raw_path),
        "bouts": len(output),
        "bouts_by_wheelchair_category": output["wheelchair_category"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "bouts_by_handedness_matchup": output["handedness_matchup"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "bouts_by_competition_phase_source_code": output["competition_phase_source_code"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "bouts_by_ending_minute_source_code": output["ending_minute_source_code"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "retained_source_measure_columns": len(numeric.columns),
        "retained_numeric_measurement_cells": int(numeric.notna().sum().sum()),
        "sparse_missing_numeric_cells": int(numeric.isna().sum().sum()),
        "athlete_identifiers_available": False,
        "source_video_links_available": False,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "codebook": str(codebook_path),
        "codebook_sha256": sha256_file(codebook_path),
        "claim_boundary": (
            "Forty-four observed post-2022 international women's wheelchair-foil direct-"
            "elimination bouts, balanced 22/22 across functional categories A and B. The SAV "
            "contains bout-level temporal and sequence aggregates, not source videos, touches, "
            "athlete identities, or one row per unique athlete. Repeated-athlete overlap cannot "
            "be audited, so these rows support descriptive analysis but not a leakage-safe "
            "bout-level train/test benchmark. Undocumented variable codes remain source codes."
        ),
        "repository_policy": "privacy-minimized CC BY row table local; code and aggregates public",
    }
    write_json(summary_path, summary)
    return summary
