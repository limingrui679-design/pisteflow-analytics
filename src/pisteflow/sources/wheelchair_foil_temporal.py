"""Figshare CC BY temporal structure data for women's wheelchair foil."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DOI = "https://doi.org/10.6084/m9.figshare.30598058.v1"
ARTICLE_URL = "https://figshare.com/articles/dataset/30598058"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/59495930"
LICENSE = "CC BY 4.0"
EXPECTED_BYTES = 22_833
EXPECTED_MD5 = "3db41b5c083454efb33541f5e576f43b"

_REQUIRED = {
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
    "STACC",
    "STPAU",
    "STDES",
    "TSACC",
    "TSPAU",
    "TSDES",
}


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach stable English fields without discarding the coded source variables."""

    missing = _REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Wheelchair-foil temporal data missing columns: {sorted(missing)}")
    result = frame.copy()
    result.insert(0, "source_row", range(1, len(result) + 1))
    result.insert(1, "bout_code", [f"WFF-{value:03d}" for value in result["Orden"].astype(int)])
    result["category"] = result["Cat_Discapa"].astype("string")
    result["paired_handedness_code"] = result["Mano"].astype("string")
    result["competition_phase"] = result["Fase"].astype("string")
    result["ending_period_code"] = result["MO_FIN_ASAL_PERIODO"].astype("string")
    result["ending_minute_code"] = result["MO_FIN_ASAL_PER_MINUT"].astype("string")
    result["used_full_bout_time"] = result["Consume_tpo_asalto"].eq("SÍ")
    result["total_bout_seconds"] = result["TTASAL"]
    result["action_seconds"] = result["TTACC"]
    result["pause_seconds"] = result["TTPAU"]
    result["rest_seconds"] = result["TTDES"]
    result["action_sequences"] = result["STACC"]
    result["pause_sequences"] = result["STPAU"]
    result["rest_sequences"] = result["STDES"]
    result["mean_action_sequence_seconds"] = result["TSACC"]
    result["mean_pause_sequence_seconds"] = result["TSPAU"]
    result["mean_rest_sequence_seconds"] = result["TSDES"]
    result["total_equals_action_plus_pause"] = result["TTASAL"].eq(
        result["TTACC"] + result["TTPAU"]
    )
    result["weapon"] = "foil"
    result["gender"] = "women"
    result["adaptive_format"] = "wheelchair"
    result["source_doi"] = DOI
    return result


def _string_keyed_labels(labels: dict[str, dict[float, str]]) -> dict[str, dict[str, str]]:
    return {
        column: {str(code): value for code, value in values.items()}
        for column, values in labels.items()
    }


def read_snapshot(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "The wheelchair-foil collector requires the 'collectors' extra."
        ) from exc
    frame, metadata = pyreadstat.read_sav(
        path,
        apply_value_formats=True,
        formats_as_category=False,
    )
    if frame.shape != (44, 107):
        raise ValueError(f"Wheelchair-foil temporal snapshot changed: {frame.shape}")
    normalized = normalize_frame(frame)
    if normalized["category"].value_counts().to_dict() != {"A": 22, "B": 22}:
        raise ValueError("Wheelchair-foil category counts changed")
    metadata_payload: dict[str, Any] = {
        "column_labels": metadata.column_names_to_labels,
        "value_labels": _string_keyed_labels(metadata.variable_value_labels),
        "source_columns": list(frame.columns),
    }
    return normalized, metadata_payload


def fetch_and_prepare(
    raw_path: Path,
    output_path: Path,
    codebook_path: Path,
    report_path: Path,
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
    frame, metadata = read_snapshot(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    write_json(codebook_path, metadata)
    report: dict[str, Any] = {
        "claim_boundary": (
            "Forty-four observed women's wheelchair-foil direct-elimination bouts from "
            "international competitions after 2022, evenly split between functional "
            "categories A and B. Rows describe bouts, not unique athletes, and contain no "
            "athlete identifiers, point-by-point actions, or source-video links."
        ),
        "source_url": ARTICLE_URL,
        "dataset_doi": DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "figshare_file_id": 59_495_930,
        "figshare_supplied_md5": EXPECTED_MD5,
        "raw_sha256": sha256_file(raw_path),
        "bouts": len(frame),
        "source_variables": 107,
        "category_counts": frame["category"].value_counts().sort_index().to_dict(),
        "phase_counts": frame["competition_phase"].value_counts().sort_index().to_dict(),
        "all_bouts_end_in_second_period": bool(
            frame["ending_period_code"].eq("PER2").all()
        ),
        "total_action_pause_identity_rows": int(
            frame["total_equals_action_plus_pause"].sum()
        ),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "codebook": str(codebook_path),
        "codebook_sha256": sha256_file(codebook_path),
    }
    write_json(report_path, report)
    return report
