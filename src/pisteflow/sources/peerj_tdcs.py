"""CC BY crossover cognitive measurements from 16 professional female fencers."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

PMCID = "PMC11064870"
DOI = "https://doi.org/10.7717/peerj.17288"
ARTICLE_URL = f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/"
SUPPLEMENT_URL = (
    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/supplementaryFiles"
)
LICENSE = "CC BY 4.0"
RAR_FILENAME = "peerj-12-17288-s001.rar"
RAR_SHA256 = "4a98627a3e48e7abb485ad174ecd015d3637c82c9cadb5240fe537f103b414e0"
NT_FILENAME = "NT date.xlsx"
SST_FILENAME = "SST date.xlsx"
WORKBOOK_SHA256 = {
    NT_FILENAME: "c34bc9f35fc3c88ef93a0f110027e35db473030ee332285bc56240ef4e09aada",
    SST_FILENAME: "af3c4f385fc196b6b61052f5c705f6c2d4f35322e56bac6b9187db522f65389b",
}
_NT_COLUMNS = ["name", "type of stimulus", "test  modes", "pre", "post"]
_SST_COLUMNS = [
    "name",
    "type of stimulus",
    "Go_RT pre",
    "Go_RT post",
    "SSRT pre",
    "SSRT post",
]
_STIMULUS = {"Sham": "sham", "sham": "sham", "a-tDCS": "anodal_tdcs"}
_MODES = {"Core": "core", "Dynamic": "dynamic", "Selective": "selective"}


def _participant_key(raw_id: object) -> str:
    numeric = int(float(str(raw_id)))
    digest = hashlib.sha256(f"peerj-17288:{numeric}".encode()).hexdigest()[:20]
    return f"PTD-{digest}"


def normalize_attention(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != _NT_COLUMNS:
        raise ValueError("PeerJ attention workbook columns changed")
    if len(frame) != 96 or frame.duplicated().any():
        raise ValueError("PeerJ attention row layout changed")
    result = frame.rename(
        columns={
            "type of stimulus": "stimulation",
            "test  modes": "attention_mode",
            "pre": "pre_score",
            "post": "post_score",
        }
    ).copy()
    if set(result["name"].astype(int)) != set(range(1, 17)):
        raise ValueError("PeerJ attention participant IDs changed")
    result["stimulation"] = result["stimulation"].map(_STIMULUS)
    result["attention_mode"] = result["attention_mode"].map(_MODES)
    if result[["stimulation", "attention_mode"]].isna().any().any():
        raise ValueError("PeerJ attention condition labels changed")
    group_sizes = result.groupby(["name", "stimulation"], dropna=False).size()
    if len(group_sizes) != 32 or not group_sizes.eq(3).all():
        raise ValueError("PeerJ attention crossover structure changed")
    result.insert(0, "participant_key", result["name"].map(_participant_key))
    result = result.drop(columns="name")
    result["change_score"] = result["post_score"] - result["pre_score"]
    result["weapon"] = "unknown"
    result["population"] = "professional_female_fencers"
    result["source_doi"] = DOI
    return result


def normalize_stop_signal(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != _SST_COLUMNS:
        raise ValueError("PeerJ stop-signal workbook columns changed")
    if len(frame) != 32 or frame.duplicated().any():
        raise ValueError("PeerJ stop-signal row layout changed")
    result = frame.rename(
        columns={
            "type of stimulus": "stimulation",
            "Go_RT pre": "go_rt_pre_ms",
            "Go_RT post": "go_rt_post_ms",
            "SSRT pre": "stop_signal_rt_pre_ms",
            "SSRT post": "stop_signal_rt_post_ms",
        }
    ).copy()
    if set(result["name"].astype(int)) != set(range(1, 17)):
        raise ValueError("PeerJ stop-signal participant IDs changed")
    result["stimulation"] = result["stimulation"].map(_STIMULUS)
    if result["stimulation"].isna().any():
        raise ValueError("PeerJ stop-signal condition labels changed")
    group_sizes = result.groupby("name", dropna=False).size()
    if len(group_sizes) != 16 or not group_sizes.eq(2).all():
        raise ValueError("PeerJ stop-signal crossover structure changed")
    result.insert(0, "participant_key", result["name"].map(_participant_key))
    result = result.drop(columns="name")
    result["go_rt_change_ms"] = result["go_rt_post_ms"] - result["go_rt_pre_ms"]
    result["stop_signal_rt_change_ms"] = (
        result["stop_signal_rt_post_ms"] - result["stop_signal_rt_pre_ms"]
    )
    result["weapon"] = "unknown"
    result["population"] = "professional_female_fencers"
    result["source_doi"] = DOI
    return result


def _extract_workbooks(supplement_zip: Path, extraction_dir: Path) -> tuple[Path, Path, Path]:
    extraction_dir.mkdir(parents=True, exist_ok=True)
    rar_path = extraction_dir / RAR_FILENAME
    with zipfile.ZipFile(supplement_zip) as archive:
        names = archive.namelist()
        matches = [name for name in names if Path(name).name == RAR_FILENAME]
        if len(matches) != 1:
            raise ValueError(f"Expected one {RAR_FILENAME}, found {len(matches)}")
        rar_path.write_bytes(archive.read(matches[0]))
    if sha256_file(rar_path) != RAR_SHA256:
        raise ValueError("PeerJ embedded RAR checksum changed")
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise RuntimeError("bsdtar is required to extract the PeerJ RAR supplement")
    subprocess.run(
        [bsdtar, "-xf", str(rar_path), "-C", str(extraction_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    workbooks = {path.name: path for path in extraction_dir.rglob("*.xlsx")}
    if set(workbooks) != {NT_FILENAME, SST_FILENAME}:
        raise ValueError(f"PeerJ workbook members changed: {sorted(workbooks)}")
    for filename, expected in WORKBOOK_SHA256.items():
        if sha256_file(workbooks[filename]) != expected:
            raise ValueError(f"PeerJ workbook checksum changed: {filename}")
    return rar_path, workbooks[NT_FILENAME], workbooks[SST_FILENAME]


def _mean_changes(frame: pd.DataFrame, change_columns: list[str]) -> dict[str, Any]:
    grouped = frame.groupby("stimulation", sort=True)[change_columns].mean()
    return {
        stimulation: {column: float(value) for column, value in row.items()}
        for stimulation, row in grouped.iterrows()
    }


def fetch_and_prepare(
    supplement_zip: Path,
    extraction_dir: Path,
    attention_output: Path,
    stop_signal_output: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        supplement_zip.unlink(missing_ok=True)
    download_resumable(SUPPLEMENT_URL, supplement_zip, user_agent=user_agent)
    rar_path, nt_path, sst_path = _extract_workbooks(supplement_zip, extraction_dir)
    attention = normalize_attention(pd.read_excel(nt_path))
    stop_signal = normalize_stop_signal(pd.read_excel(sst_path))
    attention_output.parent.mkdir(parents=True, exist_ok=True)
    attention.to_parquet(attention_output, index=False)
    stop_signal.to_parquet(stop_signal_output, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "CC BY raw measurements from an article-described randomized crossover study of "
            "16 professional female fencers. The supplement provides stimulation condition, "
            "pre/post Go reaction time, stop-signal reaction time, and three attention modes, "
            "but no weapon, match result, intervention order, athlete name, or competition "
            "outcome. These acute laboratory cognitive changes are not evidence of improved "
            "fencing performance, clinical efficacy, or a training prescription."
        ),
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "pmcid": PMCID,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "supplement_url": SUPPLEMENT_URL,
        "supplement_zip_sha256": sha256_file(supplement_zip),
        "embedded_rar_sha256": sha256_file(rar_path),
        "workbook_sha256": {
            NT_FILENAME: sha256_file(nt_path),
            SST_FILENAME: sha256_file(sst_path),
        },
        "participants": int(attention["participant_key"].nunique()),
        "weapon": "unknown",
        "attention_condition_rows": len(attention),
        "stop_signal_condition_rows": len(stop_signal),
        "stimulation_conditions": ["anodal_tdcs", "sham"],
        "attention_modes": ["core", "dynamic", "selective"],
        "descriptive_mean_changes_not_causal_estimates": {
            "attention": _mean_changes(attention, ["change_score"]),
            "stop_signal": _mean_changes(
                stop_signal, ["go_rt_change_ms", "stop_signal_rt_change_ms"]
            ),
        },
        "attention_output": str(attention_output),
        "attention_output_sha256": sha256_file(attention_output),
        "stop_signal_output": str(stop_signal_output),
        "stop_signal_output_sha256": sha256_file(stop_signal_output),
        "repository_policy": "privacy-minimized row tables local-only; code and aggregates public",
    }
    write_json(report_path, summary)
    return summary
