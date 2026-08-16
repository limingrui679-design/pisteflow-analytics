"""CC BY cognitive, fitness, and blood tables for an 18-fencer cohort."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

ARTICLE_DOI = "https://doi.org/10.1371/journal.pone.0251907"
DEMOGRAPHICS_PAGE = (
    "https://plos.figshare.com/articles/dataset/"
    "Demographic_and_physical_fitness_data_/14733487"
)
OUTCOMES_PAGE = (
    "https://plos.figshare.com/articles/dataset/"
    "Behavioral_and_blood_parameters_data_/14733490"
)
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
FILES = {
    "demographics": {
        "url": "https://ndownloader.figshare.com/files/28303576",
        "filename": "pone.0251907.s001.xlsx",
        "bytes": 12_284,
        "sha256": "8c808d65d3216bebaa9641a3e6b690465d737cabe2bfde618aea0ffe6479b7e0",
    },
    "outcomes": {
        "url": "https://ndownloader.figshare.com/files/28303579",
        "filename": "pone.0251907.s002.xlsx",
        "bytes": 22_650,
        "sha256": "fdb84378b7870c5fead739cdae93924a87c9414197f7b91204b14329ff9776a1",
    },
}


def _safe_column(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return text or "unnamed"


def _fencer_rows(frame: pd.DataFrame) -> pd.DataFrame:
    group = frame.iloc[:, 0].astype("string")
    result = frame.loc[group.str.fullmatch(r"F\d+", na=False)].copy()
    if len(result) != 18 or result.iloc[:, 0].nunique() != 18:
        raise ValueError("PLOS cognition fencer cohort membership changed")
    return result


def _rename_numeric_columns(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    columns = ["source_group", *[f"{prefix}_{_safe_column(value)}" for value in frame.columns[1:]]]
    if len(set(columns)) != len(columns):
        raise ValueError(f"PLOS cognition {prefix} normalized columns collide")
    result.columns = columns
    for column in result.columns[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def normalize(
    demographics: pd.DataFrame,
    behavior: pd.DataFrame,
    blood: pd.DataFrame,
) -> pd.DataFrame:
    """Join the three fencer-only sheets and remove source codes/exact ages."""

    if demographics.shape != (60, 15):
        raise ValueError(f"PLOS cognition demographics snapshot changed: {demographics.shape}")
    if behavior.shape != (63, 35) or blood.shape != (57, 6):
        raise ValueError("PLOS cognition outcome sheet dimensions changed")
    demographic_fencers = _rename_numeric_columns(
        _fencer_rows(demographics), prefix="fitness"
    )
    behavior_fencers = _rename_numeric_columns(_fencer_rows(behavior), prefix="cognition")
    blood_fencers = _rename_numeric_columns(_fencer_rows(blood), prefix="blood")
    keys = [
        set(frame["source_group"])
        for frame in [demographic_fencers, behavior_fencers, blood_fencers]
    ]
    if not (keys[0] == keys[1] == keys[2]):
        raise ValueError("PLOS cognition fencer codes differ across sheets")
    result = demographic_fencers.merge(
        behavior_fencers, on="source_group", validate="one_to_one"
    ).merge(blood_fencers, on="source_group", validate="one_to_one")
    result = result.sort_values(
        "source_group",
        key=lambda series: series.str.extract(r"(\d+)")[0].astype(int),
    ).reset_index(drop=True)
    result.insert(0, "participant_key", [f"PFC-{index:02d}" for index in range(1, 19)])
    age = result.pop("fitness_age")
    result.insert(
        1,
        "age_band",
        pd.cut(
            age,
            bins=[0, 17, 20, 24, float("inf")],
            labels=["under_18", "18_20", "21_24", "25_plus"],
        ).astype("string"),
    )
    gender = result.pop("fitness_gender")
    result.insert(2, "sex", gender.map({0: "man", 1: "woman"}).astype("string"))
    result["upstream_post_ctsb_outlier"] = result["source_group"].eq("F2")
    result["weapon"] = "unknown"
    result["source_url"] = ARTICLE_DOI
    return result.drop(columns="source_group")


def fetch_and_prepare(
    cache_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire both supplements and build a privacy-minimized fencer table."""

    paths: dict[str, Path] = {}
    for key, specification in FILES.items():
        path = cache_directory / str(specification["filename"])
        if force:
            path.unlink(missing_ok=True)
        download_resumable(
            str(specification["url"]),
            path,
            user_agent=user_agent,
            expected_bytes=int(str(specification["bytes"])),
        )
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"PLOS fencer cognition checksum changed: {key}")
        paths[key] = path
    demographics = pd.read_excel(paths["demographics"], sheet_name="Sheet1")
    behavior = pd.read_excel(paths["outcomes"], sheet_name="Sheet1")
    blood = pd.read_excel(paths["outcomes"], sheet_name="Sheet2")
    processed = normalize(demographics, behavior, blood)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, index=False)

    summary: dict[str, Any] = {
        "source_url": ARTICLE_DOI,
        "demographics_page": DEMOGRAPHICS_PAGE,
        "outcomes_page": OUTCOMES_PAGE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "participants": len(processed),
        "weapon": "not_deposited",
        "by_sex": processed["sex"].value_counts().sort_index().to_dict(),
        "age_bands": processed["age_band"].value_counts().sort_index().to_dict(),
        "columns": len(processed.columns),
        "upstream_marked_outliers": int(processed["upstream_post_ctsb_outlier"].sum()),
        "claim_boundary": (
            "Eighteen fencers from a three-group cross-sectional exercise/cognition study. The "
            "joined CC BY table includes physical fitness, cognitive-task, and pre/post blood "
            "measures, but no fencing weapon, bout, skill label, athlete identity, or longitudinal "
            "training intervention. Exact study codes and ages are removed. The source styles F2 "
            "post-CTSB as an outlier; the value is retained with an explicit flag. Repeated "
            "pre/post columns are measures on the same 18 people, not additional participants."
        ),
        "assets": {
            key: {
                "url": specification["url"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for (key, specification), path in zip(FILES.items(), paths.values(), strict=True)
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    write_json(report_path, summary)
    return summary
