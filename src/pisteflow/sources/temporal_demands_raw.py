"""CC BY bout- and period-level temporal data for elite three-weapon fencing."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable
from pisteflow.sources.temporal_demands import XML_URL, parse_table

DATASET_DOI = "https://doi.org/10.18745/uhra.herts.ac.uk.00015826"
ARTICLE_DOI = "https://doi.org/10.1371/journal.pone.0285033"
PAGE_URL = "https://uhra.herts.ac.uk/id/eprint/15826/"
ARCHIVE_URL = (
    "https://uhra.herts.ac.uk/id/eprint/15826/2/"
    "UH%20Exchange%20File-SyA6TG2uBfJqzgS7%20%281%29.zip"
)
README_URL = "https://uhra.herts.ac.uk/id/eprint/15826/1/README-V2.txt"
LICENSE = "CC BY (version not stated in deposited README)"
ARCHIVE_BYTES = 10_466
README_BYTES = 10_721

_CODE = {
    1: ("epee", "men", "Men's Epee.csv"),
    2: ("epee", "women", "Women's Epee.csv"),
    3: ("sabre", "men", "Men's Sabre.csv"),
    4: ("sabre", "women", "Women's Sabre.csv"),
    5: ("foil", "men", "Men's Foil.csv"),
    6: ("foil", "women", "Women's Foil.csv"),
}
_TIME_COLUMNS = {
    "TBT (s)": "total_bout_time_s",
    "WT (s)": "work_time_s",
    "TPT (s)": "total_pause_time_s",
    "PRT (s)": "period_rest_time_s",
    "PTBP (s)": "pause_between_periods_s",
    "AAT (s)": "average_action_time_s",
    "AHT (s)": "average_halt_time_s",
}
_BOUT_COLUMNS = [
    "Bout",
    "Weapon",
    "SEX",
    "Cod_Weapon",
    "TBT (s)",
    "WT (s)",
    "TPT (s)",
    "PRT (s)",
    "PTBP (s)",
    "Halte (n)",
    "AAT (s)",
    "AHT (s)",
]
_PERIOD_COLUMNS = [
    "Bout",
    "Period",
    "TBT (s)",
    "WT (s)",
    "TPT (s)",
    "PRT (s)",
    "PTBP (s)",
    "Halte (n)",
    "AAT (s)",
    "AHT (s)",
]


def _attach_times(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source, output in _TIME_COLUMNS.items():
        result[f"{output}_raw"] = pd.to_numeric(result[source], errors="coerce")
        result[output] = result[f"{output}_raw"] / 1_000
    result["pause_between_periods_missing_upstream"] = result["PTBP (s)"].isna()
    result["pause_between_periods_s_derived_zero"] = result[
        "pause_between_periods_s"
    ].fillna(0)
    result["pause_identity_holds"] = np.isclose(
        result["total_pause_time_s"],
        result["period_rest_time_s"] + result["pause_between_periods_s_derived_zero"],
        rtol=0,
        atol=1e-9,
    )
    result["time_unit_interpretation"] = "milliseconds_in_source_despite_seconds_header"
    return result.drop(columns=list(_TIME_COLUMNS))


def normalize_bouts(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 83 one-row-per-bout records."""

    if list(frame.columns) != _BOUT_COLUMNS:
        raise ValueError("Elite-temporal total-data columns changed")
    result = frame.copy()
    result.insert(0, "bout_code", result["Bout"].map(lambda value: f"TDE-{int(value):03d}"))
    result = result.rename(columns={"Bout": "upstream_bout_id", "Halte (n)": "halts_n"})
    metadata = result["Cod_Weapon"].map(_CODE)
    if metadata.isna().any():
        raise ValueError("Elite-temporal modality code changed")
    result[["weapon", "gender", "expected_period_file"]] = pd.DataFrame(
        metadata.tolist(), index=result.index
    )
    expected_weapon = result["Cod_Weapon"].map({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3})
    expected_sex = result["Cod_Weapon"].map({1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2})
    if not result["Weapon"].eq(expected_weapon).all() or not result["SEX"].eq(expected_sex).all():
        raise ValueError("Elite-temporal modality fields disagree")
    result = _attach_times(result)
    result["source_dataset_doi"] = DATASET_DOI
    return result.drop(columns=["Weapon", "SEX", "Cod_Weapon"])


def normalize_periods(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Normalize all six modality-specific period tables."""

    by_filename = {filename: (weapon, gender) for weapon, gender, filename in _CODE.values()}
    rows: list[pd.DataFrame] = []
    for filename, (weapon, gender) in by_filename.items():
        frame = frames[filename]
        if list(frame.columns) != _PERIOD_COLUMNS:
            raise ValueError(f"Elite-temporal period columns changed: {filename}")
        result = frame.rename(
            columns={
                "Bout": "upstream_bout_id",
                "Period": "period_code",
                "Halte (n)": "halts_n",
            }
        ).copy()
        result.insert(
            0,
            "bout_code",
            result["upstream_bout_id"].map(lambda value: f"TDE-{int(value):03d}"),
        )
        result["weapon"] = weapon
        result["gender"] = gender
        result["period"] = result["period_code"].map(
            {
                1: "period_1",
                2: "period_2",
                3: "period_3",
                4: "priority_minute",
                5: "non_combativity_extra_period",
            }
        )
        if result["period"].isna().any():
            raise ValueError(f"Elite-temporal period code changed: {filename}")
        result = _attach_times(result)
        result["source_period_file"] = filename
        result["source_dataset_doi"] = DATASET_DOI
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def read_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        basenames = {Path(name).name: name for name in members}
        expected = {"Total Data.csv"} | {value[2] for value in _CODE.values()}
        if set(basenames) != expected:
            raise ValueError(f"Elite-temporal archive members changed: {members}")
        total = pd.read_csv(io.BytesIO(archive.read(basenames["Total Data.csv"])))
        period_frames = {
            filename: pd.read_csv(io.BytesIO(archive.read(basenames[filename])))
            for filename in expected
            if filename != "Total Data.csv"
        }
    bouts = normalize_bouts(total)
    periods = normalize_periods(period_frames)
    if len(bouts) != 83 or len(periods) != 200:
        raise ValueError("Elite-temporal deposited row counts changed")
    if set(periods["bout_code"]) != set(bouts["bout_code"]):
        raise ValueError("Elite-temporal bout and period files disagree")
    return bouts, periods, members


def _published_mean_comparison(bouts: pd.DataFrame, xml_path: Path) -> dict[str, Any]:
    published = parse_table(xml_path.read_bytes())
    rows: list[dict[str, Any]] = []
    for _, benchmark in published.iterrows():
        subset = bouts.loc[
            bouts["weapon"].eq(benchmark["weapon"])
            & bouts["gender"].eq(benchmark["gender"])
        ]
        for _source, metric in _TIME_COLUMNS.items():
            raw_metric = (
                "pause_between_periods_s_derived_zero"
                if metric == "pause_between_periods_s"
                else metric
            )
            observed = float(subset[raw_metric].mean())
            reported = float(benchmark[f"{metric}_mean"])
            rows.append(
                {
                    "weapon": benchmark["weapon"],
                    "gender": benchmark["gender"],
                    "metric": metric,
                    "deposit_mean_s": observed,
                    "published_mean_s": reported,
                    "difference_s": observed - reported,
                }
            )
    return {
        "rows": rows,
        "max_absolute_difference_s": max(abs(row["difference_s"]) for row in rows),
    }


def fetch_and_prepare(
    archive_path: Path,
    readme_path: Path,
    article_xml_path: Path,
    processed_dir: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        archive_path.unlink(missing_ok=True)
        readme_path.unlink(missing_ok=True)
        article_xml_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=ARCHIVE_BYTES,
    )
    download_resumable(
        README_URL,
        readme_path,
        user_agent=user_agent,
        expected_bytes=README_BYTES,
    )
    download_resumable(XML_URL, article_xml_path, user_agent=user_agent)
    if "Licenses/restrictions placed on the data: CC-BY" not in readme_path.read_text(
        encoding="utf-8-sig", errors="replace"
    ):
        raise ValueError("Elite-temporal deposited license statement changed")
    bouts, periods, members = read_archive(archive_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bouts": processed_dir / "bouts.parquet",
        "periods": processed_dir / "periods.parquet",
    }
    bouts.to_parquet(outputs["bouts"], index=False)
    periods.to_parquet(outputs["periods"], index=False)
    comparison = _published_mean_comparison(bouts, article_xml_path)
    report: dict[str, Any] = {
        "claim_boundary": (
            "Bout- and period-level video-coding summaries for 83 elite direct-elimination "
            "bouts at the 2014 World Championships. The deposit contains 83 bout rows and "
            "200 period rows, not the article's 5,900 underlying Allez-to-Halte coded events. "
            "Headers say seconds, but values are milliseconds: dividing by 1,000 reproduces "
            "published modality means apart from small source/publication differences. Missing "
            "PTBP cells are retained and a separate zero-filled field is derived only where "
            "the TPT = PRT + PTBP identity proves zero."
        ),
        "source_url": PAGE_URL,
        "dataset_doi": DATASET_DOI,
        "article_doi": ARTICLE_DOI,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "readme_sha256": sha256_file(readme_path),
        "article_xml_sha256": sha256_file(article_xml_path),
        "archive_members": members,
        "bout_rows": len(bouts),
        "period_rows": len(periods),
        "underlying_allez_halte_records_reported_not_deposited": 5_900,
        "weapon_bouts": bouts["weapon"].value_counts().sort_index().to_dict(),
        "gender_bouts": bouts["gender"].value_counts().sort_index().to_dict(),
        "modality_bouts": (
            bouts.groupby(["gender", "weapon"]).size().rename("bouts").reset_index().to_dict(
                orient="records"
            )
        ),
        "period_code_rows": periods["period"].value_counts().sort_index().to_dict(),
        "bout_pause_identity_rows": int(bouts["pause_identity_holds"].sum()),
        "period_pause_identity_rows": int(periods["pause_identity_holds"].sum()),
        "missing_bout_ptbp_cells": int(bouts["pause_between_periods_missing_upstream"].sum()),
        "published_mean_comparison": comparison,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "output_sha256": {key: sha256_file(path) for key, path in outputs.items()},
    }
    write_json(report_path, report)
    return report
