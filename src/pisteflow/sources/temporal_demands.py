"""Extract the published six-modality benchmark from a CC BY PLOS article."""

# ruff: noqa: RUF001 - exact published modality labels use typographic apostrophes

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

PMCID = "PMC10234521"
DOI = "https://doi.org/10.1371/journal.pone.0285033"
ARTICLE_URL = "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0285033"
XML_URL = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML"
LICENSE = "CC BY 4.0"
_MODALITIES = [
    ("Women’s épée", "epee", "women"),
    ("Men’s épée", "epee", "men"),
    ("Women’s foil", "foil", "women"),
    ("Men’s foil", "foil", "men"),
    ("Women’s sabre", "sabre", "women"),
    ("Men’s sabre", "sabre", "men"),
]
_METRICS = {
    "TBT (s)": "total_bout_time_s",
    "WT (s)": "work_time_s",
    "TPT (s)": "total_pause_time_s",
    "PRT (s)": "period_rest_time_s",
    "PTBP (s)": "pause_between_periods_s",
    "Halte (n)": "halts_n",
    "AAT (s)": "average_action_time_s",
    "AHT (s)": "average_halt_time_s",
}


def parse_table(xml: bytes) -> pd.DataFrame:
    """Parse Table 1 into six aggregate benchmark rows."""

    soup = BeautifulSoup(xml, "html.parser")
    target: Tag | None = None
    for table_wrap in soup.find_all("table-wrap"):
        caption = table_wrap.find("caption")
        if caption and "Temporal structure of the final phase" in caption.get_text(" ", strip=True):
            target = table_wrap
            break
    if target is None:
        raise ValueError("Temporal-demands article Table 1 is missing")
    raw_rows = [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)]
        for row in target.find_all("tr")
    ]
    header = raw_rows[0]
    if header[1:7] != [name for name, _weapon, _gender in _MODALITIES]:
        raise ValueError("Temporal-demands modality headers changed")
    rows: list[dict[str, Any]] = [
        {
            "modality": name,
            "weapon": weapon,
            "gender": gender,
            "event": "2014 World Fencing Championships final rounds",
            "stage": "direct_elimination_final_rounds",
            "source_doi": DOI,
            "source_license": LICENSE,
        }
        for name, weapon, gender in _MODALITIES
    ]
    for cells in raw_rows[1:]:
        label = cells[0]
        if label == "Bouts (n)":
            if len(cells) != 8:
                raise ValueError("Temporal-demands bout-count row changed")
            for index, row in enumerate(rows):
                row["bouts_n"] = int(cells[index + 1])
        elif label in _METRICS:
            if len(cells) != 22:
                raise ValueError(f"Temporal-demands metric row changed: {label}")
            for index, row in enumerate(rows):
                offset = 1 + index * 3
                if cells[offset + 1] != "±":
                    raise ValueError(f"Temporal-demands uncertainty token changed: {label}")
                key = _METRICS[label]
                row[f"{key}_mean"] = float(cells[offset])
                row[f"{key}_sd"] = float(cells[offset + 2])
        elif label == "W/R":
            if len(cells) != 8:
                raise ValueError("Temporal-demands work/rest row changed")
            for index, row in enumerate(rows):
                ratio = cells[index + 1]
                left, right = ratio.split(":", 1)
                if float(left) != 1.0:
                    raise ValueError("Temporal-demands work/rest numerator changed")
                row["rest_per_work_ratio"] = float(right.strip())
    frame = pd.DataFrame(rows)
    required = {"bouts_n", "average_action_time_s_mean", "rest_per_work_ratio"}
    if required.difference(frame.columns):
        raise ValueError("Temporal-demands benchmark is incomplete")
    return frame


def fetch_and_prepare(
    snapshot_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        snapshot_path.unlink(missing_ok=True)
    download_resumable(XML_URL, snapshot_path, user_agent=user_agent)
    frame = parse_table(snapshot_path.read_bytes())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Six published modality-level aggregates from video coding of 83 direct-"
            "elimination bouts at the 2014 World Championships. A separate CC BY public "
            "deposit supplies 83 bout-level and 200 period-level summaries, which PisteFlow "
            "processes independently. The article's 5,900 underlying Allez-to-Halte event "
            "records are still not deposited and are not reconstructed."
        ),
        "article_url": ARTICLE_URL,
        "source_doi": DOI,
        "pmcid": PMCID,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "snapshot_sha256": sha256_file(snapshot_path),
        "benchmark_rows": len(frame),
        "bouts_observed": int(frame["bouts_n"].sum()),
        "separate_deposit_bout_rows": 83,
        "separate_deposit_period_rows": 200,
        "underlying_event_records_reported_not_deposited": 5_900,
        "by_weapon_bouts": frame.groupby("weapon")["bouts_n"].sum().astype(int).to_dict(),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
