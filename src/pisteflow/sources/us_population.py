"""Privacy-minimized audit of a public AskFRED-derived participation snapshot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.classification import infer_weapon_candidates, primary_weapon

SOURCE_URL = "https://github.com/wesleyyuan17/us-fencing-population-study"
LICENSE = "No repository license; local analysis only"
EXPECTED_COLUMNS = [
    "Date",
    "Location",
    "Event",
    "NumCompetitors",
    "EventRating",
    "Place",
    "Name",
    "ClubAbbr",
    "Club",
    "Rating",
    "RatingEarned",
    "StartDate",
    "EndDate",
]


def _stable_id(namespace: str, value: str) -> str:
    normalized = " ".join(value.split()).casefold()
    digest = hashlib.sha256(f"pisteflow:{namespace}:{normalized}".encode()).hexdigest()[:20]
    return f"{namespace}:{digest}"


def _gender(event: str) -> str:
    normalized = event.casefold().replace("\u2019", "'")
    if any(token in normalized for token in ("women's", "womens", "girls", "female")):
        return "female"
    if any(token in normalized for token in ("men's", "mens", "boys", "male")):
        return "male"
    return "mixed_or_unknown"


def prepare(source_directory: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    raw_path = source_directory / "data.csv"
    readme_path = source_directory / "README.md"
    processed_source_path = source_directory / "processed_data.csv"
    if not raw_path.is_file() or not readme_path.is_file():
        raise ValueError("US fencing population snapshot is incomplete")

    frame = pd.read_csv(raw_path, low_memory=False)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("US fencing population source schema changed")
    source_rows = len(frame)
    exact_duplicate_rows = int(frame.duplicated().sum())
    deduplicated = frame.drop_duplicates().copy()
    required = ["Name", "Place", "NumCompetitors", "StartDate", "EndDate", "Event"]
    complete = deduplicated[required].notna().all(axis=1)
    missing_required_rows = int((~complete).sum())
    eligible = deduplicated.loc[complete].copy()
    eligible["Place"] = pd.to_numeric(eligible["Place"], errors="raise").astype("int32")
    eligible["NumCompetitors"] = pd.to_numeric(
        eligible["NumCompetitors"], errors="raise"
    ).astype("int32")
    invalid_placement = eligible["Place"].lt(1) | eligible["Place"].gt(
        eligible["NumCompetitors"]
    )
    invalid_placement_rows = int(invalid_placement.sum())
    eligible = eligible.loc[~invalid_placement].copy()
    eligible["StartDate"] = pd.to_datetime(eligible["StartDate"], errors="raise")
    eligible["EndDate"] = pd.to_datetime(eligible["EndDate"], errors="raise")
    reversed_date = eligible["EndDate"].lt(eligible["StartDate"])
    reversed_date_rows = int(reversed_date.sum())
    eligible = eligible.loc[~reversed_date].copy()

    unique_labels = eligible["Event"].astype(str).drop_duplicates().tolist()
    label_to_weapon = {
        label: primary_weapon(infer_weapon_candidates(label)) for label in unique_labels
    }
    event_keys = (
        eligible["StartDate"].dt.strftime("%Y-%m-%d")
        + "|"
        + eligible["EndDate"].dt.strftime("%Y-%m-%d")
        + "|"
        + eligible["Location"].fillna("").astype(str)
        + "|"
        + eligible["Event"].astype(str)
    )
    names = eligible["Name"].astype(str)
    output = pd.DataFrame(
        {
            "entry_id": [f"askfred-entry:{index}" for index in eligible.index],
            "event_instance_id": event_keys.map(lambda value: _stable_id("askfred-event", value)),
            "athlete_id": names.map(lambda value: _stable_id("askfred-athlete", value)),
            "start_date": eligible["StartDate"],
            "end_date": eligible["EndDate"],
            "weapon": eligible["Event"].astype(str).map(
                lambda label: label_to_weapon[label].value
            ),
            "gender_label": eligible["Event"].astype(str).map(_gender),
            "place": eligible["Place"],
            "field_size_reported": eligible["NumCompetitors"],
            "event_rating": eligible["EventRating"].astype("string"),
            "rating_entered": eligible["Rating"].astype("string"),
            "rating_earned": eligible["RatingEarned"].astype("string"),
        }
    ).sort_values(["start_date", "event_instance_id", "place", "entry_id"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    weapon_rows = output["weapon"].value_counts().sort_index().to_dict()
    weapon_events = (
        output[["event_instance_id", "weapon"]]
        .drop_duplicates()["weapon"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "source_rows": source_rows,
        "exact_duplicate_rows_excluded": exact_duplicate_rows,
        "missing_required_rows_excluded": missing_required_rows,
        "invalid_place_gt_field_rows_excluded": invalid_placement_rows,
        "reversed_date_rows_excluded": reversed_date_rows,
        "eligible_name_free_entries": len(output),
        "unique_published_name_spellings_before_pseudonymization": int(names.nunique()),
        "event_instances": int(output["event_instance_id"].nunique()),
        "date_min": output["start_date"].min().date().isoformat(),
        "date_max": output["end_date"].max().date().isoformat(),
        "entries_by_weapon": {str(key): int(value) for key, value in weapon_rows.items()},
        "events_by_weapon": {str(key): int(value) for key, value in weapon_events.items()},
        "raw_sha256": sha256_file(raw_path),
        "readme_sha256": sha256_file(readme_path),
        "upstream_processed_sha256": (
            sha256_file(processed_source_path) if processed_source_path.is_file() else None
        ),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
        "claim_boundary": (
            "AskFRED-derived published event-result entries from 2002-2022, not a USA Fencing "
            "membership census. The source itself documents incomplete queries, senior/open-event "
            "selection, parsing errors, and post-2020 disruption. Names are pseudonymized; clubs, "
            "locations, and event titles are removed from the local analytic table. Exact repeats, "
            "missing athlete rows, impossible placements, and reversed date ranges are excluded. "
            "No repository license was found, so row-level data and derivatives remain local."
        ),
    }
    if sum(summary["entries_by_weapon"].values()) != len(output):
        raise ValueError("Weapon inference summary no longer covers every retained row")
    write_json(report_path, summary)
    return summary
