"""Local-only adapter for the public OSF sabre eye-tracking deposits.

The OSF node is publicly downloadable but declares no reuse license.  Raw and
processed rows therefore stay in ignored local directories.  Only hashes,
aggregate diagnostics, processing code, and explicit claim boundaries belong
in the repository.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://osf.io/ynkx3/"
PAPER_URL = "https://osf.io/download/jhcxe/"
REGISTERED_REPORT_URL = "https://osf.io/yzhkf/"


@dataclass(frozen=True)
class Asset:
    url: str
    filename: str
    expected_bytes: int
    sha256: str


ASSETS: dict[str, Asset] = {
    "main": Asset(
        "https://osf.io/download/fz6xt/",
        "main-study-eye-tracking.xlsx",
        38_420_360,
        "bf48ac71ab5a3fb0d9cbe0bce79ece39c7107bd15f03dfc5396e39720b182338",
    ),
    "main_aggregate": Asset(
        "https://osf.io/download/f5gjx/",
        "main-study-participant-aggregates.csv",
        571,
        "bcaa7eaf4a748b829e845ea5709dba65fd0b12ef493a887e87540644951392cb",
    ),
    "pilot": Asset(
        "https://osf.io/download/xvftk/",
        "pilot-study-eye-tracking.xlsx",
        43_978,
        "c4dc0fed037d52ada957327dda8189ee0ff7efca4c040fce0746a16b563d3268",
    ),
    "paper": Asset(
        PAPER_URL,
        "registered-report-manuscript.pdf",
        685_434,
        "2826efa09a7125be124bbf44ea5805d2a1c927ff3d7a3a0d3e9542e995acfd0b",
    ),
}

MAIN_COLUMNS = {
    "Source.Name",
    "Participant_ID",
    "Computer timestamp",
    "Time_trans",
    "Validity left",
    "Validity right",
    "Eye movement type",
    "Gaze event duration",
    "Eye movement type index",
    "win/lost",
    "offensive/defensive",
    "fixation location",
    "fixation count",
    "location count",
}
PILOT_COLUMNS = {
    "Participants",
    "Recording_timestamp",
    "Validity_ left_eye",
    "Validity_right_eye",
    "Eye_movement_type",
    "Gaze_duration",
    "Performance",
    "Phase",
    "Number_of_fixations",
    "Location_of_final_fixation",
    "Number_of_locations ",
}


def _source_local_key(namespace: str, value: object) -> str:
    normalized = " ".join(str(value).split()).casefold()
    digest = hashlib.sha256(f"osf-sabre-eye-v1|{namespace}|{normalized}".encode()).hexdigest()
    return f"OSE-{digest[:20]}"


def _binary_number(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin([0, 1]))


def _clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_main(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-sample, run-level event, and labelled-touch tables."""

    if frame.shape != (623_397, 14):
        raise ValueError(f"OSF sabre main-study snapshot changed: {frame.shape}")
    if set(frame.columns) != MAIN_COLUMNS:
        raise ValueError("OSF sabre main-study columns changed")
    if frame.duplicated().any():
        raise ValueError("OSF sabre main-study unexpectedly contains exact duplicates")
    if frame.duplicated(["Source.Name", "Computer timestamp"]).any():
        raise ValueError("OSF sabre main-study has duplicate recording timestamps")
    if frame["Participant_ID"].nunique() != 10 or frame["Source.Name"].nunique() != 18:
        raise ValueError("OSF sabre main-study participant or recording count changed")

    ordered = frame.copy()
    ordered.insert(0, "source_row", np.arange(1, len(ordered) + 1, dtype=np.int32))
    ordered["participant_key"] = ordered["Participant_ID"].map(
        lambda value: _source_local_key("main-participant", value)
    )
    ordered["recording_key"] = ordered["Source.Name"].map(
        lambda value: _source_local_key("main-recording", value)
    )
    ordered = ordered.sort_values(
        ["recording_key", "Computer timestamp", "source_row"],
        kind="stable",
    ).reset_index(drop=True)
    if not ordered.groupby("recording_key")["Computer timestamp"].is_monotonic_increasing.all():
        raise ValueError("OSF sabre recording timestamps are not monotonic")

    same_recording = ordered["recording_key"].eq(ordered["recording_key"].shift())
    same_type = ordered["Eye movement type"].eq(ordered["Eye movement type"].shift())
    same_index = ordered["Eye movement type index"].eq(
        ordered["Eye movement type index"].shift()
    )
    new_run = ~(same_recording & same_type & same_index)
    ordered["recording_event_run"] = new_run.groupby(ordered["recording_key"]).cumsum()

    ordered["left_eye_valid"] = ordered["Validity left"].eq("Valid")
    ordered["right_eye_valid"] = ordered["Validity right"].eq("Valid")
    ordered["both_eyes_valid"] = ordered["left_eye_valid"] & ordered["right_eye_valid"]
    ordered["either_eye_valid"] = ordered["left_eye_valid"] | ordered["right_eye_valid"]

    samples = ordered[
        [
            "source_row",
            "participant_key",
            "recording_key",
            "recording_event_run",
            "Computer timestamp",
            "Time_trans",
            "left_eye_valid",
            "right_eye_valid",
            "both_eyes_valid",
            "either_eye_valid",
            "Eye movement type",
            "Gaze event duration",
            "Eye movement type index",
        ]
    ].rename(
        columns={
            "Computer timestamp": "computer_timestamp_ms",
            "Time_trans": "recording_time_seconds",
            "Eye movement type": "eye_movement_type",
            "Gaze event duration": "gaze_event_duration_ms",
            "Eye movement type index": "upstream_movement_type_index",
        }
    )
    samples["source_url"] = SOURCE_URL

    events = (
        samples.groupby(["recording_key", "recording_event_run"], sort=False)
        .agg(
            participant_key=("participant_key", "first"),
            eye_movement_type=("eye_movement_type", "first"),
            upstream_movement_type_index=("upstream_movement_type_index", "first"),
            upstream_duration_ms=("gaze_event_duration_ms", "first"),
            first_timestamp_ms=("computer_timestamp_ms", "min"),
            last_timestamp_ms=("computer_timestamp_ms", "max"),
            samples=("source_row", "size"),
            both_eyes_valid_fraction=("both_eyes_valid", "mean"),
            either_eye_valid_fraction=("either_eye_valid", "mean"),
        )
        .reset_index()
    )
    events["source_url"] = SOURCE_URL

    labeled = ordered.loc[ordered["win/lost"].notna()].copy()
    if len(labeled) != 420:
        raise ValueError("OSF sabre main-study labelled-row count changed")
    won = _binary_number(labeled["win/lost"])
    offensive = _binary_number(labeled["offensive/defensive"])
    fixation_count = pd.to_numeric(labeled["fixation count"], errors="coerce")
    location_count = pd.to_numeric(labeled["location count"], errors="coerce")
    labeled["outcome_eligible"] = won.notna()
    labeled["phase_eligible"] = offensive.notna()
    labeled["supervised_touch_eligible"] = won.notna() & offensive.notna()
    labeled["won_touch"] = won.astype("Int8")
    labeled["offensive_phase"] = offensive.astype("Int8")
    labeled["fixation_count_numeric"] = fixation_count.astype("Int8")
    labeled["location_count_numeric"] = location_count.astype("Int8")
    labeled["fixation_location_raw"] = _clean_string(labeled["fixation location"])
    labeled["touch_index_within_participant"] = (
        labeled.groupby("participant_key", sort=False).cumcount() + 1
    ).astype("int16")
    touches = labeled[
        [
            "source_row",
            "participant_key",
            "recording_key",
            "touch_index_within_participant",
            "Computer timestamp",
            "Time_trans",
            "Validity left",
            "Validity right",
            "Gaze event duration",
            "won_touch",
            "offensive_phase",
            "fixation_count_numeric",
            "location_count_numeric",
            "fixation_location_raw",
            "outcome_eligible",
            "phase_eligible",
            "supervised_touch_eligible",
        ]
    ].rename(
        columns={
            "Computer timestamp": "computer_timestamp_ms",
            "Time_trans": "recording_time_seconds",
            "Validity left": "left_eye_validity",
            "Validity right": "right_eye_validity",
            "Gaze event duration": "quiet_eye_duration_ms",
        }
    )
    touches["weapon"] = "sabre"
    touches["source_url"] = SOURCE_URL
    return samples, events, touches


def normalize_pilot(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 14-participant pilot without asserting cross-study independence."""

    if frame.shape != (575, 11):
        raise ValueError(f"OSF sabre pilot snapshot changed: {frame.shape}")
    if set(frame.columns) != PILOT_COLUMNS:
        raise ValueError("OSF sabre pilot columns changed")
    if frame.duplicated().any() or frame.duplicated(
        ["Participants", "Recording_timestamp"]
    ).any():
        raise ValueError("OSF sabre pilot contains duplicate touch rows")
    if frame["Participants"].nunique() != 14:
        raise ValueError("OSF sabre pilot participant count changed")
    if not set(frame["Performance"]).issubset({0, 1}) or not set(frame["Phase"]).issubset(
        {0, 1}
    ):
        raise ValueError("OSF sabre pilot outcome or phase labels changed")

    result = pd.DataFrame(
        {
            "source_row": np.arange(1, len(frame) + 1, dtype=np.int16),
            "participant_key": frame["Participants"].map(
                lambda value: _source_local_key("pilot-participant", value)
            ),
            "computer_timestamp_ms": frame["Recording_timestamp"],
            "left_eye_validity": _clean_string(frame["Validity_ left_eye"]),
            "right_eye_validity": _clean_string(frame["Validity_right_eye"]),
            "quiet_eye_duration_ms": frame["Gaze_duration"],
            "won_touch": frame["Performance"].astype("int8"),
            "offensive_phase": frame["Phase"].astype("int8"),
            "fixation_count_numeric": frame["Number_of_fixations"].astype("int8"),
            "location_count_numeric": pd.to_numeric(
                frame["Number_of_locations "], errors="coerce"
            ).astype("Int8"),
            "fixation_location_code": frame["Location_of_final_fixation"].astype("int8"),
        }
    )
    result["touch_index_within_participant"] = (
        result.groupby("participant_key", sort=False).cumcount() + 1
    ).astype("int16")
    result["weapon"] = "sabre"
    result["cross_study_identity_unresolved"] = True
    result["source_url"] = SOURCE_URL
    return result


def _validate_main_aggregates(touches: pd.DataFrame, path: Path) -> dict[str, Any]:
    aggregate = pd.read_csv(path, encoding="utf-8-sig")
    if aggregate.shape != (10, 7):
        raise ValueError(f"OSF sabre aggregate snapshot changed: {aggregate.shape}")
    participant_map = {
        row.Participant: _source_local_key("main-participant", row.Participant)
        for row in aggregate.itertuples(index=False)
    }
    aggregate["participant_key"] = aggregate["Participant"].map(participant_map)
    eligible = touches.loc[touches["supervised_touch_eligible"]].copy()
    checks = {
        "QE_Won": eligible.loc[eligible["won_touch"].eq(1)]
        .groupby("participant_key")["quiet_eye_duration_ms"]
        .mean(),
        "QE_Lost": eligible.loc[eligible["won_touch"].eq(0)]
        .groupby("participant_key")["quiet_eye_duration_ms"]
        .mean(),
        "No_fix_off": eligible.loc[eligible["offensive_phase"].eq(1)]
        .groupby("participant_key")["fixation_count_numeric"]
        .mean(),
        "No_fix_def": eligible.loc[eligible["offensive_phase"].eq(0)]
        .groupby("participant_key")["fixation_count_numeric"]
        .mean(),
        "No_loc_off": eligible.loc[eligible["offensive_phase"].eq(1)]
        .groupby("participant_key")["location_count_numeric"]
        .mean(),
        "No_loc_def": eligible.loc[eligible["offensive_phase"].eq(0)]
        .groupby("participant_key")["location_count_numeric"]
        .mean(),
    }
    expected_mismatches = {"No_fix_def": 6, "No_loc_def": 4}
    audit: dict[str, Any] = {}
    for column, computed in checks.items():
        expected = aggregate.set_index("participant_key")[column]
        differences = (computed.sort_index() - expected.sort_index()).abs()
        mismatches = int(differences.gt(0.011).sum())
        if mismatches != expected_mismatches.get(column, 0):
            raise ValueError(f"OSF sabre participant aggregate mismatch profile changed: {column}")
        audit[column] = {
            "participants_outside_0_011_tolerance": mismatches,
            "maximum_absolute_difference": float(differences.max()),
        }
    return audit


def fetch_and_prepare(
    cache_directory: Path,
    processed_directory: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the pinned deposit and write local-only privacy-minimized tables."""

    paths: dict[str, Path] = {}
    for key, asset in ASSETS.items():
        path = cache_directory / asset.filename
        if force:
            path.unlink(missing_ok=True)
        download_resumable(
            asset.url,
            path,
            user_agent=user_agent,
            expected_bytes=asset.expected_bytes,
        )
        if sha256_file(path) != asset.sha256:
            raise ValueError(f"OSF sabre eye-tracking checksum changed: {key}")
        paths[key] = path

    main_frame = pd.read_excel(paths["main"], sheet_name="raw data")
    samples, events, touches = normalize_main(main_frame)
    aggregate_audit = _validate_main_aggregates(touches, paths["main_aggregate"])
    pilot_frame = pd.read_excel(paths["pilot"], sheet_name="Sheet1")
    pilot = normalize_pilot(pilot_frame)

    processed_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main_samples": processed_directory / "main-gaze-samples.parquet",
        "main_events": processed_directory / "main-gaze-events.parquet",
        "main_touches": processed_directory / "main-labelled-touches.parquet",
        "pilot_touches": processed_directory / "pilot-labelled-touches.parquet",
    }
    samples.to_parquet(outputs["main_samples"], index=False)
    events.to_parquet(outputs["main_events"], index=False)
    touches.to_parquet(outputs["main_touches"], index=False)
    pilot.to_parquet(outputs["pilot_touches"], index=False)

    eligible = touches.loc[touches["supervised_touch_eligible"]]
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "registered_report_url": REGISTERED_REPORT_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "access": "public_download",
        "license": "not_declared_on_osf_node_or_deposited_files",
        "repository_policy": "raw_and_processed_rows_local_only",
        "claim_boundary": (
            "Repeated eye-tracker samples and coded touches from sabre assaults, not independent "
            "athletes, weapon-comparative evidence, competition video, or causal training effects. "
            "The main study has ten included national-level sabre fencers and 379 supervised "
            "touch rows; 623,397 sensor rows must be grouped by participant and recording. The "
            "pilot has 14 coded participant labels and 575 touches, but participant overlap with "
            "the main study is unpublished, so the two studies cannot support person-independent "
            "joint evaluation. The deposited defensive-phase participant aggregates do not fully "
            "recompute from the touch rows and are reported as discrepancies. No reuse license is "
            "declared, and the paper reports minors may participate, so all row tables remain "
            "local-only."
        ),
        "main": {
            "sensor_rows": len(samples),
            "recordings": int(samples["recording_key"].nunique()),
            "participants": int(samples["participant_key"].nunique()),
            "run_level_gaze_events": len(events),
            "both_eyes_valid_rows": int(samples["both_eyes_valid"].sum()),
            "either_eye_valid_rows": int(samples["either_eye_valid"].sum()),
            "labelled_rows": len(touches),
            "supervised_touch_rows": len(eligible),
            "won_touches": int(eligible["won_touch"].eq(1).sum()),
            "lost_touches": int(eligible["won_touch"].eq(0).sum()),
            "offensive_touches": int(eligible["offensive_phase"].eq(1).sum()),
            "defensive_touches": int(eligible["offensive_phase"].eq(0).sum()),
            "excluded_placeholder_rows": int((~touches["supervised_touch_eligible"]).sum()),
            "participant_aggregate_reconciliation": aggregate_audit,
            "paper_sample": {
                "athletes": 10,
                "women": 5,
                "mean_age_years": 16.9,
                "mean_fencing_experience_years": 8.8,
                "assault_minutes_per_athlete": 10,
                "included_gaze_coverage_range_percent": [74, 97],
                "mean_gaze_coverage_percent": 86.66,
                "phase_coding_disagreement_before_consensus_percent": 3.5,
                "quiet_eye_recoding_disagreement_percent": 13,
                "other_gaze_variable_disagreement_upper_bound_percent": 10,
            },
        },
        "pilot": {
            "touch_rows": len(pilot),
            "participant_labels": int(pilot["participant_key"].nunique()),
            "won_touches": int(pilot["won_touch"].eq(1).sum()),
            "lost_touches": int(pilot["won_touch"].eq(0).sum()),
            "cross_study_identity_overlap": "unresolved",
        },
        "assets": {
            key: {
                "url": asset.url,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for (key, asset), path in zip(ASSETS.items(), paths.values(), strict=True)
        },
        "outputs": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in outputs.items()
        },
    }
    write_json(report_path, summary)
    return summary
