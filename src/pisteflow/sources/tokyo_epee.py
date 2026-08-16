"""Prepare manually coded touch events from ten Tokyo 2020 Olympic epee bouts."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = "https://github.com/OscarFawcett/Epee-Fencing-Tokyo"
PINNED_COMMIT = "a6872cc45bb4b715e0f86406bfae165de463794c"
WORKBOOK_URL = (
    "https://raw.githubusercontent.com/OscarFawcett/Epee-Fencing-Tokyo/"
    f"{PINNED_COMMIT}/FencingData.xlsx"
)
EXPECTED_BYTES = 20_742
EXPECTED_SHA256 = "6fd6e342bf1a0a44434294adfe3b56221c83a1848fbddc31b55f8f9cd42d3b76"
LICENSE = "MIT"
_COLUMNS = [
    "match",
    "player1",
    "player2",
    "league",
    "body",
    "blade_contacts",
    "double",
    "area",
    "time",
    "period",
    "scorer",
]


def _normalized_name(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _athlete_id(value: object) -> str:
    normalized = _normalized_name(value)
    return "tokyo-epee-athlete-" + sha256(f"tokyo-epee:{normalized}".encode()).hexdigest()[:16]


def _bout_id(row: pd.Series) -> str:
    key = "|".join(
        [
            _normalized_name(row["match"]),
            _normalized_name(row["league"]),
            _normalized_name(row["player1"]),
            _normalized_name(row["player2"]),
        ]
    )
    return "tokyo-epee-bout-" + sha256(key.encode()).hexdigest()[:16]


def _reported_side(row: pd.Series) -> str | None:
    if pd.isna(row["scorer"]):
        return None
    scorer = _normalized_name(row["scorer"])
    matches = []
    for side, column in [("player_1", "player1"), ("player_2", "player2")]:
        player = _normalized_name(row[column])
        if scorer == player or scorer in player.split():
            matches.append(side)
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve a Tokyo epee scorer alias: {row['scorer']}")
    return matches[0]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove direct names while preserving every coded event and source order."""

    if list(frame.columns) != _COLUMNS:
        raise ValueError("Tokyo epee workbook columns changed")
    required = [
        "match",
        "player1",
        "player2",
        "league",
        "blade_contacts",
        "double",
        "area",
        "period",
    ]
    if frame[required].isna().any().any():
        raise ValueError("Tokyo epee workbook has missing required event fields")
    if not set(frame["league"].unique()).issubset({"M", "W"}):
        raise ValueError("Tokyo epee sex codes changed")

    result = pd.DataFrame(index=frame.index)
    result["source_row"] = pd.Series(range(2, len(frame) + 2), index=frame.index, dtype="int32")
    result["bout_id"] = frame.apply(_bout_id, axis=1)
    result["player_1_id"] = frame["player1"].map(_athlete_id)
    result["player_2_id"] = frame["player2"].map(_athlete_id)
    result["match_stage"] = frame["match"].astype("string").str.strip()
    result["sex"] = frame["league"].map({"M": "men", "W": "women"}).astype("string")
    result["weapon"] = "epee"
    result["body_target"] = frame["body"].astype("string").str.strip().fillna("unknown")
    result["blade_contacts"] = pd.to_numeric(frame["blade_contacts"], errors="raise").astype(
        "int16"
    )
    result["double_touch"] = frame["double"].astype(bool)
    result["piste_area"] = frame["area"].astype("string").str.strip()
    result["time_remaining_seconds"] = pd.to_numeric(frame["time"], errors="coerce")
    result["time_missing"] = result["time_remaining_seconds"].isna()
    result["period"] = pd.to_numeric(frame["period"], errors="raise").astype("int8")
    result["reported_scorer_side"] = frame.apply(_reported_side, axis=1).astype("string")
    result["event_type"] = "uncoded_or_no_touch_exchange"
    result.loc[result["reported_scorer_side"].notna(), "event_type"] = "coded_touch_event"
    result.loc[
        result["double_touch"] & result["reported_scorer_side"].isna(), "event_type"
    ] = "coded_double_touch_without_single_scorer"
    result["awarded_side"] = result["reported_scorer_side"]
    result.loc[result["double_touch"], "awarded_side"] = "both"
    result["competition"] = "Tokyo 2020 Olympic Games"
    result["source_url"] = SOURCE_PAGE
    result["source_license"] = LICENSE
    result["event_id"] = [f"tokyo-epee-event-{number:03d}" for number in range(1, len(result) + 1)]
    return result.reset_index(drop=True)


def prepare(workbook_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if sha256_file(workbook_path) != EXPECTED_SHA256:
        raise ValueError("Tokyo epee workbook checksum changed")
    source = pd.read_excel(workbook_path, sheet_name="Complete Dataset")
    result = normalize(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    scorer_rows = result["reported_scorer_side"].notna()
    touch_rows = scorer_rows | result["double_touch"]
    awarded_touches = int(
        (touch_rows & ~result["double_touch"]).sum() + 2 * result["double_touch"].sum()
    )
    summary: dict[str, Any] = {
        "claim_boundary": (
            "A manually coded MIT-licensed table of 225 ordered rows from ten Tokyo 2020 "
            "Olympic epee bouts. Twenty rows have no scorer: one is a double touch and the "
            "remaining 19 are retained as uncoded or no-touch exchanges. For double touches, "
            "the source scorer field is preserved only as a reported side while the derived "
            "awarded side is both. These data do not cover "
            "all Olympic epee bouts, foil, sabre, or automatic sensor measurements."
        ),
        "source_url": SOURCE_PAGE,
        "workbook_url": WORKBOOK_URL,
        "pinned_commit": PINNED_COMMIT,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "workbook_sha256": sha256_file(workbook_path),
        "output_sha256": sha256_file(output_path),
        "rows": len(result),
        "bouts": int(result["bout_id"].nunique()),
        "unique_pseudonymous_athletes": len(
            set(result["player_1_id"]) | set(result["player_2_id"])
        ),
        "rows_with_reported_scorer": int(scorer_rows.sum()),
        "derived_touch_event_rows": int(touch_rows.sum()),
        "uncoded_or_no_touch_exchange_rows": int((~touch_rows).sum()),
        "double_touch_rows_without_single_scorer": int(
            (result["double_touch"] & ~scorer_rows).sum()
        ),
        "double_touch_rows": int(result["double_touch"].sum()),
        "derived_awarded_touches": awarded_touches,
        "rows_with_missing_time": int(result["time_missing"].sum()),
        "by_sex": result.groupby("sex").size().astype(int).to_dict(),
        "by_match_stage": result.groupby("match_stage").size().astype(int).to_dict(),
        "body_target_counts": result.groupby("body_target").size().astype(int).to_dict(),
        "piste_area_counts": result.groupby("piste_area").size().astype(int).to_dict(),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    workbook_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        workbook_path.unlink(missing_ok=True)
    download_resumable(
        WORKBOOK_URL,
        workbook_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    return prepare(workbook_path, output_path, report_path)
