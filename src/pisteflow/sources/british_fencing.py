"""Privacy-minimized preparation of British Fencing's public performance tables."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = "https://www.britishfencing.com/international-results-raw-data-archives/"
LICENSE = "No explicit reusable data license; British Fencing site footer states copyright"

# The first link for each weapon is the alphabetical-order view. The second official
# link is only a different ordering of the same rows and is deliberately not duplicated.
SHEETS: dict[str, dict[str, str]] = {
    "2021-22": {
        "foil": "8da0cacc9b6949eaa30867a8833f888e",
        "sabre": "09a7539059b24d5da097ea3e241385cb",
        "epee": "24b4fc9ce7c148e9a8baa1622069197d",
    },
    "2022-23": {
        "foil": "0d7a7a13de9d4064a39442b29b335824",
        "sabre": "7d094ece34264523a8b2673d515ed316",
        "epee": "7639156c86034a1590889fff09971b59",
    },
    "2023-24": {
        "foil": "e57d9574898e42e7bfa82529f18b65b8",
        "sabre": "ceafae1986a043fc8f37fe0e596f715a",
        "epee": "b9cea9d5dbaa4ccb8c48488954fcdd22",
    },
}

_COLUMN_MAP = {
    "Names": "source_name",
    "Weapon Category": "weapon_category",
    "Competition Date": "competition_date",
    "Season": "source_season",
    "Ranking Status": "ranking_status",
    "Event": "event",
    "Final Placing": "final_placing",
    "Starting Fencers": "starting_fencers",
    "% finish": "finish_pct",
    "Rank": "source_rank",
    "No of Poule Fights": "pool_bouts",
    "Poule V": "pool_victories",
    "% Poules": "pool_win_pct",
    "Poules HS": "pool_hits_scored",
    "Poules HR": "pool_hits_received",
    "Ind": "pool_indicator",
    "No of 1 point losses": "one_point_losses",
    "Promoted to DE": "promoted_to_de",
    "Received a Bye": "received_bye",
    "Number of DE Fights": "de_bouts",
    "DE Fights": "de_bouts",
    "Number of DE Victories": "de_victories",
    "DE Victories": "de_victories",
    "% of Wins": "de_win_pct",
    "Final DE Score": "final_de_score",
}

_NUMERIC_COLUMNS = [
    "final_placing",
    "starting_fencers",
    "source_rank",
    "pool_bouts",
    "pool_victories",
    "pool_hits_scored",
    "pool_hits_received",
    "pool_indicator",
    "one_point_losses",
    "de_bouts",
    "de_victories",
]
_PERCENT_COLUMNS = ["finish_pct", "pool_win_pct", "de_win_pct"]


def _cell_value(cell: Tag) -> str:
    checkbox = cell.find("img", alt=True)
    if checkbox is not None and checkbox.get("alt") in {"true", "false"}:
        return str(checkbox["alt"])
    return cell.get_text(" ", strip=True)


def _athlete_id(name: str) -> str:
    normalized = " ".join(name.casefold().split())
    return "bf-" + sha256(f"british-fencing:{normalized}".encode()).hexdigest()[:16]


def parse_sheet(html: str, *, season: str, weapon: str) -> pd.DataFrame:
    """Parse one public Smartsheet view without retaining names."""

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="clsGridGantt")
    if not isinstance(table, Tag):
        raise ValueError("British Fencing Smartsheet table is missing")
    table_rows = table.find_all("tr")
    if not table_rows:
        raise ValueError("British Fencing Smartsheet table is empty")
    header_cells = table_rows[0].find_all("td", recursive=False)
    headers = [cell.get_text(" ", strip=True) for cell in header_cells][1:]
    unknown_headers = set(headers).difference(_COLUMN_MAP)
    if unknown_headers:
        raise ValueError(f"Unexpected British Fencing columns: {sorted(unknown_headers)}")

    records: list[dict[str, str]] = []
    for row in table_rows[1:]:
        cells = row.find_all("td", recursive=False)
        if not cells or "clsNumberCell" not in (cells[0].get("class") or []):
            continue
        values = [_cell_value(cell) for cell in cells[1:]]
        if len(values) != len(headers):
            raise ValueError("British Fencing row width does not match its header")
        records.append(dict(zip(headers, values, strict=True)))
    if not records:
        raise ValueError("British Fencing Smartsheet has no performance rows")

    frame = pd.DataFrame(records).rename(columns=_COLUMN_MAP)
    frame["athlete_id"] = frame.pop("source_name").map(_athlete_id)
    frame["season"] = season
    frame["weapon"] = weapon
    frame["competition_date"] = pd.to_datetime(
        frame["competition_date"], format="%d/%m/%y", errors="coerce"
    )
    if frame["competition_date"].isna().any():
        raise ValueError("British Fencing sheet contains an invalid competition date")
    for column in _NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in _PERCENT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(
            frame[column].astype("string").str.rstrip("%"), errors="coerce"
        )
    for column in ("promoted_to_de", "received_bye"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = frame[column].map({"true": True, "false": False, "": pd.NA}).astype(
            "boolean"
        )
    if "source_season" in frame:
        source_season = frame.pop("source_season").astype("string").str.strip()
        populated = source_season[source_season.ne("")]
        if not populated.empty and not populated.str.startswith(season[:4], na=False).all():
            raise ValueError("British Fencing source season does not match requested season")
    category = frame["weapon_category"].astype("string").str.upper().str.strip()
    frame["gender"] = category.str.extract(r"([MW])[^MW]*$", expand=False).map(
        {"M": "men", "W": "women"}
    )
    frame["source_url"] = "https://publish.smartsheet.com/" + SHEETS[season][weapon]
    frame["source_license"] = LICENSE
    return frame.sort_values(
        ["competition_date", "weapon", "event", "athlete_id"], kind="stable"
    ).reset_index(drop=True)


def fetch_and_prepare(
    cache_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch all nine distinct official tables and write a local-only analytic table."""

    frames: list[pd.DataFrame] = []
    snapshot_hashes: dict[str, str] = {}
    for season, weapons in SHEETS.items():
        for weapon, token in weapons.items():
            snapshot = cache_directory / season / f"{weapon}-{token}.html"
            if force:
                snapshot.unlink(missing_ok=True)
            download_resumable(
                f"https://publish.smartsheet.com/{token}",
                snapshot,
                user_agent=user_agent,
            )
            frames.append(
                parse_sheet(
                    snapshot.read_text(errors="replace"),
                    season=season,
                    weapon=weapon,
                )
            )
            snapshot_hashes[f"{season}:{weapon}"] = sha256_file(snapshot)

    result = pd.concat(frames, ignore_index=True)
    result["performance_row_id"] = [f"bf-row-{index + 1:05d}" for index in range(len(result))]
    result = result.sort_values(
        ["competition_date", "weapon", "event", "athlete_id"], kind="stable"
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Public GBR international athlete-event performance rows for 2021-22 through "
            "2023-24. Names are removed and replaced by stable pseudonymous IDs, but this is "
            "not anonymization. The row table is local-analysis-only because no reusable data "
            "license was found; no causal athlete-development or selection claim is made."
        ),
        "source_url": SOURCE_PAGE,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "seasons": len(SHEETS),
        "official_distinct_sheets": len(SHEETS) * 3,
        "athlete_event_rows": len(result),
        "unique_pseudonymous_athletes": int(result["athlete_id"].nunique()),
        "by_season": result.groupby("season").size().astype(int).to_dict(),
        "by_weapon": result.groupby("weapon").size().astype(int).to_dict(),
        "by_gender": result.groupby("gender", dropna=False).size().astype(int).to_dict(),
        "pool_bouts_reported": int(result["pool_bouts"].fillna(0).sum()),
        "de_bouts_reported": int(result["de_bouts"].fillna(0).sum()),
        "events": int(result[["competition_date", "event", "weapon"]].drop_duplicates().shape[0]),
        "snapshot_sha256": snapshot_hashes,
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
