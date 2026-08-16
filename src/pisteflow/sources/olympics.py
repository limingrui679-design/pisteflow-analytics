"""CC BY 4.0 Olympic athlete-event history, filtered to fencing."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json
from pisteflow.sources.classification import infer_weapon_candidates, primary_weapon

SOURCE_URL = (
    "https://raw.githubusercontent.com/moderndive/olympicAthletes/main/data/olympic_athletes.rda"
)
SOURCE_PAGE = "https://github.com/moderndive/olympicAthletes"
LICENSE = "CC BY 4.0"


def _download(url: str, destination: Path, user_agent: str, attempts: int = 6) -> None:
    last_error: Exception | None = None
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=180),
        headers={"User-Agent": user_agent},
    ) as client:
        for attempt in range(attempts):
            try:
                response = client.get(url)
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else float(2**attempt)
                    )
                    time.sleep(min(delay, 45.0))
                    continue
                response.raise_for_status()
                atomic_write_bytes(destination, response.content)
                return
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError(f"Olympic data download failed after {attempts} attempts") from last_error


def fetch_and_prepare(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download the licensed source and create a fencing-only Parquet table."""

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if force or not raw_path.exists():
        _download(SOURCE_URL, raw_path, user_agent)

    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("Install the collectors extra: pip install -e '.[collectors]'") from exc

    objects = pyreadr.read_r(str(raw_path))
    if not objects:
        raise ValueError(f"No data frame found in {raw_path}")
    frame = next(iter(objects.values())).copy()
    required = {
        "id",
        "name",
        "sex",
        "age",
        "height",
        "weight",
        "team",
        "noc",
        "games",
        "year",
        "season",
        "sport",
        "event",
        "medal",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Olympic source is missing columns: {sorted(missing)}")

    fencing = frame[frame["sport"].astype(str).str.casefold() == "fencing"].copy()
    if fencing.empty:
        raise ValueError("Olympic source contains no fencing rows")
    candidates = fencing["event"].astype(str).map(infer_weapon_candidates)
    fencing["weapon"] = candidates.map(primary_weapon).map(lambda weapon: weapon.value)
    fencing["weapon_candidates"] = candidates.map(
        lambda values: ",".join(weapon.value for weapon in values)
    )
    fencing["event_type"] = (
        fencing["event"]
        .astype(str)
        .str.contains(
            "team",
            case=False,
            regex=False,
        )
        .map({True: "team", False: "individual"})
    )
    fencing["medal"] = fencing["medal"].fillna("none").astype(str).str.lower()
    fencing["is_medal"] = fencing["medal"].isin(["gold", "silver", "bronze"]).astype(int)
    for column in ("id", "age", "height", "weight", "year"):
        fencing[column] = pd.to_numeric(fencing[column], errors="coerce")

    history = (
        fencing.groupby(["id", "year"], as_index=False)
        .agg(
            games_in_year=("games", "nunique"),
            medal_rows_in_year=("is_medal", "sum"),
            event_rows_in_year=("event", "size"),
        )
        .sort_values(["id", "year"])
    )
    history["prior_games"] = (
        history.groupby("id")["games_in_year"].cumsum() - history["games_in_year"]
    )
    history["prior_medal_rows"] = (
        history.groupby("id")["medal_rows_in_year"].cumsum() - history["medal_rows_in_year"]
    )
    history["prior_event_rows"] = (
        history.groupby("id")["event_rows_in_year"].cumsum() - history["event_rows_in_year"]
    )
    history["prior_medal_rate"] = history["prior_medal_rows"] / history["prior_event_rows"].clip(
        lower=1
    )
    fencing = fencing.merge(
        history[
            [
                "id",
                "year",
                "prior_games",
                "prior_medal_rows",
                "prior_event_rows",
                "prior_medal_rate",
            ]
        ],
        on=["id", "year"],
        how="left",
        validate="many_to_one",
    )
    fencing["source_url"] = SOURCE_PAGE
    fencing["source_license"] = LICENSE

    keep = [
        "id",
        "name",
        "sex",
        "age",
        "height",
        "weight",
        "team",
        "noc",
        "games",
        "year",
        "season",
        "city_local_latin",
        "city_english",
        "sport",
        "event",
        "weapon",
        "weapon_candidates",
        "event_type",
        "medal",
        "is_medal",
        "prior_games",
        "prior_medal_rows",
        "prior_event_rows",
        "prior_medal_rate",
        "source_url",
        "source_license",
    ]
    keep = [column for column in keep if column in fencing.columns]
    fencing = fencing[keep].sort_values(["year", "event", "name"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fencing.to_parquet(output_path, index=False)

    summary = {
        "claim_boundary": (
            "Historical Olympic athlete-event records; per-player team medals and source "
            "missingness "
            "must not be interpreted as current athlete ability or causal performance effects."
        ),
        "source_url": SOURCE_PAGE,
        "source_data_url": SOURCE_URL,
        "license": LICENSE,
        "fetched_at": datetime.now(UTC).isoformat(),
        "raw_sha256": sha256_file(raw_path),
        "processed_sha256": sha256_file(output_path),
        "rows": len(fencing),
        "unique_athletes": int(fencing["id"].nunique()),
        "year_min": int(fencing["year"].min()),
        "year_max": int(fencing["year"].max()),
        "by_weapon": fencing.groupby("weapon").size().astype(int).to_dict(),
        "by_event_type": fencing.groupby("event_type").size().astype(int).to_dict(),
        "medal_rows": int(fencing["is_medal"].sum()),
        "missingness": {
            column: round(float(fencing[column].isna().mean()), 6)
            for column in ("age", "height", "weight")
        },
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
