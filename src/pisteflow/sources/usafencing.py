"""Privacy-minimized preparation of the public USA Fencing tournament dataset."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_bytes, sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = (
    "https://www.kaggle.com/datasets/rivkalipkovitz/usa-fencing-national-tournament-dataset"
)
ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rivkalipkovitz/usa-fencing-national-tournament-dataset"
)
LICENSE = "AGPL-3.0"
_WEAPONS = {"epee": "epee", "foil": "foil", "saber": "sabre", "sabre": "sabre"}
_GENDERS = {"f": "women", "m": "men"}


def _csv(archive: ZipFile, name: str) -> pd.DataFrame:
    try:
        with archive.open(name) as handle:
            return pd.read_csv(handle, low_memory=False)
    except KeyError as exc:
        raise ValueError(f"USA Fencing archive is missing {name}") from exc


def normalize_bouts(
    pool_bouts: pd.DataFrame,
    de_bouts: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join event context and remove names, clubs, countries, and referee fields."""

    event_columns = {
        "event_id",
        "tournament_id",
        "date",
        "category",
        "gender",
        "weapon",
        "num_fencers",
    }
    missing_events = event_columns.difference(events.columns)
    if missing_events:
        raise ValueError(f"USA Fencing events missing columns: {sorted(missing_events)}")
    bout_columns = {
        "bout_id",
        "event_id",
        "fencer1_id",
        "fencer1_name",
        "fencer2_id",
        "fencer2_name",
        "score1",
        "score2",
        "winner",
    }
    for stage, frame in (("pool", pool_bouts), ("direct_elimination", de_bouts)):
        missing = bout_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"USA Fencing {stage} bouts missing columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for stage, source in (("pool", pool_bouts), ("direct_elimination", de_bouts)):
        frame = source[list(bout_columns)].copy()
        frame["stage"] = stage
        left = frame["fencer1_name"].astype("string").str.strip().str.casefold()
        right = frame["fencer2_name"].astype("string").str.strip().str.casefold()
        winner = frame["winner"].astype("string").str.strip().str.casefold()
        frame["left_won"] = pd.Series(pd.NA, index=frame.index, dtype="Int8")
        frame.loc[winner.eq(left) & winner.notna(), "left_won"] = 1
        frame.loc[winner.eq(right) & winner.notna(), "left_won"] = 0
        frame["left_score"] = pd.to_numeric(frame["score1"], errors="coerce").astype("Int16")
        frame["right_score"] = pd.to_numeric(frame["score2"], errors="coerce").astype("Int16")
        frame = frame.rename(
            columns={"fencer1_id": "left_fencer_id", "fencer2_id": "right_fencer_id"}
        )
        frames.append(
            frame[
                [
                    "bout_id",
                    "event_id",
                    "stage",
                    "left_fencer_id",
                    "right_fencer_id",
                    "left_score",
                    "right_score",
                    "left_won",
                ]
            ]
        )

    event_frame = events[list(event_columns)].copy()
    event_frame["event_date"] = pd.to_datetime(event_frame.pop("date"), errors="coerce")
    event_frame["weapon"] = event_frame["weapon"].astype("string").str.casefold().map(_WEAPONS)
    event_frame["gender"] = event_frame["gender"].astype("string").str.casefold().map(_GENDERS)
    if event_frame["weapon"].isna().any():
        values = sorted(events.loc[event_frame["weapon"].isna(), "weapon"].astype(str).unique())
        raise ValueError(f"Unknown USA Fencing weapon values: {values}")
    if event_frame["event_id"].duplicated().any():
        raise ValueError("USA Fencing event IDs are not unique")

    bouts = pd.concat(frames, ignore_index=True)
    if bouts["bout_id"].duplicated().any():
        raise ValueError("USA Fencing bout IDs are not unique across pool and DE data")
    bouts = bouts.merge(event_frame, on="event_id", how="left", validate="many_to_one")
    if bouts["weapon"].isna().any():
        raise ValueError("Some USA Fencing bouts do not match an event")
    bouts["year"] = bouts["event_date"].dt.year.astype("Int16")
    bouts["source_dataset"] = "USA Fencing National Tournament Dataset"
    bouts["source_license"] = LICENSE
    bouts = bouts.sort_values(["event_date", "event_id", "stage", "bout_id"], kind="stable")
    event_frame["source_dataset"] = "USA Fencing National Tournament Dataset"
    event_frame["source_license"] = LICENSE
    event_frame = event_frame.sort_values(["event_date", "event_id"], kind="stable")
    return bouts.reset_index(drop=True), event_frame.reset_index(drop=True)


def fetch_and_prepare(
    archive_path: Path,
    bouts_output: Path,
    events_output: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download, validate, minimize, and normalize the three-weapon dataset."""

    if force and archive_path.exists():
        archive_path.unlink()
    download_resumable(ARCHIVE_URL, archive_path, user_agent=user_agent)
    with ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"USA Fencing archive has a corrupt member: {bad}")
        pool = _csv(archive, "data/ts_pool_bouts.csv")
        de = _csv(archive, "data/ts_de_bouts.csv")
        events = _csv(archive, "data/ts_events.csv")
        entries = _csv(archive, "data/ts_event_entries.csv")
        pools = _csv(archive, "data/ts_pools.csv")
        tournaments = _csv(archive, "data/ts_tournaments.csv")
        license_sha256 = sha256_bytes(archive.read("LICENSE"))

    bouts, event_frame = normalize_bouts(pool, de, events)
    bouts_output.parent.mkdir(parents=True, exist_ok=True)
    bouts.to_parquet(bouts_output, index=False)
    event_frame.to_parquet(events_output, index=False)
    completed = bouts["left_won"].notna()
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Historical USA national-event results from 2016-2020, not all US fencing. "
            "Public names, clubs, countries, and referee fields are removed from processed tables; "
            "pseudonymous IDs must not be used to publish rankings of minors or causal "
            "ability claims."
        ),
        "source_url": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "license_sha256": license_sha256,
        "bouts_sha256": sha256_file(bouts_output),
        "events_sha256": sha256_file(events_output),
        "tournaments": len(tournaments),
        "events": len(event_frame),
        "entries": len(entries),
        "pools": len(pools),
        "pool_bouts": int((bouts["stage"] == "pool").sum()),
        "direct_elimination_bouts": int((bouts["stage"] == "direct_elimination").sum()),
        "bouts": len(bouts),
        "completed_bouts": int(completed.sum()),
        "unique_pseudonymous_fencers": int(
            pd.concat([bouts["left_fencer_id"], bouts["right_fencer_id"]]).nunique()
        ),
        "year_min": int(bouts["year"].min()),
        "year_max": int(bouts["year"].max()),
        "by_weapon": bouts.groupby("weapon").size().astype(int).to_dict(),
        "by_gender": bouts.groupby("gender").size().astype(int).to_dict(),
        "by_stage": bouts.groupby("stage").size().astype(int).to_dict(),
        "outputs": {"bouts": str(bouts_output), "events": str(events_output)},
    }
    write_json(report_path, summary)
    return summary
