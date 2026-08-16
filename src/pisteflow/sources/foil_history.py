"""Prepare the GPL snapshot of historical FIE foil pool bouts conservatively."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_bytes, sha256_file, write_json
from pisteflow.sources.net import download_resumable

COMMIT = "4b0511eba35eafd85c9be48ededaa1ff1bd546eb"
SOURCE_PAGE = "https://github.com/amichaelsen/fie-fencing-dataset"
ARCHIVE_URL = f"https://github.com/amichaelsen/fie-fencing-dataset/archive/{COMMIT}.zip"
LICENSE = "GPL-3.0 in source repository"
EXPECTED_BYTES = 2_326_738
_ROOT = "fie-fencing-dataset-main/final_output"


def _pseudonymous_id(value: object, gender: str) -> str:
    digest = sha256(f"fie-foil:{gender}:{value}".encode()).hexdigest()
    return "foil-" + digest[:16]


def normalize_bouts(frame: pd.DataFrame, *, gender: str) -> pd.DataFrame:
    """Remove directly reversible FIE IDs while preserving the upstream measurements."""

    required = {
        "fencer_ID",
        "opp_ID",
        "fencer_age",
        "opp_age",
        "fencer_score",
        "opp_score",
        "winner_ID",
        "fencer_curr_pts",
        "opp_curr_pts",
        "tournament_ID",
        "pool_ID",
        "upset",
        "date",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FIE foil bout table missing columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "left_fencer_id": [
                _pseudonymous_id(value, gender) for value in frame["fencer_ID"]
            ],
            "right_fencer_id": [_pseudonymous_id(value, gender) for value in frame["opp_ID"]],
            "left_age_upstream": pd.to_numeric(frame["fencer_age"], errors="coerce"),
            "right_age_upstream": pd.to_numeric(frame["opp_age"], errors="coerce"),
            "left_score": pd.to_numeric(frame["fencer_score"], errors="coerce"),
            "right_score": pd.to_numeric(frame["opp_score"], errors="coerce"),
            "left_points_before": pd.to_numeric(frame["fencer_curr_pts"], errors="coerce"),
            "right_points_before": pd.to_numeric(frame["opp_curr_pts"], errors="coerce"),
            "tournament_id": frame["tournament_ID"].astype("string"),
            "pool_id": pd.to_numeric(frame["pool_ID"], errors="coerce"),
            "upset_upstream": frame["upset"].astype("boolean"),
            "event_date": pd.to_datetime(frame["date"], errors="coerce"),
        }
    )
    winner = frame["winner_ID"].astype("string")
    left = frame["fencer_ID"].astype("string")
    right = frame["opp_ID"].astype("string")
    result["left_won"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[winner.eq(left), "left_won"] = 1
    result.loc[winner.eq(right), "left_won"] = 0
    if result["left_won"].isna().any():
        raise ValueError("FIE foil winner IDs do not match bout participants")
    if result["event_date"].isna().any():
        raise ValueError("FIE foil bout table contains invalid dates")
    result["gender"] = gender
    result["weapon"] = "foil"
    result["stage"] = "pool"
    result["year"] = result["event_date"].dt.year.astype("Int16")
    result["ages_plausible"] = result["left_age_upstream"].between(13, 70) & result[
        "right_age_upstream"
    ].between(13, 70)
    result["source_url"] = SOURCE_PAGE
    result["source_license"] = LICENSE
    result = result.sort_values(
        ["event_date", "tournament_id", "pool_id", "left_fencer_id", "right_fencer_id"],
        kind="stable",
    ).reset_index(drop=True)
    canonical = result.assign(
        low=result[["left_fencer_id", "right_fencer_id"]].min(axis=1),
        high=result[["left_fencer_id", "right_fencer_id"]].max(axis=1),
    )
    if canonical.duplicated(["tournament_id", "pool_id", "low", "high"]).any():
        raise ValueError("FIE foil table contains duplicate canonical pool bouts")
    return result


def prepare(archive_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    """Normalize the pinned men's and women's snapshots into one local-only table."""

    frames: list[pd.DataFrame] = []
    with ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"FIE foil archive has a corrupt member: {bad}")
        license_bytes = archive.read("fie-fencing-dataset-main/LICENSE")
        if b"GNU GENERAL PUBLIC LICENSE" not in license_bytes:
            raise ValueError("FIE foil repository license changed")
        for source_gender, normalized_gender in (("mens", "men"), ("womens", "women")):
            member = (
                f"{_ROOT}/all_{source_gender}_foil/"
                f"all_{source_gender}_foil_bout_data_May_12_2021.csv"
            )
            with archive.open(member) as handle:
                frames.append(normalize_bouts(pd.read_csv(handle), gender=normalized_gender))
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["event_date", "gender", "tournament_id", "pool_id"], kind="stable"
    )
    result["bout_id"] = [f"foil-history-{index + 1:06d}" for index in range(len(result))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Historical international foil pool-bout snapshot assembled by the upstream "
            "repository from FIE pages through May 2021. It is neither a current FIE feed nor "
            "complete international fencing. Direct FIE IDs are replaced, implausible upstream "
            "ages and 0-0 rows are retained and counted, and row data remains local-only."
        ),
        "source_url": SOURCE_PAGE,
        "source_commit": COMMIT,
        "archive_url": ARCHIVE_URL,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "license_sha256": sha256_bytes(license_bytes),
        "bouts": len(result),
        "by_gender": result.groupby("gender").size().astype(int).to_dict(),
        "unique_pseudonymous_fencers": int(
            pd.concat([result["left_fencer_id"], result["right_fencer_id"]]).nunique()
        ),
        "tournaments": int(result["tournament_id"].nunique()),
        "year_min": int(result["year"].min()),
        "year_max": int(result["year"].max()),
        "zero_zero_rows": int(((result["left_score"] == 0) & (result["right_score"] == 0)).sum()),
        "rows_with_implausible_upstream_age": int((~result["ages_plausible"]).sum()),
        "output_sha256": sha256_file(output_path),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    return prepare(archive_path, output_path, report_path)
