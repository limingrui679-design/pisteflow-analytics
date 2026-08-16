"""Deidentified athlete and ranking panels from a historical women's-foil snapshot."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_PAGE = "https://www.kaggle.com/datasets/amichaelsen/fie-fencing-womens-foil-data"
ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "amichaelsen/fie-fencing-womens-foil-data?datasetVersionNumber=1"
)
LICENSE = "Kaggle metadata: CC0; upstream rows were collected from FIE pages"
EXPECTED_BYTES = 637_405
EXPECTED_SHA256 = "8985a51fe6b439bf4f64cc960f207cc3e0cef1272b09a7838625abd6c6295355"
_BIO_MEMBER = "all_womens_foil_fencer_bio_data_May_13_2021_cleaned.csv"
_RANK_MEMBER = "all_womens_foil_fencer_rankings_data_May_13_2021_cleaned.csv"
_TOURNAMENT_MEMBER = "all_womens_foil_tournament_data_May_13_2021_cleaned.csv"
_BOUT_MEMBER = "all_womens_foil_bout_data_May_13_2021_cleaned.csv"
_EXPECTED_MEMBERS = {_BIO_MEMBER, _RANK_MEMBER, _TOURNAMENT_MEMBER, _BOUT_MEMBER}
_BIO_COLUMNS = ["id", "name", "country_code", "country", "hand", "age", "url", "date_accessed"]
_RANK_COLUMNS = ["id", "weapon", "category", "season", "rank", "points"]
_TOURNAMENT_COLUMNS = [
    "competition_ID",
    "season",
    "name",
    "category",
    "country",
    "start_date",
    "end_date",
    "weapon",
    "gender",
    "timezone",
    "url",
    "unique_ID",
    "missing_results_flag",
]


def _athlete_id(value: object) -> str:
    digest = sha256(f"fie-womens-foil-snapshot:{value}".encode()).hexdigest()[:20]
    return f"wfs-athlete:{digest}"


def _event_id(value: object) -> str:
    digest = sha256(f"fie-womens-foil-event:{value}".encode()).hexdigest()[:20]
    return f"wfs-event:{digest}"


def normalize_bios(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != _BIO_COLUMNS:
        raise ValueError("Women's-foil bio columns changed")
    if frame["id"].duplicated().any() or frame[["id", "name", "url"]].isna().any().any():
        raise ValueError("Women's-foil bio identities are missing or duplicated")
    result = pd.DataFrame(
        {
            "athlete_id": frame["id"].map(_athlete_id),
            "country_code": frame["country_code"].astype("string"),
            "country": frame["country"].astype("string"),
            "handedness": frame["hand"].map({"Left": "left", "Right": "right"}),
            "age_upstream": pd.to_numeric(frame["age"], errors="raise").astype("int16"),
            "snapshot_accessed_at": pd.to_datetime(frame["date_accessed"], errors="raise"),
        }
    )
    result["handedness"] = result["handedness"].fillna("unknown").astype("string")
    result["age_between_13_and_80"] = result["age_upstream"].between(13, 80)
    return result.sort_values("athlete_id", kind="stable").reset_index(drop=True)


def normalize_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != _RANK_COLUMNS:
        raise ValueError("Women's-foil ranking columns changed")
    if frame.isna().any().any():
        raise ValueError("Women's-foil ranking rows contain missing values")
    result = frame.copy()
    result["athlete_id"] = result.pop("id").map(_athlete_id)
    result["weapon"] = result["weapon"].str.casefold().replace({"épée": "epee"})
    if not set(result["weapon"]).issubset({"foil", "epee", "sabre"}):
        raise ValueError("Women's-foil ranking weapon labels changed")
    if not set(result["category"]).issubset({"Cadet", "Junior", "Senior", "Veterans"}):
        raise ValueError("Women's-foil ranking categories changed")
    season_parts = result["season"].str.extract(r"^(\d{4})/(\d{4})$")
    if season_parts.isna().any().any():
        raise ValueError("Women's-foil ranking season format changed")
    result["season_start"] = season_parts[0].astype("int16")
    result["season_end"] = season_parts[1].astype("int16")
    if not result["season_end"].sub(result["season_start"]).eq(1).all():
        raise ValueError("Women's-foil ranking season spans changed")
    result = result.drop(columns=["season"])
    result["rank"] = pd.to_numeric(result["rank"], errors="raise").astype("int32")
    result["points"] = pd.to_numeric(result["points"], errors="raise").astype("float64")
    if result["rank"].lt(1).any() or result["points"].lt(0).any():
        raise ValueError("Women's-foil ranking values changed")
    duplicate_key = [
        "athlete_id",
        "weapon",
        "category",
        "season_start",
        "season_end",
        "rank",
        "points",
    ]
    result["exact_repeated_row"] = result.duplicated(duplicate_key, keep=False)
    result["training_eligible"] = ~result.duplicated(duplicate_key, keep="first")
    return result.sort_values(
        ["season_start", "weapon", "category", "rank", "athlete_id"], kind="stable"
    ).reset_index(drop=True)


def normalize_tournaments(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != _TOURNAMENT_COLUMNS:
        raise ValueError("Women's-foil tournament columns changed")
    result = pd.DataFrame(
        {
            "event_id": frame["unique_ID"].map(_event_id),
            "season": pd.to_numeric(frame["season"], errors="raise").astype("int16"),
            "event_type": frame["name"].astype("string"),
            "category": frame["category"].astype("string"),
            "country": frame["country"].astype("string"),
            "start_date": pd.to_datetime(frame["start_date"], errors="raise"),
            "end_date": pd.to_datetime(frame["end_date"], errors="raise"),
            "weapon": frame["weapon"].str.casefold().astype("string"),
            "gender": frame["gender"].map({"Womens": "women"}).astype("string"),
            "timezone": frame["timezone"].astype("string"),
        }
    )
    result["upstream_date_order_valid"] = result["start_date"].le(result["end_date"])
    if result["event_id"].duplicated().any() or not result["weapon"].eq("foil").all():
        raise ValueError("Women's-foil tournament identity or weapon changed")
    if not result["gender"].eq("women").all():
        raise ValueError("Women's-foil tournament gender changed")
    return result.sort_values(["start_date", "event_id"], kind="stable").reset_index(drop=True)


def prepare(
    archive_path: Path,
    bios_output: Path,
    rankings_output: Path,
    tournaments_output: Path,
    report_path: Path,
) -> dict[str, Any]:
    archive_changed = (
        archive_path.stat().st_size != EXPECTED_BYTES
        or sha256_file(archive_path) != EXPECTED_SHA256
    )
    if archive_changed:
        raise ValueError("Women's-foil Kaggle archive size or checksum changed")
    with ZipFile(archive_path) as archive:
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        if names != _EXPECTED_MEMBERS:
            raise ValueError("Women's-foil Kaggle archive member inventory changed")
        bios_raw = pd.read_csv(BytesIO(archive.read(_BIO_MEMBER)))
        rankings_raw = pd.read_csv(BytesIO(archive.read(_RANK_MEMBER)))
        tournaments_raw = pd.read_csv(BytesIO(archive.read(_TOURNAMENT_MEMBER)))
        bouts_raw = pd.read_csv(BytesIO(archive.read(_BOUT_MEMBER)))
        uncompressed_bytes = sum(item.file_size for item in archive.infolist())

    if bios_raw.shape != (2_108, 8) or rankings_raw.shape != (10_937, 6):
        raise ValueError("Women's-foil athlete table sizes changed")
    if tournaments_raw.shape != (218, 13) or bouts_raw.shape != (49_131, 13):
        raise ValueError("Women's-foil event or bout table sizes changed")
    bios = normalize_bios(bios_raw)
    rankings = normalize_rankings(rankings_raw)
    tournaments = normalize_tournaments(tournaments_raw)
    known_ids = set(bios_raw["id"])
    endpoint_ids = set(bouts_raw["fencer_ID"]) | set(bouts_raw["opp_ID"])
    if not set(rankings_raw["id"]).issubset(known_ids) or not endpoint_ids.issubset(known_ids):
        raise ValueError("Women's-foil identity crosswalk coverage changed")

    for path, frame in [
        (bios_output, bios),
        (rankings_output, rankings),
        (tournaments_output, tournaments),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    eligible = rankings.loc[rankings["training_eligible"]]
    weapon_count_by_athlete = eligible.groupby("athlete_id")["weapon"].nunique()
    report: dict[str, Any] = {
        "claim_boundary": (
            "A May 2021 third-party snapshot whose Kaggle metadata declares CC0, but whose "
            "rows originated from FIE pages. To avoid relying on that rights assertion, row "
            "tables stay local and only code, hashes, and aggregates are published. Athlete "
            "IDs are stable hashes of reversible source IDs and therefore pseudonymous, not "
            "anonymous. Ages are upstream snapshot values; six values outside 13-80 are "
            "retained and flagged. The 49,131 bout rows duplicate the historical foil-bout "
            "source already prepared elsewhere and are audited here but not emitted or counted "
            "again as training samples. One source tournament has a start date one year after "
            "its end date; both values are retained unchanged and explicitly flagged."
        ),
        "source_url": SOURCE_PAGE,
        "license": LICENSE,
        "prepared_at": datetime.now(UTC).isoformat(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_uncompressed_bytes": uncompressed_bytes,
        "archive_sha256": EXPECTED_SHA256,
        "athlete_bios": len(bios),
        "bio_name_spellings": int(bios_raw["name"].nunique()),
        "handedness": dict(sorted(Counter(bios["handedness"]).items())),
        "ages_outside_13_to_80": int((~bios["age_between_13_and_80"]).sum()),
        "ranking_rows_source": len(rankings),
        "ranking_exact_repeated_rows_after_first": int((~rankings["training_eligible"]).sum()),
        "ranking_rows_training_eligible": len(eligible),
        "ranking_athletes": int(eligible["athlete_id"].nunique()),
        "ranking_rows_by_weapon": eligible.groupby("weapon").size().astype(int).to_dict(),
        "athletes_with_multiple_ranked_weapons": int(weapon_count_by_athlete.gt(1).sum()),
        "ranking_season_start_min": int(eligible["season_start"].min()),
        "ranking_season_start_max": int(eligible["season_start"].max()),
        "tournaments": len(tournaments),
        "tournaments_with_inverted_upstream_dates": int(
            (~tournaments["upstream_date_order_valid"]).sum()
        ),
        "tournament_date_min": tournaments["start_date"].min().date().isoformat(),
        "tournament_date_max": tournaments["end_date"].max().date().isoformat(),
        "bout_rows_audited_not_reemitted": len(bouts_raw),
        "bout_unique_athletes_covered_by_bios": len(endpoint_ids),
        "direct_names_removed_from_outputs": True,
        "direct_source_ids_removed_from_outputs": True,
        "outputs": {
            "bios": str(bios_output),
            "rankings": str(rankings_output),
            "tournaments": str(tournaments_output),
        },
        "output_sha256": {
            "bios": sha256_file(bios_output),
            "rankings": sha256_file(rankings_output),
            "tournaments": sha256_file(tournaments_output),
        },
    }
    write_json(report_path, report)
    return report


def fetch_and_prepare(
    archive_path: Path,
    bios_output: Path,
    rankings_output: Path,
    tournaments_output: Path,
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
    return prepare(archive_path, bios_output, rankings_output, tournaments_output, report_path)
