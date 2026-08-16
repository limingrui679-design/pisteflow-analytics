"""Privacy-minimized adapter for the CC BY OSF junior three-weapon corpus.

The deposit is unusually large and useful, but it is not a flawless ground-
truth table.  This adapter pins the archive, removes direct athlete names and
nationalities from analytic outputs, and makes all known contradictions
explicit rather than silently repairing them.
"""

from __future__ import annotations

import hashlib
import io
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://osf.io/jtb8h/"
DOWNLOAD_URL = "https://osf.io/download/6a6b595c49a5aae78922f40d/"
ARCHIVE_BYTES = 4_106_249
ARCHIVE_SHA256 = "0bf14df0ff6624f691a92d68d0de6bd5475af10065ba63be80fefcf1dffc0305"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ROOT = "Three_Weapons_OSF_deposit/"
MEMBERS = {
    "bouts": f"{ROOT}data/pool_bouts.csv",
    "entries": f"{ROOT}data/pool_seeds.csv",
    "rankings": f"{ROOT}data/rankings.csv",
    "codebook": f"{ROOT}data/CODEBOOK.md",
    "data_license": f"{ROOT}LICENSE-CC-BY-4.0.txt",
}
EXPECTED_ROWS = {"bouts": 158_652, "entries": 54_637, "rankings": 66_090}
WEAPONS = {"E": "epee", "F": "foil", "S": "sabre"}


def _normalize_identity(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split()).casefold()


def athlete_key(name: object, nationality: object) -> str:
    """Return a stable source-local key without exposing either input string."""

    normalized = f"{_normalize_identity(name)}|{_normalize_identity(nationality)}"
    digest = hashlib.sha256(f"osf-three-weapons-v1|{normalized}".encode()).hexdigest()
    return f"OTW-{digest[:20]}"


def _validate_shared(frame: pd.DataFrame, *, expected_rows: int) -> pd.DataFrame:
    if len(frame) != expected_rows:
        raise ValueError(f"OSF three-weapons row count changed: {len(frame)}")
    required = {"year", "comp_id", "date", "category", "weapon", "gender"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OSF three-weapons shared columns changed: {sorted(missing)}")
    if not set(frame["weapon"]).issubset(WEAPONS):
        raise ValueError("OSF three-weapons weapon codes changed")
    if not set(frame["gender"]).issubset({"M", "F"}):
        raise ValueError("OSF three-weapons gender codes changed")
    parsed_dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("OSF three-weapons contains invalid dates")
    result = frame.copy()
    result["weapon"] = result["weapon"].map(WEAPONS)
    result["competition_key"] = (
        result["year"].astype(str) + ":" + result["comp_id"].astype(str)
    )
    return result


def normalize_bouts(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a direct-identifier-free bout table with honest outcome flags."""

    required = {
        "pool",
        "A_name",
        "A_nat",
        "A_seed",
        "A_pos",
        "B_name",
        "B_nat",
        "B_seed",
        "B_pos",
        "scoreA",
        "scoreB",
        "winner",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OSF bout columns changed: {sorted(missing)}")
    result = _validate_shared(frame, expected_rows=EXPECTED_ROWS["bouts"])
    if result.duplicated().any():
        raise ValueError("OSF bout snapshot unexpectedly contains exact duplicates")
    if not result["scoreA"].between(0, 5).all() or not result["scoreB"].between(0, 5).all():
        raise ValueError("OSF bout score is outside 0-5")
    if not set(result["winner"]).issubset({"A", "B"}):
        raise ValueError("OSF bout winner labels changed")

    result.insert(
        0,
        "source_row",
        pd.Series(range(1, len(result) + 1), dtype="int32"),
    )
    result["athlete_a_key"] = [
        athlete_key(name, nat)
        for name, nat in zip(result["A_name"], result["A_nat"], strict=True)
    ]
    result["athlete_b_key"] = [
        athlete_key(name, nat)
        for name, nat in zip(result["B_name"], result["B_nat"], strict=True)
    ]
    tied = result["scoreA"].eq(result["scoreB"])
    derived = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    derived.loc[~tied & result["scoreA"].gt(result["scoreB"])] = "A"
    derived.loc[~tied & result["scoreB"].gt(result["scoreA"])] = "B"
    conflict = ~tied & result["winner"].ne(derived)
    result["score_derived_winner"] = derived
    result["outcome_issue"] = pd.Series("none", index=result.index, dtype="string")
    result.loc[tied, "outcome_issue"] = "tied_score_with_declared_winner"
    result.loc[conflict, "outcome_issue"] = "declared_winner_score_conflict"
    result["outcome_eligible"] = ~(tied | conflict)
    result["athlete_a_won"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result.loc[result["outcome_eligible"], "athlete_a_won"] = result.loc[
        result["outcome_eligible"], "winner"
    ].eq("A")
    result["source_url"] = SOURCE_URL
    return result.drop(columns=["A_name", "A_nat", "B_name", "B_nat"])


def normalize_entries(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the privacy-minimized per-competition entry table."""

    required = {"seed", "pool", "pos", "name", "nat", "vm", "ts", "tr", "ind"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OSF entry columns changed: {sorted(missing)}")
    result = _validate_shared(frame, expected_rows=EXPECTED_ROWS["entries"])
    if result.duplicated().any():
        raise ValueError("OSF entry snapshot unexpectedly contains exact duplicates")
    result.insert(0, "source_row", pd.Series(range(1, len(result) + 1), dtype="int32"))
    result["athlete_key"] = [
        athlete_key(name, nat)
        for name, nat in zip(result["name"], result["nat"], strict=True)
    ]
    identity_counts = result.groupby(["competition_key", "athlete_key"])[
        "source_row"
    ].transform("size")
    result["same_spelling_event_collision"] = identity_counts.gt(1)
    result["source_url"] = SOURCE_URL
    return result.drop(columns=["name", "nat"])


def normalize_rankings(
    frame: pd.DataFrame,
    *,
    bout_competition_keys: set[str],
) -> pd.DataFrame:
    """Remove exact duplicates and unreliable direct age/identity fields."""

    required = {"event", "rank", "points", "name", "nat", "age", "birth_year"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OSF ranking columns changed: {sorted(missing)}")
    result = _validate_shared(frame, expected_rows=EXPECTED_ROWS["rankings"])
    result.insert(0, "source_row", pd.Series(range(1, len(result) + 1), dtype="int32"))
    duplicated = result.drop(columns="source_row").duplicated(keep="first")
    result["source_exact_duplicate_removed"] = duplicated
    result = result.loc[~duplicated].copy()
    result["athlete_key"] = [
        athlete_key(name, nat)
        for name, nat in zip(result["name"], result["nat"], strict=True)
    ]
    result["pool_corpus_match"] = result["competition_key"].isin(bout_competition_keys)
    result["source_url"] = SOURCE_URL
    # The deposited age field is inconsistent with both event year and birth year
    # for most rows.  Neither age nor birth year is needed for the tournament task.
    return result.drop(columns=["name", "nat", "age", "birth_year"])


def _read_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(member)))


def fetch_and_prepare(
    archive_path: Path,
    processed_directory: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download, validate, pseudonymize, and report the pinned OSF corpus."""

    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        DOWNLOAD_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=ARCHIVE_BYTES,
    )
    if sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("OSF three-weapons archive checksum changed")

    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("OSF three-weapons archive failed CRC validation")
        names = set(archive.namelist())
        missing = set(MEMBERS.values()).difference(names)
        if missing:
            raise ValueError(f"OSF three-weapons archive members changed: {sorted(missing)}")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("OSF three-weapons archive contains an unsafe member path")
        license_text = archive.read(MEMBERS["data_license"]).decode("utf-8")
        if "Creative Commons Attribution 4.0" not in license_text:
            raise ValueError("OSF three-weapons data license changed")
        raw_bouts = _read_csv(archive, MEMBERS["bouts"])
        raw_entries = _read_csv(archive, MEMBERS["entries"])
        raw_rankings = _read_csv(archive, MEMBERS["rankings"])

    bouts = normalize_bouts(raw_bouts)
    entries = normalize_entries(raw_entries)
    bout_keys = set(bouts["competition_key"])
    rankings = normalize_rankings(raw_rankings, bout_competition_keys=bout_keys)

    processed_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bouts": processed_directory / "junior-pool-bouts.parquet",
        "entries": processed_directory / "junior-pool-entries.parquet",
        "rankings": processed_directory / "junior-event-rankings.parquet",
    }
    bouts.to_parquet(outputs["bouts"], index=False)
    entries.to_parquet(outputs["entries"], index=False)
    rankings.to_parquet(outputs["rankings"], index=False)

    bout_competitions = bouts[["competition_key", "weapon"]].drop_duplicates()
    ranking_keys = set(rankings["competition_key"])
    expected_age = raw_rankings["year"] - raw_rankings["birth_year"]
    age_difference = (raw_rankings["age"] - expected_age).abs()
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "download_url": DOWNLOAD_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": archive_path.stat().st_size,
        "claim_boundary": (
            "Third-party CC BY transcription of public FIE Junior World Cup PDFs, not an "
            "official FIE API export. Direct names and nationalities are removed from local "
            "analytic tables. The archive omits its PDF-to-CSV parser, has 71 tied-score rows, "
            "one non-tie declared-winner conflict, one exact ranking duplicate, six same-name "
            "event collisions, and an age field that usually disagrees with event year minus "
            "birth year. Outcome training must use outcome_eligible and group by athlete keys."
        ),
        "bouts": {
            "rows": len(bouts),
            "competitions": int(bout_competitions["competition_key"].nunique()),
            "by_weapon": bouts["weapon"].value_counts().sort_index().to_dict(),
            "by_gender": bouts["gender"].value_counts().sort_index().to_dict(),
            "years": [int(bouts["year"].min()), int(bouts["year"].max())],
            "dates": [str(bouts["date"].min()), str(bouts["date"].max())],
            "distinct_pseudonymous_athlete_keys": int(
                pd.concat([bouts["athlete_a_key"], bouts["athlete_b_key"]]).nunique()
            ),
            "tied_score_rows": int(
                bouts["outcome_issue"].eq("tied_score_with_declared_winner").sum()
            ),
            "declared_winner_score_conflicts": int(
                bouts["outcome_issue"].eq("declared_winner_score_conflict").sum()
            ),
            "outcome_eligible_rows": int(bouts["outcome_eligible"].sum()),
        },
        "entries": {
            "rows": len(entries),
            "competitions": int(entries["competition_key"].nunique()),
            "distinct_pseudonymous_athlete_keys": int(entries["athlete_key"].nunique()),
            "same_spelling_event_collision_rows": int(
                entries["same_spelling_event_collision"].sum()
            ),
            "same_spelling_event_collision_groups": int(
                entries.loc[entries["same_spelling_event_collision"]]
                .groupby(["competition_key", "athlete_key"])
                .ngroups
            ),
        },
        "rankings": {
            "source_rows": len(raw_rankings),
            "analytic_rows": len(rankings),
            "exact_duplicates_removed": len(raw_rankings) - len(rankings),
            "competitions": int(rankings["competition_key"].nunique()),
            "years": [int(rankings["year"].min()), int(rankings["year"].max())],
            "individual_event_rows": int(rankings["event"].eq("I").sum()),
            "pool_corpus_match_rows": int(rankings["pool_corpus_match"].sum()),
            "pool_competitions_missing_rankings": len(bout_keys - ranking_keys),
            "age_birth_rows_removed": len(raw_rankings),
            "age_exactly_matches_year_minus_birth_year": int(age_difference.eq(0).sum()),
            "age_within_one_year_of_year_minus_birth_year": int(age_difference.le(1).sum()),
        },
        "outputs": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
        },
    }
    write_json(report_path, summary)
    return summary
