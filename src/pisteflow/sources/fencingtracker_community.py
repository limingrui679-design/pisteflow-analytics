"""Pinned, privacy-minimized audit of a public 2024 FencingTracker snapshot."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/benakc/fencing"
SOURCE_PLATFORM_URL = "https://fencingtracker.com/"
SNAPSHOT = "50c4bd615c1088df623a80dc3ef4841b393c32b8"
CSV_URL = (
    "https://raw.githubusercontent.com/benakc/fencing/"
    f"{SNAPSHOT}/Analysis%20Part%201/fencingtracker_bouts_2024.csv"
)
EXPECTED_BYTES = 71_303_834
EXPECTED_SHA256 = "13d6d354cf34cd15b0952d455968dfbecbec69f059cd7ec9b34bbea5336ee21e"
EXPECTED_ROWS = 261_803
EXPECTED_EVENT_IDS = 3_701
EXPECTED_SOURCE_BOUT_IDS = 261_803
EXPECTED_COLUMNS = [
    "bout_id",
    "tournament_name",
    "event_id",
    "event_name",
    "weapon",
    "date",
    "fencer_name",
    "fencer_id",
    "fencer_gender",
    "fencer_rating",
    "fencer_birth_year",
    "fencer_club",
    "opponent_name",
    "opponent_id",
    "opponent_gender",
    "opponent_rating",
    "opponent_elo",
    "opponent_club",
    "fencer_elo",
    "result",
    "score",
    "bout_type",
    "bout_type_raw",
    "is_de",
    "win_probability",
    "fencer_divisions",
]
LICENSE = (
    "No repository or dataset license. FencingTracker describes its database as public, but "
    "does not publish a data-reuse grant; source rows and normalized rows remain local."
)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _normalized_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def _numeric_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(int(float(value)))


def _name_id_crosswalk(frame: pd.DataFrame) -> dict[str, str]:
    candidates: defaultdict[str, set[str]] = defaultdict(set)
    for name_column, id_column in (
        ("fencer_name", "fencer_id"),
        ("opponent_name", "opponent_id"),
    ):
        for name, source_id in zip(frame[name_column], frame[id_column], strict=True):
            numeric = _numeric_id(source_id)
            if pd.notna(name) and numeric is not None:
                candidates[_normalized_name(str(name))].add(numeric)
    return {
        name: next(iter(source_ids))
        for name, source_ids in candidates.items()
        if len(source_ids) == 1
    }


def _athlete_id(name: str, source_id: Any, crosswalk: dict[str, str]) -> tuple[str, str]:
    numeric = _numeric_id(source_id)
    if numeric is not None:
        identity = f"source-id:{numeric}"
        provenance = "source_id"
    else:
        normalized = _normalized_name(name)
        recovered = crosswalk.get(normalized)
        if recovered is not None:
            identity = f"source-id:{recovered}"
            provenance = "name_crosswalk"
        else:
            identity = f"name-only:{normalized}"
            provenance = "name_only"
    return _stable_id("fencingtracker-athlete", identity), provenance


def _event_gender(event_name: str) -> str:
    normalized = event_name.casefold()
    if "mixed" in normalized:
        return "mixed"
    if re.search(r"\bwomen(?:'s)?\b", normalized):
        return "women"
    if re.search(r"\bmen(?:'s)?\b", normalized):
        return "men"
    return "unknown"


def _person_gender(source_value: Any, event_gender: str) -> str:
    if event_gender in {"men", "women"}:
        return event_gender
    normalized = str(source_value).strip().upper()
    return {"M": "men", "F": "women"}.get(normalized, "unknown")


def _rating_letter(value: Any) -> str | None:
    if pd.isna(value):
        return None
    match = re.match(r"^\s*([A-EU])", str(value).upper())
    return match.group(1) if match else None


def acquire(path: Path, *, user_agent: str, force: bool = False) -> None:
    if force:
        path.unlink(missing_ok=True)
    download_resumable(CSV_URL, path, user_agent=user_agent, expected_bytes=EXPECTED_BYTES)
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"FencingTracker community snapshot checksum changed: {actual}")


def prepare(source_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if source_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("FencingTracker community snapshot byte size changed")
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_SHA256:
        raise ValueError("FencingTracker community snapshot checksum changed")

    source = pd.read_csv(source_path, low_memory=False)
    if list(source.columns) != EXPECTED_COLUMNS:
        raise ValueError("FencingTracker community snapshot schema changed")
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"FencingTracker community row count changed: {len(source)}")
    if int(source["event_id"].nunique()) != EXPECTED_EVENT_IDS:
        raise ValueError("FencingTracker community event count changed")
    if int(source["bout_id"].nunique()) != EXPECTED_SOURCE_BOUT_IDS:
        raise ValueError("FencingTracker community source bout IDs are not unique")

    exact_duplicate_rows = int(source.duplicated().sum())
    malformed_required = source[
        [
            "bout_id",
            "event_id",
            "event_name",
            "date",
            "fencer_name",
            "opponent_name",
            "result",
            "score",
            "bout_type",
            "bout_type_raw",
        ]
    ].isna().any(axis=1)
    missing_required_rows = int(malformed_required.sum())
    frame = source.loc[~malformed_required].copy()

    score_parts = frame["score"].astype(str).str.extract(r"^\s*(\d+)\s*:\s*(\d+)\s*$")
    malformed_score = score_parts.isna().any(axis=1)
    malformed_score_rows = int(malformed_score.sum())
    frame = frame.loc[~malformed_score].copy()
    score_parts = score_parts.loc[~malformed_score].astype("int16")
    frame["left_score"] = score_parts[0]
    frame["right_score"] = score_parts[1]
    frame["date"] = pd.to_datetime(frame["date"], format="mixed", errors="raise")

    weapon_map = {"epee": "epee", "foil": "foil", "saber": "sabre", "sabre": "sabre"}
    frame["normalized_weapon"] = frame["weapon"].astype("string").str.casefold().map(weapon_map)
    unknown_weapon_rows = int(frame["normalized_weapon"].isna().sum())
    frame["normalized_weapon"] = frame["normalized_weapon"].fillna("unknown")

    crosswalk = _name_id_crosswalk(frame)
    output_rows: list[dict[str, Any]] = []
    identity_provenance: Counter[str] = Counter()
    fencer_gender_mismatches = 0
    opponent_gender_mismatches = 0
    for _, row in frame.iterrows():
        left_id, left_provenance = _athlete_id(
            str(row["fencer_name"]), row["fencer_id"], crosswalk
        )
        right_id, right_provenance = _athlete_id(
            str(row["opponent_name"]), row["opponent_id"], crosswalk
        )
        identity_provenance[left_provenance] += 1
        identity_provenance[right_provenance] += 1
        event_gender = _event_gender(str(row["event_name"]))
        if event_gender == "men":
            fencer_gender_mismatches += int(str(row["fencer_gender"]).strip() != "M")
            opponent_gender_mismatches += int(str(row["opponent_gender"]).strip() != "M")
        elif event_gender == "women":
            fencer_gender_mismatches += int(str(row["fencer_gender"]).strip() != "F")
            opponent_gender_mismatches += int(str(row["opponent_gender"]).strip() != "F")

        left_score = int(row["left_score"])
        right_score = int(row["right_score"])
        source_result = str(row["result"]).strip().upper()
        score_winner = (
            "left" if left_score > right_score else "right" if right_score > left_score else None
        )
        source_winner = (
            "left" if source_result == "W" else "right" if source_result == "L" else None
        )
        if score_winner is None:
            result_status = "tied_score_reported"
        elif source_winner is None:
            result_status = "unknown_source_result"
        elif score_winner != source_winner:
            result_status = "score_result_conflict"
        else:
            result_status = "decisive_consistent"
        model_eligible = (
            result_status == "decisive_consistent" and row["normalized_weapon"] != "unknown"
        )
        source_bout_id = str(row["bout_id"])
        output_rows.append(
            {
                "bout_id": _stable_id("fencingtracker-bout", source_bout_id),
                "event_id": _stable_id("fencingtracker-event", str(int(row["event_id"]))),
                "date": row["date"],
                "weapon": str(row["normalized_weapon"]),
                "event_gender": event_gender,
                "bout_type": str(row["bout_type"]).casefold(),
                "round": str(row["bout_type_raw"]),
                "left_fencer_id": left_id,
                "right_fencer_id": right_id,
                "left_rating_letter": _rating_letter(row["fencer_rating"]),
                "right_rating_letter": _rating_letter(row["opponent_rating"]),
                "left_gender": _person_gender(row["fencer_gender"], event_gender),
                "right_gender": _person_gender(row["opponent_gender"], event_gender),
                "left_score": left_score,
                "right_score": right_score,
                "winner_side": source_winner,
                "result_status": result_status,
                "model_eligible": model_eligible,
            }
        )

    output = pd.DataFrame(output_rows)
    if output["bout_id"].duplicated().any():
        raise ValueError("FencingTracker community generated duplicate bout IDs")

    low_id = output[["left_fencer_id", "right_fencer_id"]].min(axis=1)
    high_id = output[["left_fencer_id", "right_fencer_id"]].max(axis=1)
    left_is_low = output["left_fencer_id"].eq(low_id)
    low_score = output["left_score"].where(left_is_low, output["right_score"])
    high_score = output["right_score"].where(left_is_low, output["left_score"])
    canonical = (
        output["event_id"].astype(str)
        + "|"
        + low_id.astype(str)
        + "|"
        + high_id.astype(str)
        + "|"
        + output["bout_type"].astype(str)
        + "|"
        + output["round"].astype(str)
        + "|"
        + low_score.astype(str)
        + ":"
        + high_score.astype(str)
    )
    canonical_duplicate_rows = int(canonical.duplicated().sum())
    if canonical_duplicate_rows:
        keep = ~canonical.duplicated()
        output = output.loc[keep].copy()

    output = output.sort_values(["date", "event_id", "bout_type", "bout_id"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    eligible = output.loc[output["model_eligible"]]
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "source_platform_url": SOURCE_PLATFORM_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        "source_rows": len(source),
        "exact_duplicate_rows": exact_duplicate_rows,
        "missing_required_rows_excluded": missing_required_rows,
        "malformed_score_rows_excluded": malformed_score_rows,
        "canonical_duplicate_rows_excluded": canonical_duplicate_rows,
        "retained_name_free_bouts": len(output),
        "model_eligible_decisive_bouts": len(eligible),
        "event_instances": int(output["event_id"].nunique()),
        "tournament_title_spellings_not_retained": int(source["tournament_name"].nunique()),
        "pseudonymous_athletes": len(
            set(output["left_fencer_id"]) | set(output["right_fencer_id"])
        ),
        "identity_slots_by_provenance": dict(sorted(identity_provenance.items())),
        "source_rows_missing_opponent_numeric_id": int(source["opponent_id"].isna().sum()),
        "event_gender_overrides": {
            "fencer_field_mismatches": fencer_gender_mismatches,
            "opponent_field_mismatches": opponent_gender_mismatches,
        },
        "unknown_weapon_rows": unknown_weapon_rows,
        "date_min": output["date"].min().date().isoformat(),
        "date_max": output["date"].max().date().isoformat(),
        "bouts_by_weapon": {
            str(key): int(value)
            for key, value in output["weapon"].value_counts().sort_index().items()
        },
        "eligible_bouts_by_weapon": {
            str(key): int(value)
            for key, value in eligible["weapon"].value_counts().sort_index().items()
        },
        "bouts_by_type": {
            str(key): int(value)
            for key, value in output["bout_type"].value_counts().sort_index().items()
        },
        "result_status_counts": dict(sorted(Counter(output["result_status"]).items())),
        "source_sha256": source_sha256,
        "output_sha256": sha256_file(output_path),
        "output": "data/processed/fencingtracker-community-2024-bouts.parquet",
        "claim_boundary": (
            "A pinned third-party snapshot of public FencingTracker bout histories during "
            "calendar 2024, not an official USA Fencing feed or a complete census. Source names, "
            "clubs, birth years, raw IDs, event titles, Elo values, and win-probability fields are "
            "excluded from the normalized table. Non-tied score/result conflicts, tied reported "
            "scores, and weapon-unknown rows remain auditable but are excluded from outcome "
            "training. Event labels override inconsistent person-gender fields in single-gender "
            "events. Reporter-side orientation is not random and must be randomized inside each "
            "training fold. No repository or dataset reuse license was found, so row-level source "
            "and normalized data remain local and only aggregate audits may be published."
        ),
    }
    write_json(report_path, summary)
    return summary


def fetch_and_prepare(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    acquire(source_path, user_agent=user_agent, force=force)
    return prepare(source_path, output_path, report_path)
