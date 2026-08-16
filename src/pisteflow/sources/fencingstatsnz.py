"""Pinned, privacy-minimized audit of the FencingStatsNZ bout ledger."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/nolanpetersonnz/FencingStatsNZ"
SNAPSHOT = "3f944038a171533836fb2034194308920f338993"
CSV_URL = f"https://raw.githubusercontent.com/nolanpetersonnz/FencingStatsNZ/{SNAPSHOT}/ingest/bouts.csv"
EXPECTED_BYTES = 2_756_145
EXPECTED_SHA256 = "baa00c73cc67c29abde75d6cb59f7d28517260a0bc604b514fbd78da37cd71ba"
LICENSE = (
    "Repository is MIT; underlying Fencing New Zealand public-results data has no separately "
    "verified reuse license, so row-level inputs and derivatives remain local"
)
EXPECTED_COLUMNS = [
    "date",
    "competition",
    "weapon",
    "bout_type",
    "fencer_a",
    "club_a",
    "fencer_b",
    "club_b",
    "score_a",
    "score_b",
    "de_round",
    "gender",
    "fencer_a_uid",
    "fencer_b_uid",
    "flag",
]


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _normalized_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def acquire(path: Path, *, user_agent: str, force: bool = False) -> None:
    if force:
        path.unlink(missing_ok=True)
    download_resumable(
        CSV_URL,
        path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"FencingStatsNZ checksum changed: {actual}")


def _name_uid_crosswalk(frame: pd.DataFrame) -> dict[str, str]:
    candidates: defaultdict[str, set[str]] = defaultdict(set)
    for side in ("a", "b"):
        for name, uid in zip(frame[f"fencer_{side}"], frame[f"fencer_{side}_uid"], strict=True):
            if pd.notna(name) and pd.notna(uid):
                candidates[_normalized_name(str(name))].add(str(int(float(uid))))
    return {
        name: next(iter(values))
        for name, values in candidates.items()
        if len(values) == 1
    }


def _athlete_id(name: str, uid: Any, crosswalk: dict[str, str]) -> tuple[str, bool]:
    if pd.notna(uid):
        identity = f"uid:{int(float(uid))}"
        recovered = False
    else:
        normalized = _normalized_name(name)
        matched_uid = crosswalk.get(normalized)
        identity = f"uid:{matched_uid}" if matched_uid is not None else f"name:{normalized}"
        recovered = matched_uid is not None
    return _stable_id("fenz-athlete", identity), recovered


def prepare(source_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if source_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("FencingStatsNZ snapshot byte size changed")
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_SHA256:
        raise ValueError("FencingStatsNZ snapshot checksum changed")

    source = pd.read_csv(source_path, low_memory=False)
    if list(source.columns) != EXPECTED_COLUMNS:
        raise ValueError("FencingStatsNZ source schema changed")
    source_rows = len(source)
    exact_duplicate_rows = int(source.duplicated().sum())
    frame = source.drop_duplicates().copy()
    required = [
        "date",
        "competition",
        "weapon",
        "bout_type",
        "fencer_a",
        "fencer_b",
        "score_a",
        "score_b",
        "gender",
    ]
    complete = frame[required].notna().all(axis=1)
    missing_required_rows = int((~complete).sum())
    frame = frame.loc[complete].copy()

    allowed_weapons = {"epee", "foil", "sabre"}
    allowed_bout_types = {"pool", "de"}
    allowed_genders = {"Mens": "men", "Womens": "women"}
    if not set(frame["weapon"]).issubset(allowed_weapons):
        raise ValueError("FencingStatsNZ contains an unknown weapon")
    if not set(frame["bout_type"]).issubset(allowed_bout_types):
        raise ValueError("FencingStatsNZ contains an unknown bout type")
    if not set(frame["gender"]).issubset(allowed_genders):
        raise ValueError("FencingStatsNZ contains an unknown gender label")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    frame["score_a"] = pd.to_numeric(frame["score_a"], errors="raise").astype("int16")
    frame["score_b"] = pd.to_numeric(frame["score_b"], errors="raise").astype("int16")
    if frame[["score_a", "score_b"]].lt(0).any().any():
        raise ValueError("FencingStatsNZ contains a negative score")

    crosswalk = _name_uid_crosswalk(frame)
    recovered_uid_rows = 0
    output_rows: list[dict[str, Any]] = []
    for source_index, row in frame.iterrows():
        athlete_a, recovered_a = _athlete_id(
            str(row["fencer_a"]), row["fencer_a_uid"], crosswalk
        )
        athlete_b, recovered_b = _athlete_id(
            str(row["fencer_b"]), row["fencer_b_uid"], crosswalk
        )
        recovered_uid_rows += int(recovered_a) + int(recovered_b)
        score_a = int(row["score_a"])
        score_b = int(row["score_b"])
        flag = None if pd.isna(row["flag"]) else str(row["flag"])
        if flag == "wd_a":
            result_status = "withdrawal_a"
        elif flag == "wd_b":
            result_status = "withdrawal_b"
        elif score_a == score_b:
            result_status = "tied_anomaly"
        else:
            result_status = "decisive"
        model_eligible = result_status == "decisive"
        competition_key = f"{row['date'].date().isoformat()}|{row['competition']}"
        output_rows.append(
            {
                "bout_id": f"fenz-bout:{source_index}",
                "event_id": _stable_id("fenz-event", competition_key),
                "date": row["date"],
                "weapon": str(row["weapon"]),
                "bout_type": str(row["bout_type"]),
                "de_round": None if pd.isna(row["de_round"]) else str(row["de_round"]),
                "gender": allowed_genders[str(row["gender"])],
                "fencer_a_id": athlete_a,
                "fencer_b_id": athlete_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner_side": (
                    "a" if model_eligible and score_a > score_b else
                    "b" if model_eligible else None
                ),
                "result_status": result_status,
                "model_eligible": model_eligible,
            }
        )

    output = pd.DataFrame(output_rows).sort_values(
        ["date", "event_id", "bout_type", "bout_id"], kind="stable"
    )
    if output["bout_id"].duplicated().any():
        raise ValueError("FencingStatsNZ generated duplicate bout IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    eligible = output.loc[output["model_eligible"]]
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "license": LICENSE,
        "source_rows": source_rows,
        "exact_duplicate_rows_excluded": exact_duplicate_rows,
        "missing_required_rows_excluded": missing_required_rows,
        "retained_name_free_bouts": len(output),
        "model_eligible_decisive_bouts": len(eligible),
        "event_instances": int(output["event_id"].nunique()),
        "pseudonymous_athletes": len(
            set(output["fencer_a_id"]) | set(output["fencer_b_id"])
        ),
        "published_name_spellings": len(
            set(frame["fencer_a"].astype(str)) | set(frame["fencer_b"].astype(str))
        ),
        "published_numeric_uids": len(
            set(frame["fencer_a_uid"].dropna()) | set(frame["fencer_b_uid"].dropna())
        ),
        "missing_uid_slots_recovered_by_unambiguous_name_crosswalk": recovered_uid_rows,
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
        "output": "data/processed/fencingstatsnz-bouts.parquet",
        "claim_boundary": (
            "A pinned 2024-2026 New Zealand public-results ledger, not an official live feed or "
            "a complete population census. Names, clubs, competition titles, and source IDs are "
            "removed from the local analytic table. Withdrawal rows and unflagged tied-score "
            "anomalies are retained for audit but excluded from predictive evaluation. The code "
            "repository is MIT, while underlying FeNZ result-data reuse rights were not "
            "independently verified; source rows and derivatives therefore remain local."
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
