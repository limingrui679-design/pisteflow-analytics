"""Audit and privacy-minimize the February 2026 NAC hot-hand snapshot."""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/eax2101-droid/fencing-hot-hand-data-pipeline"
SNAPSHOT = "9e748e7ca956a893f3526b5ef8fc30402dd49de1"
ARCHIVE_URL = f"https://codeload.github.com/eax2101-droid/fencing-hot-hand-data-pipeline/tar.gz/{SNAPSHOT}"
EXPECTED_BYTES = 190_086
EXPECTED_SHA256 = "f5bf19468fc163fb6ec4cce10720be776b395e94e28a13a8a6fcb9f1265f73d9"
ASKFRED_EVENT_URL = (
    "https://www.askfred.net/events/fbd67c95-846a-4179-ad89-4aa217bbc89a/round/1"
)
USA_FENCING_EVENT_URL = "https://www.usafencing.org/2026-february-north-american-cup"
LICENSE = (
    "No license file or explicit data license was present at the pinned snapshot; "
    "source rows and privacy-minimized derivatives remain local."
)
CSV_SUFFIX = "data/processed/fencer_bout_level_data.csv"
EXPECTED_COLUMNS = [
    "tournament",
    "pool_id",
    "fencer_num",
    "fencer",
    "opponent_num",
    "opponent",
    "bout_order",
    "bout_type",
    "cell_result",
    "result",
    "touches_scored",
    "touches_allowed",
    "score_margin",
    "fencer_rating_pre",
    "opp_rating_pre",
    "strength_diff",
    "win",
    "fencer_bout_order",
    "prev_win",
    "recent_form_3",
]
CELL_PATTERN = re.compile(r"^(?P<status>[VD])(?P<score>\d+)$")


def _stable_id(pool_id: str, fencer_num: int) -> str:
    digest = hashlib.sha256(
        f"pisteflow:hot-hand:{pool_id}:{fencer_num}".encode()
    ).hexdigest()[:20]
    return f"hot-hand-athlete:{digest}"


def acquire(path: Path, *, user_agent: str, force: bool = False) -> None:
    if force:
        path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_BYTES,
    )
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"hot-hand snapshot checksum changed: {actual}")


def _source_frame(archive_path: Path) -> pd.DataFrame:
    if archive_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError("hot-hand archive byte size changed")
    if sha256_file(archive_path) != EXPECTED_SHA256:
        raise ValueError("hot-hand archive checksum changed")
    with tarfile.open(archive_path, "r:gz") as archive:
        candidates = [member for member in archive.getmembers() if member.name.endswith(CSV_SUFFIX)]
        if len(candidates) != 1:
            raise ValueError("hot-hand archive must contain exactly one bout-level CSV")
        member = candidates[0]
        if not member.isfile():
            raise ValueError("hot-hand bout-level CSV member is not a regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("hot-hand bout-level CSV could not be read")
        frame = pd.read_csv(io.BytesIO(handle.read()))
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("hot-hand source schema changed")
    return frame


def _validate_cell(row: pd.Series) -> None:
    match = CELL_PATTERN.fullmatch(str(row["cell_result"]))
    if match is None:
        raise ValueError(f"malformed hot-hand cell: {row['cell_result']}")
    win = match.group("status") == "V"
    if win != bool(row["win"]):
        raise ValueError("hot-hand cell and win fields disagree")


def _cell_score(value: Any) -> int:
    match = CELL_PATTERN.fullmatch(str(value))
    if match is None:
        raise ValueError(f"malformed hot-hand cell: {value}")
    return int(match.group("score"))


def _optional_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def prepare(archive_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    source = _source_frame(archive_path)
    if source.duplicated().any():
        raise ValueError("hot-hand source contains exact duplicate rows")
    source.apply(_validate_cell, axis=1)
    source["parsed_score"] = source["cell_result"].map(_cell_score)
    upstream_touch_score_errors = int(
        (source["parsed_score"] != source["touches_scored"]).sum()
    )
    source["pair_key"] = source.apply(
        lambda row: (
            str(row["pool_id"]),
            min(int(row["fencer_num"]), int(row["opponent_num"])),
            max(int(row["fencer_num"]), int(row["opponent_num"])),
        ),
        axis=1,
    )
    group_sizes = source.groupby("pair_key", sort=True).size()
    if not group_sizes.eq(2).all():
        raise ValueError("every hot-hand bout must have two mirrored perspective rows")

    rows: list[dict[str, Any]] = []
    upstream_winner_margin_errors = 0
    for (pool_id, left_num, right_num), pair in source.groupby("pair_key", sort=True):
        left = pair.loc[pair["fencer_num"] == left_num]
        right = pair.loc[pair["fencer_num"] == right_num]
        if len(left) != 1 or len(right) != 1:
            raise ValueError("hot-hand pair orientation is not unique")
        left_row = left.iloc[0]
        right_row = right.iloc[0]
        if int(left_row["opponent_num"]) != right_num or int(right_row["opponent_num"]) != left_num:
            raise ValueError("hot-hand mirrored opponent IDs disagree")
        if int(left_row["bout_order"]) != int(right_row["bout_order"]):
            raise ValueError("hot-hand mirrored bout order disagrees")
        if bool(left_row["win"]) == bool(right_row["win"]):
            raise ValueError("hot-hand mirrored win flags are not complementary")
        if str(left_row["opponent"]) != str(right_row["fencer"]):
            raise ValueError("hot-hand mirrored participant names disagree")
        if str(right_row["opponent"]) != str(left_row["fencer"]):
            raise ValueError("hot-hand mirrored participant names disagree")
        if (
            pd.notna(left_row["fencer_rating_pre"])
            and pd.notna(right_row["opp_rating_pre"])
            and float(left_row["fencer_rating_pre"]) != float(right_row["opp_rating_pre"])
        ):
            raise ValueError("hot-hand mirrored pre-event ratings disagree")
        if (
            pd.notna(right_row["fencer_rating_pre"])
            and pd.notna(left_row["opp_rating_pre"])
            and float(right_row["fencer_rating_pre"]) != float(left_row["opp_rating_pre"])
        ):
            raise ValueError("hot-hand mirrored pre-event ratings disagree")
        score_left = int(left_row["parsed_score"])
        score_right = int(right_row["parsed_score"])
        if int(left_row["score_margin"]) != score_left - score_right:
            upstream_winner_margin_errors += 1
        if int(right_row["score_margin"]) != score_right - score_left:
            upstream_winner_margin_errors += 1
        rows.append(
            {
                "bout_id": f"hot-hand:{pool_id}:{left_num}:{right_num}",
                "event_id": "february-nac-2026-junior-mens-foil",
                "event_date_start": "2026-02-12",
                "event_date_end": "2026-02-15",
                "weapon": "foil",
                "gender": "men",
                "category": "junior",
                "pool_id": str(pool_id),
                "bout_order": int(left_row["bout_order"]),
                "competitor_a_id": _stable_id(str(pool_id), left_num),
                "competitor_b_id": _stable_id(str(pool_id), right_num),
                "score_a": score_left,
                "score_b": score_right,
                "winner_side": "a" if bool(left_row["win"]) else "b",
                "rating_a_pre": _optional_float(left_row["fencer_rating_pre"]),
                "rating_b_pre": _optional_float(right_row["fencer_rating_pre"]),
                "prev_win_a": _optional_float(left_row["prev_win"]),
                "prev_win_b": _optional_float(right_row["prev_win"]),
                "recent_form_3_a": _optional_float(left_row["recent_form_3"]),
                "recent_form_3_b": _optional_float(right_row["recent_form_3"]),
            }
        )

    output = pd.DataFrame(rows).sort_values(["pool_id", "bout_order", "bout_id"], kind="stable")
    if output["bout_id"].duplicated().any():
        raise ValueError("hot-hand preparation generated duplicate bout IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    source_names = set(source["fencer"].astype(str)) | set(source["opponent"].astype(str))
    payload = output_path.read_bytes().decode("latin1", errors="ignore")
    if any(name in payload for name in source_names):
        raise ValueError("hot-hand identity minimization failed")

    complete_ratings = output[["rating_a_pre", "rating_b_pre"]].notna().all(axis=1)
    complete_prev = output[["prev_win_a", "prev_win_b"]].notna().all(axis=1)
    complete_recent = output[["recent_form_3_a", "recent_form_3_b"]].notna().all(axis=1)
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "askfred_event_url": ASKFRED_EVENT_URL,
        "usa_fencing_event_url": USA_FENCING_EVENT_URL,
        "license": LICENSE,
        "source_perspective_rows": len(source),
        "retained_unique_bouts": len(output),
        "pools": int(output["pool_id"].nunique()),
        "pseudonymous_athletes": len(
            set(output["competitor_a_id"]) | set(output["competitor_b_id"])
        ),
        "weapon": "foil",
        "gender": "men",
        "category": "junior",
        "event_date_range": ["2026-02-12", "2026-02-15"],
        "bouts_with_complete_pre_event_ratings": int(complete_ratings.sum()),
        "bouts_with_complete_previous_outcomes": int(complete_prev.sum()),
        "bouts_with_complete_three_bout_form": int(complete_recent.sum()),
        "upstream_touches_scored_rows_corrected": upstream_touch_score_errors,
        "upstream_winner_side_score_margin_rows_corrected": upstream_winner_margin_errors,
        "source_sha256": sha256_file(archive_path),
        "output_sha256": sha256_file(output_path),
        "output": "data/processed/february-nac-2026-hot-hand-bouts.parquet",
        "claim_boundary": (
            "A pinned, independently mirrored-row-validated subset of 273 pool bouts from 13 "
            "pools in the 2026 February NAC Junior Men's Foil event. AskFRED independently "
            "confirms the event, weapon, category, and published pool grid; USA Fencing confirms "
            "the tournament date range. The source contains every bout twice, so PisteFlow "
            "reconstructs one bout from the complementary cells and never counts 546 rows as "
            "546 independent bouts. Published names are replaced with pool-scoped pseudonyms. "
            "The upstream winner-perspective score fields assume five touches in one V3 bout, "
            "and winner-side score margins generally assume five touches allowed; both are "
            "corrected from the compact cells and their mirrored counterparts. This one-event "
            "observational snapshot "
            "cannot establish a causal hot-hand effect or broad athlete generalization."
        ),
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
    acquire(archive_path, user_agent=user_agent, force=force)
    return prepare(archive_path, output_path, report_path)
