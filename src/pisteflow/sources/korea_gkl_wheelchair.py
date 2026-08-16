"""Official Korean wheelchair-fencing event and name-minimized result rows."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

SOURCE_PAGE = "https://www.data.go.kr/data/15048179/fileData.do"
DOWNLOAD_URL = (
    "https://www.data.go.kr/cmm/cmm/fileDownload.do?"
    "atchFileId=FILE_000000003649293&fileDetailSn=1&insertDataPrcus=N"
)
LICENSE = "Korean Public Data Portal: unrestricted use scope"
SNAPSHOT_DATE = "2026-03-22"
EXPECTED_BYTES = 13_682
EXPECTED_SHA256 = "d629406c917bf985918380f8dfda30b5bc1682539def807f5a7ae5543079da4d"

SOURCE_COLUMNS = ["대회일자", "참가대회", "순서", "종목", "성적"]
_WEAPON_PATTERN = r"(?:플러뢰|플뢰레|플러레|에\s*뻬|에빼|에뻬|에베|에페|샤브르|사브르)"
_ATHLETE_PATTERN = re.compile(
    rf"(?P<name>[가-힣]{{3}})\s*(?P<class>[ABC])?\s*(?=\[?\s*{_WEAPON_PATTERN})"
)
_INDIVIDUAL_RESULT_PATTERN = re.compile(
    rf"\[?\s*(?P<weapon>{_WEAPON_PATTERN})\s*(?P<class>[ABC])?\s*\]?"
    r"\s*(?P<value>\d+)\s*(?P<unit>위|강)"
)
_TEAM_RESULT_PATTERN = re.compile(
    rf"[\[\s]*(?P<weapon>{_WEAPON_PATTERN})\s*\]?"
    r"(?:\s*\([^)]*\))?\s*(?P<value>\d+)\s*(?P<unit>위|강)"
)
_WEAPON_MAP = {
    "플러뢰": "foil",
    "플뢰레": "foil",
    "플러레": "foil",
    "에 뻬": "epee",
    "에빼": "epee",
    "에뻬": "epee",
    "에베": "epee",
    "에페": "epee",
    "샤브르": "sabre",
    "사브르": "sabre",
}


def _athlete_code(name: str) -> str:
    digest = sha256(f"korea-gkl-wheelchair-fencing:{name}".encode()).hexdigest()[:18]
    return f"kgw-{digest}"


def _acquire_snapshot(raw_path: Path, *, user_agent: str) -> None:
    """Use curl for this portal because its TLS handshake stalls in Python's SSL stack."""

    if raw_path.exists() and raw_path.stat().st_size == EXPECTED_BYTES:
        return
    executable = shutil.which("curl")
    if executable is None:
        raise RuntimeError("curl is required to download this Korean public-data snapshot")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(raw_path.suffix + ".part")
    subprocess.run(
        [
            executable,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "4",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "--user-agent",
            user_agent,
            "--output",
            str(temporary),
            DOWNLOAD_URL,
        ],
        check=True,
    )
    if temporary.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            "Korean wheelchair-fencing CSV size changed: "
            f"{temporary.stat().st_size} != {EXPECTED_BYTES}"
        )
    temporary.replace(raw_path)


def _event_code(order: int, date_range: str, competition: str) -> str:
    digest = sha256(f"{order}|{date_range}|{competition}".encode()).hexdigest()[:12]
    return f"kgw-event-{order:03d}-{digest}"


def _normalized_weapon(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    weapon = _WEAPON_MAP.get(normalized)
    if weapon is None:
        raise ValueError(f"Unknown Korean weapon spelling: {value!r}")
    return weapon


def _placement(unit: str, value: int) -> tuple[str, str]:
    if unit == "위":
        return "exact_rank", f"rank_{value}"
    if unit == "강":
        return "round_reached", f"last_{value}"
    raise ValueError(f"Unknown Korean placement unit: {unit!r}")


def _parse_date_range(value: str) -> tuple[str, str]:
    parts = value.split("~")
    if len(parts) != 2:
        raise ValueError(f"Unexpected Korean event date range: {value!r}")
    parsed = [pd.Timestamp(part).date().isoformat() for part in parts]
    if parsed[1] < parsed[0]:
        raise ValueError(f"Korean event ends before it starts: {value!r}")
    return parsed[0], parsed[1]


def _result_row(
    *,
    event_code: str,
    event_order: int,
    scope: str,
    athlete_code: str | None,
    weapon: str,
    wheelchair_class: str | None,
    value: int,
    unit: str,
) -> dict[str, Any]:
    placement_type, notation = _placement(unit, value)
    identity = athlete_code or "team"
    digest = sha256(
        f"{event_code}|{scope}|{identity}|{weapon}|{wheelchair_class}|{notation}".encode()
    ).hexdigest()[:18]
    return {
        "result_code": f"kgw-result-{digest}",
        "event_code": event_code,
        "event_order": event_order,
        "scope": scope,
        "athlete_code": athlete_code,
        "weapon": weapon,
        "wheelchair_class": wheelchair_class,
        "placement_type": placement_type,
        "placement_value": value,
        "placement_notation": notation,
        "source_page": SOURCE_PAGE,
    }


def _parse_results(text: str, event_code: str, event_order: int) -> list[dict[str, Any]]:
    normalized = text.replace("\uff3b", "[").replace("\uff3d", "]")
    sections = re.split(r"\*?\s*단체전", normalized, maxsplit=1)
    individual = sections[0]
    markers = list(_ATHLETE_PATTERN.finditer(individual))
    rows: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(individual)
        segment = individual[marker.end() : stop]
        athlete_code = _athlete_code(marker.group("name"))
        default_class = marker.group("class")
        for match in _INDIVIDUAL_RESULT_PATTERN.finditer(segment):
            rows.append(
                _result_row(
                    event_code=event_code,
                    event_order=event_order,
                    scope="individual",
                    athlete_code=athlete_code,
                    weapon=_normalized_weapon(match.group("weapon")),
                    wheelchair_class=match.group("class") or default_class,
                    value=int(match.group("value")),
                    unit=match.group("unit"),
                )
            )
    if len(sections) == 2:
        for match in _TEAM_RESULT_PATTERN.finditer(sections[1]):
            rows.append(
                _result_row(
                    event_code=event_code,
                    event_order=event_order,
                    scope="team",
                    athlete_code=None,
                    weapon=_normalized_weapon(match.group("weapon")),
                    wheelchair_class=None,
                    value=int(match.group("value")),
                    unit=match.group("unit"),
                )
            )
    return rows


def normalize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an event table and a direct-name-free athlete/team result table."""

    if list(frame.columns) != SOURCE_COLUMNS:
        raise ValueError("Korean wheelchair-fencing source columns changed")
    order = pd.to_numeric(frame["순서"], errors="raise").astype(int)
    if order.duplicated().any():
        raise ValueError("Korean wheelchair-fencing event orders are duplicated")
    if not frame["종목"].eq("펜싱").all():
        raise ValueError("Korean wheelchair-fencing source includes another sport")

    events: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        event_order = int(order.iloc[position])
        date_range = str(row["대회일자"]).strip()
        competition = str(row["참가대회"]).strip()
        result_text = str(row["성적"]).strip()
        event_start, event_end = _parse_date_range(date_range)
        code = _event_code(event_order, date_range, competition)
        events.append(
            {
                "event_code": code,
                "event_order": event_order,
                "event_start": event_start,
                "event_end": event_end,
                "event_name_local": competition,
                "activity_kind": (
                    "training"
                    if "전지훈련" in competition or "훈련" in result_text
                    else "competition"
                ),
                "source_page": SOURCE_PAGE,
            }
        )
        results.extend(_parse_results(result_text, code, event_order))

    event_frame = pd.DataFrame(events).sort_values("event_order").reset_index(drop=True)
    result_frame = pd.DataFrame(results).sort_values(
        ["event_order", "scope", "weapon", "result_code"]
    ).reset_index(drop=True)
    if event_frame["event_code"].duplicated().any():
        raise ValueError("Korean wheelchair-fencing event codes are duplicated")
    if not result_frame.empty and result_frame["result_code"].duplicated().any():
        raise ValueError("Korean wheelchair-fencing result codes are duplicated")
    return event_frame, result_frame


def normalize_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(path, encoding="cp949")
    if source.shape != (77, 5):
        raise ValueError(f"Korean wheelchair-fencing snapshot changed: {source.shape}")
    events, results = normalize_frame(source)
    if set(events["event_order"]) != set(range(1, 78)):
        raise ValueError("Korean wheelchair-fencing event sequence changed")
    if len(results) != 458:
        raise ValueError(f"Korean wheelchair-fencing parsed-result count changed: {len(results)}")
    if results["scope"].value_counts().to_dict() != {"individual": 425, "team": 33}:
        raise ValueError("Korean wheelchair-fencing result-scope counts changed")
    if results["weapon"].value_counts().to_dict() != {
        "foil": 171,
        "epee": 159,
        "sabre": 128,
    }:
        raise ValueError("Korean wheelchair-fencing weapon counts changed")
    return events, results


def fetch_and_prepare(
    raw_path: Path,
    event_output_path: Path,
    result_output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the official CSV and write traceable, direct-name-free tables."""

    if force:
        raw_path.unlink(missing_ok=True)
    _acquire_snapshot(raw_path, user_agent=user_agent)
    if sha256_file(raw_path) != EXPECTED_SHA256:
        raise ValueError("Korean wheelchair-fencing CSV checksum changed")
    events, results = normalize_snapshot(raw_path)
    event_output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(event_output_path, index=False)
    results.to_parquet(result_output_path, index=False)
    report: dict[str, Any] = {
        "claim_boundary": (
            "Official public records for one Korean corporate wheelchair-fencing team, not a "
            "national census or a complete competition archive. The snapshot has 77 event or "
            "training entries from 2016-2026 and 458 parsable athlete-weapon or team-weapon "
            "placements. Athlete names are replaced by stable project-local codes. Placements "
            "are not bouts, scores, touches, independent athletes, or causal training outcomes. "
            "The portal description says 2016-2024, but the downloaded file itself includes "
            "2025 and 2026 rows; the file is treated as authoritative for row coverage."
        ),
        "source_page": SOURCE_PAGE,
        "snapshot_date_in_title": SNAPSHOT_DATE,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "event_rows": len(events),
        "event_date_range": [events["event_start"].min(), events["event_end"].max()],
        "event_rows_by_activity_kind": dict(Counter(events["activity_kind"])),
        "parsed_result_rows": len(results),
        "parsed_result_rows_by_scope": dict(Counter(results["scope"])),
        "parsed_result_rows_by_weapon": dict(Counter(results["weapon"])),
        "parsed_result_rows_by_placement_type": dict(
            Counter(results["placement_type"])
        ),
        "pseudonymous_athlete_codes": int(results["athlete_code"].dropna().nunique()),
        "direct_athlete_names_removed": True,
        "raw_sha256": sha256_file(raw_path),
        "event_output": str(event_output_path),
        "event_output_sha256": sha256_file(event_output_path),
        "result_output": str(result_output_path),
        "result_output_sha256": sha256_file(result_output_path),
    }
    write_json(report_path, report)
    return report
