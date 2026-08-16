"""Privacy-minimized parser for the frozen AskFRED v1 public-API archive."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pisteflow.provenance import canonical_json_bytes, sha256_file, write_json

SOURCE_URL = "https://github.com/boscojwho/askFRED-v1-archive"
SNAPSHOT = "9305ccb90af040e8a867a4c418b0b358948742db"
LICENSE = (
    "No repository or underlying data license. The archivist expressly disclaims ownership; "
    "source files and normalized row-level derivatives remain local."
)
EXPECTED_ENDPOINT_TOTALS = {
    "event": 165_564,
    "roundresult": 192_939,
    "tournament": 42_043,
}

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("bout_id", pa.string(), nullable=False),
        pa.field("round_id", pa.string(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("tournament_id", pa.string(), nullable=False),
        pa.field("date", pa.string()),
        pa.field("weapon", pa.string(), nullable=False),
        pa.field("event_gender", pa.string(), nullable=False),
        pa.field("event_age_limit", pa.string()),
        pa.field("event_rating_limit", pa.string()),
        pa.field("event_entries", pa.int32()),
        pa.field("is_team_event", pa.bool_(), nullable=False),
        pa.field("round_type", pa.string(), nullable=False),
        pa.field("round_sequence", pa.int32()),
        pa.field("table_size", pa.int32()),
        pa.field("container_index", pa.int32(), nullable=False),
        pa.field("bout_sequence", pa.int32()),
        pa.field("left_competitor_id", pa.string()),
        pa.field("right_competitor_id", pa.string()),
        pa.field("left_gender", pa.string()),
        pa.field("right_gender", pa.string()),
        pa.field("left_seed", pa.int32()),
        pa.field("right_seed", pa.int32()),
        pa.field("left_score", pa.int32()),
        pa.field("right_score", pa.int32()),
        pa.field("winner_side", pa.string()),
        pa.field("result_status", pa.string(), nullable=False),
        pa.field("outcome_eligible", pa.bool_(), nullable=False),
        pa.field("chronology_eligible", pa.bool_(), nullable=False),
    ]
)


def repository_snapshot(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot = result.stdout.strip()
    if snapshot != SNAPSHOT:
        raise ValueError(f"AskFRED v1 checkout changed: {snapshot} != {SNAPSHOT}")
    return snapshot


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _fingerprint(value: Any) -> bytes:
    return hashlib.blake2b(canonical_json_bytes(value), digest_size=16).digest()


def _normalize_weapon(value: Any) -> str:
    normalized = str(value).strip().casefold()
    return {
        "epee": "epee",
        "épée": "epee",
        "foil": "foil",
        "saber": "sabre",
        "sabre": "sabre",
    }.get(normalized, "unknown")


def _normalize_event_gender(value: Any) -> str:
    return {
        "men": "men",
        "women": "women",
        "mixed": "mixed",
    }.get(str(value).strip().casefold(), "unknown")


def _normalize_person_gender(value: Any, event_gender: str) -> str | None:
    if event_gender in {"men", "women"}:
        return event_gender
    return {"M": "men", "F": "women"}.get(str(value).strip().upper())


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"AskFRED endpoint directory is missing: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"AskFRED endpoint directory is empty: {directory}")
    return files


def _load_endpoint_map(
    directory: Path,
    collection_key: str,
    transform: Any,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    fingerprints: dict[int, bytes] = {}
    occurrences = 0
    duplicate_records = 0
    conflicting_records = 0
    malformed_files: list[str] = []
    advertised_totals: Counter[int] = Counter()
    page_sizes: Counter[int] = Counter()
    files = _page_files(directory)
    source_bytes = 0
    for path in files:
        source_bytes += path.stat().st_size
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_files.append(path.name)
            continue
        if not isinstance(payload, Mapping):
            malformed_files.append(path.name)
            continue
        total = _optional_int(payload.get("total_matched"))
        per_page = _optional_int(payload.get("per_page"))
        if total is not None:
            advertised_totals[total] += 1
        if per_page is not None:
            page_sizes[per_page] += 1
        rows = payload.get(collection_key)
        if not isinstance(rows, list):
            malformed_files.append(path.name)
            continue
        for raw in rows:
            if not isinstance(raw, Mapping) or _optional_int(raw.get("id")) is None:
                raise ValueError(f"Malformed AskFRED {collection_key} record in {path.name}")
            occurrences += 1
            record_id = int(raw["id"])
            fingerprint = _fingerprint(raw)
            if record_id in fingerprints:
                duplicate_records += 1
                if fingerprints[record_id] != fingerprint:
                    conflicting_records += 1
                continue
            fingerprints[record_id] = fingerprint
            records[record_id] = transform(raw)
    if conflicting_records:
        raise ValueError(f"AskFRED {collection_key} repeated IDs have conflicting payloads")
    audit = {
        "files": len(files),
        "source_bytes": source_bytes,
        "malformed_files": len(malformed_files),
        "malformed_file_examples": malformed_files[:10],
        "record_occurrences": occurrences,
        "unique_records": len(records),
        "duplicate_identical_records": duplicate_records,
        "conflicting_records": conflicting_records,
        "advertised_totals": {
            str(key): int(value) for key, value in sorted(advertised_totals.items())
        },
        "page_sizes": {str(key): int(value) for key, value in sorted(page_sizes.items())},
    }
    return records, audit


def _event_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tournament_id": _optional_int(raw.get("tournament_id")),
        "weapon": _normalize_weapon(raw.get("weapon")),
        "gender": _normalize_event_gender(raw.get("gender")),
        "age_limit": str(raw["age_limit"]) if raw.get("age_limit") is not None else None,
        "rating_limit": (
            str(raw["rating_limit"]) if raw.get("rating_limit") is not None else None
        ),
        "entries": _optional_int(raw.get("entries")),
        "is_team": bool(raw.get("is_team", False)),
    }


def _tournament_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    start_date = raw.get("start_date")
    return {"start_date": str(start_date) if start_date else None}


def _rounds_from_page(path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("rounds"), list):
        raise ValueError(f"Malformed AskFRED round page: {path.name}")
    rounds = payload["rounds"]
    if not all(isinstance(row, Mapping) for row in rounds):
        raise ValueError(f"Malformed AskFRED round record: {path.name}")
    return payload, rounds


def _competitor_identity(
    person: Mapping[str, Any],
    cache: dict[tuple[str, str], str],
) -> tuple[str, str]:
    source_id = _optional_int(person.get("id"))
    usfa_id = str(person.get("usfa_id", "")).strip()
    first_name = str(person.get("first_name", "")).strip()
    last_name = str(person.get("last_name", "")).strip()
    if source_id is not None:
        provenance = "source_id"
        identity = str(source_id)
    elif usfa_id:
        provenance = "federation_id"
        identity = usfa_id
    else:
        provenance = "name_only"
        identity = " ".join(f"{first_name} {last_name}".casefold().split())
    key = provenance, identity
    pseudonym = cache.get(key)
    if pseudonym is None:
        pseudonym = _stable_id("askfred-competitor", f"{provenance}:{identity}")
        cache[key] = pseudonym
    return pseudonym, provenance


def _iter_bouts(round_record: Mapping[str, Any]) -> Iterator[tuple[int, int | None, Any]]:
    if isinstance(round_record.get("pools"), list):
        containers = round_record["pools"]
    elif isinstance(round_record.get("de_tables"), list):
        containers = round_record["de_tables"]
    else:
        containers = []
    for container_index, container in enumerate(containers, start=1):
        if not isinstance(container, Mapping) or not isinstance(container.get("bouts"), list):
            continue
        table_size = _optional_int(container.get("of"))
        for bout in container["bouts"]:
            if not isinstance(bout, Mapping):
                continue
            yield container_index, table_size, bout


def prepare(
    checkout: Path,
    output_path: Path,
    report_path: Path,
    *,
    repository_revision: str,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    if repository_revision != SNAPSHOT:
        raise ValueError(f"AskFRED v1 revision changed: {repository_revision} != {SNAPSHOT}")
    endpoint_root = checkout / "data/endpoints"
    events, event_audit = _load_endpoint_map(
        endpoint_root / "event", "events", _event_record
    )
    tournaments, tournament_audit = _load_endpoint_map(
        endpoint_root / "tournament", "tournaments", _tournament_record
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary_output.unlink(missing_ok=True)
    writer = pq.ParquetWriter(  # type: ignore[no-untyped-call]
        temporary_output,
        OUTPUT_SCHEMA,
        compression="zstd",
        use_dictionary=True,
    )
    batch: list[dict[str, Any]] = []

    round_files = _page_files(endpoint_root / "roundresult")
    round_fingerprints: dict[int, bytes] = {}
    bout_ids: set[int] = set()
    competitor_cache: dict[tuple[str, str], str] = {}
    round_occurrences = 0
    duplicate_rounds = 0
    round_conflicts = 0
    bout_occurrences = 0
    round_types: Counter[str] = Counter()
    competitor_counts: Counter[int] = Counter()
    identity_slot_provenance: Counter[str] = Counter()
    unique_identity_provenance: Counter[str] = Counter()
    competitor_statuses: Counter[str] = Counter()
    result_statuses: Counter[str] = Counter()
    bouts_by_weapon: Counter[str] = Counter()
    eligible_by_weapon: Counter[str] = Counter()
    bouts_by_round_type: Counter[str] = Counter()
    team_bouts = 0
    outcome_eligible = 0
    chronology_eligible = 0
    event_mapped_rounds = 0
    dated_rounds = 0
    round_event_ids: set[int] = set()
    round_tournament_ids: set[int] = set()
    advertised_round_totals: Counter[int] = Counter()
    round_page_sizes: Counter[int] = Counter()
    source_bytes = 0

    def flush() -> None:
        if not batch:
            return
        writer.write_table(  # type: ignore[no-untyped-call]
            pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA)
        )
        batch.clear()

    try:
        for path in round_files:
            source_bytes += path.stat().st_size
            payload, rounds = _rounds_from_page(path)
            total = _optional_int(payload.get("total_matched"))
            per_page = _optional_int(payload.get("per_page"))
            if total is not None:
                advertised_round_totals[total] += 1
            if per_page is not None:
                round_page_sizes[per_page] += 1
            for round_record in rounds:
                round_occurrences += 1
                round_id = _optional_int(round_record.get("id"))
                event_id = _optional_int(round_record.get("event_id"))
                tournament_id = _optional_int(round_record.get("tournament_id"))
                if round_id is None or event_id is None or tournament_id is None:
                    raise ValueError(f"AskFRED round is missing stable IDs in {path.name}")
                fingerprint = _fingerprint(round_record)
                if round_id in round_fingerprints:
                    duplicate_rounds += 1
                    if round_fingerprints[round_id] != fingerprint:
                        round_conflicts += 1
                    continue
                round_fingerprints[round_id] = fingerprint
                round_event_ids.add(event_id)
                round_tournament_ids.add(tournament_id)
                event = events.get(event_id)
                tournament = tournaments.get(tournament_id)
                if event is not None:
                    event_mapped_rounds += 1
                if tournament is not None and tournament.get("start_date"):
                    dated_rounds += 1
                weapon = str(event["weapon"]) if event is not None else "unknown"
                event_gender = str(event["gender"]) if event is not None else "unknown"
                is_team = bool(event["is_team"]) if event is not None else False
                date = str(tournament["start_date"]) if tournament else None
                round_type = str(round_record.get("round_type", "unknown"))
                round_types[round_type] += 1
                for container_index, table_size, bout in _iter_bouts(round_record):
                    bout_occurrences += 1
                    bout_id = _optional_int(bout.get("id"))
                    if bout_id is None:
                        raise ValueError(f"AskFRED bout has no stable ID in round {round_id}")
                    if bout_id in bout_ids:
                        raise ValueError(f"AskFRED bout ID repeats across unique rounds: {bout_id}")
                    bout_ids.add(bout_id)
                    people = bout.get("fencers")
                    if not isinstance(people, list) or not all(
                        isinstance(person, Mapping) for person in people
                    ):
                        people = []
                    competitor_counts[len(people)] += 1
                    if len(people) > 2:
                        raise ValueError(f"AskFRED bout has more than two competitors: {bout_id}")
                    person_values: list[dict[str, Any]] = []
                    for person in people:
                        person_id, provenance = _competitor_identity(person, competitor_cache)
                        identity_slot_provenance[provenance] += 1
                        competitor_statuses[str(person.get("status", "")).upper()] += 1
                        person_values.append(
                            {
                                "id": person_id,
                                "provenance": provenance,
                                "gender": _normalize_person_gender(
                                    person.get("gender"), event_gender
                                ),
                                "seed": _optional_int(person.get("seed")),
                                "score": _optional_int(person.get("score")),
                                "status": str(person.get("status", "")).upper(),
                            }
                        )
                    if len(person_values) == 0:
                        result_status = "empty_placeholder"
                        winner_side = None
                    elif len(person_values) == 1:
                        result_status = "bye_or_single_entry"
                        winner_side = "left" if person_values[0]["status"] == "V" else None
                    else:
                        pair_left, pair_right = person_values
                        left_score = pair_left["score"]
                        right_score = pair_right["score"]
                        statuses = [pair_left["status"], pair_right["status"]]
                        if left_score is None or right_score is None:
                            result_status = "missing_score"
                            winner_side = (
                                "left" if statuses == ["V", "D"] else
                                "right" if statuses == ["D", "V"] else None
                            )
                        elif left_score == right_score:
                            result_status = "tied_score"
                            winner_side = (
                                "left" if statuses == ["V", "D"] else
                                "right" if statuses == ["D", "V"] else None
                            )
                        elif statuses.count("V") == 1 and statuses.count("D") == 1:
                            score_winner = "left" if left_score > right_score else "right"
                            status_winner = "left" if statuses[0] == "V" else "right"
                            winner_side = status_winner
                            result_status = (
                                "decisive_consistent"
                                if score_winner == status_winner
                                else "score_status_conflict"
                            )
                        else:
                            result_status = "nonstandard_status"
                            winner_side = None
                    eligible = (
                        len(person_values) == 2
                        and result_status == "decisive_consistent"
                        and weapon in {"epee", "foil", "sabre"}
                        and not is_team
                    )
                    chronological = eligible and date is not None
                    result_statuses[result_status] += 1
                    bouts_by_weapon[weapon] += 1
                    bouts_by_round_type[round_type] += 1
                    team_bouts += int(is_team)
                    outcome_eligible += int(eligible)
                    chronology_eligible += int(chronological)
                    if eligible:
                        eligible_by_weapon[weapon] += 1
                    left: dict[str, Any] | None = person_values[0] if person_values else None
                    right: dict[str, Any] | None = (
                        person_values[1] if len(person_values) == 2 else None
                    )
                    batch.append(
                        {
                            "bout_id": _stable_id("askfred-bout", str(bout_id)),
                            "round_id": _stable_id("askfred-round", str(round_id)),
                            "event_id": _stable_id("askfred-event", str(event_id)),
                            "tournament_id": _stable_id(
                                "askfred-tournament", str(tournament_id)
                            ),
                            "date": date,
                            "weapon": weapon,
                            "event_gender": event_gender,
                            "event_age_limit": event.get("age_limit") if event else None,
                            "event_rating_limit": event.get("rating_limit") if event else None,
                            "event_entries": event.get("entries") if event else None,
                            "is_team_event": is_team,
                            "round_type": round_type,
                            "round_sequence": _optional_int(round_record.get("seq")),
                            "table_size": table_size,
                            "container_index": container_index,
                            "bout_sequence": _optional_int(bout.get("seq")),
                            "left_competitor_id": left["id"] if left else None,
                            "right_competitor_id": right["id"] if right else None,
                            "left_gender": left["gender"] if left else None,
                            "right_gender": right["gender"] if right else None,
                            "left_seed": left["seed"] if left else None,
                            "right_seed": right["seed"] if right else None,
                            "left_score": left["score"] if left else None,
                            "right_score": right["score"] if right else None,
                            "winner_side": winner_side,
                            "result_status": result_status,
                            "outcome_eligible": eligible,
                            "chronology_eligible": chronological,
                        }
                    )
                    if len(batch) >= batch_size:
                        flush()
        flush()
    finally:
        writer.close()  # type: ignore[no-untyped-call]
    if round_conflicts:
        temporary_output.unlink(missing_ok=True)
        raise ValueError("AskFRED repeated round IDs have conflicting payloads")
    temporary_output.replace(output_path)

    for provenance, _ in competitor_cache:
        unique_identity_provenance[provenance] += 1
    source_file_bytes = (
        source_bytes + int(event_audit["source_bytes"]) + int(tournament_audit["source_bytes"])
    )
    expected_round_total = EXPECTED_ENDPOINT_TOTALS["roundresult"]
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": repository_revision,
        "license": LICENSE,
        "source_files": (
            len(round_files) + int(event_audit["files"]) + int(tournament_audit["files"])
        ),
        "source_file_bytes": source_file_bytes,
        "endpoint_audits": {
            "event": event_audit,
            "tournament": tournament_audit,
            "roundresult": {
                "files": len(round_files),
                "source_bytes": source_bytes,
                "advertised_totals": {
                    str(key): int(value)
                    for key, value in sorted(advertised_round_totals.items())
                },
                "page_sizes": {
                    str(key): int(value) for key, value in sorted(round_page_sizes.items())
                },
                "round_occurrences": round_occurrences,
                "unique_rounds": len(round_fingerprints),
                "duplicate_identical_rounds": duplicate_rounds,
                "conflicting_rounds": round_conflicts,
                "unavailable_rounds_against_advertised_total": (
                    expected_round_total - len(round_fingerprints)
                ),
            },
        },
        "unique_bout_records": len(bout_ids),
        "bout_occurrences": bout_occurrences,
        "competitor_slots_per_bout": {
            str(key): int(value) for key, value in sorted(competitor_counts.items())
        },
        "unique_pseudonymous_competitors": len(competitor_cache),
        "identity_slots_by_provenance": dict(sorted(identity_slot_provenance.items())),
        "unique_identities_by_provenance": dict(sorted(unique_identity_provenance.items())),
        "competitor_status_counts": dict(sorted(competitor_statuses.items())),
        "result_status_counts": dict(sorted(result_statuses.items())),
        "round_type_counts": dict(sorted(round_types.items())),
        "bouts_by_round_type": dict(sorted(bouts_by_round_type.items())),
        "bouts_by_weapon": dict(sorted(bouts_by_weapon.items())),
        "outcome_eligible_bouts_by_weapon": dict(sorted(eligible_by_weapon.items())),
        "team_bout_records": team_bouts,
        "outcome_eligible_individual_bouts": outcome_eligible,
        "chronology_eligible_individual_bouts": chronology_eligible,
        "rounds_with_event_metadata": event_mapped_rounds,
        "rounds_with_tournament_date": dated_rounds,
        "round_subset_event_ids": len(round_event_ids),
        "round_subset_tournament_ids": len(round_tournament_ids),
        "output": "data/processed/askfred-v1-round-bouts.parquet",
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "grouping_rule": (
            "Hold out complete tournaments or competitors. All rounds and bouts from one event "
            "remain in one split; presentation side must be randomized inside each fold."
        ),
        "claim_boundary": (
            "A frozen, incomplete archive of the deprecated AskFRED v1 public API. The archive "
            "advertises 192,939 round objects but contains 192,533 unique round IDs after "
            "identical pagination overlap is removed; missing pages are not reconstructed. Every "
            "source bout ID is counted once. Names, federation IDs, birth years, clubs, venues, "
            "addresses, comments, fees, and raw source IDs are absent from the normalized table. "
            "Byes, empty bracket placeholders, tied scores, nonstandard statuses, team events, "
            "unknown weapons, and score/status conflicts remain auditable but are excluded from "
            "individual outcome training. This is a historical community archive, not an official "
            "live USA Fencing feed, a complete participation census, or permission to republish "
            "the underlying rows."
        ),
    }
    write_json(report_path, summary)
    return summary
