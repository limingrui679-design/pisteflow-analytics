"""CC0 Wikidata registry of humans explicitly connected to sport fencing."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from pisteflow.provenance import atomic_write_bytes, sha256_file, write_json, write_jsonl
from pisteflow.sources.classification import infer_weapon_candidates

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
ENTITY_API = "https://www.wikidata.org/w/api.php"
FENCER_OCCUPATION = "Q13381863"
FOIL_FENCER_OCCUPATION = "Q88191276"
EPEE_FENCER_OCCUPATION = "Q88207871"
SABRE_FENCER_OCCUPATION = "Q117536057"
WHEELCHAIR_FENCER_OCCUPATION = "Q62056391"
FENCING_SPORT = "Q12100"
HUMAN = "Q5"
SOURCE_PAGE = "https://www.wikidata.org/wiki/Q13381863"
LICENSE = "CC0 1.0"
_ENTITY_PATTERN = re.compile(r"/entity/(Q\d+)$")
_FENCER_OCCUPATIONS = {
    FENCER_OCCUPATION,
    FOIL_FENCER_OCCUPATION,
    EPEE_FENCER_OCCUPATION,
    SABRE_FENCER_OCCUPATION,
    WHEELCHAIR_FENCER_OCCUPATION,
}
_OCCUPATION_WEAPONS = {
    FOIL_FENCER_OCCUPATION: "foil",
    EPEE_FENCER_OCCUPATION: "epee",
    SABRE_FENCER_OCCUPATION: "sabre",
}

_HEIGHT_UNITS = {
    "Q11573": 100.0,  # metre -> centimetre
    "Q174728": 1.0,  # centimetre
    "Q174789": 0.1,  # millimetre -> centimetre
}
_WEIGHT_UNITS = {
    "Q11570": 1.0,  # kilogram
    "Q41803": 0.001,  # gram -> kilogram
}
_FEMALE_GENDERS = {"Q6581072"}
_MALE_GENDERS = {"Q6581097"}


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str],
    attempts: int = 7,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                delay = response.headers.get("Retry-After")
                seconds = float(delay) if delay and delay.isdigit() else float(2**attempt)
                time.sleep(min(seconds, 45.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError(f"Expected a JSON object from {url}")
            return payload
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(float(2**attempt), 45.0))
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def fetch_fencer_ids(
    destination: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> list[str]:
    """Fetch human items with a fencing occupation or a sport-fencing statement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force and destination.exists():
        payload = _load_json(destination)
    else:
        occupation_values = " ".join(f"wd:{value}" for value in sorted(_FENCER_OCCUPATIONS))
        query = f"""
SELECT DISTINCT ?athlete WHERE {{
  ?athlete wdt:P31 wd:{HUMAN} .
  {{
    VALUES ?occupation {{ {occupation_values} }}
    ?athlete wdt:P106 ?occupation .
  }} UNION {{
    ?athlete wdt:P641 wd:{FENCING_SPORT} .
  }}
}}
""".strip()
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60, read=180),
            headers={"User-Agent": user_agent},
        ) as client:
            payload = _request_json(
                client,
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
            )
        atomic_write_bytes(destination, _json_bytes(payload))

    bindings = payload.get("results", {}).get("bindings", [])
    identifiers: set[str] = set()
    for binding in bindings if isinstance(bindings, list) else []:
        value = binding.get("athlete", {}).get("value") if isinstance(binding, dict) else None
        match = _ENTITY_PATTERN.search(str(value))
        if match:
            identifiers.add(match.group(1))
    return sorted(identifiers, key=lambda value: int(value[1:]))


def _json_bytes(value: Any) -> bytes:
    import json

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _entity_batch(
    ids: list[str],
    *,
    user_agent: str,
) -> dict[str, dict[str, Any]]:
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=120),
        headers={"User-Agent": user_agent},
    ) as client:
        payload = _request_json(
            client,
            ENTITY_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(ids),
                "props": "labels|descriptions|claims|sitelinks",
                "languages": "en",
                "languagefallback": "1",
                "sitefilter": "enwiki",
                "format": "json",
                "formatversion": "2",
            },
        )
    entities = payload.get("entities", {})
    return {
        str(identifier): entity
        for identifier, entity in entities.items()
        if isinstance(entity, dict)
    }


def fetch_entities(
    identifiers: list[str],
    cache_directory: Path,
    *,
    user_agent: str,
    workers: int = 4,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch entity JSON in bounded batches and retain raw snapshots outside Git."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    batches = [identifiers[start : start + 50] for start in range(0, len(identifiers), 50)]
    collected: dict[str, dict[str, Any]] = {}
    pending: list[tuple[int, list[str], Path]] = []
    for index, batch in enumerate(batches):
        cache_path = cache_directory / f"batch-{index:04d}.json"
        if cache_path.exists() and not force:
            cached = _load_json(cache_path)
            entities = cached.get("entities", {})
            if isinstance(entities, dict) and set(entities) == set(batch):
                collected.update(
                    {
                        str(identifier): entity
                        for identifier, entity in entities.items()
                        if isinstance(entity, dict)
                    }
                )
                continue
        pending.append((index, batch, cache_path))

    failures: list[tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_entity_batch, batch, user_agent=user_agent): (index, cache_path)
            for index, batch, cache_path in pending
        }
        for future in as_completed(futures):
            index, cache_path = futures[future]
            try:
                entities = future.result()
            except Exception as exc:  # preserve every successful batch before failing the run
                failures.append((index, exc))
                continue
            atomic_write_bytes(cache_path, _json_bytes({"batch": index, "entities": entities}))
            collected.update(entities)
    if failures:
        failed_batches = ", ".join(str(index) for index, _ in sorted(failures))
        raise RuntimeError(
            "Wikidata entity retrieval was incomplete; successful batches were cached and "
            f"can be resumed. Failed batch indexes: {failed_batches}"
        ) from failures[0][1]
    return collected


def _statement_values(entity: dict[str, Any], property_id: str) -> list[Any]:
    claims = entity.get("claims", {})
    statements = claims.get(property_id, []) if isinstance(claims, dict) else []
    values: list[Any] = []
    for statement in statements if isinstance(statements, list) else []:
        if not isinstance(statement, dict) or statement.get("rank") == "deprecated":
            continue
        mainsnak = statement.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {}) if isinstance(mainsnak, dict) else {}
        if isinstance(datavalue, dict) and "value" in datavalue:
            values.append(datavalue["value"])
    return values


def _entity_ids(entity: dict[str, Any], property_id: str) -> list[str]:
    values: list[str] = []
    for value in _statement_values(entity, property_id):
        if isinstance(value, dict) and value.get("id"):
            values.append(str(value["id"]))
    return list(dict.fromkeys(values))


def _quantity(entity: dict[str, Any], property_id: str, units: dict[str, float]) -> float | None:
    for value in _statement_values(entity, property_id):
        if not isinstance(value, dict):
            continue
        unit_match = _ENTITY_PATTERN.search(str(value.get("unit", "")))
        if not unit_match or unit_match.group(1) not in units:
            continue
        try:
            amount = float(str(value["amount"]))
        except (KeyError, TypeError, ValueError):
            continue
        converted = amount * units[unit_match.group(1)]
        if converted > 0:
            return round(converted, 3)
    return None


def _birth_year(entity: dict[str, Any]) -> int | None:
    for value in _statement_values(entity, "P569"):
        if not isinstance(value, dict):
            continue
        match = re.match(r"^([+-]\d+)-", str(value.get("time", "")))
        if match:
            year = int(match.group(1))
            if 1600 <= year <= datetime.now(UTC).year:
                return year
    return None


def _text(entity: dict[str, Any], field: str) -> str | None:
    values = entity.get(field, {})
    english = values.get("en") if isinstance(values, dict) else None
    value = english.get("value") if isinstance(english, dict) else None
    return str(value) if value else None


def _gender(gender_ids: list[str]) -> str:
    if any(identifier in _FEMALE_GENDERS for identifier in gender_ids):
        return "female"
    if any(identifier in _MALE_GENDERS for identifier in gender_ids):
        return "male"
    return "other" if gender_ids else "unknown"


def _label_lookup(
    identifiers: set[str],
    cache_directory: Path,
    *,
    user_agent: str,
    workers: int,
) -> dict[str, str]:
    if not identifiers:
        return {}
    entities = fetch_entities(
        sorted(identifiers, key=lambda value: int(value[1:])),
        cache_directory,
        user_agent=user_agent,
        workers=workers,
    )
    return {
        identifier: label
        for identifier, entity in entities.items()
        if (label := _text(entity, "labels")) is not None
    }


def _entity_row(
    identifier: str,
    entity: dict[str, Any],
    labels: dict[str, str],
    *,
    retrieved_at: str,
) -> dict[str, Any] | None:
    """Normalize one human entity and reject stale or non-fencing memberships."""

    if entity.get("missing") or HUMAN not in _entity_ids(entity, "P31"):
        return None
    occupation_ids = _entity_ids(entity, "P106")
    sport_ids = _entity_ids(entity, "P641")
    fencing_occupation_ids = sorted(set(occupation_ids) & _FENCER_OCCUPATIONS)
    has_fencing_sport = FENCING_SPORT in sport_ids
    if not fencing_occupation_ids and not has_fencing_sport:
        return None

    name = _text(entity, "labels") or identifier
    description = _text(entity, "descriptions")
    gender_ids = _entity_ids(entity, "P21")
    country_ids = _entity_ids(entity, "P27")
    occupation_weapons = {
        _OCCUPATION_WEAPONS[item]
        for item in fencing_occupation_ids
        if item in _OCCUPATION_WEAPONS
    }
    description_weapons = {weapon.value for weapon in infer_weapon_candidates(description)}
    weapons = [
        weapon
        for weapon in ("epee", "foil", "sabre")
        if weapon in occupation_weapons | description_weapons
    ]
    image_values = _statement_values(entity, "P18")
    sitelinks = entity.get("sitelinks", {})
    enwiki = sitelinks.get("enwiki") if isinstance(sitelinks, dict) else None
    article_title = enwiki.get("title") if isinstance(enwiki, dict) else None
    return {
        "athlete_id": f"wikidata:{identifier}",
        "wikidata_id": identifier,
        "name": name,
        "description": description,
        "gender": _gender(gender_ids),
        "gender_ids": gender_ids,
        "gender_labels": [labels[item] for item in gender_ids if item in labels],
        "birth_year": _birth_year(entity),
        "country_ids": country_ids,
        "countries": [labels[item] for item in country_ids if item in labels],
        "weapons": weapons,
        "weapons_from_occupation": sorted(occupation_weapons),
        "weapons_from_description": sorted(description_weapons),
        "fencing_occupation_ids": fencing_occupation_ids,
        "has_fencing_occupation": bool(fencing_occupation_ids),
        "has_fencing_sport_statement": has_fencing_sport,
        "wheelchair_fencer_occupation": WHEELCHAIR_FENCER_OCCUPATION
        in fencing_occupation_ids,
        "height_cm": _quantity(entity, "P2048", _HEIGHT_UNITS),
        "weight_kg": _quantity(entity, "P2067", _WEIGHT_UNITS),
        "image_file": str(image_values[0]) if image_values else None,
        "wikipedia_title": str(article_title) if article_title else None,
        "source_url": f"https://www.wikidata.org/wiki/{identifier}",
        "source_license": LICENSE,
        "retrieved_at": retrieved_at,
    }


def build_manifest(
    ids_path: Path,
    cache_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    workers: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Create a compact, license-explicit registry of real fencer identities."""

    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    identifiers = fetch_fencer_ids(ids_path, user_agent=user_agent, force=force)
    entities = fetch_entities(
        identifiers,
        cache_directory / "fencers",
        user_agent=user_agent,
        workers=workers,
        force=force,
    )

    related_ids: set[str] = set()
    for related_entity in entities.values():
        related_ids.update(_entity_ids(related_entity, "P21"))
        related_ids.update(_entity_ids(related_entity, "P27"))
    labels = _label_lookup(
        related_ids,
        cache_directory / "labels",
        user_agent=user_agent,
        workers=workers,
    )

    rows: list[dict[str, Any]] = []
    skipped_nonhuman_or_stale = 0
    for identifier in identifiers:
        entity = entities.get(identifier)
        if entity is None:
            skipped_nonhuman_or_stale += 1
            continue
        row = _entity_row(identifier, entity, labels, retrieved_at=retrieved_at)
        if row is None:
            skipped_nonhuman_or_stale += 1
            continue
        rows.append(row)

    manifest_sha256 = write_jsonl(output_path, rows)
    explicit_weapon = sum(bool(row["weapons"]) for row in rows)
    summary = {
        "claim_boundary": (
            "Wikidata human identities and self-described metadata are a CC0 registry, not "
            "verified current rankings, performance measurements, proof of active status, or "
            "proof that an attached image depicts fencing action. Sport-only and occupation-only "
            "memberships remain distinguishable."
        ),
        "source_url": SOURCE_PAGE,
        "query_occupations": sorted(_FENCER_OCCUPATIONS),
        "query_sport": FENCING_SPORT,
        "query_human": HUMAN,
        "license": LICENSE,
        "retrieved_at": retrieved_at,
        "ids": len(identifiers),
        "records": len(rows),
        "skipped_nonhuman_missing_or_stale": skipped_nonhuman_or_stale,
        "with_name": sum(row["name"] != row["wikidata_id"] for row in rows),
        "with_birth_year": sum(row["birth_year"] is not None for row in rows),
        "with_country": sum(bool(row["countries"]) for row in rows),
        "with_gender": sum(row["gender"] != "unknown" for row in rows),
        "with_height": sum(row["height_cm"] is not None for row in rows),
        "with_weight": sum(row["weight_kg"] is not None for row in rows),
        "with_image": sum(row["image_file"] is not None for row in rows),
        "with_wikipedia": sum(row["wikipedia_title"] is not None for row in rows),
        "with_any_weapon_evidence": explicit_weapon,
        "with_weapon_occupation": sum(bool(row["weapons_from_occupation"]) for row in rows),
        "with_weapon_description": sum(bool(row["weapons_from_description"]) for row in rows),
        "with_fencing_occupation": sum(row["has_fencing_occupation"] for row in rows),
        "with_fencing_sport_statement": sum(
            row["has_fencing_sport_statement"] for row in rows
        ),
        "with_both_membership_routes": sum(
            row["has_fencing_occupation"] and row["has_fencing_sport_statement"]
            for row in rows
        ),
        "with_wheelchair_fencer_occupation": sum(
            row["wheelchair_fencer_occupation"] for row in rows
        ),
        "by_explicit_weapon": {
            weapon: sum(weapon in row["weapons"] for row in rows)
            for weapon in ("epee", "foil", "sabre")
        },
        "ids_snapshot_sha256": sha256_file(ids_path),
        "manifest_sha256": manifest_sha256,
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
