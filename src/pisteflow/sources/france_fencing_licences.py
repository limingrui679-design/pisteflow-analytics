"""Prepare France's open annual, geocoded fencing-licence census."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

DATASET_URL = (
    "https://www.data.gouv.fr/datasets/donnees-geocodees-issues-du-recensement-"
    "des-licences-et-clubs-aupres-des-federations-sportives-agreees-par-le-"
    "ministere-charge-des-sports"
)
LICENSE = "Licence Ouverte / Open Licence 2.0 (Etalab)"
FENCING_FEDERATION_CODE = "110"
DATA_SPORTS_HISTORY_URL = (
    "https://data.sports.gouv.fr/api/explore/v2.1/catalog/datasets/lics_histo/"
    "exports/csv?where=code_fede_2024%3D110&lang=fr&timezone=UTC&use_labels=false&"
    "delimiter=%3B"
)
DATA_SPORTS_2024_DETAIL_URL = (
    "https://data.sports.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "lics_dep_age_sexe/exports/csv?where=code_fede_2024%3D110&lang=fr&timezone=UTC&"
    "use_labels=false&delimiter=%3B"
)
DATA_SPORTS_HISTORY_BYTES = 3_916
DATA_SPORTS_HISTORY_SHA256 = "fe73cc924967b6634719edb48347484a05e6a2b15b109155c54a57c4308fd5df"
DATA_SPORTS_2024_DETAIL_BYTES = 604_472
DATA_SPORTS_2024_DETAIL_SHA256 = (
    "15614ff0b701a0cbb61ddf24e042090beecb1deec63fa909ad3d8355493203e8"
)


@dataclass(frozen=True)
class Resource:
    """One pinned annual licence resource from the data.gouv.fr dataset."""

    year: int
    resource_id: str
    url: str
    filename: str
    expected_bytes: int
    expected_sha1: str | None
    encoding: str


_CURRENT_BASE = (
    "https://static.data.gouv.fr/resources/donnees-geocodees-issues-du-"
    "recensement-des-licences-et-clubs-aupres-des-federations-sportives-"
    "agreees-par-le-ministere-charge-des-sports"
)
_LEGACY_BASE = (
    "https://static.data.gouv.fr/resources/recensement-des-licences-et-clubs-"
    "aupres-des-federations-sportives-agreees-par-le-ministere-charge-d"
)

RESOURCES = (
    Resource(
        2023,
        "3da4eb46-8a73-47d5-b9b5-3f260d4f045b",
        f"{_CURRENT_BASE}/20251229-163107/lic-data-2023.csv",
        "lic-data-2023.csv",
        256_370_260,
        None,
        "utf-8-sig",
    ),
    Resource(
        2022,
        "ce39c9d6-2e7f-4a05-9f95-9e6c06b38219",
        f"{_CURRENT_BASE}/20241230-162428/lic-data-2022.csv",
        "lic-data-2022.csv",
        247_714_299,
        "d43d26c031c2df38e89780990af927b0ae569f2e",
        "utf-8-sig",
    ),
    Resource(
        2021,
        "b62c05d2-6b45-44e0-80a4-6e10e3f14ec6",
        f"{_CURRENT_BASE}/20231207-134903/lic-data-2021.csv",
        "lic-data-2021.csv",
        225_825_478,
        "2700462852304582730cfdd1c33823ddab4f9b30",
        "utf-8-sig",
    ),
    Resource(
        2020,
        "1eb1e7c9-27c4-4834-b8b6-cf63b45ee74b",
        f"{_CURRENT_BASE}/20231207-134516/lic-data-2020.csv",
        "lic-data-2020.csv",
        240_546_787,
        "97c893f1614b4451c463b8d6f1220908a9466027",
        "utf-8-sig",
    ),
    Resource(
        2019,
        "9b002e32-0ec9-40fc-9292-cddeae4416c6",
        f"{_CURRENT_BASE}/20231207-134205/lic-data-2019.csv",
        "lic-data-2019.csv",
        266_754_279,
        "fa2b79d7c9c3f9593fe8ed26b4150718f4a429d9",
        "utf-8-sig",
    ),
    Resource(
        2018,
        "f3970a7b-df0e-4c3e-9f36-c93da58f4a3e",
        f"{_CURRENT_BASE}/20220217-115513/lic-data-2018.csv",
        "lic-data-2018.csv",
        206_338_975,
        "60523cd62e52a00bdd55912e20ba0f117d0b0632",
        "cp1252",
    ),
    Resource(
        2017,
        "523892b6-1eca-4680-ac4a-e93d3a896161",
        f"{_CURRENT_BASE}/20201029-165909/lic-data-2017.csv",
        "lic-data-2017.csv",
        222_161_061,
        "6c985ec65bd1a3ea835904ae322aca961959c2fd",
        "cp1252",
    ),
    Resource(
        2016,
        "67e7edcc-112e-415a-aea0-07fcabd03e83",
        f"{_CURRENT_BASE}/20201030-145527/lic-data-2016.csv",
        "lic-data-2016.csv",
        230_646_125,
        "6a238b038c5d183865bc5fb86156397fe9c63fda",
        "cp1252",
    ),
    Resource(
        2015,
        "e15a8ec3-254d-45ce-90bb-1f4ca45eda8c",
        f"{_LEGACY_BASE}/20180131-163516/Licences_2015.csv",
        "Licences_2015.csv",
        197_275_413,
        "55928c23e600d8847dd16e12ae10ff158ca574cc",
        "cp1252",
    ),
    Resource(
        2014,
        "22f45258-4496-4f8c-8619-936248121342",
        f"{_LEGACY_BASE}/20161215-152726/Licences_2014.csv",
        "Licences_2014.csv",
        178_371_632,
        "05f151e29d2c954e4866751020627487f199dbc0",
        "cp1252",
    ),
    Resource(
        2013,
        "6f02feb7-e128-4554-b0ed-bfdd7a1959ce",
        f"{_LEGACY_BASE}/20160826-142541/Licences_2013.csv",
        "Licences_2013.csv",
        178_584_859,
        "985b2d6954b16a8801c0e1b748ea257038f14533",
        "cp1252",
    ),
    Resource(
        2012,
        "b924c7bc-040d-47b1-b9db-eebcf4fe4efc",
        "https://static.data.gouv.fr/resources/licences_2012.csv",
        "licences_2012.csv",
        159_729_783,
        None,
        "cp1252",
    ),
    Resource(
        2011,
        "15fe6bcc-0833-4795-9c7a-103fc40e1286",
        "https://www.data.gouv.fr/storage/f/2014-05-09T15-49-45/lic-2011.csv",
        "lic-2011.csv",
        37_652_175,
        None,
        "cp1252",
    ),
)

_GEOGRAPHY_CODE_COLUMNS = ("Code Commune", "code_commune", "newcog2", "cog2", "plg_depcom")
_GEOGRAPHY_NAME_COLUMNS = ("Commune", "libelle", "LIBELLE", "libgeo", "LIBGEO")
_REGION_COLUMNS = ("Région", "region")
_DEPARTMENT_COLUMNS = ("Département", "departement")
_FEDERATION_NAME_COLUMNS = ("Fédération", "nom_fed")
_AGE_TOKEN_MAP = {
    "19": "0-19",
    "2060": "20-60",
    "61": "61+",
    "0_4": "0-4",
    "5_9": "5-9",
    "10_14": "10-14",
    "15_19": "15-19",
    "20_29": "20-29",
    "30_44": "30-44",
    "45_59": "45-59",
    "60_74": "60-74",
    "75": "75+",
    "75_99": "75-99",
}


def selected_resources(start_year: int, end_year: int) -> tuple[Resource, ...]:
    """Return the pinned inclusive year interval in ascending order."""

    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    selected = tuple(
        resource
        for resource in sorted(RESOURCES, key=lambda item: item.year)
        if start_year <= resource.year <= end_year
    )
    if not selected or selected[0].year != start_year or selected[-1].year != end_year:
        raise ValueError("requested years must be within the complete 2011-2023 resource range")
    return selected


def _normalized_line(line: str) -> str:
    value = line.rstrip("\r\n")
    # The 2012 export wraps the entire semicolon-delimited row in exactly one
    # pair of quotes.  Later exports instead quote every individual field; do
    # not strip their first and last field quotes.
    if (
        value.startswith('"')
        and value.endswith('"')
        and value.count('"') == 2
        and value.count(";") > value.count(",")
    ):
        value = value[1:-1].replace('""', '"')
    return value


def _parse_line(line: str, delimiter: str) -> list[str]:
    return next(csv.reader([_normalized_line(line)], delimiter=delimiter, quotechar='"'))


def _delimiter(header_line: str) -> str:
    header = _normalized_line(header_line)
    return ";" if header.count(";") > header.count(",") else ","


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _federation_column(fieldnames: list[str]) -> str:
    if "Code" in fieldnames:
        return "Code"
    candidates = [field for field in fieldnames if field.casefold().startswith("fed_")]
    if len(candidates) != 1:
        raise ValueError(f"unable to identify one federation-code column: {candidates}")
    return candidates[0]


def _is_fencing_code(value: object) -> bool:
    text = str(value).strip().strip('"')
    if text.endswith(".0"):
        text = text[:-2]
    return text == FENCING_FEDERATION_CODE


def _count(value: object, *, column: str) -> int | None:
    text = str(value).strip()
    if not text:
        return 0
    if text in {".", "NA", "N/A"}:
        return None
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"non-integral licence count in {column}: {value!r}")
    return int(text)


def _current_age_key(column: str) -> str | None:
    match = re.fullmatch(r"([FH]) - (.+?)(?: ans)?", column)
    if match is None:
        return None
    sex = {"F": "female", "H": "male"}[match.group(1)]
    age = match.group(2).replace(" à ", "-").replace(" ", "").casefold()
    if age == "nr":
        age = "not_reported"
    return f"{sex}_age_{age}"


def _legacy_metric_key(column: str, year: int) -> str | None:
    lower = column.casefold()
    if lower == f"lic_h_{year}":
        return "male_total"
    if not lower.startswith("l_") or not lower.endswith(f"_{year}"):
        return None
    payload = lower[2 : -len(f"_{year}")].strip("_").replace("__", "_")
    if payload == "":
        return "total"
    if payload in {"f", "h"}:
        return {"f": "female_total", "h": "male_total"}[payload]

    for prefix in ("qp", "zus"):
        if payload == prefix:
            return "priority_area_total"
        if payload in {f"{prefix}_f", f"{prefix}_h"}:
            sex = {"f": "female", "h": "male"}[payload[-1]]
            return f"priority_area_{sex}"

    sex = "all"
    if payload.endswith(("_f", "_h")):
        sex = {"f": "female", "h": "male"}[payload[-1]]
        payload = payload[:-2]
    elif year == 2011 and payload[-1:] in {"f", "h"}:
        sex = {"f": "female", "h": "male"}[payload[-1]]
        payload = payload[:-1]
    age = _AGE_TOKEN_MAP.get(payload)
    return None if age is None else f"{sex}_age_{age}"


def metric_key(column: str, year: int) -> str | None:
    """Map one source count column to a stable licence metric."""

    clean = column.removeprefix("\ufeff").strip()
    if clean == "Total":
        return "total"
    return _current_age_key(clean) or _legacy_metric_key(clean, year)


def _text(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        value = row.get(candidate, "").strip()
        if value:
            return value
    return None


def _geography_code(row: dict[str, str]) -> str:
    value = _text(row, _GEOGRAPHY_CODE_COLUMNS)
    if value is None:
        raise ValueError("fencing licence row has no geography code")
    if value.isdigit() and len(value) < 5:
        value = value.zfill(5)
    return value


def _department(code: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if code.startswith(("97", "98")) and len(code) >= 3:
        return code[:3]
    return code[:2] if len(code) >= 2 else None


def normalize_record(row: dict[str, str], *, year: int, source_row: int) -> dict[str, Any]:
    """Normalize one municipality or municipality-QPV fencing row."""

    code = _geography_code(row)
    explicit_department = _text(row, _DEPARTMENT_COLUMNS)
    result: dict[str, Any] = {
        "year": year,
        "source_row": source_row,
        "geography_code": code,
        "geography_name": _text(row, _GEOGRAPHY_NAME_COLUMNS),
        "department": _department(code, explicit_department),
        "region": _text(row, _REGION_COLUMNS),
        "qpv_code": row.get("Code QPV") or None,
        "qpv_name": row.get("Nom QPV") or None,
        "geography_status": row.get("Statut géo") or None,
        "federation_code": FENCING_FEDERATION_CODE,
        "federation_name": _text(row, _FEDERATION_NAME_COLUMNS),
    }
    for column, value in row.items():
        key = metric_key(column, year)
        if key is not None:
            result[key] = _count(value, column=column)

    for key, value in tuple(result.items()):
        if not key.startswith("all_age_") or value is None:
            continue
        age = key.removeprefix("all_age_")
        female_key = f"female_age_{age}"
        male_key = f"male_age_{age}"
        if female_key in result and male_key not in result:
            male_value = int(value) - int(result[female_key])
            if male_value < 0:
                raise ValueError(f"negative derived male age count for {year}, {age}")
            result[male_key] = male_value
            result[f"{male_key}_origin"] = "derived_all_minus_source_female"

    if result.get("male_total") is None and result.get("total") is not None and result.get(
        "female_total"
    ) is not None:
        male_total = int(result["total"]) - int(result["female_total"])
        if male_total < 0:
            raise ValueError(f"negative derived male total for {year}, row {source_row}")
        result["male_total"] = male_total
        result["male_total_origin"] = "derived_total_minus_source_female_total"

    for sex in ("female", "male"):
        age_values = [
            int(value)
            for key, value in result.items()
            if key.startswith(f"{sex}_age_")
            and not key.endswith("_origin")
            and value is not None
        ]
        if f"{sex}_total" not in result and age_values:
            result[f"{sex}_total"] = sum(age_values)
            result[f"{sex}_total_origin"] = "derived_sum_of_source_age_cells"
        elif f"{sex}_total" in result and f"{sex}_total_origin" not in result:
            result[f"{sex}_total_origin"] = "source"
    if "total" not in result and result.get("female_total") is not None and result.get(
        "male_total"
    ) is not None:
        result["total"] = int(result["female_total"]) + int(result["male_total"])
        result["total_origin"] = "derived_sum_of_source_sex_totals"
    else:
        result["total_origin"] = "source"
    if result.get("total") is None:
        raise ValueError(f"fencing licence row has no usable total in {year}, row {source_row}")
    if result.get("female_total") is not None and result.get("male_total") is not None:
        sex_balance_delta = (
            int(result["total"]) - int(result["female_total"]) - int(result["male_total"])
        )
        result["sex_balance_delta"] = sex_balance_delta
        result["sex_balance_delta_origin"] = "derived_total_minus_source_sex_totals"
    return result


def _write_filtered_rows(
    source: TextIO,
    destination: Path,
    *,
    year: int,
) -> tuple[list[dict[str, Any]], int]:
    header_line = source.readline()
    if not header_line:
        raise ValueError(f"empty France licence source for {year}")
    delimiter = _delimiter(header_line)
    fieldnames = [
        field.removeprefix("\ufeff").strip()
        for field in _parse_line(header_line, delimiter)
    ]
    federation_column = _federation_column(fieldnames)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, Any]] = []
    source_rows = 0
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for source_row, line in enumerate(source, start=2):
            if not line.strip():
                continue
            values = _parse_line(line, delimiter)
            if len(values) != len(fieldnames):
                raise ValueError(
                    f"France licence row width changed in {year}, line {source_row}: "
                    f"{len(values)} != {len(fieldnames)}"
                )
            source_rows += 1
            row = dict(zip(fieldnames, values, strict=True))
            if not _is_fencing_code(row[federation_column]):
                continue
            writer.writerow(row)
            normalized.append(normalize_record(row, year=year, source_row=source_row))
    if not normalized:
        raise ValueError(f"no federation-code 110 rows found in France licence source {year}")
    return normalized, source_rows


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_data_sports(
    cache_directory: Path,
    history_output_path: Path,
    detail_2024_output_path: Path,
    *,
    user_agent: str,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history_path = cache_directory / "data-sports-fencing-history-2002-2024.csv"
    detail_path = cache_directory / "data-sports-fencing-2024-department-age-sex.csv"
    for path in (history_path, detail_path):
        if force:
            path.unlink(missing_ok=True)
    download_resumable(
        DATA_SPORTS_HISTORY_URL,
        history_path,
        user_agent=user_agent,
        expected_bytes=DATA_SPORTS_HISTORY_BYTES,
    )
    download_resumable(
        DATA_SPORTS_2024_DETAIL_URL,
        detail_path,
        user_agent=user_agent,
        expected_bytes=DATA_SPORTS_2024_DETAIL_BYTES,
    )
    if sha256_file(history_path) != DATA_SPORTS_HISTORY_SHA256:
        raise ValueError("Data.Sports fencing history snapshot changed")
    if sha256_file(detail_path) != DATA_SPORTS_2024_DETAIL_SHA256:
        raise ValueError("Data.Sports fencing 2024 detail snapshot changed")

    history = pd.read_csv(history_path, sep=";", encoding="utf-8-sig")
    history_expected = {
        "code_fede_2024",
        "code_fede_ref",
        "federation",
        "annee",
        "evol_20ans",
        "evol_10ans",
        "evol_5ans",
        "lics",
        "lics_100_2002",
        "lics_100_2016",
        "fede_gp",
    }
    if set(history.columns) != history_expected or len(history) != 23:
        raise ValueError("Data.Sports fencing history schema or row count changed")
    history["year"] = pd.to_datetime(history.pop("annee"), errors="raise").dt.year
    if set(history["year"]) != set(range(2002, 2025)):
        raise ValueError("Data.Sports fencing history no longer covers every 2002-2024 year")
    if not history["code_fede_2024"].eq(110).all() or history["year"].duplicated().any():
        raise ValueError("Data.Sports fencing history federation filter changed")
    history = history.sort_values("year", kind="stable").reset_index(drop=True)

    detail = pd.read_csv(detail_path, sep=";", encoding="utf-8-sig")
    detail_expected = {
        "code_fede_2024",
        "code_fede_ref",
        "sexe",
        "trage_full",
        "trage",
        "dep",
        "dep_l",
        "newreg",
        "newreg_l",
        "lics",
        "pop",
        "lics_pop",
        "rank_fede_dep",
        "lics_pop_fr",
        "is",
        "fede_gp",
    }
    if set(detail.columns) != detail_expected or len(detail) != 3_960:
        raise ValueError("Data.Sports fencing 2024 detail schema or row count changed")
    if not detail["code_fede_2024"].eq(110).all() or set(detail["sexe"]) != {"F", "H"}:
        raise ValueError("Data.Sports fencing 2024 detail federation or sex filter changed")
    detail = detail.rename(
        columns={
            "sexe": "sex",
            "trage_full": "source_age_band",
            "trage": "population_age_band",
            "dep": "department",
            "dep_l": "department_label",
            "newreg": "region",
            "newreg_l": "region_label",
            "lics": "licence_count",
            "pop": "population",
            "lics_pop": "licences_per_1000",
            "rank_fede_dep": "department_rate_rank",
            "lics_pop_fr": "france_licences_per_1000",
            "is": "specificity_index",
        }
    )
    detail["sex"] = detail["sex"].map({"F": "female", "H": "male"})
    detail["year"] = 2024
    detail = detail.sort_values(
        ["sex", "source_age_band", "department"], kind="stable"
    ).reset_index(drop=True)
    detailed_total = int(detail["licence_count"].sum())
    history_2024_total = int(history.loc[history["year"].eq(2024), "lics"].iloc[0])
    if detailed_total != history_2024_total:
        raise ValueError(
            "Data.Sports 2024 department-age-sex rows do not reconcile with history: "
            f"{detailed_total} != {history_2024_total}"
        )
    history_output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(history_output_path, index=False)
    detail_2024_output_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(detail_2024_output_path, index=False)
    summary = {
        "history": {
            "source_url": DATA_SPORTS_HISTORY_URL,
            "rows": len(history),
            "years": [2002, 2024],
            "fencing_licence_counts": {
                str(int(row.year)): int(row.lics)
                for row in history[["year", "lics"]].itertuples(index=False)
            },
            "source_bytes": history_path.stat().st_size,
            "source_sha256": sha256_file(history_path),
            "output": str(history_output_path),
            "output_sha256": sha256_file(history_output_path),
        },
        "detail_2024": {
            "source_url": DATA_SPORTS_2024_DETAIL_URL,
            "rows": len(detail),
            "departments_or_out_of_scope_geographies": int(detail["department"].nunique()),
            "source_age_bands": int(detail["source_age_band"].nunique()),
            "sexes": sorted(str(value) for value in detail["sex"].unique()),
            "licence_count": detailed_total,
            "licence_count_by_sex": {
                str(key): int(value)
                for key, value in detail.groupby("sex")["licence_count"].sum().items()
            },
            "zero_licence_cells": int(detail["licence_count"].eq(0).sum()),
            "source_bytes": detail_path.stat().st_size,
            "source_sha256": sha256_file(detail_path),
            "output": str(detail_2024_output_path),
            "output_sha256": sha256_file(detail_2024_output_path),
        },
    }
    return history, detail, summary


def fetch_and_prepare(
    cache_directory: Path,
    filtered_directory: Path,
    output_path: Path,
    history_output_path: Path,
    detail_2024_output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    start_year: int = 2011,
    end_year: int = 2023,
    keep_source_archives: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the selected annual sources and retain all code-110 aggregate rows."""

    frames: list[pd.DataFrame] = []
    annual: dict[str, Any] = {}
    source_files: dict[str, Any] = {}
    for resource in selected_resources(start_year, end_year):
        source_path = cache_directory / resource.filename
        if force:
            source_path.unlink(missing_ok=True)
        download_resumable(
            resource.url,
            source_path,
            user_agent=user_agent,
            expected_bytes=resource.expected_bytes,
        )
        source_sha1 = _sha1_file(source_path)
        if resource.expected_sha1 is not None and source_sha1 != resource.expected_sha1:
            raise ValueError(f"France licence SHA-1 changed for {resource.year}")
        source_sha256 = sha256_file(source_path)
        filtered_path = filtered_directory / f"lic-data-{resource.year}-fencing.csv"
        with source_path.open(
            "r", encoding=resource.encoding, errors="strict", newline=""
        ) as source:
            records, scanned_rows = _write_filtered_rows(
                source,
                filtered_path,
                year=resource.year,
            )
        frame = pd.DataFrame(records)
        frames.append(frame)
        total = int(frame["total"].fillna(0).sum())
        female = int(frame["female_total"].fillna(0).sum())
        male = int(frame["male_total"].fillna(0).sum())
        sex_balance = frame["sex_balance_delta"].fillna(0).astype(int)
        age_totals = {
            column: int(frame[column].fillna(0).sum())
            for column in sorted(frame.columns)
            if column.startswith(("female_age_", "male_age_"))
            and not column.endswith("_origin")
        }
        annual[str(resource.year)] = {
            "source_rows_scanned_all_federations": scanned_rows,
            "fencing_geography_rows": len(frame),
            "unique_geography_codes": int(frame["geography_code"].nunique()),
            "licence_count": total,
            "female_licence_count": female,
            "male_licence_count": male,
            "net_total_minus_sex_counts": int(sex_balance.sum()),
            "rows_with_positive_sex_balance_gap": int(sex_balance.gt(0).sum()),
            "rows_with_negative_sex_balance_gap": int(sex_balance.lt(0).sum()),
            "positive_sex_balance_gap_sum": int(sex_balance.clip(lower=0).sum()),
            "negative_sex_balance_gap_absolute_sum": int(
                (-sex_balance.clip(upper=0)).sum()
            ),
            "age_sex_counts": age_totals,
        }
        source_files[str(resource.year)] = {
            "resource_id": resource.resource_id,
            "url": resource.url,
            "bytes": resource.expected_bytes,
            "sha1": source_sha1,
            "sha256": source_sha256,
            "filtered_rows": len(frame),
            "filtered_sha256": sha256_file(filtered_path),
        }
        if not keep_source_archives:
            source_path.unlink()

    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result.sort_values(
        ["year", "geography_code", "qpv_code", "source_row"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    history, _, data_sports = _prepare_data_sports(
        cache_directory,
        history_output_path,
        detail_2024_output_path,
        user_agent=user_agent,
        force=force,
    )
    history_lookup = history.set_index("year")["lics"].astype(int)
    reconciliation = {
        year: {
            "geocoded_licence_count": int(values["licence_count"]),
            "official_revised_history_count": int(history_lookup.loc[int(year)]),
            "difference_geocoded_minus_history": int(values["licence_count"])
            - int(history_lookup.loc[int(year)]),
        }
        for year, values in annual.items()
    }
    report: dict[str, Any] = {
        "source_url": DATASET_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": LICENSE,
        "federation_code": FENCING_FEDERATION_CODE,
        "years": [resource.year for resource in selected_resources(start_year, end_year)],
        "source_files": source_files,
        "annual": annual,
        "data_sports": data_sports,
        "geocoded_vs_revised_history_reconciliation": reconciliation,
        "normalized_geography_rows": len(result),
        "licence_counts_summed_across_years": int(result["total"].fillna(0).sum()),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source_archives_retained": keep_source_archives,
        "claim_boundary": (
            "Annual French Fencing Federation licence counts by residence geography, sex, "
            "and age band for the selected years. These are memberships, not unique people, "
            "bout observations, training sessions, performance outcomes, or proof of where a "
            "person practised. The same person may appear in multiple years, so the across-year "
            "sum is a repeated annual-licence exposure total and never an athlete count. From "
            "2019 the geography is municipality-QPV, schemas and age bands differ over time, "
            "and the producer describes geocoded values as estimates with historical revisions "
            "not necessarily propagated backward. Raw filtered files preserve each source schema."
            " Some historical geography rows have source totals that do not equal female plus "
            "male counts; signed row-level balance deltas are preserved rather than relabelled "
            "as an unreported-sex category."
            " The 2002-2024 national history and 2024 department-age-sex table are coarser "
            "official views used for time-series analysis and reconciliation; they overlap the "
            "geocoded series and are never added to it as independent licences."
        ),
    }
    write_json(report_path, report)
    return report
