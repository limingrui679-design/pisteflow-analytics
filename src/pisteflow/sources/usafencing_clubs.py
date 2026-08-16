"""Prepare the CC0 Harvard Dataverse USA Fencing club geography snapshot."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

SOURCE_URL = "https://doi.org/10.7910/DVN/AN2RB5"
VERSION = "1.0"
WORKBOOK_FILE_ID = 14_030_182
README_FILE_ID = 14_030_181
EXPECTED_WORKBOOK_BYTES = 247_254
EXPECTED_WORKBOOK_SHA256 = "3289ff196f53a8783d37f7f23eec72130fb9fd612efbcf9e5774d3f31a6dece2"
EXPECTED_README_BYTES = 654
EXPECTED_README_SHA256 = "d7e0b15d23ace9a673fb36edaeec8c51d5783a64903d395424f34029835f2bc6"
EXPECTED_ROWS = 589

EXPECTED_COLUMNS = [
    "id",
    "name",
    "short_name",
    "slug",
    "membership_type",
    "inactive",
    "leased_daily",
    "nonprofit",
    "specializes_in",
    "specialties",
    "website",
    "url_short",
    "detail_url",
    "division",
    "region",
    "address",
    "city",
    "state",
    "zip",
    "country",
    "formatted_address_plain",
    "lat",
    "lng",
    "google_place",
    "google_directions",
]

_SPECIALTIES = {"Foil", "Epee", "Saber", "Para"}
_DIRECT_IDENTIFIERS = {
    "source_id",
    "name",
    "short_name",
    "slug",
    "website",
    "url_short",
    "detail_url",
    "address",
    "formatted_address_plain",
    "google_place",
    "google_directions",
}


def _stable_id(value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:usafencing-club:{value}".encode()).hexdigest()[:20]
    return f"usafencing-club:{digest}"


def parse_specialties(value: object) -> tuple[str, ...]:
    """Normalize the exact upstream semicolon-separated specialty vocabulary."""

    if pd.isna(value):
        return ()
    parts = tuple(part.strip() for part in str(value).split(";") if part.strip())
    unexpected = set(parts) - _SPECIALTIES
    if unexpected:
        raise ValueError(f"Unexpected USA Fencing club specialty: {sorted(unexpected)}")
    if len(parts) != len(set(parts)):
        raise ValueError("USA Fencing club specialty list contains a duplicate token")
    return parts


def normalize_zip_code(value: object) -> str | None:
    """Recover five-digit formatting lost when Excel stores a US ZIP as numeric."""

    if pd.isna(value):
        return None
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Non-integral USA Fencing club ZIP value: {value}")
        value = int(value)
    text = str(value).strip()
    if text.isdigit() and len(text) <= 5:
        return text.zfill(5)
    if re_full_zip(text):
        return text
    raise ValueError(f"Unexpected USA Fencing club ZIP value: {value!r}")


def re_full_zip(value: str) -> bool:
    return len(value) == 10 and value[:5].isdigit() and value[5] == "-" and value[6:].isdigit()


def _valid_website(value: object) -> bool:
    if pd.isna(value):
        return False
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _flag(value: object, field: str) -> bool:
    if pd.isna(value) or value not in (0, 1, False, True):
        raise ValueError(f"Unexpected USA Fencing club {field} flag: {value!r}")
    return bool(value)


def _membership_type(value: object) -> str:
    if pd.isna(value):
        return "unknown_missing"
    if value == 0:
        return "unknown_numeric_zero"
    return str(value).strip()


def prepare(
    workbook_path: Path,
    readme_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Validate the source workbook and publish a direct-name-free geography table."""

    if workbook_path.stat().st_size != EXPECTED_WORKBOOK_BYTES:
        raise ValueError("USA Fencing clubs workbook size changed")
    if sha256_file(workbook_path) != EXPECTED_WORKBOOK_SHA256:
        raise ValueError("USA Fencing clubs workbook checksum changed")
    if readme_path.stat().st_size != EXPECTED_README_BYTES:
        raise ValueError("USA Fencing clubs README size changed")
    if sha256_file(readme_path) != EXPECTED_README_SHA256:
        raise ValueError("USA Fencing clubs README checksum changed")

    source = pd.read_excel(workbook_path, sheet_name="usafencing_clubs")
    if list(source.columns) != EXPECTED_COLUMNS:
        raise ValueError("USA Fencing clubs workbook columns changed")
    if len(source) != EXPECTED_ROWS:
        raise ValueError("USA Fencing clubs workbook row count changed")
    if source["id"].isna().any() or source["id"].duplicated().any():
        raise ValueError("USA Fencing clubs source IDs are missing or duplicated")
    if source["slug"].isna().any() or source["slug"].duplicated().any():
        raise ValueError("USA Fencing clubs slugs are missing or duplicated")

    rows: list[dict[str, Any]] = []
    for value in source.to_dict(orient="records"):
        specialties = parse_specialties(value["specialties"])
        latitude = None if pd.isna(value["lat"]) else float(value["lat"])
        longitude = None if pd.isna(value["lng"]) else float(value["lng"])
        if (latitude is None) != (longitude is None):
            raise ValueError("USA Fencing club row has only one coordinate")
        if latitude is not None and longitude is not None and not (
            -90 <= latitude <= 90 and -180 <= longitude <= 180
        ):
            raise ValueError("USA Fencing club coordinate is outside valid global bounds")
        rows.append(
            {
                "club_id": _stable_id(f"{int(value['id'])}:{value['slug']}"),
                "membership_type": _membership_type(value["membership_type"]),
                "inactive": _flag(value["inactive"], "inactive"),
                "leased_daily": _flag(value["leased_daily"], "leased_daily"),
                "nonprofit": _flag(value["nonprofit"], "nonprofit"),
                "specialties": "; ".join(specialties) if specialties else None,
                "has_epee": "Epee" in specialties,
                "has_foil": "Foil" in specialties,
                "has_sabre": "Saber" in specialties,
                "has_para": "Para" in specialties,
                "division": str(value["division"]),
                "region": str(value["region"]),
                "city": None if pd.isna(value["city"]) else str(value["city"]),
                "state": None if pd.isna(value["state"]) else str(value["state"]),
                "zip_code": normalize_zip_code(value["zip"]),
                "country": None if pd.isna(value["country"]) else str(value["country"]),
                "latitude": latitude,
                "longitude": longitude,
                "has_valid_website": _valid_website(value["website"]),
            }
        )

    frame = pd.DataFrame(rows)
    if _DIRECT_IDENTIFIERS.intersection(frame.columns):
        raise ValueError("Direct club identifiers leaked into the public analytic table")
    if frame["club_id"].duplicated().any():
        raise ValueError("USA Fencing club pseudonymous IDs are duplicated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    coordinate_rows = frame.dropna(subset=["latitude", "longitude"])
    coordinate_pairs = coordinate_rows[["latitude", "longitude"]].drop_duplicates()
    specialty_combinations = Counter(
        "unknown" if pd.isna(value) else str(value) for value in frame["specialties"]
    )
    membership_counts = Counter(frame["membership_type"])
    short_name_excel_dates = sum(
        isinstance(value, (date, datetime)) for value in source["short_name"]
    )
    report: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "version": VERSION,
        "license": "CC0 1.0",
        "source_files": {
            "workbook": {
                "dataverse_file_id": WORKBOOK_FILE_ID,
                "bytes": workbook_path.stat().st_size,
                "sha256": sha256_file(workbook_path),
            },
            "readme": {
                "dataverse_file_id": README_FILE_ID,
                "bytes": readme_path.stat().st_size,
                "sha256": sha256_file(readme_path),
            },
        },
        "club_rows": len(frame),
        "pseudonymous_club_ids": int(frame["club_id"].nunique()),
        "source_name_duplicates_beyond_first": int(source["name"].duplicated().sum()),
        "source_exact_duplicates_excluding_id": int(
            source.drop(columns=["id"]).duplicated().sum()
        ),
        "membership_type_counts": dict(sorted(membership_counts.items())),
        "inactive_rows": int(frame["inactive"].sum()),
        "leased_daily_rows": int(frame["leased_daily"].sum()),
        "nonprofit_rows": int(frame["nonprofit"].sum()),
        "specialty_combination_counts": dict(sorted(specialty_combinations.items())),
        "weapon_specialty_counts": {
            "epee": int(frame["has_epee"].sum()),
            "foil": int(frame["has_foil"].sum()),
            "sabre": int(frame["has_sabre"].sum()),
            "para": int(frame["has_para"].sum()),
            "unknown": int(frame["specialties"].isna().sum()),
        },
        "divisions": int(frame["division"].nunique()),
        "regions": dict(sorted(Counter(frame["region"]).items())),
        "states_or_territories_nonmissing": int(frame["state"].nunique(dropna=True)),
        "rows_missing_state": int(frame["state"].isna().sum()),
        "rows_missing_country": int(frame["country"].isna().sum()),
        "rows_with_coordinate_pairs": len(coordinate_rows),
        "unique_coordinate_pairs": len(coordinate_pairs),
        "reused_coordinate_rows_beyond_first": len(coordinate_rows) - len(coordinate_pairs),
        "rows_missing_both_coordinates": int(frame["latitude"].isna().sum()),
        "rows_with_valid_website": int(frame["has_valid_website"].sum()),
        "source_url_short_placeholder_rows": int(source["url_short"].fillna("").eq("://").sum()),
        "source_short_name_excel_date_cells": short_name_excel_dates,
        "output": "data/processed/usafencing-clubs-geography.parquet",
        "output_sha256": sha256_file(output_path),
        "claim_boundary": (
            "This CC0 snapshot contains 589 public club records, not athletes, memberships, "
            "training sessions, or performance outcomes. The published analytic table removes club "
            "names, source IDs, slugs, street addresses, URLs, and map links while retaining "
            "pseudonymous club rows, organizational attributes, specialty coverage, and geography. "
            "Coordinates are upstream geocodes and are not independently field-verified; repeated "
            "coordinates may represent shared facilities rather than duplicate clubs."
        ),
    }
    write_json(report_path, report)
    return report
