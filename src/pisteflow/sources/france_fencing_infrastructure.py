"""Prepare open French fencing-club and facility geography tables."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

LICENSE = "Licence Ouverte / Open Licence 2.0 (Etalab)"
CLUBS_URL = (
    "https://data.sports.gouv.fr/api/explore/v2.1/catalog/datasets/clubs_dep/exports/"
    "csv?select=code_fede_2024%2Ccode_fede_ref%2Cannee%2Cnewreg%2Cnewreg_l%2Cdep%2C"
    "dep_l%2Cfrance%2Cn_actifs_clubs%2Cn_actifs_epa%2Cn_actifs%2Cn_tot_clubs%2C"
    "n_tot_epa%2Cn_tot%2Cpop%2Cn_actifs_pop%2Cn_tot_pop&where=code_fede_ref%3D110&"
    "lang=fr&timezone=UTC&use_labels=false&delimiter=%3B"
)
EQUIPMENT_URL = (
    "https://data.sports.gouv.fr/api/explore/v2.1/catalog/datasets/equipements-sportifs/"
    "exports/csv?select=equip_numero%2Cequip_maj_date%2Cnew_name%2Cnew_code%2Cdep_code%2C"
    "dep_nom%2Creg_code%2Creg_nom%2Cdens_niveau%2Cdens_lib%2Cinst_hs_bool%2C"
    "equip_type_name%2Cequip_x%2Cequip_y%2Cequip_prop_type%2Cequip_gest_type%2C"
    "equip_ouv_public_bool%2Cequip_nature%2Cequip_sol%2Cequip_long%2Cequip_larg%2C"
    "equip_haut%2Cequip_surf%2Cequip_eclair%2Cequip_piste_nb%2Cequip_trib_nb%2C"
    "equip_vest_sport%2Cequip_vest_ens%2Cequip_douche%2Cequip_sanit%2Cequip_acc_libre%2C"
    "equip_saison%2Cequip_type_famille%2Cequip_type_code%2Cinst_acc_handi_bool%2C"
    "inst_trans_bool%2Cequip_pmr_vest%2Cequip_pmr_acc%2Cequip_pmr_aire%2Cequip_pmr_chem%2C"
    "equip_pmr_douche%2Cequip_pmr_sanit%2Cequip_pshs_aire%2Cequip_pshs_chem%2C"
    "equip_pshs_sanit%2Cequip_pshs_sign%2Cequip_pshs_vest%2Caps_name%2Cequip_qpv%2C"
    "equip_qpv_200m&where=search(aps_name%2C%22Escrime%22)&lang=fr&timezone=UTC&"
    "use_labels=false&delimiter=%3B"
)
CLUBS_BYTES = 37_195
CLUBS_SHA256 = "61a0b69455c3b629272bdc941928507856ee40caa41c616ea41a7cd0a7c17b56"
EQUIPMENT_BYTES = 547_910
EQUIPMENT_SHA256 = "016e931e38bfed6d9705ac911f5cca1630f45b03069caf0c98da0238c5967789"

_CLUB_COLUMNS = {
    "code_fede_2024",
    "code_fede_ref",
    "annee",
    "newreg",
    "newreg_l",
    "dep",
    "dep_l",
    "france",
    "n_actifs_clubs",
    "n_actifs_epa",
    "n_actifs",
    "n_tot_clubs",
    "n_tot_epa",
    "n_tot",
    "pop",
    "n_actifs_pop",
    "n_tot_pop",
}
_BOOLEAN_COLUMNS = (
    "inst_hs_bool",
    "equip_ouv_public_bool",
    "equip_eclair",
    "equip_douche",
    "equip_sanit",
    "equip_acc_libre",
    "equip_saison",
    "inst_acc_handi_bool",
    "inst_trans_bool",
    "equip_pmr_vest",
    "equip_pmr_acc",
    "equip_pmr_aire",
    "equip_pmr_chem",
    "equip_pmr_douche",
    "equip_pmr_sanit",
    "equip_pshs_aire",
    "equip_pshs_chem",
    "equip_pshs_sanit",
    "equip_pshs_sign",
    "equip_pshs_vest",
)


def normalize_boolean(value: object) -> bool | None:
    """Normalize the portal's text booleans without treating missing as false."""

    if pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"true", "oui"}:
        return True
    if normalized in {"false", "non"}:
        return False
    raise ValueError(f"unexpected Data.Sports boolean: {value!r}")


def _read_pinned(path: Path, expected_bytes: int, expected_sha256: str) -> pd.DataFrame:
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ValueError(f"Data.Sports infrastructure snapshot changed: {path.name}")
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def prepare_clubs(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the three-year French Fencing Federation club panel."""

    if set(frame.columns) != _CLUB_COLUMNS or len(frame) != 330:
        raise ValueError("French fencing club panel schema or row count changed")
    if not frame["code_fede_ref"].eq(110).all():
        raise ValueError("French fencing club federation filter changed")
    frame = frame.rename(
        columns={
            "annee": "year",
            "newreg": "region",
            "newreg_l": "region_label",
            "dep": "department",
            "dep_l": "department_label",
            "france": "in_france",
            "n_actifs_clubs": "active_clubs",
            "n_actifs_epa": "active_approved_professional_establishments",
            "n_actifs": "active_establishments",
            "n_tot_clubs": "all_clubs",
            "n_tot_epa": "all_approved_professional_establishments",
            "n_tot": "all_establishments",
            "pop": "population",
            "n_actifs_pop": "active_establishments_per_100000",
            "n_tot_pop": "all_establishments_per_100000",
        }
    )
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if set(frame["year"]) != {2022, 2023, 2024}:
        raise ValueError("French fencing club panel year coverage changed")
    if frame.duplicated(["year", "department"]).any():
        raise ValueError("French fencing club panel has duplicate department-year rows")
    return frame.sort_values(["year", "department"], kind="stable").reset_index(drop=True)


def prepare_equipment(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and type the current public fencing-facility snapshot."""

    invalid_identity = "equip_numero" not in frame or frame.get(
        "equip_numero", pd.Series(dtype="string")
    ).duplicated().any()
    if len(frame) != 1_124 or invalid_identity:
        raise ValueError("French fencing equipment row count or public IDs changed")
    if not frame["aps_name"].astype("string").str.contains("Escrime", case=False, na=False).all():
        raise ValueError("French fencing equipment activity filter changed")
    result = frame.rename(
        columns={
            "equip_numero": "equipment_id",
            "equip_maj_date": "record_updated_on",
            "new_name": "municipality_name",
            "new_code": "municipality_code",
            "dep_code": "department",
            "dep_nom": "department_name",
            "reg_code": "region",
            "reg_nom": "region_name",
            "equip_x": "longitude",
            "equip_y": "latitude",
            "aps_name": "activity_names",
        }
    ).copy()
    result["record_updated_on"] = pd.to_datetime(
        result["record_updated_on"], errors="raise"
    ).dt.date
    result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
    result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    one_coordinate_missing = result["longitude"].isna() ^ result["latitude"].isna()
    if one_coordinate_missing.any():
        raise ValueError("French fencing equipment row has only one coordinate")
    for column in _BOOLEAN_COLUMNS:
        result[column] = result[column].map(normalize_boolean).astype("boolean")
    return result.sort_values("equipment_id", kind="stable").reset_index(drop=True)


def fetch_and_prepare(
    cache_directory: Path,
    clubs_output_path: Path,
    equipment_output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire and prepare the pinned club and equipment snapshots."""

    clubs_path = cache_directory / "france-fencing-clubs-2022-2024.csv"
    equipment_path = cache_directory / "france-fencing-equipment.csv"
    for path in (clubs_path, equipment_path):
        if force:
            path.unlink(missing_ok=True)
    download_resumable(
        CLUBS_URL,
        clubs_path,
        user_agent=user_agent,
        expected_bytes=CLUBS_BYTES,
    )
    download_resumable(
        EQUIPMENT_URL,
        equipment_path,
        user_agent=user_agent,
        expected_bytes=EQUIPMENT_BYTES,
    )
    clubs = prepare_clubs(_read_pinned(clubs_path, CLUBS_BYTES, CLUBS_SHA256))
    equipment = prepare_equipment(
        _read_pinned(equipment_path, EQUIPMENT_BYTES, EQUIPMENT_SHA256)
    )
    clubs_output_path.parent.mkdir(parents=True, exist_ok=True)
    clubs.to_parquet(clubs_output_path, index=False)
    equipment_output_path.parent.mkdir(parents=True, exist_ok=True)
    equipment.to_parquet(equipment_output_path, index=False)

    club_years: dict[str, Any] = {}
    for year, group in clubs.groupby("year", sort=True):
        club_years[str(int(year))] = {
            "department_rows": len(group),
            "active_clubs": int(group["active_clubs"].fillna(0).sum()),
            "all_clubs": int(group["all_clubs"].fillna(0).sum()),
            "active_establishments": int(group["active_establishments"].fillna(0).sum()),
            "all_establishments": int(group["all_establishments"].fillna(0).sum()),
        }
    equipment_types = Counter(equipment["equip_type_name"].fillna("unknown").astype(str))
    report: dict[str, Any] = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": LICENSE,
        "clubs": {
            "source_url": CLUBS_URL,
            "rows": len(clubs),
            "years": club_years,
            "source_bytes": clubs_path.stat().st_size,
            "source_sha256": sha256_file(clubs_path),
            "output": str(clubs_output_path),
            "output_sha256": sha256_file(clubs_output_path),
        },
        "equipment": {
            "source_url": EQUIPMENT_URL,
            "rows": len(equipment),
            "unique_public_equipment_ids": int(equipment["equipment_id"].nunique()),
            "equipment_type_counts": dict(sorted(equipment_types.items())),
            "dedicated_fencing_room_rows": int(
                equipment["equip_type_name"].eq("Salle d'escrime").sum()
            ),
            "regions": int(equipment["region"].nunique(dropna=True)),
            "departments": int(equipment["department"].nunique(dropna=True)),
            "rows_with_coordinates": int(equipment["latitude"].notna().sum()),
            "listed_open_to_public": int(equipment["equip_ouv_public_bool"].fillna(False).sum()),
            "listed_transport_accessible": int(
                equipment["inst_trans_bool"].fillna(False).sum()
            ),
            "listed_motor_accessible": int(
                equipment["inst_acc_handi_bool"].fillna(False).sum()
            ),
            "source_bytes": equipment_path.stat().st_size,
            "source_sha256": sha256_file(equipment_path),
            "output": str(equipment_output_path),
            "output_sha256": sha256_file(equipment_output_path),
        },
        "claim_boundary": (
            "Official open aggregate establishment counts and public equipment records whose "
            "activity field contains fencing. These are infrastructure and access-context rows, "
            "not athletes, members, visits, training sessions, bouts, coaching quality, current "
            "availability guarantees, or adoption evidence. A multisport facility listing fencing "
            "does not establish a dedicated piste or an active fencing programme."
        ),
    }
    write_json(report_path, report)
    return report
