"""Descriptive spatial-access analysis over official French open data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pisteflow.provenance import sha256_file, write_json

RANDOM_SEED = 20260820
BOOTSTRAP_REPETITIONS = 2_000
YOUTH_BANDS = {"0 à 4 ans", "5 à 9 ans", "10 à 14 ans", "15 à 19 ans"}
CLAIM_BOUNDARY = (
    "Cross-sectional department context from official French open-data snapshots. Licence "
    "rows are annual memberships, not unique people or participation visits; residence does "
    "not establish where somebody fences. Active establishments are not capacity, quality, "
    "or service availability. Facility records list places associated with fencing, including "
    "multisport venues, and do not prove an active club, available piste, programme schedule, "
    "or actual accessibility. Licence and club measures are from 2024 while equipment records "
    "were updated on varying dates through 2026. Department-level rank correlations and "
    "bootstrap intervals are descriptive associations, not causal access barriers, policy "
    "effects, individual risk estimates, adoption evidence, or resource-allocation advice."
)


def normalize_department(value: object) -> str:
    """Normalize numeric portal department codes without changing Corsican or overseas codes."""

    if pd.isna(value):
        raise ValueError("department code is missing")
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) < 2:
        text = text.zfill(2)
    if not text:
        raise ValueError("department code is empty")
    return text


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing {label} fields: {sorted(missing)}")


def prepare_department_frame(
    licences: pd.DataFrame,
    clubs: pd.DataFrame,
    equipment: pd.DataFrame,
) -> pd.DataFrame:
    """Build one auditable row per in-scope department without imputing missing booleans."""

    _require_columns(
        licences,
        {
            "department",
            "department_label",
            "sex",
            "population_age_band",
            "licence_count",
            "population",
            "year",
        },
        "licence",
    )
    _require_columns(
        clubs,
        {
            "department",
            "department_label",
            "year",
            "in_france",
            "active_establishments",
            "population",
            "active_establishments_per_100000",
        },
        "club",
    )
    _require_columns(
        equipment,
        {
            "equipment_id",
            "department",
            "equip_type_name",
            "equip_ouv_public_bool",
            "inst_acc_handi_bool",
            "inst_trans_bool",
            "equip_qpv",
            "equip_qpv_200m",
            "record_updated_on",
        },
        "equipment",
    )

    licence_rows = licences.loc[licences["year"].eq(2024)].copy()
    club_rows = clubs.loc[clubs["year"].eq(2024) & clubs["in_france"].eq(1)].copy()
    if licence_rows.empty or club_rows.empty:
        raise ValueError("2024 in-scope licence or club rows are empty")
    licence_rows["department"] = licence_rows["department"].map(normalize_department)
    club_rows["department"] = club_rows["department"].map(normalize_department)
    if club_rows["department"].duplicated().any():
        raise ValueError("2024 club input has duplicate in-scope departments")

    in_scope_departments = set(club_rows["department"])
    licence_rows = licence_rows.loc[licence_rows["department"].isin(in_scope_departments)]
    missing_licence_departments = in_scope_departments.difference(licence_rows["department"])
    if missing_licence_departments:
        raise ValueError(
            "Licence input is missing in-scope departments: "
            f"{sorted(missing_licence_departments)}"
        )
    if licence_rows.duplicated(["department", "sex", "population_age_band"]).any():
        raise ValueError("Licence input has duplicate department-sex-age cells")

    licence_rows["female_licences"] = np.where(
        licence_rows["sex"].eq("female"), licence_rows["licence_count"], 0
    )
    licence_rows["youth_licences"] = np.where(
        licence_rows["population_age_band"].isin(YOUTH_BANDS),
        licence_rows["licence_count"],
        0,
    )
    licence_summary = (
        licence_rows.groupby("department", as_index=False)
        .agg(
            department_label=("department_label", "first"),
            licence_memberships=("licence_count", "sum"),
            female_licence_memberships=("female_licences", "sum"),
            youth_licence_memberships=("youth_licences", "sum"),
            licence_population=("population", "sum"),
        )
        .sort_values("department", kind="stable")
    )
    licence_summary["licences_per_1000"] = np.where(
        licence_summary["licence_population"].gt(0),
        licence_summary["licence_memberships"]
        / licence_summary["licence_population"]
        * 1_000,
        np.nan,
    )
    licence_summary["female_membership_share"] = np.where(
        licence_summary["licence_memberships"].gt(0),
        licence_summary["female_licence_memberships"]
        / licence_summary["licence_memberships"],
        np.nan,
    )
    licence_summary["youth_membership_share"] = np.where(
        licence_summary["licence_memberships"].gt(0),
        licence_summary["youth_licence_memberships"]
        / licence_summary["licence_memberships"],
        np.nan,
    )

    club_summary = club_rows[
        [
            "department",
            "active_establishments",
            "population",
            "active_establishments_per_100000",
        ]
    ].rename(columns={"population": "club_population"})
    departments = licence_summary.merge(
        club_summary,
        on="department",
        how="inner",
        validate="one_to_one",
    )
    if len(departments) != len(club_summary):
        raise ValueError("Licence and club department coverage did not reconcile")

    equipment_rows = equipment.copy()
    if equipment_rows["equipment_id"].duplicated().any():
        raise ValueError("Equipment input has duplicate public equipment IDs")
    equipment_rows["department"] = equipment_rows["department"].map(normalize_department)
    equipment_rows = equipment_rows.loc[
        equipment_rows["department"].isin(in_scope_departments)
    ].copy()
    equipment_rows["dedicated_fencing_room"] = equipment_rows["equip_type_name"].eq(
        "Salle d'escrime"
    )
    equipment_rows["near_priority_area"] = (
        equipment_rows[["equip_qpv", "equip_qpv_200m"]].notna().any(axis=1)
    )
    equipment_rows["public_open_observed"] = equipment_rows["equip_ouv_public_bool"].notna()
    equipment_rows["motor_access_observed"] = equipment_rows["inst_acc_handi_bool"].notna()
    equipment_rows["transport_access_observed"] = equipment_rows["inst_trans_bool"].notna()
    equipment_rows["public_open_true"] = equipment_rows["equip_ouv_public_bool"].eq(True)
    equipment_rows["motor_access_true"] = equipment_rows["inst_acc_handi_bool"].eq(True)
    equipment_rows["transport_access_true"] = equipment_rows["inst_trans_bool"].eq(True)

    equipment_summary = equipment_rows.groupby("department", as_index=False).agg(
        equipment_records=("equipment_id", "size"),
        dedicated_fencing_rooms=("dedicated_fencing_room", "sum"),
        near_priority_area_records=("near_priority_area", "sum"),
        public_open_true=("public_open_true", "sum"),
        public_open_observed=("public_open_observed", "sum"),
        motor_access_true=("motor_access_true", "sum"),
        motor_access_observed=("motor_access_observed", "sum"),
        transport_access_true=("transport_access_true", "sum"),
        transport_access_observed=("transport_access_observed", "sum"),
        equipment_updated_min=("record_updated_on", "min"),
        equipment_updated_max=("record_updated_on", "max"),
    )
    departments = departments.merge(
        equipment_summary,
        on="department",
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "equipment_records",
        "dedicated_fencing_rooms",
        "near_priority_area_records",
        "public_open_true",
        "public_open_observed",
        "motor_access_true",
        "motor_access_observed",
        "transport_access_true",
        "transport_access_observed",
    ]
    departments[count_columns] = departments[count_columns].fillna(0).astype(int)
    departments["equipment_records_per_100000"] = np.where(
        departments["club_population"].gt(0),
        departments["equipment_records"] / departments["club_population"] * 100_000,
        np.nan,
    )
    departments["licences_per_active_establishment"] = np.where(
        departments["active_establishments"].gt(0),
        departments["licence_memberships"] / departments["active_establishments"],
        np.nan,
    )
    departments["dedicated_room_share"] = np.where(
        departments["equipment_records"].gt(0),
        departments["dedicated_fencing_rooms"] / departments["equipment_records"],
        np.nan,
    )
    departments["public_open_share_observed"] = np.where(
        departments["public_open_observed"].gt(0),
        departments["public_open_true"] / departments["public_open_observed"],
        np.nan,
    )
    departments["motor_access_share_observed"] = np.where(
        departments["motor_access_observed"].gt(0),
        departments["motor_access_true"] / departments["motor_access_observed"],
        np.nan,
    )
    departments["transport_access_share_observed"] = np.where(
        departments["transport_access_observed"].gt(0),
        departments["transport_access_true"] / departments["transport_access_observed"],
        np.nan,
    )
    departments["population_relative_difference"] = np.where(
        departments["club_population"].gt(0),
        (departments["licence_population"] - departments["club_population"])
        / departments["club_population"],
        np.nan,
    )
    return departments.sort_values("department", kind="stable").reset_index(drop=True)


def _bootstrap_spearman(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    values = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 4:
        raise ValueError(f"At least four complete departments are required for {left} vs {right}")
    left_values = values[left].to_numpy(dtype=float)
    right_values = values[right].to_numpy(dtype=float)
    if np.unique(left_values).size < 2 or np.unique(right_values).size < 2:
        raise ValueError(f"Correlation inputs are constant for {left} vs {right}")
    point = float(spearmanr(left_values, right_values).statistic)
    rng = np.random.default_rng(random_seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(values), size=len(values))
        if (
            np.unique(left_values[sample]).size < 2
            or np.unique(right_values[sample]).size < 2
        ):
            continue
        estimate = float(spearmanr(left_values[sample], right_values[sample]).statistic)
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        raise ValueError(f"No valid bootstrap correlations for {left} vs {right}")
    lower, upper = np.percentile(np.asarray(estimates), [2.5, 97.5])
    return {
        "left": left,
        "right": right,
        "departments": len(values),
        "spearman_rho": point,
        "bootstrap_unit": "department",
        "requested_repetitions": repetitions,
        "valid_repetitions": len(estimates),
        "percentile_95_interval": [float(lower), float(upper)],
    }


def _ranked_rows(frame: pd.DataFrame, column: str, *, ascending: bool) -> list[dict[str, Any]]:
    rows = frame.loc[frame[column].notna()].sort_values(
        [column, "department"], ascending=[ascending, True], kind="stable"
    )
    return [
        {
            "department": str(row.department),
            "department_label": str(row.department_label),
            column: float(getattr(row, column)),
        }
        for row in rows.head(5).itertuples(index=False)
    ]


def evaluate_spatial_access(
    departments: pd.DataFrame,
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Summarize distributions and department-resampled associations."""

    required = {
        "department",
        "department_label",
        "licence_memberships",
        "female_licence_memberships",
        "youth_licence_memberships",
        "licence_population",
        "licences_per_1000",
        "female_membership_share",
        "youth_membership_share",
        "club_population",
        "active_establishments",
        "active_establishments_per_100000",
        "equipment_records",
        "equipment_records_per_100000",
        "dedicated_fencing_rooms",
        "near_priority_area_records",
        "public_open_true",
        "public_open_observed",
        "motor_access_true",
        "motor_access_observed",
        "transport_access_true",
        "transport_access_observed",
        "licences_per_active_establishment",
        "population_relative_difference",
    }
    _require_columns(departments, required, "prepared spatial-analysis")
    if departments["department"].duplicated().any():
        raise ValueError("Prepared spatial input has duplicate departments")
    if bootstrap_repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")

    total_memberships = int(departments["licence_memberships"].sum())
    total_population = float(departments["licence_population"].sum())
    total_equipment = int(departments["equipment_records"].sum())
    rate_quantiles = departments["licences_per_1000"].quantile([0.1, 0.5, 0.9])
    pressure_quantiles = departments["licences_per_active_establishment"].quantile(
        [0.1, 0.5, 0.9]
    )
    associations = {
        "membership_rate_vs_active_establishment_density": _bootstrap_spearman(
            departments,
            "licences_per_1000",
            "active_establishments_per_100000",
            repetitions=bootstrap_repetitions,
            random_seed=random_seed,
        ),
        "membership_rate_vs_equipment_record_density": _bootstrap_spearman(
            departments,
            "licences_per_1000",
            "equipment_records_per_100000",
            repetitions=bootstrap_repetitions,
            random_seed=random_seed + 1,
        ),
        "female_membership_share_vs_active_establishment_density": _bootstrap_spearman(
            departments,
            "female_membership_share",
            "active_establishments_per_100000",
            repetitions=bootstrap_repetitions,
            random_seed=random_seed + 2,
        ),
    }
    return {
        "card_type": "descriptive_spatial_analysis",
        "claim_boundary": CLAIM_BOUNDARY,
        "evaluation_design": (
            "2024 in-scope department aggregation; current facility context; deterministic "
            "department bootstrap for rank-correlation uncertainty"
        ),
        "random_seed": random_seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "departments": len(departments),
        "departments_with_equipment_records": int(departments["equipment_records"].gt(0).sum()),
        "departments_with_active_establishments": int(
            departments["active_establishments"].gt(0).sum()
        ),
        "national_context": {
            "licence_memberships": total_memberships,
            "licence_population": total_population,
            "memberships_per_1000": (
                total_memberships / total_population * 1_000 if total_population else None
            ),
            "female_membership_share": (
                float(departments["female_licence_memberships"].sum()) / total_memberships
                if total_memberships
                else None
            ),
            "youth_membership_share": (
                float(departments["youth_licence_memberships"].sum()) / total_memberships
                if total_memberships
                else None
            ),
            "active_establishments": int(departments["active_establishments"].sum()),
            "equipment_records": total_equipment,
            "dedicated_fencing_rooms": int(departments["dedicated_fencing_rooms"].sum()),
            "listed_near_priority_area": int(
                departments["near_priority_area_records"].sum()
            ),
            "listed_open_to_public": {
                "true": int(departments["public_open_true"].sum()),
                "observed": int(departments["public_open_observed"].sum()),
            },
            "listed_motor_disability_access": {
                "true": int(departments["motor_access_true"].sum()),
                "observed": int(departments["motor_access_observed"].sum()),
            },
            "listed_public_transport_access": {
                "true": int(departments["transport_access_true"].sum()),
                "observed": int(departments["transport_access_observed"].sum()),
            },
        },
        "department_distribution": {
            "memberships_per_1000": {
                "p10": float(rate_quantiles.loc[0.1]),
                "median": float(rate_quantiles.loc[0.5]),
                "p90": float(rate_quantiles.loc[0.9]),
            },
            "memberships_per_active_establishment": {
                "p10": float(pressure_quantiles.loc[0.1]),
                "median": float(pressure_quantiles.loc[0.5]),
                "p90": float(pressure_quantiles.loc[0.9]),
            },
            "highest_membership_rate": _ranked_rows(
                departments, "licences_per_1000", ascending=False
            ),
            "lowest_membership_rate": _ranked_rows(
                departments, "licences_per_1000", ascending=True
            ),
            "highest_memberships_per_active_establishment": _ranked_rows(
                departments, "licences_per_active_establishment", ascending=False
            ),
        },
        "associations": associations,
        "data_quality": {
            "max_absolute_population_relative_difference": float(
                departments["population_relative_difference"].abs().max()
            ),
            "facility_boolean_missingness_is_not_imputed_false": True,
            "out_of_scope_geographies_excluded_using_club_in_france_flag": True,
            "equipment_records_without_in_scope_department": (
                "excluded from department analysis and retained in the source manifest"
            ),
        },
    }


def analyze_and_report(
    licences_path: Path,
    clubs_path: Path,
    equipment_path: Path,
    department_output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Run the full analysis and write local department rows plus aggregate public evidence."""

    licences = pd.read_parquet(licences_path)
    clubs = pd.read_parquet(clubs_path)
    equipment = pd.read_parquet(equipment_path)
    departments = prepare_department_frame(licences, clubs, equipment)
    report = evaluate_spatial_access(departments)
    department_output_path.parent.mkdir(parents=True, exist_ok=True)
    departments.to_parquet(department_output_path, index=False)
    equipment_dates = pd.to_datetime(equipment["record_updated_on"], errors="coerce").dropna()
    report.update(
        {
            "input_sha256": {
                "licences": sha256_file(licences_path),
                "clubs": sha256_file(clubs_path),
                "equipment": sha256_file(equipment_path),
            },
            "department_output_sha256": sha256_file(department_output_path),
            "department_output_policy": (
                "local aggregate table; public report contains summary only"
            ),
            "source_snapshots": {
                "licences_year": 2024,
                "clubs_year": 2024,
                "equipment_record_updated_min": equipment_dates.min().date().isoformat(),
                "equipment_record_updated_max": equipment_dates.max().date().isoformat(),
            },
        }
    )
    write_json(report_path, report)
    return report
