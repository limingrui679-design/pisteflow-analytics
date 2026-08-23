from __future__ import annotations

import numpy as np
import pandas as pd

from pisteflow.analysis.spatial_access import (
    CLAIM_BOUNDARY,
    evaluate_spatial_access,
    normalize_department,
    prepare_department_frame,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    licences: list[dict[str, object]] = []
    clubs: list[dict[str, object]] = []
    equipment: list[dict[str, object]] = []
    for index, department in enumerate(["01", "02", "03", "04"], start=1):
        population = 100_000 + index * 10_000
        for sex in ["female", "male"]:
            for band in ["10 à 14 ans", "30 à 34 ans"]:
                licences.append(
                    {
                        "department": department,
                        "department_label": f"{department} - Test",
                        "sex": sex,
                        "population_age_band": band,
                        "licence_count": index**2 + 2 if sex == "female" else index * 5,
                        "population": population / 4,
                        "year": 2024,
                    }
                )
        clubs.append(
            {
                "department": department,
                "department_label": f"{department} - Test",
                "year": 2024,
                "in_france": 1,
                "active_establishments": index,
                "population": population,
                "active_establishments_per_100000": index / population * 100_000,
            }
        )
        for equipment_index in range(index):
            equipment.append(
                {
                    "equipment_id": f"E-{department}-{equipment_index}",
                    "department": str(index),
                    "equip_type_name": (
                        "Salle d'escrime" if equipment_index == 0 else "Salle multisports"
                    ),
                    "equip_ouv_public_bool": True if equipment_index == 0 else None,
                    "inst_acc_handi_bool": equipment_index % 2 == 0,
                    "inst_trans_bool": equipment_index % 2 == 1,
                    "equip_qpv": "QPV" if equipment_index == 0 else None,
                    "equip_qpv_200m": None,
                    "record_updated_on": "2025-03-31",
                }
            )
    return pd.DataFrame(licences), pd.DataFrame(clubs), pd.DataFrame(equipment)


def test_normalize_department_preserves_non_numeric_codes() -> None:
    assert normalize_department("1") == "01"
    assert normalize_department(2.0) == "02"
    assert normalize_department("2A") == "2A"
    assert normalize_department("971") == "971"


def test_prepare_department_frame_preserves_units_and_missingness() -> None:
    departments = prepare_department_frame(*_inputs())

    assert list(departments["department"]) == ["01", "02", "03", "04"]
    first = departments.iloc[0]
    assert first["licence_memberships"] == 16
    assert first["female_licence_memberships"] == 6
    assert first["youth_licence_memberships"] == 8
    assert first["licence_population"] == 110_000
    assert first["equipment_records"] == 1
    assert first["dedicated_fencing_rooms"] == 1
    assert first["public_open_observed"] == 1
    assert np.isclose(first["licences_per_active_establishment"], 16.0)


def test_evaluate_spatial_access_is_bounded_and_bootstrapped() -> None:
    departments = prepare_department_frame(*_inputs())
    report = evaluate_spatial_access(
        departments,
        bootstrap_repetitions=100,
        random_seed=7,
    )

    assert report["departments"] == 4
    assert report["national_context"]["equipment_records"] == 10
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    association = report["associations"][
        "membership_rate_vs_active_establishment_density"
    ]
    assert association["requested_repetitions"] == 100
    assert association["valid_repetitions"] > 0
    assert len(association["percentile_95_interval"]) == 2
