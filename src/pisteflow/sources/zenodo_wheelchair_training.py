"""Prepare two license-pinned wheelchair-fencing group-aggregate studies."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

LICENSE = "CC BY 4.0"
COHORT_BOUNDARY = (
    "Both papers report 14 Class A para fencers from the same two institutions, use seven-person "
    "control and experimental groups, and describe a 14-week intervention. They may reuse one "
    "cohort, so the repository reports at most 14 distinct participants rather than adding them "
    "to 28."
)
EVIDENCE_BOUNDARY = (
    "Only publisher-reported group aggregates are available. No participant-level rows, protocol "
    "files, exact test statistics, or independent replication are supplied; these rows support "
    "descriptive sensitivity analysis, not athlete-level prediction or causal effectiveness claims."
)

SOURCE_SPECS: dict[str, dict[str, str | int]] = {
    "high_stress": {
        "record_id": "21195576",
        "doi": "10.5281/zenodo.21195576",
        "title": (
            "Studying decision-making and reflex speed in wheelchair fencers under high-stress "
            "conditions"
        ),
        "filename": "21195576-high-stress.pdf",
        "bytes": 659_520,
        "sha256": "7fd148351c958445e6bd70baaaa3bdbfca5a892d03901f49942e03a4962d8b16",
    },
    "biomechanics": {
        "record_id": "21200227",
        "doi": "10.5281/zenodo.21200227",
        "title": "Optimizing the biomechanics of technical movements of wheelchair fencers",
        "filename": "21200227-biomechanics.pdf",
        "bytes": 664_609,
        "sha256": "2269bac317923882e2793e0767711a2244250d36bd95f92d736cce42eca78853",
    },
}


def _row(
    source: str,
    table: str,
    page: int,
    metric: str,
    group: str,
    condition: str,
    value: float,
    unit: str,
    *,
    sd: float | None = None,
    reported_change: float | None = None,
    reported_change_unit: str | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[source]
    return {
        "source": source,
        "zenodo_record_id": spec["record_id"],
        "doi": spec["doi"],
        "table": table,
        "source_pdf_page": page,
        "metric": metric,
        "group": group,
        "condition": condition,
        "value": value,
        "unit": unit,
        "sd": sd,
        "reported_change": reported_change,
        "reported_change_unit": reported_change_unit,
        "group_n": n,
        "weapon": "unknown",
        "classification": "wheelchair_fencing_class_a",
        "measurement_level": "published_group_aggregate",
        "independently_validated": False,
    }


def published_rows() -> list[dict[str, Any]]:
    """Return a faithful long-form transcription of the visible published tables."""

    rows: list[dict[str, Any]] = []

    high_stress_accuracy = {
        "rapid_situation_comprehension": (62.0, 81.0),
        "correct_tactical_solution_selection": (65.0, 84.0),
        "decision_execution_accuracy": (69.0, 86.0),
        "timely_decision_execution": (64.0, 83.0),
        "overall_effectiveness": (65.0, 83.5),
    }
    for metric, (accuracy_control, accuracy_experimental) in high_stress_accuracy.items():
        rows.extend(
            [
                _row(
                    "high_stress",
                    "table_1",
                    3,
                    metric,
                    "control",
                    "high_stress",
                    accuracy_control,
                    "percent",
                    n=7,
                ),
                _row(
                    "high_stress",
                    "table_1",
                    3,
                    metric,
                    "experimental",
                    "high_stress",
                    accuracy_experimental,
                    "percent",
                    n=7,
                ),
            ]
        )

    stress_stability = {
        "low_stress": (78.0, 90.0),
        "moderate_stress": (66.0, 85.0),
        "high_stress": (54.0, 79.0),
        "very_high_stress": (47.0, 72.0),
    }
    for condition, (stability_control, stability_experimental) in stress_stability.items():
        rows.extend(
            [
                _row(
                    "high_stress",
                    "table_2",
                    4,
                    "reflex_stability",
                    "control",
                    condition,
                    stability_control,
                    "percent",
                    n=7,
                ),
                _row(
                    "high_stress",
                    "table_2",
                    4,
                    "reflex_stability",
                    "experimental",
                    condition,
                    stability_experimental,
                    "percent",
                    n=7,
                ),
            ]
        )

    before_after = {
        "stress_resilience": (58.0, 81.0, 23.0),
        "decision_speed_score": (61.0, 85.0, 24.0),
        "decision_accuracy": (63.0, 88.0, 25.0),
        "reflex_control": (60.0, 84.0, 24.0),
    }
    for metric, (before, after, change) in before_after.items():
        rows.extend(
            [
                _row(
                    "high_stress",
                    "table_3",
                    5,
                    metric,
                    "not_specified_in_table",
                    "before_training",
                    before,
                    "percent",
                    reported_change=change,
                    reported_change_unit="percentage_points_labelled_as_percent",
                ),
                _row(
                    "high_stress",
                    "table_3",
                    5,
                    metric,
                    "not_specified_in_table",
                    "after_training",
                    after,
                    "percent",
                    reported_change=change,
                    reported_change_unit="percentage_points_labelled_as_percent",
                ),
            ]
        )

    characteristics = {
        "age": ((22.1, 1.3), (21.8, 1.5), "years"),
        "sport_experience": ((4.2, 0.8), (4.5, 0.7), "years"),
    }
    for metric, (control_characteristic, experimental_characteristic, unit) in (
        characteristics.items()
    ):
        rows.extend(
            [
                _row(
                    "biomechanics",
                    "table_1",
                    3,
                    metric,
                    "control",
                    "baseline",
                    control_characteristic[0],
                    unit,
                    sd=control_characteristic[1],
                    n=7,
                ),
                _row(
                    "biomechanics",
                    "table_1",
                    3,
                    metric,
                    "experimental",
                    "baseline",
                    experimental_characteristic[0],
                    unit,
                    sd=experimental_characteristic[1],
                    n=7,
                ),
            ]
        )

    biomechanics = {
        "attack_time": ((0.84, 0.06, 0.82, 0.05, -2.4), (0.85, 0.05, 0.72, 0.04, -16.5), "seconds"),
        "hit_accuracy": ((65.1, 4.3, 66.8, 4.1, 2.6), (64.8, 3.9, 79.4, 3.6, 22.2), "percent"),
        "counterattack_reaction_time": (
            (420.0, 35.0, 410.0, 32.0, -2.3),
            (425.0, 30.0, 360.0, 28.0, -15.3),
            "milliseconds",
        ),
        "trunk_stability": ((3.8, 0.4, 4.0, 0.5, 5.3), (3.7, 0.5, 4.8, 0.4, 29.7), "score"),
    }
    for metric, (control_biomechanics, experimental_biomechanics, unit) in (
        biomechanics.items()
    ):
        for group, biomechanics_values in (
            ("control", control_biomechanics),
            ("experimental", experimental_biomechanics),
        ):
            before, before_sd, after, after_sd, change = biomechanics_values
            rows.extend(
                [
                    _row(
                        "biomechanics",
                        "table_2",
                        3,
                        metric,
                        group,
                        "before_training",
                        before,
                        unit,
                        sd=before_sd,
                        reported_change=change,
                        reported_change_unit="percent_as_printed",
                        n=7,
                    ),
                    _row(
                        "biomechanics",
                        "table_2",
                        3,
                        metric,
                        group,
                        "after_training",
                        after,
                        unit,
                        sd=after_sd,
                        reported_change=change,
                        reported_change_unit="percent_as_printed",
                        n=7,
                    ),
                ]
            )

    tactical = {
        "attacks": ((42.6, 43.1, 1.2), (43.1, 47.9, 11.1)),
        "counterattacks": ((36.3, 36.4, 0.3), (37.0, 40.1, 8.4)),
        "parry_riposte": ((25.9, 26.0, 0.4), (26.0, 30.9, 18.8)),
        "attacks_with_blade_action": ((13.7, 13.8, 0.7), (13.9, 18.8, 35.3)),
    }
    for metric, (control_tactical, experimental_tactical) in tactical.items():
        for group, tactical_values in (
            ("control", control_tactical),
            ("experimental", experimental_tactical),
        ):
            before, after, change = tactical_values
            rows.extend(
                [
                    _row(
                        "biomechanics",
                        "table_3",
                        4 if metric != "attacks_with_blade_action" else 5,
                        metric,
                        group,
                        "before_training",
                        before,
                        "percent",
                        reported_change=change,
                        reported_change_unit="percent_as_printed",
                        n=7,
                    ),
                    _row(
                        "biomechanics",
                        "table_3",
                        4 if metric != "attacks_with_blade_action" else 5,
                        metric,
                        group,
                        "after_training",
                        after,
                        "percent",
                        reported_change=change,
                        reported_change_unit="percent_as_printed",
                        n=7,
                    ),
                ]
            )

    return rows


def _validate_sources(source_directory: Path) -> list[dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    for source, spec in SOURCE_SPECS.items():
        path = source_directory / str(spec["filename"])
        if not path.exists():
            raise FileNotFoundError(f"Missing pinned Zenodo PDF: {path}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != spec["bytes"] or actual_sha256 != spec["sha256"]:
            raise ValueError(f"Pinned Zenodo PDF changed: {path.name}")
        source_rows.append(
            {
                "source": source,
                "record_id": spec["record_id"],
                "doi": spec["doi"],
                "title": spec["title"],
                "filename": path.name,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "license": LICENSE,
            }
        )
    return source_rows


def prepare(source_directory: Path, output_path: Path, summary_path: Path) -> dict[str, Any]:
    """Validate fixed PDFs and write the visible tables as a long-form Parquet dataset."""

    sources = _validate_sources(source_directory)
    rows = published_rows()
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    reported_change_discrepancies = 0
    for (source, table, metric, group), part in frame.dropna(
        subset=["reported_change"]
    ).groupby(["source", "table", "metric", "group"]):
        del source, table, metric, group
        before = part.loc[part["condition"].eq("before_training"), "value"]
        after = part.loc[part["condition"].eq("after_training"), "value"]
        if before.empty or after.empty:
            continue
        recomputed = (float(after.iloc[0]) - float(before.iloc[0])) / float(before.iloc[0]) * 100
        printed = float(part["reported_change"].iloc[0])
        if abs(recomputed - printed) > 0.25:
            reported_change_discrepancies += 1

    summary: dict[str, Any] = {
        "sources": sources,
        "rows": len(frame),
        "metrics": int(frame["metric"].nunique()),
        "rows_by_source": dict(Counter(frame["source"])),
        "published_participants_per_paper": 14,
        "maximum_distinct_participants_across_both_papers": 14,
        "cohort_overlap_status": "likely_same_cohort_not_verified",
        "reported_change_groups_over_0_25pp_from_recomputed_relative_change": (
            reported_change_discrepancies
        ),
        "license": LICENSE,
        "cohort_boundary": COHORT_BOUNDARY,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "claim_boundary": (
            "Publisher-reported aggregate tables from two CC BY 4.0 Zenodo records. The source "
            "weapon is not reported, and the two papers are not counted as independent athlete "
            "cohorts. Printed change columns are preserved even when they differ from a recomputed "
            "relative-percent change."
        ),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(summary_path, summary)
    return summary
