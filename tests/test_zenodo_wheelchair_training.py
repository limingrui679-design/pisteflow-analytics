from __future__ import annotations

from collections import Counter

from pisteflow.sources.zenodo_wheelchair_training import published_rows


def test_published_rows_preserve_source_and_group_boundaries() -> None:
    rows = published_rows()

    assert len(rows) == 62
    assert Counter(row["source"] for row in rows) == {
        "high_stress": 26,
        "biomechanics": 36,
    }
    assert {row["weapon"] for row in rows} == {"unknown"}
    assert {row["measurement_level"] for row in rows} == {
        "published_group_aggregate"
    }


def test_high_stress_unspecified_before_after_group_is_not_invented() -> None:
    rows = [
        row
        for row in published_rows()
        if row["source"] == "high_stress" and row["table"] == "table_3"
    ]

    assert len(rows) == 8
    assert {row["group"] for row in rows} == {"not_specified_in_table"}
    assert {row["reported_change_unit"] for row in rows} == {
        "percentage_points_labelled_as_percent"
    }


def test_biomechanics_printed_change_is_retained_verbatim() -> None:
    rows = [
        row
        for row in published_rows()
        if row["source"] == "biomechanics"
        and row["metric"] == "attack_time"
        and row["group"] == "experimental"
    ]

    assert [row["value"] for row in rows] == [0.85, 0.72]
    assert {row["reported_change"] for row in rows} == {-16.5}
