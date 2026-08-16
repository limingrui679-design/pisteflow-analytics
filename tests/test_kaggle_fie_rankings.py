from __future__ import annotations

from datetime import date

import pytest

from pisteflow.sources.kaggle_fie_rankings import (
    _identity_key,
    _parse_event_label,
    _parse_points,
)


def test_event_label_preserves_date_location_and_type() -> None:
    parsed = _parse_event_label("29.04.22 Tauberbischofsheim (A)")

    assert parsed == (date(2022, 4, 29), "Tauberbischofsheim", "A")


def test_parenthesized_points_remain_excluded_from_total() -> None:
    assert _parse_points("(8.000)") == (8.0, False)
    assert _parse_points(32.0) == (32.0, True)
    assert _parse_points(float("nan")) is None


def test_identity_key_normalizes_case_and_spacing_without_exposing_name() -> None:
    first = _identity_key("  CHOI   Injeong ", "KOR")
    second = _identity_key("choi injeong", "kor")

    assert first == second
    assert "choi" not in first


def test_malformed_event_label_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unexpected ranking event label"):
        _parse_event_label("2022 event")
