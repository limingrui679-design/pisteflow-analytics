from __future__ import annotations

import pytest

from pisteflow.sources.korean_fencing_tracker import (
    priority_note_category,
    video_id_from_value,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("data/raw/usaf_0HeqT9us5wA.mp4", "0HeqT9us5wA"),
        ("usaf_UzQ8Ci7lft8", "UzQ8Ci7lft8"),
        ("jr_foil_final_li_lin_hKUXgUsDOKE", "hKUXgUsDOKE"),
    ],
)
def test_video_id_from_value_uses_terminal_source_id(value: str, expected: str) -> None:
    assert video_id_from_value(value) == expected


def test_video_id_from_value_rejects_non_video_token() -> None:
    with pytest.raises(ValueError, match="11-character video ID"):
        video_id_from_value("local-report")


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("양측 램프(double) — 자동 입력", "automatic_double_light_score_proxy"),
        ("단독 램프(single_left) — 검토 필요", "needs_human_review_single_light"),
        ("램프 미판독 — 검토 필요", "needs_human_review_no_lamp"),
        ("득점자와 모순 — 검토 필요", "needs_human_review_conflict"),
        ("different note", "unclassified_upstream_note"),
    ],
)
def test_priority_note_category_preserves_review_boundary(note: str, expected: str) -> None:
    assert priority_note_category(note) == expected
