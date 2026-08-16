import pytest

from pisteflow.sources.hzume_epee_annotations import merged_interval_seconds, parse_label


def test_parse_label_preserves_side_action_and_outcome() -> None:
    assert parse_label("l_lunge_hit") == ("l", "lunge", "hit")
    assert parse_label("r_parry_success") == ("r", "parry", "success")


def test_parse_label_rejects_undocumented_semantics() -> None:
    with pytest.raises(ValueError, match="Unexpected Hzume epee label"):
        parse_label("right_attack_winner")


def test_merged_interval_seconds_does_not_double_count_overlap() -> None:
    assert merged_interval_seconds([(1.0, 3.0), (2.0, 4.0), (6.0, 7.5)]) == 4.5
    assert merged_interval_seconds([]) == 0.0
