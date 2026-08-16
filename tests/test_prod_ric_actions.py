import pytest

from pisteflow.sources.prod_ric_actions import EXPECTED_CLIPS, classify_relative_path


def test_classify_relative_path_preserves_upstream_directory_label() -> None:
    spec = classify_relative_path("src/data/videoes/lunge/lunge1.mov")
    assert spec.action_label == "lunge"
    assert spec.weapon == "weapon_agnostic_footwork"


def test_repeated_demonstrator_and_bout_clips_share_leakage_groups() -> None:
    assert (
        EXPECTED_CLIPS["src/data/videoes/advance/fastadvance.mov"].source_group_id
        == EXPECTED_CLIPS["src/data/videoes/lunge/lunge1.mov"].source_group_id
    )
    assert (
        EXPECTED_CLIPS["testvideoes/testvideo1.mov"].source_group_id
        == EXPECTED_CLIPS["testvideoes/testvideo2.mov"].source_group_id
    )


def test_classify_relative_path_rejects_unreviewed_media() -> None:
    with pytest.raises(ValueError, match="Unexpected prod-ric video path"):
        classify_relative_path("src/data/videoes/idle/synthetic.mp4")
