from pathlib import Path

from pisteflow.sources import github_fencing_media


def test_three_weapon_original_video_coverage() -> None:
    original_specs = [
        *(
            spec
            for spec in github_fencing_media.FENCSTATS_MEDIA.values()
            if spec.media_role == "original_broadcast_excerpt"
        ),
        *(
            spec
            for spec in github_fencing_media.THEEPHOP_MEDIA.values()
            if spec.derivative_of is None
        ),
        *(
            spec
            for spec in github_fencing_media.TENNIS_VIDEO_MEDIA.values()
            if spec.derivative_of is None
        ),
    ]

    assert len(original_specs) == 10
    assert {spec.weapon for spec in original_specs} == {"epee", "foil", "sabre"}


def test_visual_group_boundaries_collapse_derivatives_and_repeated_bouts() -> None:
    all_specs = [
        *github_fencing_media.FENCSTATS_MEDIA.values(),
        *github_fencing_media.THEEPHOP_MEDIA.values(),
        *github_fencing_media.TENNIS_VIDEO_MEDIA.values(),
    ]

    assert len({spec.source_group_id for spec in all_specs}) == 14
    assert (
        github_fencing_media.FENCSTATS_MEDIA[
            "data/inference/video/SlimyDependentIbadanmalimbe-mobile.mp4"
        ].source_group_id
        == github_fencing_media.FENCSTATS_MEDIA[
            "data/inference/video/WigglyAlarmedKitfox-mobile.mp4"
        ].source_group_id
    )
    assert (
        github_fencing_media.TENNIS_VIDEO_MEDIA["OUTPUTVID.avi"].source_group_id
        == github_fencing_media.TENNIS_VIDEO_MEDIA[
            "example_input_videos/video144p.avi"
        ].source_group_id
    )


def test_inventory_digest_uses_relative_paths_and_content(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "nested/b.jpg"
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    forward = github_fencing_media._inventory_digest([first, second], tmp_path)
    reverse = github_fencing_media._inventory_digest([second, first], tmp_path)

    assert forward == reverse
    second.write_bytes(b"changed")
    assert github_fencing_media._inventory_digest([first, second], tmp_path) != forward
