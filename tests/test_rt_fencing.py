from pisteflow.sources.rt_fencing import parse_member


def test_parse_rt_fencing_member_preserves_match_group() -> None:
    row = parse_member(
        "train/LNG/f-528.13.0.png",
        file_size=100,
        compressed_size=80,
        crc32=123,
    )
    assert row["action_label"] == "lunge"
    assert row["is_augmented"] is True
    assert row["group_id"] == "rtf:match:528"
    assert row["fencer_side"] == "left"


def test_parse_rt_fencing_preserves_unexpected_upstream_fencer_code() -> None:
    row = parse_member("test/NON/4.4.2.png", file_size=10, compressed_size=8, crc32=1)

    assert row["fencer_code"] == "2"
    assert row["fencer_side"] == "other_2"


def test_parse_rt_fencing_marks_packaging_copy() -> None:
    row = parse_member("train/CTR/f-2.4.1(1).png", file_size=10, compressed_size=8, crc32=1)

    assert row["is_augmented"] is True
    assert row["is_packaging_duplicate"] is True
    assert row["copy_variant"] == "paren_1"
