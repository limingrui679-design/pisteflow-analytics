from __future__ import annotations

from pisteflow.sources.womensports_fencing import _hamming, perceptual_groups


def test_perceptual_groups_join_near_hashes_transitively() -> None:
    hashes = ["0000000000000000", "0000000000000001", "0000000000000003", "ffffffffffffffff"]

    groups = perceptual_groups(hashes, threshold=1)

    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_hamming_counts_bit_differences() -> None:
    assert _hamming("0000000000000000", "000000000000000f") == 4
