import pytest

from pisteflow.sources.cmu_mocap import parse_amc_frames


def test_parse_contiguous_amc_frames() -> None:
    payload = (
        b"#!OML:ASF example.asf\r\n:FULLY-SPECIFIED\r\n:DEGREES\r\n"
        b"1\r\nroot 0\r\n2\r\nroot 1\r\n"
    )

    assert parse_amc_frames(payload) == 2


def test_reject_noncontiguous_amc_frames() -> None:
    payload = b":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0\n3\nroot 1\n"

    with pytest.raises(ValueError, match="not contiguous"):
        parse_amc_frames(payload)
