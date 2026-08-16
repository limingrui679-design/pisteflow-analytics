import pytest

from pisteflow.sources import g1_moves_fencing


def test_parse_bvh_header_counts_frames_time_and_channels() -> None:
    text = """HIERARCHY
ROOT Hips
{
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Chest
  {
    CHANNELS 3 Zrotation Xrotation Yrotation
  }
}
MOTION
Frames: 2
Frame Time: 0.0166667
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
"""

    frames, frame_time, channels = g1_moves_fencing.parse_bvh_header(text)

    assert frames == 2
    assert frame_time == pytest.approx(0.0166667)
    assert channels == 9


def test_pinned_snapshot_preserves_performer_conflict() -> None:
    assert g1_moves_fencing.README_PERFORMER != g1_moves_fencing.MANIFEST_PERFORMER
    assert sum(row["frames"] for row in g1_moves_fencing.EXPECTED_MANIFEST.values()) == 2258
    assert sum(
        row["training_frames"] for row in g1_moves_fencing.EXPECTED_MANIFEST.values()
    ) == 2256
