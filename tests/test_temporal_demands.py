from pathlib import Path

from pisteflow.sources.temporal_demands import parse_table


def test_parse_published_temporal_table() -> None:
    fixture = Path(__file__).parent / "fixtures" / "temporal-demands-table.xml"
    frame = parse_table(fixture.read_bytes())

    assert len(frame) == 6
    assert frame["bouts_n"].sum() == 83
    assert frame.loc[frame["weapon"].eq("sabre"), "average_action_time_s_mean"].max() == 1.9
