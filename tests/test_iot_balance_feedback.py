from pisteflow.sources import iot_balance_feedback


def _page(week: int) -> str:
    lines = [f"Week {week}"]
    means = iot_balance_feedback.EXPECTED_MEANS
    for prefix, group in iot_balance_feedback.GROUPS.items():
        movement, unbalance_two, unbalance_three = means[(group, week)]
        for number in range(1, 11):
            lines.append(
                f"{prefix}{number} {movement} {unbalance_two} {unbalance_three}"
            )
    return "\n".join(lines)


def test_parse_measurement_pages_preserves_repeated_measure_units() -> None:
    frame = iot_balance_feedback.parse_measurement_pages([_page(0), _page(5)])

    assert len(frame) == 60
    assert frame["participant_key"].nunique() == 30
    assert frame.groupby("participant_key").size().eq(2).all()
    assert frame.groupby(["study_group", "week"]).size().eq(10).all()
    assert set(frame["weapon"]) == {"unknown"}


def test_pinned_inner_supplement_hashes_are_explicit() -> None:
    assert iot_balance_feedback.NESTED_ZIP_BYTES == 99_873
    assert len(iot_balance_feedback.NESTED_ZIP_SHA256) == 64
    assert iot_balance_feedback.PDF_BYTES == 105_497
    assert len(iot_balance_feedback.PDF_SHA256) == 64
