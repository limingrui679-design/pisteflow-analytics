from pisteflow.sources import riposte_adapted_fencing as source


def test_normalize_attendance_rows() -> None:
    rows = [
        [
            "N°",
            "Group",
            "Loss to follow-up",
            "Number of participated sessions",
            "Total number of sessions proposed",
            "Participation rate",
        ],
        ["1", " Delayed RIPOSTE", "No", "14", "15", "93%"],
        ["2", "Early RIPOSTE", "Yes", "3 and then withdraw", "", ""],
    ]

    result = source.normalize_attendance_rows(rows)

    assert len(result) == 2
    assert result["allocation_timing"].tolist() == ["delayed", "early"]
    assert result["lost_to_follow_up"].tolist() == [False, True]
    assert result["participated_sessions"].tolist() == [14, 3]
    assert result["proposed_sessions"].tolist()[0] == 15
    assert result["participant_code"].str.startswith("rip-").all()


def test_normalize_outcome_rows_with_negative_interval_bound() -> None:
    rows = [
        ["QLQ-C30", "", "", "", "", "", "", ""],
        [
            "Dyspnea",
            "9",
            "-3.7 (-27.5 - 20.1)",
            "10",
            "-13.3 (-25.6--1.0)",
            "",
            "-9.6 (-33.5-14.3)",
            "0.40",
        ],
    ]

    result = source.normalize_outcome_rows(rows)

    assert len(result) == 1
    assert result.loc[0, "instrument"] == "QLQ-C30"
    assert result.loc[0, "delayed_change_ci_high"] == -1.0
    assert result.loc[0, "difference_of_change_mean"] == -9.6
