from pisteflow.sources.paris_2024 import parse_individual_page, parse_team_page


def test_parse_individual_bout_without_retaining_names() -> None:
    text = """
    Women's Épée Individual
    SAT 27 JUL 2024 Table of 64
    Bout Results
    BOUT: 31 PISTE: Red
      34 EHAB Nardin EGY 15 - 9 31 GABER Shirwit EGY
    Bout Duration: 17:26 Overtime: --
    FENWEPEE--------------R64-003100--_73I
    """

    row = parse_individual_page(text, 20)

    assert row["weapon"] == "epee"
    assert row["stage"] == "table_of_64"
    assert row["winner_side"] == "left"
    assert row["duration_seconds"] == 1046
    assert "EHAB" not in str(row)


def test_parse_team_match_and_all_nine_relays() -> None:
    relays = "\n".join(
        f"LEFT Athlete {i} {i} {i} -- {i} {i} RIGHT Athlete {i}" for i in range(1, 9)
    )
    relays += "\nLEFT Athlete 9 1 9 OT 10 2 RIGHT Athlete 9 (S)"
    text = f"""
    Men's Foil Team
    SUN 4 AUG 2024 Gold Medal Match
    Match Results
    MATCH: 12 PISTE: Podium
    ITA - Italy (1) 9 Final 10 JPN - Japan (2)
    Fencer Touches Score vs Score Touches Fencer
    {relays}
    Match Duration: 45:00 Overtime: 0:30
    FENMTEAMFOIL----------FNL-000100--_73T
    """

    match, relay_rows = parse_team_page(text, 341)

    assert match["stage"] == "gold_medal"
    assert match["winner_side"] == "right"
    assert len(relay_rows) == 9
    assert relay_rows[-1]["overtime_relay"] is True
    assert relay_rows[-1]["right_substitute"] is True
    assert "Athlete" not in str(relay_rows)
