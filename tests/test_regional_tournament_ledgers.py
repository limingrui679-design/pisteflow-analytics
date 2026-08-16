from __future__ import annotations

from pisteflow.sources.regional_tournament_ledgers import parse_ledger_lines


def test_ledger_parser_keeps_scored_bouts_walkovers_and_audit_anomalies() -> None:
    rows, audit = parse_ledger_lines(
        [
            "Oct 2025 Example Open Tournament (32)",
            "Left Fencer,Right Fencer,15,12",
            "Walkover Winner,Walkover Loser,>,",
            "/",
            "Tied One,Tied Two,14,14",
        ],
        event_id="001",
        source_file="001 Example.csv",
    )

    assert len(rows) == 2
    assert [row["outcome_kind"] for row in rows] == ["scored", "walkover"]
    assert rows[0]["weapon"] == "unknown"
    assert rows[0]["event_month"] == "2025-10"
    assert audit == {
        "scored_bouts": 1,
        "walkovers": 1,
        "placeholder_lines": 1,
        "tied_score_anomalies": 1,
    }


def test_unknown_gender_and_weapon_are_not_inferred() -> None:
    rows, _audit = parse_ledger_lines(
        ["GT Open Tournament", "Fencer A,Fencer B,15,9"],
        event_id="002",
        source_file="002 Example.csv",
    )

    assert rows[0]["event_month"] is None
    assert rows[0]["gender_scope"] == "mixed_or_unspecified"
    assert rows[0]["weapon"] == "unknown"
