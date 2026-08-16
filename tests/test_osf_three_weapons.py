from __future__ import annotations

import pandas as pd

from pisteflow.sources import osf_three_weapons


def _shared(rows: int) -> dict[str, list[object]]:
    return {
        "year": [2025] * rows,
        "comp_id": [12] * rows,
        "date": ["2025-01-02"] * rows,
        "category": ["J"] * rows,
        "weapon": ["E"] * rows,
        "gender": ["F"] * rows,
    }


def test_athlete_key_is_stable_and_does_not_embed_identity() -> None:
    first = osf_three_weapons.athlete_key("  FENCER   One ", "USA")
    second = osf_three_weapons.athlete_key("fencer one", "usa")

    assert first == second
    assert "fencer" not in first.casefold()
    assert "usa" not in first.casefold()


def test_normalize_bouts_flags_tie_and_conflict(monkeypatch: object) -> None:
    monkeypatch.setitem(osf_three_weapons.EXPECTED_ROWS, "bouts", 3)  # type: ignore[attr-defined]
    frame = pd.DataFrame(
        {
            **_shared(3),
            "pool": [1, 1, 1],
            "A_name": ["One", "Two", "Three"],
            "A_nat": ["USA", "GBR", "FRA"],
            "A_seed": [1, 2, 3],
            "A_pos": [1, 2, 3],
            "B_name": ["Four", "Five", "Six"],
            "B_nat": ["ITA", "GER", "POL"],
            "B_seed": [4, 5, 6],
            "B_pos": [4, 5, 6],
            "scoreA": [5, 4, 5],
            "scoreB": [3, 4, 2],
            "winner": ["A", "A", "B"],
        }
    )

    result = osf_three_weapons.normalize_bouts(frame)

    assert result["outcome_eligible"].tolist() == [True, False, False]
    assert result["outcome_issue"].tolist() == [
        "none",
        "tied_score_with_declared_winner",
        "declared_winner_score_conflict",
    ]
    assert {"A_name", "A_nat", "B_name", "B_nat"}.isdisjoint(result.columns)


def test_normalize_rankings_removes_duplicate_and_age_fields(monkeypatch: object) -> None:
    monkeypatch.setitem(osf_three_weapons.EXPECTED_ROWS, "rankings", 2)  # type: ignore[attr-defined]
    row = {
        **{key: values[0] for key, values in _shared(1).items()},
        "event": "I",
        "rank": 1,
        "points": 32.0,
        "name": "One",
        "nat": "USA",
        "age": 42,
        "birth_year": 2008,
    }
    frame = pd.DataFrame([row, row])

    result = osf_three_weapons.normalize_rankings(
        frame,
        bout_competition_keys={"2025:12"},
    )

    assert len(result) == 1
    assert result["pool_corpus_match"].tolist() == [True]
    assert {"name", "nat", "age", "birth_year"}.isdisjoint(result.columns)
