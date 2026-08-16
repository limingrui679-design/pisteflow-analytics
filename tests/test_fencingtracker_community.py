from __future__ import annotations

from pathlib import Path

import pandas as pd
from pytest import MonkeyPatch

from pisteflow.provenance import sha256_file
from pisteflow.sources import fencingtracker_community


def test_prepare_privacy_minimizes_and_fails_closed_on_outcome_conflicts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "bouts.csv"
    rows = [
        {
            "bout_id": "10_1_2_Pool_5:3",
            "tournament_name": "Named tournament",
            "event_id": 10,
            "event_name": "Women's Foil",
            "weapon": "foil",
            "date": "January 2, 2024",
            "fencer_name": "Alpha Person",
            "fencer_id": 1,
            "fencer_gender": "F",
            "fencer_rating": "A24",
            "fencer_birth_year": 2008,
            "fencer_club": "Private Club",
            "opponent_name": "Beta Person",
            "opponent_id": 2,
            "opponent_gender": "M",
            "opponent_rating": "B23",
            "opponent_elo": 2000,
            "opponent_club": "Other Club",
            "fencer_elo": 2100,
            "result": "W",
            "score": "5:3",
            "bout_type": "Pool",
            "bout_type_raw": "Pool",
            "is_de": False,
            "win_probability": 55,
            "fencer_divisions": "I",
        },
        {
            "bout_id": "10_1_none_T4_2:0",
            "tournament_name": "Named tournament",
            "event_id": 10,
            "event_name": "Women's Foil",
            "weapon": "foil",
            "date": "January 2, 2024",
            "fencer_name": "Alpha Person",
            "fencer_id": 1,
            "fencer_gender": "F",
            "fencer_rating": "A24",
            "fencer_birth_year": 2008,
            "fencer_club": "Private Club",
            "opponent_name": "Beta Person",
            "opponent_id": None,
            "opponent_gender": "M",
            "opponent_rating": "B23",
            "opponent_elo": 2000,
            "opponent_club": "Other Club",
            "fencer_elo": 2100,
            "result": "L",
            "score": "2:0",
            "bout_type": "DE",
            "bout_type_raw": "T4",
            "is_de": True,
            "win_probability": 55,
            "fencer_divisions": "I",
        },
    ]
    pd.DataFrame(rows, columns=fencingtracker_community.EXPECTED_COLUMNS).to_csv(
        source, index=False
    )
    monkeypatch.setattr(fencingtracker_community, "EXPECTED_BYTES", source.stat().st_size)
    monkeypatch.setattr(fencingtracker_community, "EXPECTED_SHA256", sha256_file(source))
    monkeypatch.setattr(fencingtracker_community, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(fencingtracker_community, "EXPECTED_EVENT_IDS", 1)
    monkeypatch.setattr(fencingtracker_community, "EXPECTED_SOURCE_BOUT_IDS", 2)

    output = tmp_path / "bouts.parquet"
    summary = fencingtracker_community.prepare(source, output, tmp_path / "summary.json")
    result = pd.read_parquet(output)

    assert summary["retained_name_free_bouts"] == 2
    assert summary["model_eligible_decisive_bouts"] == 1
    assert summary["identity_slots_by_provenance"]["name_crosswalk"] == 1
    assert summary["event_gender_overrides"]["opponent_field_mismatches"] == 2
    assert set(result["result_status"]) == {
        "decisive_consistent",
        "score_result_conflict",
    }
    assert set(result["left_gender"]) == {"women"}
    assert set(result["right_gender"]) == {"women"}
    payload = output.read_bytes().decode("latin1", errors="ignore")
    for private_value in (
        "Alpha Person",
        "Beta Person",
        "Private Club",
        "Named tournament",
    ):
        assert private_value not in payload
