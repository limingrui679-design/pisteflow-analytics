from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pisteflow.sources import askfred_v1


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prepare_deduplicates_round_pages_and_removes_direct_identity(tmp_path: Path) -> None:
    checkout = tmp_path / "archive"
    event = {
        "id": 100,
        "tournament_id": 200,
        "weapon": "Foil",
        "gender": "Women",
        "age_limit": "Junior",
        "rating_limit": "Open",
        "entries": 3,
        "is_team": False,
    }
    tournament = {"id": 200, "start_date": "2020-01-02"}
    round_record = {
        "id": 300,
        "event_id": 100,
        "tournament_id": 200,
        "round_type": "pool",
        "seq": 1,
        "pools": [
            {
                "bouts": [
                    {
                        "id": 400,
                        "seq": 1,
                        "fencers": [
                            {
                                "id": 1,
                                "usfa_id": "secret-1",
                                "first_name": "Alpha",
                                "last_name": "Person",
                                "gender": "F",
                                "birthyear": 2005,
                                "primary_club_id": 9,
                                "score": 5,
                                "status": "V",
                                "seed": 1,
                            },
                            {
                                "id": 2,
                                "usfa_id": "secret-2",
                                "first_name": "Beta",
                                "last_name": "Person",
                                "gender": "M",
                                "birthyear": 2006,
                                "primary_club_id": 8,
                                "score": 3,
                                "status": "D",
                                "seed": 2,
                            },
                        ],
                    },
                    {
                        "id": 401,
                        "seq": 2,
                        "fencers": [
                            {
                                "id": 3,
                                "first_name": "Gamma",
                                "last_name": "Person",
                                "score": 0,
                                "status": "V",
                            }
                        ],
                    },
                    {
                        "id": 402,
                        "seq": 3,
                        "fencers": [
                            {"id": 1, "score": 1, "status": "V"},
                            {"id": 2, "score": 1, "status": "D"},
                        ],
                    },
                ]
            }
        ],
    }
    _write(
        checkout / "data/endpoints/event/page-1",
        {"total_matched": 1, "per_page": 100, "events": [event]},
    )
    _write(
        checkout / "data/endpoints/tournament/page-1",
        {"total_matched": 1, "per_page": 100, "tournaments": [tournament]},
    )
    round_page = {"total_matched": 1, "per_page": 100, "rounds": [round_record]}
    _write(checkout / "data/endpoints/roundresult/page-1", round_page)
    _write(checkout / "data/endpoints/roundresult/page-duplicate", round_page)

    output = tmp_path / "bouts.parquet"
    summary = askfred_v1.prepare(
        checkout,
        output,
        tmp_path / "summary.json",
        repository_revision=askfred_v1.SNAPSHOT,
        batch_size=2,
    )
    result = pd.read_parquet(output)

    assert summary["unique_bout_records"] == 3
    assert summary["endpoint_audits"]["roundresult"]["duplicate_identical_rounds"] == 1
    assert summary["outcome_eligible_individual_bouts"] == 1
    assert summary["result_status_counts"] == {
        "bye_or_single_entry": 1,
        "decisive_consistent": 1,
        "tied_score": 1,
    }
    assert result["outcome_eligible"].sum() == 1
    assert set(result["event_gender"]) == {"women"}
    assert set(result["left_gender"].dropna()) == {"women"}
    payload = output.read_bytes().decode("latin1", errors="ignore")
    for private_value in ("Alpha", "Beta", "secret-1", "secret-2"):
        assert private_value not in payload
