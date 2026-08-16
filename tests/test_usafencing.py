import pandas as pd

from pisteflow.sources.usafencing import normalize_bouts


def test_normalize_bouts_removes_names_and_maps_all_weapons() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "tournament_id": "t1",
                "date": "2020-01-01",
                "category": "Senior",
                "gender": "F",
                "weapon": "Epee",
                "num_fencers": 2,
            },
            {
                "event_id": "e2",
                "tournament_id": "t1",
                "date": "2020-01-02",
                "category": "Senior",
                "gender": "M",
                "weapon": "Foil",
                "num_fencers": 2,
            },
            {
                "event_id": "e3",
                "tournament_id": "t1",
                "date": "2020-01-03",
                "category": "Senior",
                "gender": "M",
                "weapon": "Saber",
                "num_fencers": 2,
            },
        ]
    )
    base = {
        "fencer1_id": "left-id",
        "fencer1_name": "Alpha",
        "fencer2_id": "right-id",
        "fencer2_name": "Beta",
        "score1": "5",
        "score2": "3",
        "winner": "Alpha",
    }
    pool = pd.DataFrame([{"bout_id": "b1", "event_id": "e1", **base}])
    de = pd.DataFrame(
        [
            {"bout_id": "b2", "event_id": "e2", **base},
            {"bout_id": "b3", "event_id": "e3", **base},
        ]
    )
    bouts, normalized_events = normalize_bouts(pool, de, events)
    assert set(bouts["weapon"]) == {"epee", "foil", "sabre"}
    assert set(normalized_events["gender"]) == {"women", "men"}
    assert "fencer1_name" not in bouts.columns
    assert "winner" not in bouts.columns
    assert bouts["left_won"].tolist() == [1, 1, 1]
