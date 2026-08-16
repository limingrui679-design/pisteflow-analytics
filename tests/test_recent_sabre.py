from pathlib import Path

import pandas as pd

from pisteflow.sources.recent_sabre import EXPECTED_FILES, prepare


def test_prepare_recent_sabre_excludes_byes_and_uses_scores(tmp_path: Path) -> None:
    event = tmp_path / "source" / "Example_2024_1"
    event.mkdir(parents=True)
    rows = {
        "athletes.csv": [[1, 100, "Alpha", 25, "AA"]],
        "elimination_rounds.csv": [
            ["64", "Alpha", "AA", 15, "Beta", "BB", 12, "Alias Beta"],
            ["32", "Alpha", "AA", 15, None, None, None, "Alpha"],
        ],
        "final_ranking.csv": [[1, 32, "Alpha", 25, "AA"]],
        "pool_results.csv": [[1, "Alpha", 25, 5, 5, 25, 10, 1.0, 15, "Qualified"]],
    }
    for filename, values in rows.items():
        pd.DataFrame(values, columns=EXPECTED_FILES[filename]).to_csv(event / filename, index=False)

    output = tmp_path / "processed.parquet"
    summary = prepare(tmp_path / "source", output, tmp_path / "summary.json")
    frame = pd.read_parquet(output)

    assert summary["events"] == 1
    assert summary["complete_scored_bouts"] == 1
    assert summary["bye_or_missing_opponent_rows"] == 1
    assert summary["winner_text_mismatches_resolved_from_score"] == 1
    assert frame.loc[0, "winner_side"] == "left"
    assert "Alpha" not in output.read_bytes().decode("latin1", errors="ignore")
