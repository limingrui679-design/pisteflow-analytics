from pathlib import Path

import pandas as pd

from pisteflow.sources.us_population import EXPECTED_COLUMNS, prepare


def test_prepare_population_snapshot_minimizes_identity_and_excludes_bad_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        [
            "01/02/2020",
            "Example Club: City, ST",
            "Senior Men's Epee",
            2,
            "E1",
            1,
            "Person One",
            "EX",
            "Example Club",
            "U",
            "E2020",
            "2020-01-02",
            "2020-01-02",
        ],
        [
            "01/02/2020",
            "Example Club: City, ST",
            "Senior Men's Epee",
            2,
            "E1",
            3,
            "Person Two",
            "EX",
            "Example Club",
            "U",
            None,
            "2020-01-02",
            "2020-01-02",
        ],
        [
            "01/03/2020",
            "Example Club: City, ST",
            "Team Event",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "2020-01-03",
            "2020-01-03",
        ],
    ]
    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(source / "data.csv", index=False)
    (source / "README.md").write_text("incomplete AskFRED snapshot", encoding="utf-8")

    output = tmp_path / "entries.parquet"
    summary = prepare(source, output, tmp_path / "summary.json")
    result = pd.read_parquet(output)

    assert summary["source_rows"] == 4
    assert summary["exact_duplicate_rows_excluded"] == 1
    assert summary["missing_required_rows_excluded"] == 1
    assert summary["invalid_place_gt_field_rows_excluded"] == 1
    assert len(result) == 1
    assert result.loc[0, "weapon"] == "epee"
    assert "Person One" not in result.to_json()
    assert "Example Club" not in result.to_json()
