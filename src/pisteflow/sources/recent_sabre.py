"""Validate a six-event 2023-2024 men's-sabre repository snapshot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json

SOURCE_URL = "https://github.com/KahHongg/Fencing_win_and_score_predictor"
LICENSE = "No repository license; local analysis only"
EXPECTED_FILES = {
    "athletes.csv": ["Rank", "Points", "Name", "Age", "Nationality"],
    "elimination_rounds.csv": [
        "Round",
        "Fencer A Name",
        "Fencer A Country",
        "Fencer A Score",
        "Fencer B Name",
        "Fencer B Country",
        "Fencer B Score",
        "Winner",
    ],
    "final_ranking.csv": ["Rank", "Points", "Name", "Age", "Nationality"],
    "pool_results.csv": ["Rk.", "Name", "Age", "V", "M", "TD", "TR", "Ind.", "Diff.", "Q"],
}


def _athlete_id(name: str) -> str:
    normalized = " ".join(name.split()).casefold()
    digest = hashlib.sha256(f"pisteflow-recent-sabre:{normalized}".encode()).hexdigest()[:20]
    return f"recent-sabre:{digest}"


def _tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prepare(
    source_directory: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Create a name-free bout table and a public aggregate audit."""

    event_directories = sorted(path for path in source_directory.iterdir() if path.is_dir())
    if not event_directories:
        raise ValueError("Recent-sabre snapshot has no event directories")

    source_files: list[Path] = []
    output_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    unique_names: set[str] = set()
    total_pool_rows = 0
    total_final_rows = 0
    total_athlete_rows = 0
    total_elimination_rows = 0
    bye_or_missing_rows = 0
    winner_text_mismatches = 0

    for event_directory in event_directories:
        tables: dict[str, pd.DataFrame] = {}
        for filename, expected_columns in EXPECTED_FILES.items():
            path = event_directory / filename
            if not path.is_file():
                raise ValueError(
                    f"Recent-sabre event is missing {filename}: {event_directory.name}"
                )
            source_files.append(path)
            table = pd.read_csv(path)
            if list(table.columns) != expected_columns:
                raise ValueError(
                    f"Recent-sabre schema changed for {event_directory.name}/{filename}"
                )
            tables[filename] = table

        athletes = tables["athletes.csv"]
        final_ranking = tables["final_ranking.csv"]
        pool_results = tables["pool_results.csv"]
        eliminations = tables["elimination_rounds.csv"]
        total_athlete_rows += len(athletes)
        total_final_rows += len(final_ranking)
        total_pool_rows += len(pool_results)
        total_elimination_rows += len(eliminations)

        for table, columns in (
            (athletes, ["Name"]),
            (final_ranking, ["Name"]),
            (pool_results, ["Name"]),
            (eliminations, ["Fencer A Name", "Fencer B Name", "Winner"]),
        ):
            for column in columns:
                unique_names.update(
                    str(value).strip()
                    for value in table[column].dropna()
                    if str(value).strip()
                )

        required = [
            "Fencer A Name",
            "Fencer B Name",
            "Fencer A Score",
            "Fencer B Score",
            "Winner",
        ]
        complete = eliminations[required].notna().all(axis=1)
        bye_or_missing_rows += int((~complete).sum())
        complete_rows = eliminations.loc[complete].copy()
        complete_rows["Fencer A Score"] = pd.to_numeric(
            complete_rows["Fencer A Score"], errors="raise"
        )
        complete_rows["Fencer B Score"] = pd.to_numeric(
            complete_rows["Fencer B Score"], errors="raise"
        )
        if complete_rows["Fencer A Score"].eq(complete_rows["Fencer B Score"]).any():
            raise ValueError(
                f"Recent-sabre event contains a tied elimination: {event_directory.name}"
            )

        for source_index, row in complete_rows.iterrows():
            left_name = str(row["Fencer A Name"]).strip()
            right_name = str(row["Fencer B Name"]).strip()
            left_score = int(row["Fencer A Score"])
            right_score = int(row["Fencer B Score"])
            winner_side = "left" if left_score > right_score else "right"
            score_winner = left_name if winner_side == "left" else right_name
            winner_text = str(row["Winner"]).strip()
            if winner_text != score_winner:
                winner_text_mismatches += 1
            output_rows.append(
                {
                    "bout_id": f"recent-sabre:{event_directory.name}:{source_index}",
                    "event_id": event_directory.name,
                    "weapon": "sabre",
                    "gender": "men",
                    "round": str(row["Round"]),
                    "left_fencer_id": _athlete_id(left_name),
                    "right_fencer_id": _athlete_id(right_name),
                    "left_score": left_score,
                    "right_score": right_score,
                    "winner_side": winner_side,
                }
            )

        event_summaries.append(
            {
                "event_id": event_directory.name,
                "athlete_rows": len(athletes),
                "pool_athlete_rows": len(pool_results),
                "final_ranking_rows": len(final_ranking),
                "elimination_table_rows": len(eliminations),
                "complete_scored_bouts": len(complete_rows),
            }
        )

    frame = pd.DataFrame(output_rows).sort_values(["event_id", "bout_id"], kind="stable")
    if frame["bout_id"].duplicated().any():
        raise ValueError("Recent-sabre generated bout IDs are not unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    output_sha256 = sha256_file(output_path)
    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "weapon": "sabre",
        "gender": "men",
        "events": len(event_directories),
        "event_summaries": event_summaries,
        "athlete_snapshot_rows": total_athlete_rows,
        "pool_athlete_event_rows": total_pool_rows,
        "final_ranking_rows": total_final_rows,
        "elimination_table_rows": total_elimination_rows,
        "complete_scored_bouts": len(frame),
        "bye_or_missing_opponent_rows": bye_or_missing_rows,
        "winner_text_mismatches_resolved_from_score": winner_text_mismatches,
        "unique_published_name_spellings": len(unique_names),
        "source_csv_tree_sha256": _tree_sha256(source_files, source_directory),
        "output_sha256": output_sha256,
        "output": str(output_path),
        "claim_boundary": (
            "Six public men's-sabre event snapshots from 2023-2024. Complete scored "
            "eliminations are retained in a name-free local table; byes and missing opponents are "
            "excluded. One winner-text spelling mismatch is resolved only from the two scores. "
            "The repository has no license, so source CSVs, normalized rows, and models remain "
            "local and this is not a current official FIE feed."
        ),
    }
    write_json(report_path, summary)
    return summary
