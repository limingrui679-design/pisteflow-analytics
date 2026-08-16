"""CC BY survey datasets covering competitive-fencing injuries."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

GENERAL_DOI = "https://doi.org/10.18745/ds.00026192"
GENERAL_PAGE = "https://uhra.herts.ac.uk/id/eprint/26192/"
GENERAL_DATA_URL = "https://uhra.herts.ac.uk/id/eprint/26192/1/Epi_Data%20%281%29.xlsx"
GENERAL_README_URL = "https://uhra.herts.ac.uk/id/eprint/26192/2/README%20File.txt"
HAND_DOI = "https://doi.org/10.18745/ds.00027141"
HAND_PAGE = "https://uhra.herts.ac.uk/id/eprint/27141/"
HAND_DATA_URL = "https://uhra.herts.ac.uk/id/eprint/27141/4/fencing_hand_data.csv"
HAND_README_URL = "https://uhra.herts.ac.uk/id/eprint/27141/2/README.txt"
LICENSE = "CC BY (version not stated in deposited README)"

_INJURY_FIELDS = [
    "Body_Part",
    "Body_Side",
    "Injury_Type",
    "Mechanism",
    "Weapon_Part",
    "CompvPrac",
    "Time_Off",
    "Treatments",
    "Recovery",
]
_HAND_OPEN_TEXT = [
    "Which Injury(ies) have you had to your hands? - Tendon Injury, "
    "please elaborate on which tendon(s) - Text",
    "Which Injury(ies) have you had to your hands? - Nerve Injury, "
    "please elaborate on which nerve(s) - Text",
    "Which Injury(ies) have you had to your hands? - Muscle injury, "
    "please elaborate on which muscle(s) - Text",
    "Which Injury(ies) have you had to your hands? - Ligament injury, "
    "please elaborate on which ligament(s) - Text",
    "Which Injury(ies) have you had to your hands? - Other injury not listed - Text",
    "Mechanism of Injury",
]


def _stringify_mixed_objects(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].astype("string")
    return result


def normalize_general_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create participant and repeated-injury tables without inventing control rows."""

    required = {
        "Age",
        "Epee",
        "Foil",
        "Saber",
        "Gender",
        "Injured_While_Fencing",
        "Number_of_Injuries",
    }
    for number in range(1, 6):
        required.update(f"Injury_{number}_{field}" for field in _INJURY_FIELDS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"General injury survey missing columns: {sorted(missing)}")
    participants = frame.copy()
    participants.insert(
        0,
        "participant_code",
        [f"UFI-{index:04d}" for index in range(1, len(frame) + 1)],
    )
    participants["weapon_count"] = sum(
        participants[column].notna().astype("int8") for column in ["Epee", "Foil", "Saber"]
    )
    participants["multi_weapon"] = participants["weapon_count"].gt(1)
    participants["source_doi"] = GENERAL_DOI

    injury_rows: list[dict[str, Any]] = []
    for _, participant in participants.iterrows():
        for number in range(1, 6):
            upstream = {field: participant[f"Injury_{number}_{field}"] for field in _INJURY_FIELDS}
            if all(pd.isna(value) for value in upstream.values()):
                continue
            injury_rows.append(
                {
                    "participant_code": participant["participant_code"],
                    "injury_slot": number,
                    **{field.lower(): value for field, value in upstream.items()},
                    "source_doi": GENERAL_DOI,
                }
            )
    injuries = pd.DataFrame(injury_rows)
    return _stringify_mixed_objects(participants), _stringify_mixed_objects(injuries)


def normalize_general_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_excel(path)
    if frame.shape != (270, 86):
        raise ValueError(f"General injury survey snapshot changed: {frame.shape}")
    participants, injuries = normalize_general_frame(frame)
    if set(participants["Injured_While_Fencing"].dropna()) != {"Yes"}:
        raise ValueError("General injury deposit is no longer an injured-only subset")
    return participants, injuries


def normalize_hand_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove free text from the analytic table while flagging all source rows."""

    missing = set(_HAND_OPEN_TEXT).difference(frame.columns)
    if missing:
        raise ValueError(f"Hand injury survey missing open-text columns: {sorted(missing)}")
    exact_duplicate = frame.duplicated(keep=False)
    empty = frame.isna().all(axis=1)
    structured = frame.drop(columns=_HAND_OPEN_TEXT).copy()
    structured.insert(0, "source_row", range(1, len(frame) + 1))
    structured["exact_duplicate_group"] = exact_duplicate.to_numpy()
    structured["empty_source_row"] = empty.to_numpy()
    age = pd.to_numeric(structured["age "], errors="coerce")
    structured["age_numeric"] = age
    structured["age_below_18"] = age.lt(18)
    structured["source_doi"] = HAND_DOI
    return _stringify_mixed_objects(structured)


def normalize_hand_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.shape != (148, 18):
        raise ValueError(f"Hand injury survey snapshot changed: {frame.shape}")
    return normalize_hand_frame(frame)


def fetch_and_prepare(
    cache_dir: Path,
    processed_dir: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    paths = {
        "general_data": cache_dir / "26192" / "Epi_Data.xlsx",
        "general_readme": cache_dir / "26192" / "README.txt",
        "hand_data": cache_dir / "27141" / "fencing_hand_data.csv",
        "hand_readme": cache_dir / "27141" / "README.txt",
    }
    if force:
        for path in paths.values():
            path.unlink(missing_ok=True)
    download_resumable(
        GENERAL_DATA_URL,
        paths["general_data"],
        user_agent=user_agent,
        expected_bytes=75_514,
    )
    download_resumable(
        GENERAL_README_URL,
        paths["general_readme"],
        user_agent=user_agent,
        expected_bytes=13_434,
    )
    download_resumable(
        HAND_DATA_URL, paths["hand_data"], user_agent=user_agent, expected_bytes=20_923
    )
    download_resumable(
        HAND_README_URL,
        paths["hand_readme"],
        user_agent=user_agent,
        expected_bytes=9_980,
    )
    for path in [paths["general_readme"], paths["hand_readme"]]:
        if "CC-BY" not in path.read_text(encoding="utf-8-sig", errors="replace"):
            raise ValueError(f"Injury-survey license statement changed: {path}")

    participants, injuries = normalize_general_snapshot(paths["general_data"])
    hand = normalize_hand_snapshot(paths["hand_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "general_participants": processed_dir / "competitive-injury-participants.parquet",
        "general_injuries": processed_dir / "competitive-injury-episodes.parquet",
        "hand": processed_dir / "elite-hand-injury-structured.parquet",
    }
    participants.to_parquet(outputs["general_participants"], index=False)
    injuries.to_parquet(outputs["general_injuries"], index=False)
    hand.to_parquet(outputs["hand"], index=False)
    hand_nonempty = hand.loc[~hand["empty_source_row"]]
    summary: dict[str, Any] = {
        "claim_boundary": (
            "Anonymous self-report surveys, not clinical diagnoses or prospective injury "
            "incidence. The general deposit contains only 270 respondents who answered Yes "
            "to prior fencing injury, so it cannot train an ever-injured versus uninjured "
            "classifier. It describes at most five detailed injuries per respondent. The hand "
            "deposit has 148 CSV rows although its README says 152; seven rows are fully empty, "
            "and free-text narratives are excluded from the analytic table for privacy."
        ),
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "general": {
            "source_url": GENERAL_PAGE,
            "dataset_doi": GENERAL_DOI,
            "raw_sha256": sha256_file(paths["general_data"]),
            "readme_sha256": sha256_file(paths["general_readme"]),
            "raw_rows": len(participants),
            "raw_columns": 86,
            "detailed_injury_rows": len(injuries),
            "weapon_memberships": {
                "epee": int(participants["Epee"].notna().sum()),
                "foil": int(participants["Foil"].notna().sum()),
                "sabre": int(participants["Saber"].notna().sum()),
            },
            "multi_weapon_participants": int(participants["multi_weapon"].sum()),
            "readme_reported_completed_members": 303,
            "readme_reported_rows": 491,
            "all_deposit_rows_report_prior_injury": True,
        },
        "hand": {
            "source_url": HAND_PAGE,
            "dataset_doi": HAND_DOI,
            "raw_sha256": sha256_file(paths["hand_data"]),
            "readme_sha256": sha256_file(paths["hand_readme"]),
            "raw_rows": len(hand),
            "nonempty_rows": len(hand_nonempty),
            "readme_reported_rows": 152,
            "readme_reported_eligible_completed": 142,
            "exact_duplicate_rows": int(hand["exact_duplicate_group"].sum()),
            "age_below_18_rows": int(hand["age_below_18"].sum()),
            "weapon_counts_nonempty": (
                hand_nonempty["weapon"].value_counts().sort_index().astype(int).to_dict()
            ),
            "open_text_columns_excluded": len(_HAND_OPEN_TEXT),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "output_sha256": {key: sha256_file(path) for key, path in outputs.items()},
    }
    write_json(report_path, summary)
    return summary
