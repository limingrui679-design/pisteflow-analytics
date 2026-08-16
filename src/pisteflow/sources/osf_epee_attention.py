"""Local-only adapter for a 24-person epee attack and attention deposit."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json
from pisteflow.sources.net import download_resumable

SOURCE_URL = (
    "https://osf.io/8bnqg/?view_only=438acf07f7b04e1a8671c5b6ce62b810"
)
ARTICLE_DOI = "https://doi.org/10.1002/ejsc.12176"
FILES = {
    "attack": {
        "filename": "data_OSF_FENCING.csv",
        "url": (
            "https://osf.io/download/e2g36/"
            "?view_only=438acf07f7b04e1a8671c5b6ce62b810"
        ),
        "bytes": 1_037,
        "sha256": "d83a60147addc94ecfe30256e514df19b67631ef14c8cae4aec27f76c7cae62f",
    },
    "covat": {
        "filename": "data_OSF_COVAT.csv",
        "url": (
            "https://osf.io/download/qhaed/"
            "?view_only=438acf07f7b04e1a8671c5b6ce62b810"
        ),
        "bytes": 6_282,
        "sha256": "e433da71805d82502fc7056c0a36f659049ff06f8069117c667499f84a714e4b",
    },
    "analysis_code": {
        "filename": "Rcode-for-OSF.pdf",
        "url": (
            "https://osf.io/download/ug93v/"
            "?view_only=438acf07f7b04e1a8671c5b6ce62b810"
        ),
        "bytes": 1_580_593,
        "sha256": "5a869aad0dce5e9bae61b381682b5fd7283d4d278e5cd7ad3df7fd42291c8752",
    },
}


def _source_number(value: object) -> int:
    match = re.fullmatch(r"[DES](\d{1,2})", str(value).strip())
    if match is None:
        raise ValueError(f"unexpected OSF epee participant code: {value!r}")
    number = int(match.group(1))
    if not 1 <= number <= 12:
        raise ValueError(f"unexpected OSF epee participant number: {number}")
    return number


def _participant_key(group: str, number: int) -> str:
    prefix = {"beginner": "B", "advanced": "A"}[group]
    return f"OEA-{prefix}{number:02d}"


def _drop_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(axis=1, how="all").copy()


def normalize_attack(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and pseudonymize the participant-level attack table."""

    frame = _drop_empty_columns(frame)
    expected_columns = ["ID", "sex", "group", "age", "h", "w", "HIT", "EXTIME", "COVATrt"]
    if frame.shape != (24, 9) or list(frame.columns) != expected_columns:
        raise ValueError(f"OSF epee attack snapshot changed: {frame.shape}, {list(frame.columns)}")
    if frame.isna().any().any() or frame.duplicated().any():
        raise ValueError("OSF epee attack missing or duplicate rows changed")

    result = frame.copy()
    result["sex"] = result["sex"].astype("string").str.strip().map({"M": "man", "W": "woman"})
    result["group"] = result["group"].astype("string").str.strip()
    if result["sex"].isna().any() or set(result["group"]) != {"beginner", "advanced"}:
        raise ValueError("OSF epee sex or experience-group labels changed")
    if result["group"].value_counts().to_dict() != {"beginner": 12, "advanced": 12}:
        raise ValueError("OSF epee group sizes changed")

    numbers = result["ID"].map(_source_number)
    expected_ids = {*(f"D{i}" for i in range(1, 13)), *(f"E{i}" for i in range(1, 13))}
    if set(result["ID"].astype(str)) != expected_ids:
        raise ValueError("OSF epee attack participant identifiers changed")
    result.insert(
        0,
        "participant_key",
        [
            _participant_key(str(group), int(number))
            for group, number in zip(result["group"], numbers, strict=True)
        ],
    )
    result = result.drop(columns="ID").rename(
        columns={
            "age": "age_years",
            "h": "height_cm",
            "w": "weight_kg",
            "HIT": "attack_hits_of_30",
            "EXTIME": "attack_execution_time_seconds",
            "COVATrt": "covat_mean_reaction_time_seconds",
        }
    )
    numeric_columns = [
        "age_years",
        "height_cm",
        "weight_kg",
        "attack_hits_of_30",
        "attack_execution_time_seconds",
        "covat_mean_reaction_time_seconds",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not result["attack_hits_of_30"].between(0, 30).all():
        raise ValueError("OSF epee attack hit counts changed")
    result["weapon"] = "epee"
    result["source_url"] = SOURCE_URL
    return result


def normalize_covat(frame: pd.DataFrame, attack: pd.DataFrame) -> pd.DataFrame:
    """Validate six COVAT condition means per participant and audit the crosswalk."""

    expected_columns = ["group", "ID", "event", "direction", "cue_t", "rt"]
    if frame.shape != (144, 6) or list(frame.columns) != expected_columns:
        raise ValueError(f"OSF epee COVAT snapshot changed: {frame.shape}, {list(frame.columns)}")
    if frame.isna().any().any() or frame.duplicated().any():
        raise ValueError("OSF epee COVAT missing or duplicate rows changed")

    result = frame.copy()
    result["group"] = result["group"].map({"Beg": "beginner", "Adv": "advanced"})
    if result["group"].isna().any():
        raise ValueError("OSF epee COVAT group labels changed")
    numbers = result["ID"].map(_source_number)
    expected_ids = {*(f"D{i}" for i in range(1, 13)), *(f"S{i}" for i in range(1, 13))}
    if set(result["ID"].astype(str)) != expected_ids:
        raise ValueError("OSF epee COVAT participant identifiers changed")
    result.insert(
        0,
        "participant_key",
        [
            _participant_key(str(group), int(number))
            for group, number in zip(result["group"], numbers, strict=True)
        ],
    )
    result["reaction_time_seconds"] = pd.to_numeric(result.pop("rt"), errors="raise")
    expected_conditions = {
        ("coherent", "L", "coherent left"),
        ("coherent", "R", "coherent right"),
        ("incoherent", "L", "incoherent left"),
        ("incoherent", "R", "incoherent right"),
        ("no-cue", "L", "no-cue left"),
        ("no-cue", "R", "no-cue right"),
    }
    for _, participant in result.groupby("participant_key", sort=False):
        conditions = set(
            zip(participant["cue_t"], participant["direction"], participant["event"], strict=True)
        )
        if len(participant) != 6 or conditions != expected_conditions:
            raise ValueError("OSF epee COVAT condition grid changed")
    if not result["reaction_time_seconds"].between(0.1, 2.0).all():
        raise ValueError("OSF epee COVAT reaction-time range changed")

    covat_means = result.groupby("participant_key")["reaction_time_seconds"].mean().round(3)
    attack_means = attack.set_index("participant_key")["covat_mean_reaction_time_seconds"]
    covat_means = covat_means.sort_index()
    attack_means = attack_means.astype(float).round(3).sort_index()
    if not covat_means.index.equals(attack_means.index) or not (
        covat_means.sub(attack_means).abs() < 1e-12
    ).all():
        raise ValueError("OSF epee COVAT-to-attack aggregate reconciliation changed")

    result["crosswalk_basis"] = result["group"].map(
        {
            "beginner": "exact_source_id",
            "advanced": "inferred_numeric_suffix_and_rounded_mean",
        }
    )
    result["weapon"] = "epee"
    result["source_url"] = SOURCE_URL
    return result.drop(columns="ID")


def fetch_and_prepare(
    cache_directory: Path,
    attack_output_path: Path,
    covat_output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download the pinned view-only files and write local-only analytic tables."""

    paths: dict[str, Path] = {}
    for key, specification in FILES.items():
        path = cache_directory / str(specification["filename"])
        if force:
            path.unlink(missing_ok=True)
        download_resumable(
            str(specification["url"]),
            path,
            user_agent=user_agent,
            expected_bytes=int(str(specification["bytes"])),
        )
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"OSF epee attention checksum changed: {key}")
        paths[key] = path

    attack_raw = pd.read_csv(paths["attack"], sep=";", encoding="utf-8-sig")
    covat_raw = pd.read_csv(paths["covat"], sep=";", encoding="utf-8-sig")
    attack = normalize_attack(attack_raw)
    covat = normalize_covat(covat_raw, attack)
    attack_output_path.parent.mkdir(parents=True, exist_ok=True)
    attack.to_parquet(attack_output_path, index=False)
    covat.to_parquet(covat_output_path, index=False)

    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "article_doi": ARTICLE_DOI,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": "not_declared",
        "repository_policy": "raw_and_row_level_outputs_local_only",
        "weapon": "epee",
        "participants": int(attack["participant_key"].nunique()),
        "by_group": {
            str(key): int(value)
            for key, value in attack["group"].value_counts().sort_index().items()
        },
        "participant_rows": len(attack),
        "covat_condition_mean_rows": len(covat),
        "covat_conditions_per_participant": 6,
        "attack_trials_described_not_deposited": len(attack) * 30,
        "attack_trial_rows_deposited": 0,
        "recorded_hits_across_participant_aggregates": int(attack["attack_hits_of_30"].sum()),
        "advanced_covat_rows_with_inferred_crosswalk": int(
            covat["group"].eq("advanced").sum()
        ),
        "claim_boundary": (
            "Twenty-four adult epee participants, split evenly between beginner and advanced "
            "groups. The deposit contains one attack-summary row and six COVAT condition means "
            "per participant; it does not contain the 720 individual attack trials. This is a "
            "cross-sectional laboratory comparison, not a training intervention, bout-outcome "
            "dataset, or causal skill estimate. The advanced files use E1-E12 versus S1-S12; "
            "their local crosswalk is explicitly inferred from numeric suffixes and exact "
            "three-decimal reconciliation of the six-condition mean. No reuse license is "
            "declared, so raw and processed rows remain local."
        ),
        "assets": {
            key: {
                "url": specification["url"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for (key, specification), path in zip(FILES.items(), paths.values(), strict=True)
        },
        "outputs": {
            "attack": {
                "path": str(attack_output_path),
                "bytes": attack_output_path.stat().st_size,
                "sha256": sha256_file(attack_output_path),
            },
            "covat": {
                "path": str(covat_output_path),
                "bytes": covat_output_path.stat().st_size,
                "sha256": sha256_file(covat_output_path),
            },
        },
    }
    write_json(report_path, summary)
    return summary
