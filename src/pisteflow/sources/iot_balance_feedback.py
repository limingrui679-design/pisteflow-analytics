"""CC BY repeated-measure fencing balance data from an IoT feedback study."""

from __future__ import annotations

import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
from pypdf import PdfReader

from pisteflow.provenance import atomic_write_bytes, sha256_bytes, sha256_file, write_json
from pisteflow.sources.net import download_resumable

ARTICLE_DOI = "https://doi.org/10.3390/s23249801"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10747936/"
SUPPLEMENT_BUNDLE_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10747936/supplementaryFiles"
)
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
NESTED_ZIP_MEMBER = "sensors-23-09801-s001.zip"
NESTED_ZIP_BYTES = 99_873
NESTED_ZIP_SHA256 = "7a28995e1cbd5683f75b5b2e33049fdc7fbc9947441ebffeacdfd898cd1eb66e"
PDF_MEMBER = "sensors-2688249-supplementary.pdf"
PDF_BYTES = 105_497
PDF_SHA256 = "8f5e5191d42368a2e4d734f125cdeb027d869a58ca55a528f6b0402b7f351540"

GROUPS = {
    "CG": "control",
    "VG": "visual_feedback",
    "HG": "haptic_feedback",
}
EXPECTED_MEANS = {
    ("control", 0): (9.952, 34.366, 17.152),
    ("control", 5): (9.928, 32.775, 16.905),
    ("visual_feedback", 0): (10.661, 33.005, 17.141),
    ("visual_feedback", 5): (10.951, 27.891, 13.867),
    ("haptic_feedback", 0): (10.933, 31.028, 16.580),
    ("haptic_feedback", 5): (10.948, 29.816, 15.766),
}
ROW_PATTERN = re.compile(
    r"(?m)^(CG|VG|HG)(10|[1-9])\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*$"
)


def extract_pinned_supplement(
    bundle_path: Path,
    nested_zip_path: Path,
    pdf_path: Path,
) -> None:
    """Extract and validate the stable supplement inside Europe PMC's dynamic bundle."""

    with ZipFile(bundle_path) as bundle:
        nested_bytes = bundle.read(NESTED_ZIP_MEMBER)
    if len(nested_bytes) != NESTED_ZIP_BYTES or sha256_bytes(nested_bytes) != NESTED_ZIP_SHA256:
        raise ValueError("IoT balance nested supplement ZIP changed")
    atomic_write_bytes(nested_zip_path, nested_bytes)

    with ZipFile(io.BytesIO(nested_bytes)) as nested:
        pdf_bytes = nested.read(PDF_MEMBER)
    if len(pdf_bytes) != PDF_BYTES or sha256_bytes(pdf_bytes) != PDF_SHA256:
        raise ValueError("IoT balance supplementary PDF changed")
    atomic_write_bytes(pdf_path, pdf_bytes)


def parse_measurement_pages(page_texts: list[str]) -> pd.DataFrame:
    """Parse only Tables S1-S6; later derived tables contain printed inconsistencies."""

    if len(page_texts) < 2:
        raise ValueError("IoT balance supplement has fewer than two measurement pages")
    rows: list[dict[str, object]] = []
    for week, text in zip([0, 5], page_texts[:2], strict=True):
        matches = list(ROW_PATTERN.finditer(text))
        if len(matches) != 30:
            raise ValueError(f"IoT balance week {week} row count changed: {len(matches)}")
        for match in matches:
            prefix, number, movement, unbalance_two, unbalance_three = match.groups()
            group = GROUPS[prefix]
            rows.append(
                {
                    "participant_key": f"IBF-{prefix}{int(number):02d}",
                    "study_group": group,
                    "week": week,
                    "total_movement_time_seconds": float(movement),
                    "unbalance_time_2_of_3_axes_percent": float(unbalance_two),
                    "unbalance_time_3_of_3_axes_percent": float(unbalance_three),
                    "weapon": "unknown",
                    "source_url": ARTICLE_DOI,
                }
            )
    result = pd.DataFrame(rows)
    expected_ids = {f"IBF-{prefix}{number:02d}" for prefix in GROUPS for number in range(1, 11)}
    if set(result["participant_key"]) != expected_ids:
        raise ValueError("IoT balance participant identifiers changed")
    if result.duplicated(["participant_key", "week"]).any():
        raise ValueError("IoT balance participant-week duplicates changed")
    if result.groupby("participant_key").size().ne(2).any():
        raise ValueError("IoT balance repeated-measure structure changed")
    if result.groupby(["study_group", "week"]).size().ne(10).any():
        raise ValueError("IoT balance group sizes changed")
    for (group, week), expected in EXPECTED_MEANS.items():
        subset = result.loc[
            result["study_group"].eq(group) & result["week"].eq(week)
        ]
        actual = (
            float(subset["total_movement_time_seconds"].mean()),
            float(subset["unbalance_time_2_of_3_axes_percent"].mean()),
            float(subset["unbalance_time_3_of_3_axes_percent"].mean()),
        )
        if any(abs(left - right) > 0.000_001 for left, right in zip(actual, expected, strict=True)):
            raise ValueError(f"IoT balance group mean changed: {(group, week)}")
    return result


def _group_changes(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = [
        "total_movement_time_seconds",
        "unbalance_time_2_of_3_axes_percent",
        "unbalance_time_3_of_3_axes_percent",
    ]
    baseline = frame.loc[frame["week"].eq(0)].set_index("participant_key")
    followup = frame.loc[frame["week"].eq(5)].set_index("participant_key")
    changes = followup[metrics].sub(baseline[metrics])
    changes["study_group"] = followup["study_group"]
    means = changes.groupby("study_group")[metrics].mean()
    return {
        str(group): {str(metric): round(float(value), 6) for metric, value in row.items()}
        for group, row in means.iterrows()
    }


def fetch_and_prepare(
    cache_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire the official supplement and prepare 60 participant-time rows."""

    bundle_path = cache_directory / "europepmc-supplementary-files.zip"
    nested_zip_path = cache_directory / NESTED_ZIP_MEMBER
    pdf_path = cache_directory / PDF_MEMBER
    if force:
        for path in [bundle_path, nested_zip_path, pdf_path]:
            path.unlink(missing_ok=True)
    download_resumable(SUPPLEMENT_BUNDLE_URL, bundle_path, user_agent=user_agent)
    extract_pinned_supplement(bundle_path, nested_zip_path, pdf_path)

    page_texts = [page.extract_text() for page in PdfReader(pdf_path).pages]
    processed = parse_measurement_pages(page_texts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, index=False)

    summary: dict[str, Any] = {
        "source_url": ARTICLE_DOI,
        "pmc_url": PMC_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "repository_policy": "minor_participant_rows_local_aggregates_public",
        "participants": int(processed["participant_key"].nunique()),
        "participant_time_rows": len(processed),
        "measurement_cells": len(processed) * 3,
        "weeks": [0, 5],
        "groups": {
            str(key): int(value)
            for key, value in processed.loc[processed["week"].eq(0), "study_group"]
            .value_counts()
            .sort_index()
            .items()
        },
        "weapon": "not_reported",
        "age_range_years": [11, 14],
        "fencing_experience_years": [4, 5],
        "training_sessions_per_week": 4,
        "targeted_minutes_per_session": 30,
        "mean_change_week_5_minus_week_0": _group_changes(processed),
        "source_anomalies": [
            "Table S4 omits the printed control-group Week 5 movement-time average; the ten "
            "participant rows compute to 9.928 seconds, matching the article table.",
            "Derived Tables S10-S12 contain reversed week columns, a VG7 typo, and mislabeled "
            "haptic-group headings; they are excluded rather than treated as extra observations.",
        ],
        "claim_boundary": (
            "Thirty juvenile fencers aged 11-14 with four to five years of experience, randomly "
            "allocated ten per control, visual-feedback, and haptic-feedback group. Sixty rows "
            "represent the same 30 participants at Week 0 and Week 5, with three measurements per "
            "visit; they are not 60 athletes or 180 independent samples. Weapon is not reported. "
            "Lower unbalance percentages indicate improvement, while movement time is a separate "
            "speed outcome. Small groups, one club, minors, and five weeks prevent population, "
            "injury-prevention, or general coaching claims."
        ),
        "assets": {
            "dynamic_europepmc_bundle": {
                "url": SUPPLEMENT_BUNDLE_URL,
                "bytes": bundle_path.stat().st_size,
                "sha256": sha256_file(bundle_path),
                "validation": "container_not_pinned_inner_assets_pinned",
            },
            "nested_supplement_zip": {
                "bytes": nested_zip_path.stat().st_size,
                "sha256": sha256_file(nested_zip_path),
            },
            "supplementary_pdf": {
                "bytes": pdf_path.stat().st_size,
                "sha256": sha256_file(pdf_path),
            },
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    write_json(report_path, summary)
    return summary
