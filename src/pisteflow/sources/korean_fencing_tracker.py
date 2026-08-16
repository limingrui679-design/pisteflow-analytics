"""Audit the pinned KoreanFencingTracker competition-report snapshot.

The repository contains report artifacts derived from public fencing videos, not the
videos themselves.  This adapter keeps the three useful layers separate:

* one privacy-minimized record per source video;
* OCR-derived score-change events from the single canonical report per video;
* explicitly unreviewed phrase and priority weak labels.

No upstream fencer names or free-text notes are copied into processed outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from pisteflow.provenance import sha256_file, write_json, write_jsonl
from pisteflow.sources.net import download_resumable

SOURCE_URL = "https://github.com/AI-BZ/KoreanFencingTracker"
SNAPSHOT = "3a40997e746cb187beba1439f9e2e0d1fb92cf24"
ARCHIVE_URL = f"https://codeload.github.com/AI-BZ/KoreanFencingTracker/tar.gz/{SNAPSHOT}"
EXPECTED_ARCHIVE_BYTES = 15_988_288
EXPECTED_ARCHIVE_SHA256 = "eda32eefbc97dcbef8e5a38525bd4a185793374a891407c880ea45c6dd3a4aea"

LICENSE_BOUNDARY = (
    "The README says MIT for repository code, but the snapshot has no LICENSE file and does "
    "not grant rights to the underlying broadcast videos or report-derived data. Public video "
    "links and aggregate audit metadata may be published; row tables stay local."
)

EXPECTED_REPORT_FILES = 88
EXPECTED_UNIQUE_REPORT_HASHES = 87
EXPECTED_SOURCE_VIDEOS = 78
EXPECTED_CANONICAL_TOUCHES = 1_037
EXPECTED_CANONICAL_ANALYZED_FRAMES = 1_640_367
EXPECTED_WEAPONS = {"epee": 23, "foil": 25, "sabre": 29, "unknown": 1}
EXPECTED_REPORT_MODES = {"continuous_only": 1, "continuous_with_ocr": 9, "ocr_only": 78}
EXPECTED_PHRASES = 302
EXPECTED_PHRASE_VIDEOS = 5
EXPECTED_ACTION_VARIANT_ROWS = 172
EXPECTED_ACTION_EVENT_KEYS = 165
EXPECTED_ACTION_CONFLICTING_EVENT_KEYS = 7
EXPECTED_ACTION_LABEL_COUNTS = {"attack": 77, "riposte": 73, "unknown": 22}
EXPECTED_PRIORITY_ROWS = 76
EXPECTED_PRIORITY_CATEGORIES = {
    "automatic_double_light_score_proxy": 21,
    "needs_human_review_conflict": 1,
    "needs_human_review_no_lamp": 7,
    "needs_human_review_single_light": 47,
}

_YOUTUBE_ID = re.compile(r"(?:^|_)([A-Za-z0-9_-]{11})$")
_PRIORITY_VIDEO_IDS = {
    "labels_priority_li_lin.csv": "hKUXgUsDOKE",
    "labels_priority_nac_y12wf.csv": "k-spxGdlWfo",
    "labels_priority_usaf_gdOdpDyaWrw.csv": "gdOdpDyaWrw",
}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"pisteflow:{prefix}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _safe_extract_stripped(archive_path: Path, output_directory: Path) -> None:
    """Extract regular files after removing the single codeload root directory."""

    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_root = output_directory.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        top_levels = {
            PurePosixPath(member.name).parts[0]
            for member in members
            if PurePosixPath(member.name).parts
        }
        if len(top_levels) != 1:
            raise ValueError("KoreanFencingTracker archive does not have one stable root")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) <= 1:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("KoreanFencingTracker archive contains an unsafe member path")
            destination = (resolved_root / relative).resolve()
            if resolved_root not in destination.parents:
                raise ValueError("KoreanFencingTracker archive escapes its extraction root")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("KoreanFencingTracker archive contains a non-regular member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("KoreanFencingTracker archive member could not be opened")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def acquire(
    archive_path: Path,
    output_directory: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> Path:
    """Download and extract the pinned repository snapshot."""

    if force:
        archive_path.unlink(missing_ok=True)
    download_resumable(
        ARCHIVE_URL,
        archive_path,
        user_agent=user_agent,
        expected_bytes=EXPECTED_ARCHIVE_BYTES,
    )
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("KoreanFencingTracker archive checksum changed")
    marker = output_directory / "services/analytics/data/reports"
    if force or not marker.is_dir():
        _safe_extract_stripped(archive_path, output_directory)
    return output_directory


def video_id_from_value(value: str) -> str:
    """Extract the terminal YouTube-style ID from an upstream path or dataset ID."""

    stem = Path(value).stem
    match = _YOUTUBE_ID.search(stem)
    if match is None:
        raise ValueError(f"No terminal 11-character video ID in {value!r}")
    return match.group(1)


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"summary", "touches", "left_fencer", "right_fencer", "meta", "warnings"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"Unexpected KoreanFencingTracker report schema: {path.name}")
    if not isinstance(value["summary"], dict) or not isinstance(value["meta"], dict):
        raise ValueError(f"Malformed KoreanFencingTracker report metadata: {path.name}")
    if not isinstance(value["touches"], list) or not isinstance(value["warnings"], list):
        raise ValueError(f"Malformed KoreanFencingTracker report rows: {path.name}")
    return value


def _score_pair(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Unexpected KoreanFencingTracker score: {value!r}")
    return int(parts[0]), int(parts[1])


def _report_inventory(
    reports_directory: Path,
) -> tuple[list[dict[str, Any]], list[tuple[Path, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    canonical: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(reports_directory.glob("*_report.json")):
        report = _read_report(path)
        summary = report["summary"]
        metadata = report["meta"]
        video_id = video_id_from_value(str(summary.get("video_path", "")))
        mode = str(metadata.get("analysis_mode", "unknown"))
        touches = report["touches"]
        if int(summary.get("total_touches", -1)) != len(touches):
            raise ValueError(f"KoreanFencingTracker report touch count changed: {path.name}")
        content_sha256 = sha256_file(path)
        rows.append(
            {
                "report_artifact_id": _stable_id("korean-report", path.name),
                "report_content_sha256": content_sha256,
                "youtube_id": video_id,
                "source_group_id": f"youtube:{video_id}",
                "analysis_mode": mode,
                "touch_rows": len(touches),
                "analyzed_frames": int(summary.get("total_frames_analyzed", 0)),
                "weapon": str(summary.get("weapon") or "unknown"),
                "gender": str(summary.get("gender") or "unknown"),
                "age_group": str(summary.get("age_group") or "unknown"),
                "bout_type": str(summary.get("bout_type") or "unknown"),
                "warning_types": sorted(
                    str(item.get("type", "unknown"))
                    for item in report["warnings"]
                    if isinstance(item, dict)
                ),
                "contains_continuous_exchanges": isinstance(report.get("exchanges"), list),
                "contains_scoring_frames": isinstance(report.get("scoring_frames"), list),
                "label_origin": "upstream_automatic_video_analysis",
                "human_reviewed": False,
            }
        )
        if mode == "ocr_only":
            canonical.append((path, report))
    return rows, canonical


def _canonical_rows(
    canonical_reports: list[tuple[Path, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    video_rows: list[dict[str, Any]] = []
    touch_rows: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    for path, report in canonical_reports:
        summary = report["summary"]
        video_id = video_id_from_value(str(summary["video_path"]))
        if video_id in seen_videos:
            raise ValueError(f"Multiple canonical OCR reports for video {video_id}")
        seen_videos.add(video_id)
        report_sha256 = sha256_file(path)
        weapon = str(summary.get("weapon") or "unknown")
        gender = str(summary.get("gender") or "unknown")
        age_group = str(summary.get("age_group") or "unknown")
        bout_type = str(summary.get("bout_type") or "unknown")
        final_left, final_right = _score_pair(str(summary["final_score"]))
        warning_types = sorted(
            str(item.get("type", "unknown"))
            for item in report["warnings"]
            if isinstance(item, dict)
        )
        video_rows.append(
            {
                "video_id": f"youtube:{video_id}",
                "youtube_id": video_id,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "media_acquired": False,
                "license_status": "link_only_underlying_video_rights_not_asserted",
                "weapon": weapon,
                "gender": gender,
                "age_group": age_group,
                "bout_type": bout_type,
                "final_score_left": final_left,
                "final_score_right": final_right,
                "ocr_touch_rows": len(report["touches"]),
                "analyzed_frames": int(summary.get("total_frames_analyzed", 0)),
                "match_duration_text": str(summary.get("match_duration") or ""),
                "warning_types": warning_types,
                "canonical_report_sha256": report_sha256,
                "label_origin": "upstream_ocr_score_change",
                "human_reviewed": False,
            }
        )
        for touch in report["touches"]:
            touch_number = int(touch["touch_number"])
            score_left, score_right = _score_pair(str(touch["score_after"]))
            touch_rows.append(
                {
                    "touch_id": _stable_id("korean-touch", f"{video_id}:{touch_number}"),
                    "source_group_id": f"youtube:{video_id}",
                    "youtube_id": video_id,
                    "weapon": weapon,
                    "gender": gender,
                    "age_group": age_group,
                    "bout_type": bout_type,
                    "touch_number": touch_number,
                    "frame": int(touch["frame"]),
                    "video_timestamp": str(touch.get("video_timestamp") or ""),
                    "match_time": str(touch.get("match_time") or ""),
                    "scorer": str(touch.get("scorer") or "unknown"),
                    "score_left": score_left,
                    "score_right": score_right,
                    "action_scorer": str(touch.get("action_scorer") or "unknown"),
                    "action_scorer_confidence": float(touch.get("action_confidence") or 0.0),
                    "action_opponent": str(touch.get("action_opponent") or "unknown"),
                    "action_opponent_confidence": float(
                        touch.get("opponent_confidence") or 0.0
                    ),
                    "label_origin": "upstream_ocr_score_change",
                    "human_reviewed": False,
                }
            )
    return video_rows, touch_rows


def _phrase_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if not isinstance(annotations, list):
        raise ValueError("KoreanFencingTracker phrase dataset has no annotation list")
    rows: list[dict[str, Any]] = []
    for item in annotations:
        if not isinstance(item, dict):
            raise ValueError("KoreanFencingTracker phrase row is not an object")
        video_id = video_id_from_value(str(item["video_id"]))
        phrase_id = int(item["phrase_id"])
        start_frame = int(item["start_frame"])
        end_frame = int(item["end_frame"])
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("KoreanFencingTracker phrase frame range is invalid")
        rows.append(
            {
                "phrase_id": _stable_id("korean-phrase", f"{video_id}:{phrase_id}"),
                "source_group_id": f"youtube:{video_id}",
                "youtube_id": video_id,
                "upstream_phrase_number": phrase_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": str(item.get("start_time") or ""),
                "end_time": str(item.get("end_time") or ""),
                "trigger": str(item.get("trigger") or "unknown"),
                "outcome": str(item.get("outcome") or "unknown"),
                "action_sequence": list(item.get("action_sequence") or []),
                "upstream_confidence": float(item.get("confidence") or 0.0),
                "label_origin": "upstream_rule_generated_phrase",
                "human_reviewed": bool(item.get("reviewed", False)),
            }
        )
    return rows


def _continuous_action_rows(reports_directory: Path) -> list[dict[str, Any]]:
    """Keep distinct continuous-report action variants and expose disagreement."""

    raw_rows: list[dict[str, Any]] = []
    seen_report_hashes: set[str] = set()
    for path in sorted(reports_directory.glob("*_report.json")):
        report = _read_report(path)
        if str(report["meta"].get("analysis_mode")) != "continuous_with_ocr":
            continue
        report_sha256 = sha256_file(path)
        if report_sha256 in seen_report_hashes:
            continue
        seen_report_hashes.add(report_sha256)
        video_id = video_id_from_value(str(report["summary"]["video_path"]))
        for touch in report["touches"]:
            touch_number = int(touch["touch_number"])
            score_left, score_right = _score_pair(str(touch["score_after"]))
            raw_rows.append(
                {
                    "event_key": _stable_id("korean-action-event", f"{video_id}:{touch_number}"),
                    "report_variant_id": _stable_id("korean-report-content", report_sha256),
                    "source_group_id": f"youtube:{video_id}",
                    "youtube_id": video_id,
                    "touch_number": touch_number,
                    "frame": int(touch["frame"]),
                    "scorer": str(touch.get("scorer") or "unknown"),
                    "score_left": score_left,
                    "score_right": score_right,
                    "action_scorer": str(touch.get("action_scorer") or "unknown"),
                    "action_scorer_confidence": float(
                        touch.get("action_confidence") or 0.0
                    ),
                    "action_opponent": str(touch.get("action_opponent") or "unknown"),
                    "action_opponent_confidence": float(
                        touch.get("opponent_confidence") or 0.0
                    ),
                    "label_origin": "upstream_continuous_rule_or_model_proxy",
                    "human_reviewed": False,
                }
            )

    grouped: defaultdict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    signature_fields = (
        "event_key",
        "source_group_id",
        "youtube_id",
        "touch_number",
        "scorer",
        "score_left",
        "score_right",
        "action_scorer",
        "action_scorer_confidence",
        "action_opponent",
        "action_opponent_confidence",
    )
    for row in raw_rows:
        grouped[tuple(row[field] for field in signature_fields)].append(row)

    rows: list[dict[str, Any]] = []
    for signature, members in sorted(grouped.items(), key=lambda item: repr(item[0])):
        representative = members[0]
        variant_ids = sorted({str(row["report_variant_id"]) for row in members})
        frames = [int(row["frame"]) for row in members]
        signature_token = json.dumps(signature, ensure_ascii=False, separators=(",", ":"))
        rows.append(
            {
                "action_variant_row_id": _stable_id("korean-action-variant", signature_token),
                **{field: representative[field] for field in signature_fields},
                "frame_min": min(frames),
                "frame_max": max(frames),
                "report_variant_ids": variant_ids,
                "supporting_report_variant_count": len(variant_ids),
                "label_origin": "upstream_continuous_rule_or_model_proxy",
                "human_reviewed": False,
            }
        )

    labels_by_event: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_event[str(row["event_key"])].add(str(row["action_scorer"]))
    conflicting = {key for key, labels in labels_by_event.items() if len(labels) > 1}
    for row in rows:
        row["has_cross_variant_action_conflict"] = str(row["event_key"]) in conflicting
    return rows


def priority_note_category(note: str) -> str:
    """Convert upstream Korean free text into a compact audit category."""

    if "모순" in note:
        return "needs_human_review_conflict"
    if "미판독" in note:
        return "needs_human_review_no_lamp"
    if "단독 램프" in note:
        return "needs_human_review_single_light"
    if "양측 램프" in note:
        return "automatic_double_light_score_proxy"
    return "unclassified_upstream_note"


def _priority_rows(data_directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, video_id in sorted(_PRIORITY_VIDEO_IDS.items()):
        path = data_directory / filename
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                touch_number = int(item["touch_number"])
                category = priority_note_category(item["note"])
                priority_attacker = item["priority_attacker"].strip() or None
                lamp = item["lamp"].strip() or None
                if category == "automatic_double_light_score_proxy" and (
                    lamp != "double" or priority_attacker not in {"left", "right"}
                ):
                    raise ValueError("KoreanFencingTracker automatic priority proxy changed")
                rows.append(
                    {
                        "priority_row_id": _stable_id(
                            "korean-priority", f"{video_id}:{touch_number}"
                        ),
                        "source_group_id": f"youtube:{video_id}",
                        "youtube_id": video_id,
                        "touch_number": touch_number,
                        "exchange_number": (
                            int(item["exchange_number"])
                            if item["exchange_number"].strip()
                            else None
                        ),
                        "priority_attacker": priority_attacker,
                        "lamp": lamp,
                        "upstream_confidence": float(item["confidence"]),
                        "note_category": category,
                        "label_origin": (
                            "upstream_score_recipient_proxy"
                            if category == "automatic_double_light_score_proxy"
                            else "unlabeled_human_review_queue"
                        ),
                        "human_reviewed": False,
                    }
                )
    return rows


def _validate_snapshot(summary: dict[str, Any]) -> None:
    expected = {
        "report_files": EXPECTED_REPORT_FILES,
        "unique_report_hashes": EXPECTED_UNIQUE_REPORT_HASHES,
        "source_videos": EXPECTED_SOURCE_VIDEOS,
        "canonical_touch_rows": EXPECTED_CANONICAL_TOUCHES,
        "canonical_analyzed_frames": EXPECTED_CANONICAL_ANALYZED_FRAMES,
        "weapon_counts": EXPECTED_WEAPONS,
        "report_mode_counts": EXPECTED_REPORT_MODES,
        "phrase_rows": EXPECTED_PHRASES,
        "phrase_videos": EXPECTED_PHRASE_VIDEOS,
        "action_variant_rows": EXPECTED_ACTION_VARIANT_ROWS,
        "action_event_keys": EXPECTED_ACTION_EVENT_KEYS,
        "action_conflicting_event_keys": EXPECTED_ACTION_CONFLICTING_EVENT_KEYS,
        "action_label_counts": EXPECTED_ACTION_LABEL_COUNTS,
        "priority_rows": EXPECTED_PRIORITY_ROWS,
        "priority_note_categories": EXPECTED_PRIORITY_CATEGORIES,
    }
    actual = {key: summary[key] for key in expected}
    if actual != expected:
        raise ValueError(f"KoreanFencingTracker snapshot counts changed: {actual} != {expected}")


def prepare(
    source_directory: Path,
    video_manifest_path: Path,
    report_manifest_path: Path,
    touch_output_path: Path,
    phrase_output_path: Path,
    action_output_path: Path,
    priority_output_path: Path,
    summary_path: Path,
    *,
    archive_path: Path | None = None,
    validate_snapshot: bool = True,
) -> dict[str, Any]:
    """Normalize the pinned snapshot without copying names or source video media."""

    data_directory = source_directory / "services/analytics/data"
    report_rows, canonical_reports = _report_inventory(data_directory / "reports")
    video_rows, touch_rows = _canonical_rows(canonical_reports)
    phrase_rows = _phrase_rows(data_directory / "datasets/phrase_boundaries.json")
    action_rows = _continuous_action_rows(data_directory / "reports")
    priority_rows = _priority_rows(data_directory)

    if len({str(row["youtube_id"]) for row in video_rows}) != len(video_rows):
        raise ValueError("KoreanFencingTracker canonical video IDs are not unique")
    video_ids = {str(row["youtube_id"]) for row in video_rows}
    if not {str(row["youtube_id"]) for row in phrase_rows}.issubset(video_ids):
        raise ValueError("KoreanFencingTracker phrase rows reference an unknown video")
    if not {str(row["youtube_id"]) for row in priority_rows}.issubset(video_ids):
        raise ValueError("KoreanFencingTracker priority rows reference an unknown video")

    touch_frame = pd.DataFrame(touch_rows)
    phrase_frame = pd.DataFrame(phrase_rows)
    action_frame = pd.DataFrame(action_rows)
    priority_frame = pd.DataFrame(priority_rows)
    for path, frame in (
        (touch_output_path, touch_frame),
        (phrase_output_path, phrase_frame),
        (action_output_path, action_frame),
        (priority_output_path, priority_frame),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    video_manifest_sha256 = write_jsonl(video_manifest_path, video_rows)
    report_manifest_sha256 = write_jsonl(report_manifest_path, report_rows)
    report_hash_counts = Counter(str(row["report_content_sha256"]) for row in report_rows)
    mode_counts = dict(sorted(Counter(str(row["analysis_mode"]) for row in report_rows).items()))
    weapon_counts = dict(sorted(Counter(str(row["weapon"]) for row in video_rows).items()))
    gender_counts = dict(sorted(Counter(str(row["gender"]) for row in video_rows).items()))
    age_group_counts = dict(sorted(Counter(str(row["age_group"]) for row in video_rows).items()))
    priority_categories = dict(
        sorted(Counter(str(row["note_category"]) for row in priority_rows).items())
    )
    action_label_counts = dict(
        sorted(Counter(str(row["action_scorer"]) for row in action_rows).items())
    )
    action_event_keys = {str(row["event_key"]) for row in action_rows}
    conflicting_action_keys = {
        str(row["event_key"])
        for row in action_rows
        if bool(row["has_cross_variant_action_conflict"])
    }
    action_labels = Counter(str(row["action_scorer"]) for row in touch_rows)
    warning_counts: Counter[str] = Counter()
    for row in video_rows:
        warning_counts.update(str(value) for value in row["warning_types"])

    summary: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "snapshot": SNAPSHOT,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": archive_path.stat().st_size if archive_path is not None else None,
        "archive_sha256": sha256_file(archive_path) if archive_path is not None else None,
        "license_boundary": LICENSE_BOUNDARY,
        "report_files": len(report_rows),
        "unique_report_hashes": len(report_hash_counts),
        "exact_duplicate_report_hash_groups": sum(
            count > 1 for count in report_hash_counts.values()
        ),
        "exact_duplicate_report_file_copies": sum(
            count - 1 for count in report_hash_counts.values()
        ),
        "source_videos": len(video_rows),
        "canonical_touch_rows": len(touch_rows),
        "canonical_analyzed_frames": sum(int(row["analyzed_frames"]) for row in video_rows),
        "weapon_counts": weapon_counts,
        "gender_counts": gender_counts,
        "age_group_counts": age_group_counts,
        "report_mode_counts": mode_counts,
        "canonical_warning_counts": dict(sorted(warning_counts.items())),
        "canonical_touch_action_labels": dict(sorted(action_labels.items())),
        "phrase_rows": len(phrase_rows),
        "phrase_videos": len({str(row["youtube_id"]) for row in phrase_rows}),
        "phrase_human_reviewed_rows": sum(bool(row["human_reviewed"]) for row in phrase_rows),
        "action_variant_rows": len(action_rows),
        "action_event_keys": len(action_event_keys),
        "action_videos": len({str(row["youtube_id"]) for row in action_rows}),
        "action_conflicting_event_keys": len(conflicting_action_keys),
        "action_label_counts": action_label_counts,
        "action_human_reviewed_rows": sum(bool(row["human_reviewed"]) for row in action_rows),
        "priority_rows": len(priority_rows),
        "priority_rows_with_proxy_label": sum(
            row["priority_attacker"] in {"left", "right"} for row in priority_rows
        ),
        "priority_note_categories": priority_categories,
        "priority_human_reviewed_rows": sum(
            bool(row["human_reviewed"]) for row in priority_rows
        ),
        "video_manifest": "data/manifests/korean-fencing-tracker-videos.jsonl",
        "video_manifest_sha256": video_manifest_sha256,
        "report_manifest": "data/manifests/korean-fencing-tracker-reports.jsonl",
        "report_manifest_sha256": report_manifest_sha256,
        "touch_output": "data/processed/korean-fencing-tracker-touches.parquet",
        "touch_output_sha256": sha256_file(touch_output_path),
        "phrase_output": "data/processed/korean-fencing-tracker-phrases.parquet",
        "phrase_output_sha256": sha256_file(phrase_output_path),
        "action_output": "data/processed/korean-fencing-tracker-action-proxies.parquet",
        "action_output_sha256": sha256_file(action_output_path),
        "priority_output": "data/processed/korean-fencing-tracker-priority.parquet",
        "priority_output_sha256": sha256_file(priority_output_path),
        "claim_boundary": (
            "The 88 report artifacts collapse to 78 unique public-video IDs with one OCR-only "
            "canonical report each. Their 1,037 score-change rows and 1,640,367 analyzed frames "
            "are upstream automatic outputs, not manually verified touches or complete bouts; "
            "63 canonical reports carry a video-truncated warning. Distinct continuous-report "
            "variants add 172 automatic action rows over 165 event keys; seven keys disagree "
            "between variants. The 302 phrase rows are rule-generated and all marked unreviewed. "
            "The 21 populated priority values among 76 candidates are score-recipient proxies "
            "for double-light events, not independent referee annotations. Source media were "
            "neither bundled nor downloaded."
        ),
    }
    if validate_snapshot:
        _validate_snapshot(summary)
    write_json(summary_path, summary)
    return summary


def fetch_and_prepare(
    archive_path: Path,
    source_directory: Path,
    video_manifest_path: Path,
    report_manifest_path: Path,
    touch_output_path: Path,
    phrase_output_path: Path,
    action_output_path: Path,
    priority_output_path: Path,
    summary_path: Path,
    *,
    user_agent: str,
    force: bool = False,
) -> dict[str, Any]:
    source = acquire(
        archive_path,
        source_directory,
        user_agent=user_agent,
        force=force,
    )
    return prepare(
        source,
        video_manifest_path,
        report_manifest_path,
        touch_output_path,
        phrase_output_path,
        action_output_path,
        priority_output_path,
        summary_path,
        archive_path=archive_path,
    )
