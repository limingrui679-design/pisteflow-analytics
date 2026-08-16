from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pisteflow.sources.fencingly import (
    _geometry_fingerprint,
    deduplicate_rows,
    parse_pose_file,
)


def _pose(offset: float) -> dict[str, object]:
    values: list[float] = []
    for joint in range(25):
        values.extend([offset + joint, offset + joint + 1, 0.8])
    return {"person_id": [-1], "pose_keypoints_2d": values}


def test_fencingly_parser_validates_body25_and_extracts_features(tmp_path: Path) -> None:
    directory = tmp_path / "IZQUIERDAO"
    directory.mkdir()
    path = directory / "Example_output_filtered_data_ordenado.json"
    payload = {
        "frames": [
            {"frame_name": f"frame-{index}", "poses": [_pose(index), _pose(index + 100)]}
            for index in range(3)
        ]
    }
    blob = json.dumps(payload).encode()
    path.write_bytes(blob)

    row = parse_pose_file((path, "v1"))

    assert row["upstream_side_label"] == "left"
    assert row["source_frame_count"] == 3
    assert row["first_two_pose_slots"] == 6
    assert row["sha256"] == hashlib.sha256(blob).hexdigest()
    assert row["pair_both_detected_frame_fraction"] == 1.0


def test_fencingly_deduplication_collapses_copies_and_excludes_label_conflict() -> None:
    base = {
        "snapshot": "original",
        "source_directory": "IZQUIERDAO",
        "source_name": "A",
        "upstream_side_label": "left",
        "sha256": "a" * 64,
        "geometry_fingerprint": "c" * 64,
    }
    same = {**base, "snapshot": "v1"}
    conflict_left = {**base, "source_name": "B", "sha256": "b" * 64}
    conflict_right = {
        **conflict_left,
        "source_directory": "DERECHAO",
        "upstream_side_label": "right",
    }

    retained, audit = deduplicate_rows([base, same, conflict_left, conflict_right])

    assert len(retained) == 1
    assert retained[0]["exact_copy_count"] == 2
    assert retained[0]["appears_in_both_snapshots"] is True
    assert audit["exact_unique_sequences"] == 2
    assert audit["label_conflict_unique_sequences_excluded"] == 1


def test_fencingly_geometry_fingerprint_ignores_uniform_translation_and_scale() -> None:
    sequence = np.zeros((3, 2, 25, 3), dtype=np.float32)
    for frame in range(3):
        for person in range(2):
            for joint in range(1, 15):
                sequence[frame, person, joint] = [
                    10 + person * 100 + joint * 2 + frame,
                    20 + person * 50 + joint * 3 + frame,
                    0.8,
                ]
    transformed = sequence.copy()
    visible = transformed[..., 2] > 0
    transformed[..., 0][visible] = transformed[..., 0][visible] * 2.5 + 400
    transformed[..., 1][visible] = transformed[..., 1][visible] * 2.5 - 120

    assert _geometry_fingerprint(sequence) == _geometry_fingerprint(transformed)
