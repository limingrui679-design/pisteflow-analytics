import json

import numpy as np
import pandas as pd

from pisteflow.sources.fencing_matches_ai import extract_pose_features, normalize_descriptor


def test_descriptor_marks_repeated_windows_without_dropping_groups() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "ClipName": ["clip_1", "clip_2"],
            "URL": ["https://www.youtube.com/watch?v=abc", "https://www.youtube.com/watch?v=abc"],
            "Start_frame": [10.0, 10.0],
            "End_frame": [85.0, 85.0],
            "Light1": [2, 2],
            "Light2": [2, 2],
            "Hit_label": ["1-0-0", "1-0-0"],
        }
    )

    result, stats = normalize_descriptor(frame)

    assert result["group_id"].nunique() == 1
    assert stats["duplicate_source_window_groups"] == 1
    assert result["training_eligible"].sum() == 1


def test_pose_features_are_finite_for_partial_visibility() -> None:
    values = np.zeros((75, 2, 25, 3), dtype=float)
    values[:, :, 1, 0] = np.arange(75)[:, None]
    values[:, :, 1, 1] = 100
    values[:, :, 1, 2] = 0.9

    features = extract_pose_features(json.dumps(values.tolist()).encode())

    assert len(features) > 400
    assert all(np.isfinite(value) for value in features.values())
    assert features["pose_s1_j02_visible_fraction"] == 1.0


def test_pose_features_keep_explicit_missing_tracker_slot() -> None:
    values = np.zeros((75, 2, 25, 3), dtype=float).tolist()
    values[10][1] = []

    features = extract_pose_features(json.dumps(values).encode())

    assert all(np.isfinite(value) for value in features.values())
    assert features["pose_s1_tracker_payload_frame_fraction"] == 1.0
    assert features["pose_s2_tracker_payload_frame_fraction"] == 74 / 75
