from __future__ import annotations

import numpy as np
import pandas as pd

from pisteflow.analysis import pose_outcomes


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = pose_outcomes.LABELS
    for group_index in range(18):
        label = labels[group_index % len(labels)]
        for clip_index in range(3):
            rows.append(
                {
                    "group_id": f"group-{group_index:02d}",
                    "outcome_class": label,
                    "outcome_code": label,
                    "light_1_code": 1,
                    "light_2_code": 2,
                    "pose_s1_j02_x_mean": float(group_index),
                    "pose_s1_j02_x_std": float(group_index % 3) + clip_index / 10,
                    "pose_s1_j02_x_range": float(group_index % 4),
                    "pose_s1_j02_x_delta": float(clip_index),
                    "pose_s1_j02_visible_fraction": 1.0,
                    "pose_s1_j02_speed_mean": float(group_index % 3),
                    "pose_s1_j02_speed_max": float(group_index % 5),
                    "pose_s1_tracker_payload_frame_fraction": 1.0,
                    "pair_separation_x_mean": float(group_index % 2),
                }
            )
    return pd.DataFrame(rows)


def test_feature_policy_excludes_static_means_and_lights() -> None:
    frame = _synthetic_frame()
    features = pose_outcomes.select_kinematic_features(frame)

    assert "pose_s1_j02_x_std" in features
    assert "pair_separation_x_mean" in features
    assert "pose_s1_j02_x_mean" not in features
    assert "light_1_code" not in features


def test_grouped_evaluation_has_no_group_leakage() -> None:
    frame = _synthetic_frame()
    report, model, features = pose_outcomes.evaluate_grouped(frame, n_splits=3)

    assert report["rows"] == len(frame)
    assert report["group_overlap_all_folds"] == 0
    assert all(fold["group_overlap"] == 0 for fold in report["folds"])
    assert set(report["model"]["labels"]) == set(pose_outcomes.LABELS)
    assert np.isfinite(report["model"]["log_loss"])
    assert model.classes_.shape == (3,)
    assert features
