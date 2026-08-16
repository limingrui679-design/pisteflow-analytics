from __future__ import annotations

import pandas as pd

from pisteflow.analysis.fencingly_diagnostic import evaluate_grouped


def test_fencingly_diagnostic_holds_out_geometry_groups() -> None:
    rows: list[dict[str, object]] = []
    for group in range(12):
        for label_index, label in enumerate(["left", "right"]):
            rows.append(
                {
                    "clip_id": f"clip-{group}-{label}",
                    "geometry_group_id": f"geometry-{group}",
                    "upstream_side_label": label,
                    "pose_1_center_x_std": float(label_index) + group / 100,
                    "pose_1_center_speed_mean": float(label_index),
                    "pose_1_center_x_mean": 10_000.0 + group,
                }
            )

    report, oof, _model, features = evaluate_grouped(pd.DataFrame(rows), n_splits=3)

    assert report["group_overlap_all_folds"] == 0
    assert all(fold["group_overlap"] == 0 for fold in report["folds"])
    assert "pose_1_center_x_mean" not in features
    assert len(oof) == len(rows)
