from __future__ import annotations

import pandas as pd

from pisteflow.analysis.asmi_referee import evaluate_grouped


def test_asmi_referee_evaluation_holds_out_filename_groups() -> None:
    rows: list[dict[str, object]] = []
    labels = ["left", "right", "simultaneous"]
    for group in range(12):
        for label_index, label in enumerate(labels):
            rows.append(
                {
                    "clip_id": f"clip-{group}-{label}",
                    "source_group_id": f"source-{group}",
                    "upstream_label": label,
                    "frame_mean_000": float(label_index) + group / 100,
                    "flow_mean": float(label_index % 2),
                }
            )

    report, oof, _model, features = evaluate_grouped(pd.DataFrame(rows), n_splits=3)

    assert report["group_overlap_all_folds"] == 0
    assert all(fold["group_overlap"] == 0 for fold in report["folds"])
    assert len(oof) == len(rows)
    assert set(features) == {"frame_mean_000", "flow_mean"}
