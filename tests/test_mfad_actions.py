from __future__ import annotations

import pandas as pd

from pisteflow.analysis.mfad_actions import evaluate_grouped


def test_mfad_action_evaluation_has_no_participant_overlap() -> None:
    rows = []
    labels = ["fleche", "lunge", "one_step_forward_lunge"]
    for participant in range(9):
        for action_index, action in enumerate(labels):
            for trial in range(2):
                rows.append(
                    {
                        "subject_group_id": f"participant-{participant}",
                        "skill_level": "subelite",
                        "action": action,
                        "trial_index_within_subject_action": trial + 1,
                        "time_steps": 50 + action_index * 10 + trial,
                        "body_scale_mean": 1.0 + participant / 100,
                        "joint_01_x_range": action_index + participant / 100,
                    }
                )
    frame = pd.DataFrame(rows)

    report, _model, features = evaluate_grouped(frame, n_splits=3)

    assert report["participant_overlap_all_folds"] == 0
    assert all(fold["participant_overlap"] == 0 for fold in report["folds"])
    assert "skill_level" not in features
    assert "trial_index_within_subject_action" not in features
