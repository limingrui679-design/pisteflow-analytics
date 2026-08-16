from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pisteflow.sources.mfad import prepare


def test_prepare_mfad_groups_by_participant_and_flags_missing_sensors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "skeleton").mkdir(parents=True)
    sensor_directory = source / "sensor" / "1011"
    sensor_directory.mkdir(parents=True)
    skeleton = np.arange(4 * 18 * 3, dtype=np.float64).reshape(4, 18, 3)
    np.save(source / "skeleton" / "1_p1_a1_1.npy", skeleton)
    (sensor_directory / "hand.txt").write_text(
        "2026-01-01-00:00:00-000 0 0 0 0 0 1000\n"
        "2026-01-01-00:00:00-020 1 2 3 4 5 6\n"
        "2026-01-01-00:00:00-040 2 3 4 5 6 7\n"
        "2026-01-01-00:00:00-060 0 0 0 0 0 1000\n"
    )

    skeleton_output = tmp_path / "skeleton.parquet"
    sensor_output = tmp_path / "sensor.parquet"
    summary = prepare(
        source,
        skeleton_output,
        sensor_output,
        tmp_path / "summary.json",
    )
    skeleton_frame = pd.read_parquet(skeleton_output)
    sensor_frame = pd.read_parquet(sensor_output)

    assert summary["skeleton_sequences"] == 1
    assert summary["sensor_stream_segments"] == 1
    assert summary["missing_sensor_files_by_placement"] == {
        "left_foot": 1,
        "right_foot": 1,
    }
    assert skeleton_frame.loc[0, "subject_group_id"] == sensor_frame.loc[
        0, "subject_group_id"
    ]
    assert sensor_frame.loc[0, "observations"] == 2
