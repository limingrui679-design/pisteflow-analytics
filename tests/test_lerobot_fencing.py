import json
from pathlib import Path

from pisteflow.sources.lerobot_fencing import _metadata_summary


def test_metadata_summary_checks_episode_lengths(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "total_frames": 5,
                "total_videos": 6,
                "fps": 30,
                "robot_type": "so-100",
                "features": {
                    "camera_1": {"dtype": "video"},
                    "camera_2": {"dtype": "video"},
                    "state": {"dtype": "float32"},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0,"length":2}\n{"episode_index":1,"length":3}\n',
        encoding="utf-8",
    )

    summary = _metadata_summary(tmp_path)

    assert summary["episodes"] == 2
    assert summary["frames"] == 5
    assert summary["camera_streams"] == 2
