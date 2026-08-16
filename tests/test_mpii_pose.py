from __future__ import annotations

import numpy as np

from pisteflow.sources import mpii_pose


def test_mpii_filters_exact_sport_fencing_and_groups_source_video() -> None:
    release = {
        "annolist": [
            {
                "image": {"name": "sport.jpg"},
                "vididx": 1,
                "frame_sec": 12,
                "annorect": {
                    "x1": 1,
                    "y1": 2,
                    "x2": 3,
                    "y2": 4,
                    "scale": 2.5,
                    "objpos": {"x": 10, "y": 11},
                    "annopoints": {
                        "point": [
                            {"id": 10, "x": 5, "y": 6, "is_visible": 1},
                            {
                                "id": 8,
                                "x": 7.5,
                                "y": 8.5,
                                "is_visible": np.array([], dtype=np.uint8),
                            },
                        ]
                    },
                },
            },
            {"image": {"name": "wood.jpg"}, "vididx": 2, "frame_sec": 3},
        ],
        "act": [
            {"act_name": "fencing", "cat_name": "sports"},
            {"act_name": "carpentry, outside house, building a fence", "cat_name": "home repair"},
        ],
        "video_list": np.array(["nGearEu2PlU", "other-video"]),
        "img_train": np.array([1, 1]),
    }

    rows = mpii_pose.records_from_release(release)

    assert len(rows) == 1
    assert rows[0]["weapon"] == "sabre"
    assert rows[0]["source_group_id"] == "youtube:nGearEu2PlU"
    assert rows[0]["person_annotations"] == 1
    assert rows[0]["keypoint_annotations"] == 2
    assert rows[0]["people"][0]["keypoints"][0]["id"] == 8
    assert rows[0]["people"][0]["keypoints"][0]["is_visible"] is None
    assert rows[0]["media_downloaded"] is False
