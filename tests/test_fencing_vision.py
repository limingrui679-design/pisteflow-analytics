import json
from pathlib import Path

import pandas as pd

from pisteflow.sources.fencing_vision import prepare


def test_prepare_fencing_vision_groups_frames_by_source_family(tmp_path: Path) -> None:
    payload = {
        "licenses": [{"id": 1, "name": "MIT"}],
        "categories": [{"id": 1, "name": "weapon"}],
        "images": [
            {"id": 1, "file_name": "BoutA_mp4-1.jpg", "width": 10, "height": 10},
            {"id": 2, "file_name": "BoutA_mp4-2.jpg", "width": 10, "height": 10},
        ],
        "annotations": [
            {
                "id": 7,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 2, 3, 4],
                "area": 12,
                "segmentation": [[1, 2, 3, 4, 5, 6]],
                "iscrowd": 0,
            }
        ],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "annotations.parquet"

    summary = prepare(annotations, output, tmp_path / "summary.json")
    frame = pd.read_parquet(output)

    assert summary["images"] == 2
    assert summary["source_image_families"] == 1
    assert summary["annotations"] == 1
    assert frame.loc[0, "segmentation_points"] == 3
