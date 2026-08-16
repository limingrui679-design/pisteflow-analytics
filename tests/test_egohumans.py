from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from pisteflow.provenance import sha256_file
from pisteflow.sources import egohumans


def _fixture(path: Path) -> None:
    payload = {
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
        "images": [
            {
                "id": 1,
                "file_name": "03_fencing/001_fencing/ego/aria01/images/rgb/00001.jpg",
                "width": 1408,
                "height": 1408,
                "video_name": "fencing_001_fencing_aria01_rgb",
                "license": 1,
            },
            {
                "id": 2,
                "file_name": "01_tagging/001_tagging/ego/aria01/images/rgb/00001.jpg",
                "width": 1408,
                "height": 1408,
                "video_name": "tagging_001_tagging_aria01_rgb",
                "license": 1,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "num_keypoints": 2,
                "keypoints_3d": [0, 0, 0, 1, 1, 1, 1, 1],
                "human_name": "aria02",
                "camera_name": "aria01",
                "time_stamp": 1,
            },
            {"id": 2, "image_id": 2, "num_keypoints": 17, "keypoints_3d": []},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_filters_fencing_and_groups_by_capture_sequence(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = tmp_path / "benchmark.json"
    _fixture(source)
    monkeypatch.setattr(egohumans, "EGO_RGB_BYTES", source.stat().st_size)
    monkeypatch.setattr(egohumans, "EGO_RGB_SHA256", sha256_file(source))
    monkeypatch.setattr(egohumans, "EXPECTED_IMAGES", 1)
    monkeypatch.setattr(egohumans, "EXPECTED_ANNOTATIONS", 1)
    monkeypatch.setattr(egohumans, "EXPECTED_2D_KEYPOINTS", 2)
    monkeypatch.setattr(egohumans, "EXPECTED_3D_KEYPOINTS", 2)
    monkeypatch.setattr(egohumans, "EXPECTED_SEQUENCES", 1)
    monkeypatch.setattr(egohumans, "EXPECTED_VIEWS", 1)
    monkeypatch.setattr(egohumans, "EXPECTED_CAPTURE_ALIASES", 1)
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    preview = preview_dir / "cam03.jpg"
    preview.write_bytes(b"preview")
    monkeypatch.setattr(
        egohumans,
        "PREVIEW_FILES",
        {"cam03.jpg": (preview.stat().st_size, sha256_file(preview))},
    )

    summary = egohumans.prepare(
        source,
        tmp_path / "images.jsonl",
        tmp_path / "views.jsonl",
        tmp_path / "sequences.jsonl",
        tmp_path / "summary.json",
        preview_dir=preview_dir,
        preview_manifest_path=tmp_path / "preview.jsonl",
    )

    assert summary["images"] == 1
    assert summary["annotations"] == 1
    assert summary["2d_keypoints"] == 2
    assert summary["3d_keypoints"] == 2
    assert summary["sequences"] == 1
    assert summary["views"] == 1
    assert summary["verified_synchronized_preview_images"] == 1
    assert summary["verified_synchronized_preview_groups"] == 1
    image = json.loads((tmp_path / "images.jsonl").read_text(encoding="utf-8"))
    assert image["source_group_id"] == "egohumans:03_fencing:001"
    assert image["weapon"] == "unknown"
    assert image["media_downloaded"] is False
