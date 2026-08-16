import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from pisteflow.provenance import sha256_file
from pisteflow.sources import harmony4d
from pisteflow.sources.harmony4d import EXPECTED_PATHS, parse_inventory


def test_parse_exact_sword_archive_inventory() -> None:
    payload = json.dumps(
        [
            {
                "path": path,
                "lfs": {"size": index + 100, "oid": f"{index + 1:064x}"},
            }
            for index, path in enumerate(sorted(EXPECTED_PATHS))
        ]
        + [{"path": "train/01_wrestling.zip", "lfs": {"size": 1, "oid": "a" * 64}}]
    ).encode()

    rows = parse_inventory(payload)

    assert len(rows) == 5
    assert {row["split"] for row in rows} == {"train", "test"}
    assert all(row["weapon"] is None for row in rows)


def test_inventory_fails_closed_if_an_expected_archive_disappears() -> None:
    payload = json.dumps(
        [
            {"path": path, "lfs": {"size": 1, "oid": "a" * 64}}
            for path in sorted(EXPECTED_PATHS)[:-1]
        ]
    ).encode()

    with pytest.raises(ValueError, match="archive set changed"):
        parse_inventory(payload)


def test_prepare_voxel51_catalog_preserves_adjacent_domain_boundary(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_id": {"$oid": "1"},
                        "filepath": "data/cam01.mp4",
                        "_media_type": "video",
                        "split": "test",
                        "activity": "sword",
                        "sequence_name": "001_sword3",
                        "camera_name": "cam01",
                        "camera_type": "exo",
                        "stream": "main",
                    },
                    {
                        "_id": {"$oid": "2"},
                        "filepath": "data/sequence.fo3d",
                        "_media_type": "3d",
                        "split": "test",
                        "activity": "sword",
                        "sequence_name": "001_sword3",
                        "camera_name": "scene_3d",
                        "camera_type": "3d",
                        "stream": "main",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    preview = tmp_path / "preview.mp4"
    preview.write_bytes(b"preview")
    monkeypatch.setattr(harmony4d, "VOXEL51_SAMPLES_BYTES", samples.stat().st_size)
    monkeypatch.setattr(harmony4d, "VOXEL51_SAMPLES_SHA256", sha256_file(samples))
    monkeypatch.setattr(harmony4d, "VOXEL51_METADATA_BYTES", metadata.stat().st_size)
    monkeypatch.setattr(harmony4d, "VOXEL51_METADATA_SHA256", sha256_file(metadata))
    monkeypatch.setattr(harmony4d, "PREVIEW_BYTES", preview.stat().st_size)
    monkeypatch.setattr(harmony4d, "PREVIEW_SHA256", sha256_file(preview))
    monkeypatch.setattr(harmony4d, "VOXEL51_EXPECTED_RECORDS", 2)
    monkeypatch.setattr(harmony4d, "VOXEL51_EXPECTED_SWORD_RECORDS", 2)
    monkeypatch.setattr(harmony4d, "VOXEL51_EXPECTED_SWORD_SEQUENCES", {"001_sword3"})

    report = harmony4d.prepare_voxel51_catalog(
        samples,
        metadata,
        preview,
        tmp_path / "manifest.jsonl",
        tmp_path / "summary.json",
    )

    assert report["sword_records"] == 2
    assert report["sword_video_views"] == 1
    assert report["sword_3d_scene_records"] == 1
    assert report["sport_fencing_training_eligible"] is False
