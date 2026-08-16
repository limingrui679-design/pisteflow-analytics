from pathlib import Path

from pisteflow.provenance import sha256_file
from pisteflow.sources import fencevision_blade


def test_fencevision_blade_collapses_augmented_frames_and_flags_source_leakage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "blade_dataset"
    for split in ("train", "valid", "test"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir()
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "names: ['blade']",
                "version: 3",
                "license: CC BY 4.0",
                fencevision_blade.DATASET_URL,
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fencevision_blade, "DATA_YAML_SHA256", sha256_file(data_yaml))

    fixtures = [
        ("train", "Bout-A_f000001_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", b"a", "0 .5 .5 .2 .2"),
        ("train", "Bout-A_f000001_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", b"b", "0 .4 .4 .2 .2"),
        ("valid", "Bout-A_f000002_jpg.rf.cccccccccccccccccccccccccccccccc", b"c", ""),
        ("test", "Bout-B_f000001_jpg.rf.dddddddddddddddddddddddddddddddd", b"c", "0 .5 .5 .1 .1"),
    ]
    for split, stem, image, label in fixtures:
        (dataset / split / "images" / f"{stem}.jpg").write_bytes(image)
        (dataset / split / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")

    monkeypatch.setattr(fencevision_blade, "EXPECTED_EXPORT_IMAGES", 4)
    monkeypatch.setattr(fencevision_blade, "EXPECTED_UNIQUE_IMAGE_HASHES", 3)
    monkeypatch.setattr(fencevision_blade, "EXPECTED_ORIGINAL_FRAMES", 3)
    monkeypatch.setattr(fencevision_blade, "EXPECTED_SOURCE_VIDEOS", 2)
    monkeypatch.setattr(fencevision_blade, "EXPECTED_BOXES", 3)
    monkeypatch.setattr(fencevision_blade, "EXPECTED_POSITIVE_EXPORT_IMAGES", 3)
    monkeypatch.setattr(fencevision_blade, "EXPECTED_POSITIVE_ORIGINAL_FRAMES", 2)
    monkeypatch.setattr(
        fencevision_blade,
        "EXPECTED_SPLITS",
        {"test": 1, "train": 2, "valid": 1},
    )
    monkeypatch.setattr(fencevision_blade, "EXPECTED_VARIANTS_PER_ORIGINAL", {1: 2, 2: 1})

    summary = fencevision_blade.prepare(
        dataset,
        tmp_path / "manifest.jsonl",
        tmp_path / "summary.json",
        repository_revision=fencevision_blade.SNAPSHOT,
    )

    assert summary["export_images"] == 4
    assert summary["original_frames"] == 3
    assert summary["source_video_overlap_between_upstream_splits"] == 1
    assert summary["original_frame_overlap_between_upstream_splits"] == 0
    assert summary["exact_duplicate_extra_exports"] == 1
