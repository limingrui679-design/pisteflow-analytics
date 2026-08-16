from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pisteflow.sources.zenodo_fencing_errors import (
    ArchiveSpec,
    MultiLabelAnnotation,
    build_action_rows,
    parse_central_directory_tail,
)


def test_central_directory_tail_and_mirror_pairing(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("lunge/correct/IMG_1.MOV", b"original")
        archive.writestr("lunge/correct/IMG_1_mirrored.mp4", b"mirror")
    payload = archive_path.read_bytes()
    entries = parse_central_directory_tail(payload, len(payload))
    spec = ArchiveSpec("lunge", "lunge.zip", "lunge.zip", len(payload), "x", 1)

    rows = build_action_rows(
        spec,
        entries,
        {"IMG_1": MultiLabelAnnotation(("correct",), ())},
        archive_complete_local=False,
    )

    assert len(rows) == 1
    assert rows[0]["canonical_single_label"] == "correct"
    assert rows[0]["multi_label_row_available"] is True
    assert rows[0]["single_label_present_in_multi_labels"] is True
    assert rows[0]["mirror_is_augmentation_not_independent_sample"] is True
