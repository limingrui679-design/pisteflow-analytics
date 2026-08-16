from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from pisteflow.sources.sports_images import prepare


def test_prepare_extracts_only_fencing_images(tmp_path: Path) -> None:
    source_image = tmp_path / "image.jpg"
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(source_image)
    archive_path = tmp_path / "sports.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.write(source_image, "sports/fencing/one.jpg")
        archive.write(source_image, "sports/fencing/two.jpg")
        archive.write(source_image, "sports/boxing/one.jpg")

    summary = prepare(
        archive_path,
        tmp_path / "output",
        tmp_path / "manifest.jsonl",
        tmp_path / "summary.json",
    )

    assert summary["all_sport_images"] == 3
    assert summary["fencing_images"] == 1
    assert summary["class_counts"] == {"boxing": 1, "fencing": 2}
    assert len(list((tmp_path / "output").glob("*.jpg"))) == 1
