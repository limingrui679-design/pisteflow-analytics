from __future__ import annotations

from pathlib import Path

import pytest

from pisteflow.sources.wikidata_images import (
    _download_selection,
    _file_title,
    _weapon_status,
    image_memberships,
)


def test_image_memberships_collapse_shared_file_and_keep_athletes() -> None:
    athletes = [
        {
            "athlete_id": "wikidata:Q2",
            "wikidata_id": "Q2",
            "name": "Two",
            "weapons": ["foil"],
            "gender": "female",
            "image_file": "Shared_image.jpg",
        },
        {
            "athlete_id": "wikidata:Q1",
            "wikidata_id": "Q1",
            "name": "One",
            "weapons": ["epee"],
            "gender": "male",
            "image_file": "Shared image.jpg",
        },
    ]

    grouped = image_memberships(athletes)

    assert list(grouped) == ["File:Shared image.jpg"]
    assert [row["wikidata_id"] for row in grouped["File:Shared image.jpg"]] == ["Q1", "Q2"]
    assert _weapon_status(grouped["File:Shared image.jpg"]) == (
        "ambiguous_multi_weapon",
        ["epee", "foil"],
    )


def test_file_title_rejects_empty_and_adds_namespace() -> None:
    assert _file_title(None) is None
    assert _file_title("  ") is None
    assert _file_title("Fencer.jpg") == "File:Fencer.jpg"
    assert _file_title("File:Fencer.jpg") == "File:Fencer.jpg"


def test_download_selection_never_requests_thumbnail_upscaling() -> None:
    small = {
        "direct_media_url": (
            "https://upload.wikimedia.org/wikipedia/commons/a/ab/example.jpg"
        ),
        "mime_type": "image/jpeg",
        "width": 800,
    }
    large = {**small, "width": 1600}

    assert _download_selection(small, 1024) == (
        small["direct_media_url"],
        "commons_original_no_upscale",
        ".jpg",
    )
    url, variant, extension = _download_selection(large, 1024)
    assert "/1024px-example.jpg" in url
    assert variant == "commons_thumbnail_w1024"
    assert extension == ".jpg"


def test_download_rejects_unsafe_worker_count(tmp_path: Path) -> None:
    from pisteflow.sources.wikidata_images import download

    with pytest.raises(ValueError, match="workers"):
        download(
            tmp_path / "missing.jsonl",
            tmp_path / "images",
            tmp_path / "summary.json",
            workers=13,
        )
