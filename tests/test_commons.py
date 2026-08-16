from __future__ import annotations

from pathlib import Path

from pisteflow.sources.commons import (
    _clean_download_url,
    load_curated_titles,
    select_training_derivative,
)


def test_clean_download_url_keeps_direct_upload_host_and_drops_tracking_query() -> None:
    source = (
        "https://upload.wikimedia.org/wikipedia/commons/3/34/"
        "Club_kampioenschappen_schermen.ogv?utm_source=commons"
    )

    result = _clean_download_url(source)

    assert result == (
        "https://upload.wikimedia.org/wikipedia/commons/3/34/"
        "Club_kampioenschappen_schermen.ogv"
    )


def test_curated_segments_are_preserved_for_training(tmp_path: Path) -> None:
    source = tmp_path / "commons.yml"
    source.write_text(
        """videos:
  - title: "File:Example.webm"
    weapon: foil
    group_id: profile
    content_kind: other
    weapon_evidence: reviewed profile
    training_segments: [[1, 5], [9, 12]]
""",
        encoding="utf-8",
    )

    record = load_curated_titles(source)["File:Example.webm"]

    assert record["training_segments"] == [[1.0, 5.0], [9.0, 12.0]]
    assert record["weapon_evidence"] == "reviewed profile"


def test_select_training_derivative_prefers_largest_bounded_webm() -> None:
    rows = [
        {"src": "original.ogv", "type": "video/ogg", "height": 1080},
        {
            "src": "small.webm",
            "type": "video/webm; codecs=\"vp9, opus\"",
            "transcodekey": "240p.vp9.webm",
            "width": 320,
            "height": 240,
            "bandwidth": 300_000,
        },
        {
            "src": "training.webm",
            "type": "video/webm; codecs=\"vp9, opus\"",
            "transcodekey": "480p.vp9.webm",
            "width": 640,
            "height": 480,
            "bandwidth": 1_100_000,
        },
        {
            "src": "large.webm",
            "type": "video/webm; codecs=\"vp9, opus\"",
            "transcodekey": "1080p.vp9.webm",
            "width": 1920,
            "height": 1080,
            "bandwidth": 4_000_000,
        },
    ]

    selected = select_training_derivative(rows, maximum_height=720)

    assert selected is not None
    assert selected["src"] == "training.webm"
