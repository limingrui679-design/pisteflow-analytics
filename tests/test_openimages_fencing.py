from __future__ import annotations

from pisteflow.sources.openimages_fencing import (
    _acquisition_candidates,
    _licensed_for_backup_acquisition,
    _metadata_payload,
    _weapon_status,
)


def test_openimages_single_weapon_label_is_usable() -> None:
    assert _weapon_status(["/m/02y8z", "/m/06s7q8"]) == "sabre"


def test_openimages_multiple_weapon_labels_are_not_forced_to_one_weapon() -> None:
    assert _weapon_status(["/m/026q93", "/m/032t45"]) == "ambiguous_multi_weapon"


def test_openimages_broad_fencing_label_keeps_weapon_unknown() -> None:
    assert _weapon_status(["/m/02y8z", "/m/0h8mjgm"]) == "unknown"


def test_openimages_metadata_payload_preserves_attribution() -> None:
    payload = _metadata_payload(
        {
            "OriginalURL": "https://example.test/original.jpg",
            "OriginalLandingURL": "https://example.test/page",
            "License": "https://creativecommons.org/licenses/by/2.0/",
            "AuthorProfileURL": "https://example.test/author",
            "Author": "Example Author",
            "Title": "Fencing image",
            "OriginalSize": "1234",
            "OriginalMD5": "CY9rzUYh03PK3k6DJie09g==",
            "Thumbnail300KURL": "https://example.test/thumbnail.jpg",
            "Rotation": "0.0",
        }
    )

    assert payload["image_license_status"] == "image_metadata_joined_cc_by_2_local_copy"
    assert payload["author"] == "Example Author"
    assert payload["original_size"] == 1234
    assert payload["rotation"] == 0.0


def test_openimages_backup_requires_joined_cc_by_license() -> None:
    assert _licensed_for_backup_acquisition(
        {"image_license": "https://creativecommons.org/licenses/by/2.0/"}
    )
    assert not _licensed_for_backup_acquisition({"image_license": None})


def test_openimages_acquisition_candidates_are_ordered_and_deduplicated() -> None:
    row = {
        "download_url": "https://example.test/s3.jpg",
        "original_url": "https://example.test/original.jpg",
        "thumbnail_300k_url": "https://example.test/original.jpg",
    }

    assert _acquisition_candidates(row) == [
        ("official_s3", "https://example.test/s3.jpg"),
        ("original_url", "https://example.test/original.jpg"),
    ]
