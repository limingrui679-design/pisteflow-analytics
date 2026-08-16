from pisteflow.sources.loc_fencing_images import (
    _best_image_url,
    _rights_advisory,
    _subject_is_fencing,
    _weapon_candidates,
)


def test_exact_subject_filter_rejects_physical_fence_search_hits() -> None:
    assert _subject_is_fencing({"subject": ["Photographs", "Fencing"]})
    assert not _subject_is_fencing({"subject": ["Fences", "Wooden fences"]})


def test_rights_and_bounded_image_resolution() -> None:
    record = {
        "item": {"rights_advisory": "No known restrictions on publication."},
        "image_url": [
            "https://tile.loc.gov/a.jpg#h=500&w=640",
            "https://tile.loc.gov/b.jpg#h=800&w=1024",
            "https://tile.loc.gov/c.jpg#h=2400&w=3200",
        ],
    }

    assert _rights_advisory(record) == "No known restrictions on publication."
    assert _best_image_url(record) == "https://tile.loc.gov/b.jpg"


def test_weapon_candidates_are_only_textual_context() -> None:
    record = {
        "title": "Winner of Jr. Foils Championship",
        "description": [],
        "item": {},
    }
    assert _weapon_candidates(record) == ["foil"]
