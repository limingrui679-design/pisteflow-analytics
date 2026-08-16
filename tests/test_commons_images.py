from __future__ import annotations

from pisteflow.sources.commons_images import _blocked_first_level, _source_group, _thumbnail_url


def test_people_equipment_and_team_branches_are_blocked() -> None:
    blocked = [
        "Category:Foil fencers",
        "Category:Foils",
        "Category:Épées",
        "Category:National men's sabre teams",
        "Category:Bell-guard test",
        "Category:1899 World Fencing Championships",
    ]

    assert all(_blocked_first_level(category) for category in blocked)
    assert not _blocked_first_level("Category:Challenge international de Paris 2013")
    assert not _blocked_first_level(
        "Category:Fencing at the 2024 Summer Olympics \N{EN DASH} Women's team sabre"
    )


def test_source_group_preserves_event_branch_and_marks_multi_branch() -> None:
    one = _source_group(
        [
            "Category:Fencing (foil) > Category:Challenge Revenu > "
            "Category:Challenge Revenu 2016"
        ]
    )
    multi = _source_group(
        [
            "Category:Fencing (foil) > Category:Tournoi international Paris Île-de-France",
            "Category:Fencing (sabre) > Category:Tournoi international Paris Île-de-France",
        ]
    )

    assert one == "category_challenge_revenu"
    assert multi == "category_tournoi_international_paris_le_de_france"


def test_thumbnail_url_preserves_raster_format_and_converts_tiff() -> None:
    jpeg = _thumbnail_url(
        "https://upload.wikimedia.org/wikipedia/commons/f/f4/Fencer.jpg?tracking=1",
        "image/jpeg",
        1280,
    )
    tiff = _thumbnail_url(
        "https://upload.wikimedia.org/wikipedia/commons/2/2c/Fencer.tif",
        "image/tiff",
        1280,
    )
    assert jpeg.endswith("/thumb/f/f4/Fencer.jpg/1280px-Fencer.jpg")
    assert tiff.endswith("/thumb/2/2c/Fencer.tif/1280px-Fencer.tif.jpg")
    assert "?" not in jpeg
