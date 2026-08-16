from pisteflow.sources.openverse_fencing import (
    _creator_group,
    _is_candidate,
    _weapon_candidates,
)


def test_candidate_filter_keeps_sport_and_rejects_name_collisions() -> None:
    sport = {
        "title": "Olympic fencer during a foil competition",
        "tags": [{"name": "athlete"}, {"name": "fencing"}],
    }
    aircraft = {"title": "Sukhoi Su-24M Fencer-D aircraft", "tags": []}
    physical_fence = {"title": "MUGA sports fencing installation", "tags": []}

    assert _is_candidate(sport, strong_query=False)
    assert not _is_candidate(aircraft, strong_query=False)
    assert not _is_candidate(physical_fence, strong_query=True)


def test_weapon_candidates_merge_query_hint_and_text_without_double_counting() -> None:
    assert _weapon_candidates("Foil fencer at an epee event", {"foil"}) == ["epee", "foil"]
    assert _weapon_candidates("Paralympic competition", {"sabre"}) == ["sabre"]


def test_creator_group_is_stable_and_source_scoped() -> None:
    flickr = {"source": "flickr", "creator": "Example Photographer"}
    museum = {"source": "museum", "creator": "Example Photographer"}

    assert _creator_group(flickr) == _creator_group(dict(flickr))
    assert _creator_group(flickr) != _creator_group(museum)
