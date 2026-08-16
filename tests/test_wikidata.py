from pathlib import Path

import pytest

from pisteflow.sources import wikidata


def _entity_with_claims(claims: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {"claims": claims}


def _claim(value: object, *, rank: str = "normal") -> dict[str, object]:
    return {"rank": rank, "mainsnak": {"datavalue": {"value": value}}}


def test_quantity_converts_wikidata_units() -> None:
    entity = _entity_with_claims(
        {
            "P2048": [
                _claim(
                    {
                        "amount": "+1.82",
                        "unit": "http://www.wikidata.org/entity/Q11573",
                    }
                )
            ]
        }
    )
    assert wikidata._quantity(entity, "P2048", wikidata._HEIGHT_UNITS) == 182.0


def test_birth_year_and_deprecated_claim_filter() -> None:
    entity = _entity_with_claims(
        {
            "P569": [
                _claim({"time": "+1888-01-01T00:00:00Z"}, rank="deprecated"),
                _claim({"time": "+1994-07-09T00:00:00Z"}),
            ]
        }
    )
    assert wikidata._birth_year(entity) == 1994


def test_entity_row_requires_human_fencing_membership_and_preserves_routes() -> None:
    entity = {
        "labels": {"en": {"value": "Example Fencer"}},
        "descriptions": {"en": {"value": "example foil fencer"}},
        "claims": {
            "P31": [_claim({"id": wikidata.HUMAN})],
            "P106": [_claim({"id": wikidata.FOIL_FENCER_OCCUPATION})],
            "P641": [_claim({"id": wikidata.FENCING_SPORT})],
        },
        "sitelinks": {},
    }

    row = wikidata._entity_row("Q123", entity, {}, retrieved_at="2026-08-16T00:00:00Z")

    assert row is not None
    assert row["weapons"] == ["foil"]
    assert row["weapons_from_occupation"] == ["foil"]
    assert row["has_fencing_occupation"] is True
    assert row["has_fencing_sport_statement"] is True


def test_entity_row_rejects_fictional_or_unrelated_entities() -> None:
    fictional = {
        "claims": {
            "P31": [_claim({"id": "Q95074"})],
            "P106": [_claim({"id": wikidata.FENCER_OCCUPATION})],
        }
    }
    unrelated = {"claims": {"P31": [_claim({"id": wikidata.HUMAN})]}}

    assert wikidata._entity_row("Q1", fictional, {}, retrieved_at="now") is None
    assert wikidata._entity_row("Q2", unrelated, {}, retrieved_at="now") is None


def test_fetch_entities_caches_successful_batches_before_reporting_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifiers = [f"Q{index}" for index in range(1, 52)]

    def fake_batch(ids: list[str], *, user_agent: str) -> dict[str, dict[str, object]]:
        del user_agent
        if ids[0] == "Q51":
            raise RuntimeError("temporary outage")
        return {identifier: {"id": identifier} for identifier in ids}

    monkeypatch.setattr(wikidata, "_entity_batch", fake_batch)

    with pytest.raises(RuntimeError, match="Failed batch indexes: 1"):
        wikidata.fetch_entities(
            identifiers,
            tmp_path,
            user_agent="test",
            workers=2,
        )

    cached = wikidata._load_json(tmp_path / "batch-0000.json")
    assert set(cached["entities"]) == set(identifiers[:50])
    assert not (tmp_path / "batch-0001.json").exists()
