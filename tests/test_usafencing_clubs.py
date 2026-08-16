import pytest

from pisteflow.sources.usafencing_clubs import normalize_zip_code, parse_specialties


def test_parse_specialties_maps_only_documented_tokens() -> None:
    assert parse_specialties("Foil; Epee; Saber; Para") == (
        "Foil",
        "Epee",
        "Saber",
        "Para",
    )
    assert parse_specialties(None) == ()


def test_parse_specialties_rejects_invented_weapon() -> None:
    with pytest.raises(ValueError, match="Unexpected USA Fencing club specialty"):
        parse_specialties("Longsword")


def test_normalize_zip_code_recovers_leading_zero() -> None:
    assert normalize_zip_code(6105) == "06105"
    assert normalize_zip_code(6105.0) == "06105"
    assert normalize_zip_code("12345-6789") == "12345-6789"
    assert normalize_zip_code(None) is None
