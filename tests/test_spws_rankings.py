from __future__ import annotations

from pathlib import Path

from pisteflow.sources.spws_rankings import normalize_published_name, parse_workbook_name


def test_parse_spws_workbook_name_maps_polish_weapon_and_women_marker() -> None:
    parsed = parse_workbook_name(Path("SZABLA-K3-2024-2025.xlsx"))

    assert parsed == {
        "weapon": "sabre",
        "gender": "women",
        "category": "3",
        "season": "2024-2025",
    }


def test_parse_spws_workbook_name_keeps_2021_gender_unknown() -> None:
    parsed = parse_workbook_name(Path("szpada-open-2021.xlsx"))

    assert parsed == {
        "weapon": "epee",
        "gender": "unknown",
        "category": "OPEN",
        "season": "2021",
    }


def test_parse_spws_workbook_name_accepts_version_suffix() -> None:
    parsed = parse_workbook_name(Path("FLORET-2-2022v4.xlsx"))

    assert parsed is not None
    assert parsed["weapon"] == "foil"
    assert parsed["gender"] == "men"


def test_normalize_spws_name_is_exact_spelling_normalization_only() -> None:
    assert normalize_published_name("  ŻUKOWSKI   Wojciech ") == "żukowski wojciech"
