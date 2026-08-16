import pytest

from pisteflow.sources.france_fencing_licences import (
    _delimiter,
    _is_fencing_code,
    _parse_line,
    metric_key,
    normalize_record,
    selected_resources,
)


def test_quoted_2012_semicolon_rows_are_unwrapped() -> None:
    line = '"cog2;LIBELLE;fed_2012;l_2012"\n'
    assert _delimiter(line) == ";"
    assert _parse_line(line, ";") == ["cog2", "LIBELLE", "fed_2012", "l_2012"]


def test_individually_quoted_2019_fields_keep_full_width() -> None:
    line = '"Code Commune";"Commune";"Code";"Fédération";"Total"\n'
    assert _delimiter(line) == ";"
    assert _parse_line(line, ";") == [
        "Code Commune",
        "Commune",
        "Code",
        "Fédération",
        "Total",
    ]


def test_metric_keys_cover_current_and_legacy_age_schemas() -> None:
    assert metric_key("F - 10 à 14 ans", 2023) == "female_age_10-14"
    assert metric_key("H - NR", 2023) == "male_age_not_reported"
    assert metric_key("l_30_44_h_2018", 2018) == "male_age_30-44"
    assert metric_key("l_30_44__2012", 2012) == "all_age_30-44"
    assert metric_key("l_19f_2011", 2011) == "female_age_0-19"
    assert metric_key("popf_10_14_2017", 2018) is None


def test_normalize_current_record_reconciles_age_cells() -> None:
    row = {
        "Code Commune": "01001",
        "Commune": "L'Abergement-Clémenciat",
        "Département": "01",
        "Région": "Auvergne-Rhône-Alpes",
        "Statut géo": "1.Champ geoc",
        "Code": "110",
        "Fédération": "FF d'Escrime",
        "F - 10 à 14 ans": "2",
        "F - NR": "1",
        "H - 10 à 14 ans": "3",
        "H - NR": "0",
        "Total": "6",
    }
    record = normalize_record(row, year=2023, source_row=2)
    assert record["female_total"] == 3
    assert record["male_total"] == 3
    assert record["total"] == 6
    assert record["sex_balance_delta"] == 0
    assert record["female_total_origin"] == "derived_sum_of_source_age_cells"


def test_resource_selection_and_code_validation() -> None:
    assert [resource.year for resource in selected_resources(2021, 2023)] == [2021, 2022, 2023]
    assert _is_fencing_code("110.0")
    assert not _is_fencing_code("101")
    with pytest.raises(ValueError, match="start_year"):
        selected_resources(2023, 2022)


def test_2011_record_derives_only_missing_male_counts() -> None:
    row = {
        "plg_depcom": "01001",
        "fed_2011": "110",
        "l_2011": "7",
        "l_f_2011": "3",
        "l_19_2011": "5",
        "l_19f_2011": "2",
        "l_2060_2011": "2",
        "l_2060f_2011": "1",
        "l_61_2011": "",
        "l_61f_2011": "",
    }
    record = normalize_record(row, year=2011, source_row=2)
    assert record["male_total"] == 4
    assert record["male_age_0-19"] == 3
    assert record["male_age_20-60"] == 1
    assert record["male_age_61+"] == 0
    assert record["male_total_origin"] == "derived_total_minus_source_female_total"
