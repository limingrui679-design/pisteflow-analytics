import json

import pytest

from pisteflow.sources.youtube8m_fencing import (
    EXPECTED_MEMBERS,
    FENCING_MID,
    parse_histogram,
    parse_id_jsonp,
    parse_label_jsonp,
)


def test_parse_histogram_requires_the_pinned_fencing_row() -> None:
    payload = f"999,/m/example,Other\n{EXPECTED_MEMBERS},{FENCING_MID},Fencing\n"
    mid, count = parse_histogram(payload)

    assert mid == FENCING_MID
    assert count == EXPECTED_MEMBERS


def test_parse_label_jsonp_is_non_executing_and_complete() -> None:
    identifiers = [f"{value:04x}" for value in range(EXPECTED_MEMBERS)]
    payload = f'p("{FENCING_MID.removeprefix("/m/")}",{json.dumps(identifiers)});'

    assert parse_label_jsonp(payload) == identifiers

    with pytest.raises(ValueError, match="JSONP"):
        parse_label_jsonp("__import__('os').system('false')")


def test_parse_id_jsonp_validates_both_identifiers() -> None:
    assert parse_id_jsonp('i("HB1g","9qcAI2U4wbY");', "HB1g") == "9qcAI2U4wbY"

    with pytest.raises(ValueError, match="HB1g"):
        parse_id_jsonp('i("other","9qcAI2U4wbY");', "HB1g")
