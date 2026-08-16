from pisteflow.sources.efc_results import parse_page


def test_parse_event_catalog_row_without_decoding_icons() -> None:
    payload = b"""
    <div class="tr tr-link" onclick="window.location = 'https://www.fencing-efc.eu/results/x'">
      <div class="td">Example Cup</div><div class="td">Example City</div>
      <div class="td"><div class="text-nowrap">30/12 - 02/01/2025</div></div>
      <div class="td">Epee, Sabre</div><div class="td"><svg></svg><svg></svg></div>
      <div class="td">Seniors</div><div class="td"><svg></svg></div>
      <a class="td" href="https://www.fencing-efc.eu/results/x/download"></a>
    </div>
    """

    rows = parse_page(payload, 3)

    assert rows[0]["start_date"] == "2024-12-30"
    assert rows[0]["end_date"] == "2025-01-02"
    assert rows[0]["weapons"] == ["epee", "sabre"]
    assert rows[0]["gender_icon_count"] == 2


def test_preserves_other_as_an_auxiliary_discipline() -> None:
    payload = b"""
    <div class="tr tr-link" onclick="window.location = 'https://www.fencing-efc.eu/results/y'">
      <div class="td">Example Cup</div><div class="td">Example City</div>
      <div class="td"><div class="text-nowrap">02/01/2025</div></div>
      <div class="td">Epee, Other</div><div class="td"><svg></svg></div>
      <div class="td">Cadets</div><div class="td"><svg></svg></div>
      <a class="td"></a>
    </div>
    """

    rows = parse_page(payload, 4)

    assert rows[0]["weapons"] == ["epee"]
    assert rows[0]["auxiliary_disciplines"] == ["other"]
