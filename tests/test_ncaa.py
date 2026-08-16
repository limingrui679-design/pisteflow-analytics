from pathlib import Path

from pisteflow.sources.ncaa import discover_pages, parse_event_matrix, parse_page


def test_discover_pages_deduplicates_and_sorts() -> None:
    html = """
    <a href="./ncaa2024.html">2024</a>
    <a href="ncaa2023.html">2023</a>
    <a href="./ncaa2024.html">duplicate</a>
    <a href="other.html">ignore</a>
    """
    assert discover_pages(html) == [
        (2023, "https://ncaa.escrimeresults.com/ncaa2023.html"),
        (2024, "https://ncaa.escrimeresults.com/ncaa2024.html"),
    ]


def test_parse_event_matrix_emits_one_bout_per_pair() -> None:
    rows = [
        ["School", "Name", "No", "1", "2", "3", "V", "TS", "TR", "Ind."],
        ["A", "Athlete One", "1", "", "V5", "3", "1", "8", "8", "0"],
        ["B", "Athlete Two", "2", "2", "", "V5", "1", "7", "8", "-1"],
        ["C", "Athlete Three", "3", "V5", "4", "", "1", "9", "8", "+1"],
    ]
    athletes, bouts = parse_event_matrix(
        rows,
        year=2024,
        event_code="WF",
        source_url="https://ncaa.escrimeresults.com/ncaa2024.html",
    )
    assert len(athletes) == 3
    assert len(bouts) == 3
    assert bouts[0]["winner"] == "left"
    assert bouts[0]["left_score"] == 5
    assert bouts[0]["right_score"] == 2
    assert {row["weapon"] for row in bouts} == {"foil"}


def test_parse_page_restores_legacy_folded_matrix_rows(tmp_path: Path) -> None:
    path = tmp_path / "ncaa2007.html"
    path.write_text(
        """
        <h2><a name="MS">MEN'S SABER</a></h2>
        <table>
          <tr><td>School</td><td>Name</td><td>No</td><td>1</td><td>2</td><td>V</td><td>TS</td><td>TR</td></tr>
          <tr><td>A</td><td>Alpha</td><td>1</td><td></td><td>V5</td><td>1</td><td>5</td><td>3</td>
          <tr><td>B</td><td>Beta</td><td>2</td><td>3</td><td></td><td>0</td><td>3</td><td>5</td>
        </table>
        """,
        encoding="utf-8",
    )
    athletes, bouts, missing = parse_page(
        path,
        year=2007,
        source_url="https://example.test",
    )
    assert len(athletes) == 2
    assert len(bouts) == 1
    assert "MS" not in missing
