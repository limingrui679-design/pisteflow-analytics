from pisteflow.sources.british_fencing import parse_sheet


def test_parse_sheet_removes_names_and_reads_checkboxes() -> None:
    html = """
    <table class="clsGridGantt">
      <tr><td></td><td>Names</td><td>Weapon Category</td><td>Event</td>
      <td>Competition Date</td><td>Final Placing</td><td>Starting Fencers</td>
      <td>% finish</td><td>No of Poule Fights</td><td>Poule V</td><td>% Poules</td>
      <td>Poules HS</td><td>Poules HR</td><td>Ind</td><td>No of 1 point losses</td>
      <td>Promoted to DE</td><td>Received a Bye</td><td>DE Fights</td>
      <td>DE Victories</td><td>% of Wins</td><td>Final DE Score</td></tr>
      <tr><td class="clsNumberCell">1</td><td>Example Athlete</td><td>JWF</td>
      <td>World Cup</td><td>09/10/21</td><td>12</td><td>100</td><td>12%</td>
      <td>6</td><td>5</td><td>0.83</td><td>28</td><td>16</td><td>12</td><td>1</td>
      <td><img alt="true"/></td><td><img alt="false"/></td><td>3</td><td>2</td>
      <td>66.7%</td><td>13-15</td></tr>
    </table>
    """

    frame = parse_sheet(html, season="2021-22", weapon="foil")

    assert "source_name" not in frame.columns
    assert frame.loc[0, "athlete_id"].startswith("bf-")
    assert frame.loc[0, "gender"] == "women"
    assert bool(frame.loc[0, "promoted_to_de"]) is True
    assert bool(frame.loc[0, "received_bye"]) is False
    assert frame.loc[0, "pool_bouts"] == 6
