import pandas as pd

from pisteflow.sources import brazilian_ssbs as source


def test_normalize_frame_filters_fencing_and_removes_sensitive_fields() -> None:
    rows = [
        [5, 2, 4, 3, 5, 5, 20, "feminine", " Esgrima ", 12, "Profissional", "Yes", "A"],
        [4, 3, 2, 5, 4, 6, 31, "masculine", "Volei", 18, "amateur", "No", "B"],
    ]
    frame = pd.DataFrame(rows, columns=source.SOURCE_COLUMNS)

    result = source.normalize_frame(frame)

    assert len(result) == 1
    assert result.loc[0, "SBSS1"] == 5
    assert result.loc[0, "age_band_years"] == "20-24"
    assert result.loc[0, "practice_years_band"] == "10-14"
    assert result.loc[0, "athlete_level_source"] == "professional"
    assert bool(result.loc[0, "currently_uses_dietary_supplements"])
    assert result.loc[0, "weapon"] == "unknown"
    assert "Biological sex" not in result
    assert "Who prescribed the dietary supplement?" not in result
