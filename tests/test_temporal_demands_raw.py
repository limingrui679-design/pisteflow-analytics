import pandas as pd

from pisteflow.sources.temporal_demands_raw import normalize_bouts


def test_temporal_bouts_convert_milliseconds_and_preserve_missing_ptbp() -> None:
    frame = pd.DataFrame(
        [
            [
                136,
                3,
                1,
                5,
                900_000,
                200_000,
                700_000,
                700_000,
                None,
                20,
                5_000,
                15_000,
            ]
        ],
        columns=[
            "Bout",
            "Weapon",
            "SEX",
            "Cod_Weapon",
            "TBT (s)",
            "WT (s)",
            "TPT (s)",
            "PRT (s)",
            "PTBP (s)",
            "Halte (n)",
            "AAT (s)",
            "AHT (s)",
        ],
    )

    normalized = normalize_bouts(frame)

    assert normalized.loc[0, "weapon"] == "foil"
    assert normalized.loc[0, "total_bout_time_s"] == 900
    assert normalized.loc[0, "pause_between_periods_missing_upstream"]
    assert normalized.loc[0, "pause_between_periods_s_derived_zero"] == 0
    assert normalized.loc[0, "pause_identity_holds"]
