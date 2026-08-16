from __future__ import annotations

import pandas as pd

from pisteflow.sources import osf_sabre_eye_tracking


def test_binary_number_rejects_placeholder() -> None:
    result = osf_sabre_eye_tracking._binary_number(pd.Series([1, 0, "x", 2]))

    assert result.iloc[0] == 1
    assert result.iloc[1] == 0
    assert result.iloc[2:].isna().all()


def test_source_local_keys_separate_main_and_pilot_namespaces() -> None:
    main = osf_sabre_eye_tracking._source_local_key("main-participant", "P1")
    pilot = osf_sabre_eye_tracking._source_local_key("pilot-participant", "P1")

    assert main != pilot
    assert "p1" not in main.casefold()


def test_pinned_assets_cover_main_pilot_aggregate_and_paper() -> None:
    assert set(osf_sabre_eye_tracking.ASSETS) == {
        "main",
        "main_aggregate",
        "pilot",
        "paper",
    }
    assert osf_sabre_eye_tracking.ASSETS["main"].expected_bytes == 38_420_360
