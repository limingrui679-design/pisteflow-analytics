from __future__ import annotations

import pandas as pd

from pisteflow.sources import peerj_tdcs


def _attention_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant in range(1, 17):
        for stimulation in ("Sham", "a-tDCS"):
            for mode in ("Core", "Dynamic", "Selective"):
                rows.append(
                    {
                        "name": participant,
                        "type of stimulus": stimulation,
                        "test  modes": mode,
                        "pre": 1.0 + participant / 100,
                        "post": 1.2 + participant / 100,
                    }
                )
    return pd.DataFrame(rows, columns=peerj_tdcs._NT_COLUMNS)


def _stop_signal_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant in range(1, 17):
        for stimulation in ("sham", "a-tDCS"):
            rows.append(
                {
                    "name": participant,
                    "type of stimulus": stimulation,
                    "Go_RT pre": 400.0,
                    "Go_RT post": 390.0,
                    "SSRT pre": 250.0,
                    "SSRT post": 240.0,
                }
            )
    return pd.DataFrame(rows, columns=peerj_tdcs._SST_COLUMNS)


def test_normalize_tdcs_workbooks_preserves_crossover_groups() -> None:
    attention = peerj_tdcs.normalize_attention(_attention_frame())
    stop_signal = peerj_tdcs.normalize_stop_signal(_stop_signal_frame())

    assert len(attention) == 96
    assert len(stop_signal) == 32
    assert attention["participant_key"].nunique() == 16
    assert stop_signal["participant_key"].nunique() == 16
    assert set(attention["stimulation"]) == {"sham", "anodal_tdcs"}
    assert set(attention["attention_mode"]) == {"core", "dynamic", "selective"}
    assert attention["change_score"].round(6).eq(0.2).all()
    assert stop_signal["go_rt_change_ms"].eq(-10.0).all()
    assert "name" not in attention
    assert "name" not in stop_signal


def test_attention_rejects_missing_condition_row() -> None:
    frame = _attention_frame().iloc[:-1]
    try:
        peerj_tdcs.normalize_attention(frame)
    except ValueError as exc:
        assert "row layout" in str(exc)
    else:
        raise AssertionError("Incomplete crossover table was accepted")
