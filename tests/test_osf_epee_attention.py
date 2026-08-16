import pandas as pd

from pisteflow.sources import osf_epee_attention


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    attack_rows: list[dict[str, object]] = []
    covat_rows: list[dict[str, object]] = []
    conditions = [
        ("coherent", "L", "coherent left"),
        ("coherent", "R", "coherent right"),
        ("incoherent", "L", "incoherent left"),
        ("incoherent", "R", "incoherent right"),
        ("no-cue", "L", "no-cue left"),
        ("no-cue", "R", "no-cue right"),
    ]
    for group, attack_prefix, covat_prefix, source_group in [
        ("beginner", "D", "D", "Beg"),
        ("advanced", "E", "S", "Adv"),
    ]:
        for number in range(1, 13):
            reaction_time = 0.300 + number / 1_000
            attack_rows.append(
                {
                    "ID": f"{attack_prefix}{number}",
                    "sex": "M " if number % 2 else "W",
                    "group": group,
                    "age": 20 + number,
                    "h": 170 + number,
                    "w": 60 + number,
                    "HIT": 10 + number,
                    "EXTIME": 0.7 + number / 100,
                    "COVATrt": reaction_time,
                    "Unnamed: 9": None,
                }
            )
            for cue, direction, event in conditions:
                covat_rows.append(
                    {
                        "group": source_group,
                        "ID": f"{covat_prefix}{number}",
                        "event": event,
                        "direction": direction,
                        "cue_t": cue,
                        "rt": reaction_time,
                    }
                )
    return pd.DataFrame(attack_rows), pd.DataFrame(covat_rows)


def test_normalize_reconciles_explicit_and_inferred_crosswalks() -> None:
    attack_raw, covat_raw = _frames()
    attack = osf_epee_attention.normalize_attack(attack_raw)
    covat = osf_epee_attention.normalize_covat(covat_raw, attack)

    assert attack.shape[0] == 24
    assert covat.shape[0] == 144
    assert attack["participant_key"].nunique() == 24
    assert covat.groupby("participant_key").size().eq(6).all()
    assert covat.loc[covat["group"].eq("beginner"), "crosswalk_basis"].eq(
        "exact_source_id"
    ).all()
    assert covat.loc[covat["group"].eq("advanced"), "crosswalk_basis"].eq(
        "inferred_numeric_suffix_and_rounded_mean"
    ).all()


def test_pinned_view_only_files_have_no_declared_license() -> None:
    assert set(osf_epee_attention.FILES) == {"attack", "covat", "analysis_code"}
    assert "view_only=" in osf_epee_attention.SOURCE_URL
