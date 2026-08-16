from pisteflow.sources import plos_fencer_cognition


def test_safe_column_normalization() -> None:
    assert plos_fencer_cognition._safe_column("Basal-BDNF") == "basal_bdnf"
    assert plos_fencer_cognition._safe_column(" Cueing-Endo-V-R-RT. ") == (
        "cueing_endo_v_r_rt"
    )


def test_pinned_files_are_cc_by_supplements() -> None:
    assert set(plos_fencer_cognition.FILES) == {"demographics", "outcomes"}
    assert plos_fencer_cognition.LICENSE == "CC BY 4.0"
