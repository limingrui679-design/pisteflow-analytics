from pisteflow.sources import osf_fencer_psychology


def test_age_bands_are_privacy_coarsened() -> None:
    assert osf_fencer_psychology.age_band(17) == "under_18"
    assert osf_fencer_psychology.age_band(18) == "18_20"
    assert osf_fencer_psychology.age_band(25) == "21_29"
    assert osf_fencer_psychology.age_band(50) == "40_plus"


def test_duration_bands_do_not_preserve_exact_years() -> None:
    assert osf_fencer_psychology.duration_band(2) == "under_5"
    assert osf_fencer_psychology.duration_band(8) == "5_9"
    assert osf_fencer_psychology.duration_band(15) == "10_19"
    assert osf_fencer_psychology.duration_band(53) == "20_plus"


def test_scale_item_counts_match_deposit() -> None:
    assert len(osf_fencer_psychology.MIPS_ITEMS) == 10
    assert len(osf_fencer_psychology.ISCCS_ITEMS) == 31
    assert len(osf_fencer_psychology.AAQ_ITEMS) == 7
