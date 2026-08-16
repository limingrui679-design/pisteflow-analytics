import pandas as pd
import pytest

from pisteflow.sources import plos_elite_backpain


def test_yes_no_codes_are_explicit() -> None:
    assert plos_elite_backpain._yes_no(1) is False
    assert plos_elite_backpain._yes_no(2) is True
    assert pd.isna(plos_elite_backpain._yes_no(pd.NA))
    with pytest.raises(ValueError):
        plos_elite_backpain._yes_no(3)


def test_pain_columns_include_all_published_periods() -> None:
    assert len(plos_elite_backpain.PAIN_BINARY_COLUMNS) == 16
    assert len(plos_elite_backpain.PAIN_DURATION_COLUMNS) == 6
