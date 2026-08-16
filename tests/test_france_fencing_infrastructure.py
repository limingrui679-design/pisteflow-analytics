import pandas as pd
import pytest

from pisteflow.sources.france_fencing_infrastructure import normalize_boolean, prepare_clubs


def test_normalize_boolean_preserves_missing() -> None:
    assert normalize_boolean("true") is True
    assert normalize_boolean("Oui") is True
    assert normalize_boolean("FALSE") is False
    assert normalize_boolean("Non") is False
    assert normalize_boolean(None) is None
    with pytest.raises(ValueError, match=r"unexpected Data\.Sports boolean"):
        normalize_boolean("yes")


def test_prepare_clubs_rejects_wrong_snapshot_size() -> None:
    with pytest.raises(ValueError, match="schema or row count"):
        prepare_clubs(pd.DataFrame({"code_fede_ref": [110]}))
