from pathlib import Path

import yaml


def test_roboflow_catalog_has_unique_non_acquired_sources() -> None:
    path = Path("data/registry/roboflow_fencing_catalog.yml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload["sources"]

    identifiers = [row["id"] for row in rows]
    assert len(rows) >= 20
    assert len(identifiers) == len(set(identifiers))
    assert all(not row["acquisition_status"].startswith("acquired") for row in rows)
    assert all(row["url"].startswith("https://universe.roboflow.com/") for row in rows)
