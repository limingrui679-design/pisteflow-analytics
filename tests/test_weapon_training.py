from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pisteflow.training.weapon import train_weapon_classifier


def _feature_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, offset in (("train", 0), ("validation", 10), ("test", 20)):
        for weapon_index, weapon in enumerate(("epee", "foil", "sabre")):
            for window in range(2):
                rows.append(
                    {
                        "video_id": f"{split}-{weapon}",
                        "weapon": weapon,
                        "split": split,
                        "group_id": f"{split}-group-{weapon}",
                        "frame_mean_000": float(offset + weapon_index + window / 10),
                        "flow_mean": float(weapon_index),
                    }
                )
    return pd.DataFrame(rows)


def test_weapon_training_rejects_source_family_overlap(tmp_path: Path) -> None:
    frame = _feature_rows()
    frame.loc[frame["split"].eq("test"), "group_id"] = "train-group-epee"
    input_path = tmp_path / "features.parquet"
    frame.to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match="Source-family leakage"):
        train_weapon_classifier(input_path, tmp_path / "model")


def test_weapon_training_reports_three_video_smoke_test(tmp_path: Path) -> None:
    input_path = tmp_path / "features.parquet"
    _feature_rows().to_parquet(input_path, index=False)

    report = train_weapon_classifier(input_path, tmp_path / "model")

    assert report["test"]["videos"] == 3
    assert report["test"]["source_groups"] == 3
    assert report["features_sha256"]
