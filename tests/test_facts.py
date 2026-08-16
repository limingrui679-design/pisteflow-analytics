import pandas as pd

from pisteflow.sources.facts import normalize


def test_normalize_deduplicates_and_flags_conflicting_clip() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "video_url": [
                "storage/bout/abc/combined/1.mp4",
                "storage/bout/abc/combined/1.mp4",
                "storage/bout/abc/clips/2.mp4",
                "storage/bout/abc/clips/2.mp4",
            ],
            "label": [
                "Attack left",
                "Attack left",
                "Remise right",
                "Riposte right",
            ],
        }
    )

    result, stats = normalize(frame)

    assert len(result) == 3
    assert stats["exact_repeated_rows_after_first"] == 1
    assert stats["contradictory_video_assets"] == 1
    multiplicity = result.loc[
        result["annotation_id"] == "facts-annotation:1", "annotation_multiplicity"
    ].item()
    assert multiplicity == 2
    assert result["training_eligible"].sum() == 1
    assert set(result["clip_set"]) == {"combined", "clips"}
