import pandas as pd

from pisteflow.sources.cmu_score_detection import (
    normalize_signals,
    normalize_touch_images,
    normalize_yolo,
)


def test_touch_and_signal_group_hashes_join_without_names() -> None:
    title = "2019 Event - First v. Second"
    touch = pd.DataFrame(
        {
            "image": [{"bytes": b"pixels", "path": f"{title}_frame_000010.jpg"}],
            "label": [1],
        }
    )
    signal = pd.DataFrame(
        {
            "video_id": [title],
            "frame_id": [10],
            "timestamp": [0.3],
            "red_ratio": [0.1],
            "green_ratio": [0.2],
            "Label": [1.0],
        }
    )

    records = normalize_touch_images(touch)
    signals = normalize_signals(signal)

    assert records[0]["group_id"] == signals.loc[0, "group_id"]
    assert title not in str(records[0])
    assert bool(signals.loc[0, "touch_peak"]) is True


def test_yolo_parser_counts_boxes_and_removes_source_path() -> None:
    frame = pd.DataFrame(
        {
            "image": [
                "/private/Copy-of-2019-Event-First-v_-Second_frame_000010_jpg.rf.abc123.jpg"
            ],
            "label": ["0 0.5 0.5 0.2 0.1\n0 0.2 0.3 0.1 0.2"],
        }
    )

    result = normalize_yolo(frame, split="train")

    assert result.loc[0, "box_count"] == 2
    assert "image" not in result.columns
    assert result.loc[0, "source_frame_number"] == 10
