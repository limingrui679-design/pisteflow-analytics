import pandas as pd

from pisteflow.sources import korea_gkl_wheelchair as source


def test_normalize_frame_parses_multispelling_results_without_direct_names() -> None:
    frame = pd.DataFrame(
        [
            [
                "2025-01-01~2025-01-02",
                "Test competition",
                3,
                "펜싱",
                (
                    "박천희 [에뻬B] 3위, [플러레B] 5위, [샤브르B] 8강,"
                    "심재훈 [에빼A] 1위,단체전(박천희, 심재훈) 에빼(남성) 2위"
                ),
            ],
            [
                "2021-01-01~2021-01-03",
                "Legacy competition",
                2,
                "펜싱",
                "심재훈 A 플러레 1위, 에뻬 2위, 샤브르 3위",
            ],
            [
                "2016-01-01~2016-01-05",
                "중국 전지훈련",
                1,
                "펜싱",
                "중국 합동훈련",
            ],
        ],
        columns=source.SOURCE_COLUMNS,
    )

    events, results = source.normalize_frame(frame)

    assert len(events) == 3
    assert events["activity_kind"].tolist() == ["training", "competition", "competition"]
    assert len(results) == 8
    assert results["scope"].value_counts().to_dict() == {"individual": 7, "team": 1}
    assert results["weapon"].value_counts().to_dict() == {"epee": 4, "foil": 2, "sabre": 2}
    assert set(results["placement_notation"]) >= {"rank_1", "rank_2", "last_8"}
    assert results["athlete_code"].dropna().str.startswith("kgw-").all()
    assert "박천희" not in results.to_json(force_ascii=False)
    assert "심재훈" not in results.to_json(force_ascii=False)
