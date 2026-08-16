from pisteflow.sources.usa_team_archives import normalize_member


def test_normalize_member_removes_direct_identity_and_flags_source_negative() -> None:
    payload = {
        "Example Athlete": {
            "first_name": "Example",
            "last_name": "Athlete",
            "region": "Example Region",
            "results": {
                "1/2/2020": {
                    "rank": 1,
                    "rolling": 8.0,
                    "domestic": 10.0,
                    "international": -2.0,
                },
                "2/3/2020": {
                    "rank": 2,
                    "rolling": 12.0,
                    "domestic": 7.0,
                    "international": 5.0,
                },
            },
        }
    }

    result = normalize_member("WF_Sr_usfa_results.json", payload)

    assert len(result) == 2
    assert result["athlete_series_id"].nunique() == 1
    assert result.loc[0, "athlete_series_id"].startswith("uta-series:")
    assert not {"name", "first_name", "last_name", "region"}.intersection(result.columns)
    assert result["weapon"].eq("foil").all()
    assert result["gender"].eq("women").all()
    assert result["age_category"].eq("senior").all()
    assert result["component_balance_error"].sum() == 0
    assert result["negative_points_upstream"].sum() == 1
