from pisteflow.sources.internet_archive import parse_metadata


def test_parse_metadata_keeps_team_foil_no_derivatives_boundary() -> None:
    payload = {
        "metadata": {
            "identifier": "teamfoil042405",
            "licenseurl": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
        },
        "files": [
            {
                "name": "teamfoil042405_512kb.mp4",
                "size": "201729214",
                "md5": "abc",
                "sha1": "def",
                "length": "2776.44",
                "width": "320",
                "height": "240",
            }
        ],
    }

    result = parse_metadata("teamfoil042405", payload)

    assert result["weapon"] == "foil"
    assert result["derivative_policy"] == "local_analysis_only_no_published_derivatives"
    assert result["bytes"] == 201_729_214
    assert result["training_eligible"] is True


def test_parse_metadata_limits_sports_attack_to_reviewed_fencing_segment() -> None:
    payload = {
        "metadata": {
            "identifier": "mactvmiv-Sports_Attack_Season_4_Episode_5",
            "licenseurl": "https://creativecommons.org/licenses/by-nc/3.0/",
        },
        "files": [
            {
                "name": "Sports_Attack_Season_4_Episode_5.mp4",
                "size": "92881120",
                "md5": "abc",
                "sha1": "def",
                "length": "891.44",
                "width": "640",
                "height": "360",
            }
        ],
    }

    result = parse_metadata("mactvmiv-Sports_Attack_Season_4_Episode_5", payload)

    assert result["weapon"] == "unknown"
    assert result["content_type"] == "real_multi_weapon_instructional_demonstration"
    assert result["segment_start_seconds"] == 586.0
    assert result["segment_end_seconds"] == 839.0
    assert result["training_duration_seconds"] == 253.0
    assert result["training_eligible"] is True


def test_parse_metadata_keeps_unreviewed_2012_clips_weapon_unknown() -> None:
    payload = {
        "metadata": {
            "identifier": "Flickr-7976244921",
            "licenseurl": "https://creativecommons.org/licenses/by-nc/2.0/",
        },
        "files": [
            {
                "name": "NCFDP_Bell_Tower_Classic_Video-7976244921.mp4",
                "size": "2216669",
                "md5": "abc",
                "sha1": "def",
                "length": "21.06",
                "width": "640",
                "height": "360",
            }
        ],
    }

    result = parse_metadata("Flickr-7976244921", payload)

    assert result["source_event"] == "Bell Tower Classic 2012"
    assert result["group_id"] == "ia:bell-tower-classic-2012"
    assert result["weapon"] == "unknown"
    assert result["weapon_label_source"] == "pending_multi_frame_visual_review"
    assert result["training_eligible"] is True
