from datetime import UTC, datetime

from pisteflow.sources import youtube_channels


def test_records_from_payload_are_link_only_and_title_labeled() -> None:
    specification = {
        "id": "UC1oc3twlZObCWUh2rsYMdig",
        "name": "USA Fencing",
        "publisher_type": "official_federation",
    }
    payload = {
        "channel_id": specification["id"],
        "entries": [
            {
                "id": "abcdefghijk",
                "title": "Women's Epee Final",
                "duration": 123.0,
                "view_count": 456,
            },
            {"id": "abcdefghijk", "title": "duplicate"},
            None,
        ],
    }

    records = youtube_channels.records_from_payload(
        specification,
        payload,
        discovered_at=datetime(2026, 8, 16, tzinfo=UTC),
    )

    assert len(records) == 1
    assert records[0].video_id == "youtube:abcdefghijk"
    assert records[0].weapon.value == "epee"
    assert records[0].license_status.value == "link_only"
    assert records[0].metadata["publisher_type"] == "official_federation"


def test_channel_registry_is_identity_unique() -> None:
    identifiers = [row["id"] for row in youtube_channels.CHANNELS]
    assert len(identifiers) == len(set(identifiers)) == 113
    assert all(youtube_channels._uploads_url(value).endswith(value[2:]) for value in identifiers)
    assert len(youtube_channels.TARGETED_SEARCHES) == 31
    assert "wheelchair+fencing" in youtube_channels._search_url(
        youtube_channels.TARGETED_SEARCHES[0]["id"],
        youtube_channels.TARGETED_SEARCHES[0]["query"],
    )
