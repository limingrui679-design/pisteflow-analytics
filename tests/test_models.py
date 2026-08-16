from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pisteflow.models import LicenseStatus, SourceKind, VideoRecord, Weapon


def test_open_video_may_have_direct_media_url() -> None:
    record = VideoRecord(
        video_id="commons:1",
        title="Example",
        weapon=Weapon.EPEE,
        source_kind=SourceKind.WIKIMEDIA_COMMONS,
        source_url="https://commons.wikimedia.org/wiki/File:Example.webm",
        direct_media_url="https://upload.wikimedia.org/example.webm",
        license_status=LicenseStatus.OPEN,
        discovered_at=datetime.now(UTC),
    )
    assert record.weapon is Weapon.EPEE


def test_link_only_video_cannot_embed_download_url() -> None:
    with pytest.raises(ValidationError, match="direct_media_url"):
        VideoRecord(
            video_id="youtube:1",
            title="Example",
            weapon=Weapon.FOIL,
            source_kind=SourceKind.YOUTUBE,
            source_url="https://youtube.com/watch?v=example",
            direct_media_url="https://example.com/video.mp4",
            license_status=LicenseStatus.LINK_ONLY,
            discovered_at=datetime.now(UTC),
        )
