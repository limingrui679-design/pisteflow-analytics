from __future__ import annotations

from pisteflow.models import LicenseStatus, SourceKind, VideoRecord, Weapon
from pisteflow.video.features import _training_intervals


def test_training_intervals_use_only_reviewed_segments() -> None:
    record = VideoRecord(
        video_id="commons:example",
        title="Example",
        weapon=Weapon.FOIL,
        source_kind=SourceKind.WIKIMEDIA_COMMONS,
        source_url="https://commons.wikimedia.org/wiki/File:Example.webm",
        license_status=LicenseStatus.OPEN,
        discovered_at="2026-08-15T00:00:00Z",
        metadata={"training_segments": [[2, 6], [10, 14.5]]},
    )

    assert _training_intervals(record, 20.0) == [(2.0, 6.0), (10.0, 14.5)]
