"""Typed contracts shared by collection, training, analytics, and reporting."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

VideoSplit: TypeAlias = Literal["train", "validation", "test", "unassigned"]
ContentKind: TypeAlias = Literal[
    "bout",
    "multi_bout_stream",
    "highlight",
    "instructional",
    "interview",
    "other",
    "unknown",
]


class Weapon(StrEnum):
    EPEE = "epee"
    FOIL = "foil"
    SABRE = "sabre"
    UNKNOWN = "unknown"


class LicenseStatus(StrEnum):
    OPEN = "open"
    LINK_ONLY = "link_only"
    UNKNOWN = "unknown"
    RESTRICTED = "restricted"


class SourceKind(StrEnum):
    WIKIMEDIA_COMMONS = "wikimedia_commons"
    FENCING_TV = "fencing_tv"
    YOUTUBE = "youtube"
    PUBLIC_API = "public_api"
    DATASET = "dataset"
    LOCAL = "local"
    SYNTHETIC = "synthetic"


class VideoRecord(BaseModel):
    """A video manifest row; media redistribution is governed separately."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    weapon: Weapon
    source_kind: SourceKind
    source_url: HttpUrl
    direct_media_url: HttpUrl | None = None
    landing_page_url: HttpUrl | None = None
    license_status: LicenseStatus
    license_code: str | None = None
    attribution: str | None = None
    athlete_names: list[str] = Field(default_factory=list)
    athlete_ids: list[str] = Field(default_factory=list)
    competition: str | None = None
    event_date: date | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    discovered_at: datetime
    reviewed: bool = False
    split: VideoSplit = "unassigned"
    group_id: str | None = None
    weapon_candidates: list[Weapon] = Field(default_factory=list)
    content_kind: ContentKind = "unknown"
    source_created_at: datetime | None = None
    thumbnail_url: HttpUrl | None = None
    view_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def enforce_media_boundary(self) -> VideoRecord:
        if self.license_status is not LicenseStatus.OPEN and self.direct_media_url is not None:
            raise ValueError("direct_media_url is allowed only for verified open media")
        return self


class AthleteRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    athlete_id: str
    name: str
    country: str | None = None
    club: str | None = None
    gender: Literal["female", "male", "other", "unknown"] = "unknown"
    birth_year: int | None = Field(default=None, ge=1900, le=2100)
    weapons: list[Weapon] = Field(default_factory=list)
    source_url: HttpUrl | None = None
    observed_at: datetime


class CompetitionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    result_id: str
    athlete_id: str
    athlete_name: str | None = None
    tournament: str
    event_date: date
    city: str | None = None
    weapon: Weapon
    medal: Literal["gold", "silver", "bronze", "none"] = "none"
    level: str | None = None
    notes: str | None = None
    source_url: HttpUrl
    fetched_at: datetime


class FrameObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    left_x: float | None = None
    right_x: float | None = None
    distance: float | None = Field(default=None, ge=0)
    left_speed: float | None = Field(default=None, ge=0)
    right_speed: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)


class DatasetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    source_name: str
    source_url: HttpUrl
    fetched_at: datetime
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    record_count: int = Field(ge=0)
    schema_version: str
    license_status: LicenseStatus
    metadata: dict[str, Any] = Field(default_factory=dict)
