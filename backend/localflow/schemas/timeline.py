"""Timeline (SceneBuilder) and export bodies (Section 15)."""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TransitionType = Literal["cut", "crossfade", "fade_from_black"]
TrackKind = Literal["video", "audio"]
ExportPreset = Literal["720p", "1080p"]

MAX_AUDIO_TRACKS = 3


def clip_id() -> str:
    return "c" + secrets.token_hex(5)


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TransitionType = "cut"
    duration_s: float = Field(default=0.0, ge=0, le=5)


class Clip(BaseModel):
    """A video clip (sequential; `start_s` is derived from the order, trims and crossfades) or an audio
    clip (placed freely by `start_s`). `in_s`/`out_s` trim the source asset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=clip_id, min_length=1, max_length=64)
    asset_id: str = Field(min_length=1)
    start_s: float = Field(default=0.0, ge=0, le=36_000)
    in_s: float = Field(default=0.0, ge=0)
    out_s: float = Field(gt=0)
    transition_in: Transition = Field(default_factory=Transition, description="Video clips only")
    gain_db: float = Field(default=0.0, ge=-60, le=24)
    shot_id: str | None = None

    @model_validator(mode="after")
    def _trim(self) -> Clip:
        if self.out_s <= self.in_s:
            raise ValueError("out_s must be after in_s")
        return self

    @property
    def duration_s(self) -> float:
        return self.out_s - self.in_s


class Track(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    kind: TrackKind
    name: str = ""
    muted: bool = Field(default=False, description="Video track: mutes the clips' own audio")
    gain_db: float = Field(default=0.0, ge=-60, le=24)
    clips: list[Clip] = Field(default_factory=list)


class TimelineData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: int = Field(default=24, ge=1, le=120)
    width: int = Field(default=1280, ge=16, le=7680)
    height: int = Field(default=720, ge=16, le=7680)
    tracks: list[Track]

    @model_validator(mode="after")
    def _shape(self) -> TimelineData:
        ids = [t.id for t in self.tracks]
        if len(set(ids)) != len(ids):
            raise ValueError("track ids must be unique")
        if sum(t.kind == "video" for t in self.tracks) != 1:
            raise ValueError("a timeline has exactly one video track")
        if sum(t.kind == "audio" for t in self.tracks) > MAX_AUDIO_TRACKS:
            raise ValueError(f"at most {MAX_AUDIO_TRACKS} audio tracks")
        clips = [c.id for t in self.tracks for c in t.clips]
        if len(set(clips)) != len(clips):
            raise ValueError("clip ids must be unique")
        return self

    @property
    def video(self) -> Track:
        return next(t for t in self.tracks if t.kind == "video")

    @property
    def audio(self) -> list[Track]:
        return [t for t in self.tracks if t.kind == "audio"]


class TimelineOut(BaseModel):
    project_id: str
    timeline: TimelineData
    duration_s: float
    updated_at: datetime | None = None
    notes: list[str] = Field(default_factory=list, description="What normalization changed")


class SkippedShot(BaseModel):
    shot_id: str
    reason: str


class AutoAssembleBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_transition: Literal["cut", "crossfade"] = Field(
        default="crossfade", description="Between scenes; shots within a scene are cut"
    )
    transition_s: float = Field(default=0.5, ge=0, le=3)
    fade_in: bool = Field(default=True, description="Fade the first shot in from black")
    keep_music: bool = Field(default=True, description="Keep clips on the music track")


class AutoAssembleOut(TimelineOut):
    skipped: list[SkippedShot] = Field(default_factory=list)


class ExportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: ExportPreset = "720p"
    loudnorm: bool = Field(default=False, description="EBU R128 loudness normalization (-16 LUFS)")
