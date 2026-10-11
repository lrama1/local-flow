"""Narration (Sections 8.3 step 6, 12.3): the LLM output contract and the API bodies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from localflow.schemas.core import JobOut
from localflow.schemas.story import LlmChoice, UsageOut, _Lenient

NarrationStatus = Literal["empty", "text", "voiced", "stale"]
NarratorSource = Literal["project", "suggested", "none"]


# --- LLM output ----------------------------------------------------------------------------------
class NarrationLine(_Lenient):
    n: int = Field(ge=1, description="The shot's 1-based position in the film")
    text: str = Field(default="", max_length=2000)


class NarrationLines(_Lenient):
    lines: list[NarrationLine]


# --- API -----------------------------------------------------------------------------------------
class NarratorOut(BaseModel):
    voice_id: str | None
    voice_name: str | None
    source: NarratorSource


class NarrationShotOut(BaseModel):
    shot_id: str
    number: str = Field(description="Scene.shot, e.g. 2.1")
    scene_title: str
    duration_s: float
    text: str
    words: int
    budget_words: int
    estimated_s: float
    status: NarrationStatus
    selected_asset_id: str | None
    asset_duration_s: float | None
    clip_start_s: float | None = Field(description="Where the shot's clip starts on the timeline")
    start_s: float | None = Field(description="Where its narration starts (or would)")
    overflow_s: float = Field(description="How far the line runs into the next narrated shot")


class NarrationPlanOut(BaseModel):
    narrator: NarratorOut
    tts_model: str | None
    speed: float
    shots: list[NarrationShotOut]
    total_words: int
    notes: list[str]


class NarrationWriteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overwrite: bool = Field(default=False, description="Rewrite shots that already have narration")
    style: str | None = Field(
        default=None, max_length=500, description="Perspective and tone, e.g. a hushed storybook"
    )
    shot_ids: list[str] | None = Field(default=None, description="Only these shots; default all")
    llm: LlmChoice | None = None


class NarratedLine(BaseModel):
    shot_id: str
    number: str
    text: str
    words: int
    budget_words: int


class NarrationWriteOut(BaseModel):
    updated: list[NarratedLine]
    skipped: int = Field(description="Shots left alone (already narrated, or not asked for)")
    usage: UsageOut
    warnings: list[str]


class NarrationGenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_ids: list[str] | None = Field(default=None, description="Only these shots; default all")
    only_missing: bool = Field(default=True, description="Skip lines that are already voiced and up to date")
    speed: float = Field(default=1.0, ge=0.8, le=1.2, description="Narration pace")
    tts_model_entry_id: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class SkippedLine(BaseModel):
    shot_id: str
    reason: str


class NarrationGenerateOut(BaseModel):
    jobs: list[JobOut]
    skipped: list[SkippedLine]
    plan: NarrationPlanOut
