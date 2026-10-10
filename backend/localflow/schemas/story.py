"""Story pipeline schemas (Section 8.3): LLM output contracts and the related API bodies."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ShotType = Literal["wide", "medium", "close", "extreme_close", "insert", "pov", "over_shoulder"]
CameraMove = Literal[
    "static",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "dolly_in",
    "dolly_out",
    "truck_left",
    "truck_right",
    "orbit",
    "handheld",
    "crane_up",
    "crane_down",
    "zoom_in",
    "zoom_out",
]
GenerationMode = Literal["t2v", "i2v", "flf2v", "ref2v"]
IngredientKind = Literal["character", "location", "prop", "style"]
ShotStatus = Literal["planned", "keyframe_ready", "video_ready", "approved"]

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str, fallback: str = "item") -> str:
    s = _SLUG.sub("-", text.lower()).strip("-")[:48].strip("-")
    return s or fallback


class _Lenient(BaseModel):
    """LLM outputs: ignore unexpected extra keys instead of failing the whole response."""

    model_config = ConfigDict(extra="ignore")


# --- Step 1: refine ----------------------------------------------------------------------------
class RefinedStory(_Lenient):
    title: str = Field(min_length=1, max_length=200)
    logline: str = Field(min_length=1)
    synopsis: str = Field(min_length=1, description="<= 200 words")
    story: str = Field(min_length=1, description="Prose, beat-ordered")
    estimated_runtime_s: int = Field(ge=1, le=3600)
    tone: str = ""
    visual_style: str = ""


# --- Step 2: bible -----------------------------------------------------------------------------
class StyleSpec(_Lenient):
    look: str = ""
    palette: str = ""
    lighting: str = ""
    lens: str = ""
    era: str = ""
    negative: str = ""


class VoiceHints(_Lenient):
    gender: str = ""
    age: str = ""
    tone: str = ""


class _Keyed(_Lenient):
    key: str = Field(min_length=1, max_length=64, description="lowercase slug, unique across the bible")
    name: str = Field(min_length=1, max_length=200)

    @field_validator("key")
    @classmethod
    def _slug(cls, v: str) -> str:
        return slugify(v)


class CharacterSpec(_Keyed):
    description: str = ""
    visual_prompt: str = Field(
        min_length=1, description="appearance, wardrobe, age, distinctive features; no actions"
    )
    voice: VoiceHints = Field(default_factory=VoiceHints)


class LocationSpec(_Keyed):
    visual_prompt: str = Field(min_length=1)


class PropSpec(_Keyed):
    visual_prompt: str = Field(min_length=1)


class StoryBible(_Lenient):
    style: StyleSpec = Field(default_factory=StyleSpec)
    characters: list[CharacterSpec] = Field(default_factory=list)
    locations: list[LocationSpec] = Field(default_factory=list)
    props: list[PropSpec] = Field(default_factory=list)

    def all_keys(self) -> list[str]:
        return [x.key for x in (*self.characters, *self.locations, *self.props)]


# --- Step 3: breakdown -------------------------------------------------------------------------
class DialogueSpec(_Lenient):
    character_key: str
    line: str = Field(min_length=1)
    emotion: str = ""


class ShotSpec(_Lenient):
    duration_s: float = Field(gt=0, le=600)
    shot_type: ShotType = "medium"
    camera_move: CameraMove = "static"
    ingredient_keys: list[str] = Field(default_factory=list)
    generation_mode: GenerationMode = "i2v"
    keyframe_prompt: str = Field(min_length=1)
    end_keyframe_prompt: str | None = None
    video_prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    dialogue: list[DialogueSpec] = Field(default_factory=list)
    sfx_prompt: str | None = None
    music_cue: str | None = None
    needs_extend: bool = False
    extend_segments: int | None = Field(default=None, ge=1, le=20)
    continuity_notes: str = ""


class SceneSpec(_Lenient):
    title: str = Field(min_length=1)
    summary: str = ""
    location_key: str | None = None
    shots: list[ShotSpec] = Field(min_length=1)


class Breakdown(_Lenient):
    scenes: list[SceneSpec] = Field(min_length=1)

    def shot_count(self) -> int:
        return sum(len(s.shots) for s in self.scenes)

    def total_duration(self) -> float:
        """Screen time including extensions (each extension continues for up to the shot's duration)."""
        return sum(sh.duration_s * (1 + (sh.extend_segments or 0)) for s in self.scenes for sh in s.shots)


# --- Step 4: prompt rewrite --------------------------------------------------------------------
class ShotPrompts(_Lenient):
    keyframe_prompt: str = Field(min_length=1)
    end_keyframe_prompt: str | None = None
    video_prompt: str = Field(min_length=1)
    negative_prompt: str = ""


# --- Context blocks ----------------------------------------------------------------------------
class RenderConstraints(BaseModel):
    """Section 8.2: derived from the project's video/image models."""

    video_model: str
    max_clip_seconds: float
    fps: int
    supported_modes: list[GenerationMode]
    supports_native_audio: bool
    supports_first_last_frame: bool
    prompt_style_guide: str
    aspect_ratio: str


class CastMember(BaseModel):
    """Section 8.5 cast block entry."""

    key: str
    name: str
    visual_prompt: str
    character_info: str | None = None
    voice: dict[str, str] | None = None


# --- API bodies --------------------------------------------------------------------------------
class LlmChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic", "openrouter"]
    model: str = Field(min_length=1)


class StoryOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genre: str | None = None
    tone: str | None = None
    target_length_seconds: int | None = Field(default=None, ge=5, le=3600)
    audience: str | None = None
    style_notes: str | None = None


class RefineBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_idea: str = Field(min_length=1, max_length=20000)
    options: StoryOptions = Field(default_factory=StoryOptions)
    cast_keys: list[str] = Field(default_factory=list, description="Existing characters that must appear")
    llm: LlmChoice | None = None
    dry_run: bool = False


class BibleBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LlmChoice | None = None
    dry_run: bool = False


class BreakdownBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_length_seconds: int | None = Field(default=None, ge=5, le=3600)
    replace_existing: bool = False
    llm: LlmChoice | None = None


class StoryPatch(BaseModel):
    """Manual edits, or applying a dry-run proposal (`applied_from`). Creates a new revision."""

    model_config = ConfigDict(extra="forbid")

    raw_idea: str | None = None
    options: StoryOptions | None = None
    cast_keys: list[str] | None = None
    refined: RefinedStory | None = None
    bible: StoryBible | None = None
    applied_from: Literal["refine", "bible"] | None = None
    llm: LlmChoice | None = None


class RestoreBody(BaseModel):
    revision: int = Field(ge=1)


class RewriteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1, max_length=2000)
    llm: LlmChoice | None = None


class UsageOut(BaseModel):
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    attempts: int


class StoryOut(BaseModel):
    id: str | None
    project_id: str
    revision: int | None
    step: str
    status: str
    raw_idea: str
    options: StoryOptions
    cast_keys: list[str]
    refined: RefinedStory | None
    bible: StoryBible | None
    has_breakdown: bool
    llm_provider: str | None
    llm_model: str | None
    template_versions: dict[str, str]
    edited_fields: list[str]
    warnings: list[str]
    usage: dict[str, Any] | None
    created_at: datetime | None


class RevisionSummary(BaseModel):
    revision: int
    step: str
    status: str
    llm_model: str | None
    title: str | None
    created_at: datetime


class StoryStepResult(BaseModel):
    story: StoryOut
    saved: bool
    usage: UsageOut | None
    warnings: list[str]
    ingredients: dict[str, int] | None = None  # bible: created/updated/locked/removed counts
    scenes: int | None = None
    shots: int | None = None


class ShotRewriteResult(BaseModel):
    shot_id: str
    proposal: ShotPrompts
    usage: UsageOut
