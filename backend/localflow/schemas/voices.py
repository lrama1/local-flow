"""Voices API (Sections 12.2, 16)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from localflow.schemas.core import JobOut
from localflow.schemas.story import LlmChoice, UsageOut
from localflow.worker.adapters.speech import VoiceMix

VoiceKind = Literal["builtin", "custom"]


class VoiceOut(BaseModel):
    id: str
    kind: VoiceKind
    adapter_key: str
    native_id: str | None
    base_voice_id: str | None
    base_voice_name: str | None = None
    name: str
    description: str
    language: str
    gender: str
    params: dict[str, Any]
    quality: str = ""
    traits: str = ""
    available: bool = Field(description="Its TTS model is installed and the worker can run it")
    preview_url: str | None = Field(default=None, description="Cached sample (GET, ?token= allowed)")
    preview_text: str | None = None
    created_at: datetime
    updated_at: datetime


class VoiceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    base_voice_id: str = Field(description="A built-in voice to start from")
    description: str = Field(default="", max_length=2000)
    params: dict[str, Any] = Field(default_factory=dict, description="Adapter voice params (blend, speed)")


class VoiceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=80)
    base_voice_id: str | None = None
    description: str | None = Field(default=None, max_length=2000)
    params: dict[str, Any] | None = None


class VoiceDesignBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_voice_id: str
    description: str = Field(min_length=1, max_length=2000)
    llm: LlmChoice | None = None


class VoiceDesign(BaseModel):
    """LLM output for `voice_design` (Section 12.2)."""

    base_weight: float = Field(default=1.0, ge=0.2, le=1.0)
    mix: list[VoiceMix] = Field(default_factory=list, max_length=2)
    speed: float = Field(default=1.0, ge=0.7, le=1.4)
    notes: str = ""


class VoiceDesignOut(BaseModel):
    params: dict[str, Any]
    notes: str
    warnings: list[str] = Field(default_factory=list)
    usage: UsageOut | None = None


class PreviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, max_length=500, description="Defaults to a sample line")
    refresh: bool = Field(default=False, description="Regenerate even if a sample is cached")


class AuditionBody(BaseModel):
    """Hear unsaved settings: a base voice plus params."""

    model_config = ConfigDict(extra="forbid")

    base_voice_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    text: str | None = Field(default=None, max_length=500)


class PreviewOut(BaseModel):
    url: str | None = Field(default=None, description="Ready to play (cached sample)")
    job: JobOut | None = Field(
        default=None, description="Queued: play the job's audio asset when it completes"
    )


class VoiceSuggestion(BaseModel):
    voice: VoiceOut | None
    reason: str
