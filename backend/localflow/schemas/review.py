"""Vision review of a shot's keyframe or take (Section 8.3 step 5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from localflow.schemas.characters import RefOut
from localflow.schemas.story import LlmChoice, UsageOut

ReviewTarget = Literal["keyframe", "end_keyframe", "video"]
PromptField = Literal["keyframe_prompt", "end_keyframe_prompt", "video_prompt"]


class ReviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str | None = Field(
        default=None, description="Image or video to review; defaults to the selected keyframe, else the take"
    )
    llm: LlmChoice | None = None


class VisionReview(BaseModel):
    """What the vision model returns."""

    model_config = ConfigDict(extra="ignore")

    consistent: bool
    issues: list[str] = Field(default_factory=list)
    suggested_prompt_fix: str = ""


class ShotReviewOut(BaseModel):
    shot_id: str
    asset_id: str
    target: ReviewTarget
    prompt_field: PromptField = Field(description="The shot field `suggested_prompt_fix` replaces")
    current_prompt: str
    consistent: bool
    issues: list[str]
    suggested_prompt_fix: str
    references: list[RefOut] = Field(description="The reference images it was compared against")
    frames_s: list[float] = Field(description="For takes: the reviewed frame times (empty for images)")
    usage: UsageOut


class VisionCapability(BaseModel):
    purpose: str
    available: bool
    model: str | None = Field(description="provider/model the purpose resolves to")
    reason: str | None
