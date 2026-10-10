"""Character creator API schemas (Sections 11.7, 16)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from localflow.schemas.board import ExtraView, VoiceHintsIn
from localflow.schemas.core import JobOut
from localflow.schemas.story import LlmChoice, UsageOut, VoiceHints

TemplateCategory = Literal[
    "realistic", "fantastical", "sci-fi", "historical", "anime", "cartoon", "creature", "animal", "custom"
]
TEMPLATE_CATEGORIES: tuple[str, ...] = (
    "realistic",
    "fantastical",
    "sci-fi",
    "historical",
    "anime",
    "cartoon",
    "creature",
    "animal",
    "custom",
)


class CharacterFields(BaseModel):
    """The editable character sheet shared by templates, the wizard and the library."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    visual_prompt: str | None = None
    negative_prompt: str | None = None
    character_info: str | None = None
    body_outfit_prompt: str | None = None
    voice_hints: VoiceHintsIn | None = None


# --- templates ---------------------------------------------------------------------------------------
class TemplateOut(BaseModel):
    id: str | None = Field(description="Null for built-in templates")
    key: str
    name: str
    category: str
    builtin: bool
    visual_prompt: str
    negative_prompt: str
    character_info: str
    body_outfit_prompt: str
    voice_hints: dict[str, str]
    thumbnail_url: str | None


class TemplateCreate(BaseModel):
    """`from_ingredient_id` copies a character's fields (and its primary image as the thumbnail); explicit
    fields override what's copied."""

    model_config = ConfigDict(extra="forbid")

    from_ingredient_id: str | None = None
    name: str | None = Field(default=None, min_length=1, max_length=200)
    category: TemplateCategory = "custom"
    visual_prompt: str | None = None
    negative_prompt: str | None = None
    character_info: str | None = None
    body_outfit_prompt: str | None = None
    voice_hints: VoiceHintsIn | None = None

    @model_validator(mode="after")
    def _needs_source(self) -> TemplateCreate:
        if not self.from_ingredient_id and not (self.name and self.visual_prompt):
            raise ValueError("Give from_ingredient_id, or at least a name and a visual_prompt")
        return self


class TemplateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    category: TemplateCategory | None = None
    visual_prompt: str | None = None
    negative_prompt: str | None = None
    character_info: str | None = None
    body_outfit_prompt: str | None = None
    voice_hints: VoiceHintsIn | None = None


# --- wizard ------------------------------------------------------------------------------------------
class DraftCreate(BaseModel):
    """Starts the creation wizard: a draft character ingredient that candidates can be generated for."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["template", "prompt", "upload"] = "prompt"
    template_key: str | None = None
    fields: CharacterFields | None = None

    @model_validator(mode="after")
    def _template(self) -> DraftCreate:
        if self.source == "template" and not self.template_key:
            raise ValueError("source 'template' needs template_key")
        return self


class ExpandBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=3, max_length=4000)
    template_key: str | None = None
    llm: LlmChoice | None = None


class CharacterExpansion(BaseModel):
    """LLM output for `character_expand` (Section 8.5)."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    visual_prompt: str = Field(min_length=10)
    body_outfit_prompt: str = ""
    negative_prompt: str = ""
    character_info: str = ""
    voice: VoiceHints = Field(default_factory=VoiceHints)


class ExpandOut(BaseModel):
    name: str
    visual_prompt: str
    body_outfit_prompt: str
    negative_prompt: str
    character_info: str
    voice_hints: dict[str, str]
    usage: UsageOut


class DescribeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: list[str] = Field(min_length=1, max_length=4)
    name: str | None = Field(default=None, max_length=200)
    llm: LlmChoice | None = None


class CharacterDescription(BaseModel):
    """LLM output for `character_describe` (Section 8.5)."""

    model_config = ConfigDict(extra="ignore")

    visual_prompt: str = Field(min_length=10)
    body_outfit_prompt: str = ""


class DescribeOut(BaseModel):
    visual_prompt: str
    body_outfit_prompt: str
    usage: UsageOut


class CharacterCapabilities(BaseModel):
    expand: bool
    expand_reason: str | None = None
    describe: bool
    describe_model: str | None = None
    describe_reason: str | None = None
    edit_model: bool = Field(description="An image-edit model is installed (views, clean reference)")
    edit_model_name: str | None = None
    voices: bool = Field(description="A TTS adapter with voices is available (Section 12.2, M6)")


class ViewsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    views: list[ExtraView] = Field(default=["full_body", "three_view"], min_length=1)
    body_outfit_prompt: str | None = Field(default=None, description="Saved on the character when given")
    count: int = Field(default=2, ge=1, le=4, description="Candidates per view")
    model_entry_id: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class ViewsOut(BaseModel):
    jobs: list[JobOut]
    views: list[ExtraView]
    mode: Literal["edit", "t2i"]
    note: str | None = None


class CleanReferenceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    count: int = Field(default=1, ge=1, le=4)
    model_entry_id: str | None = None


class SetViewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str | None


# --- library -----------------------------------------------------------------------------------------
class LibraryImageOut(BaseModel):
    n: int
    view: str
    is_primary: bool
    url: str
    thumb_url: str


class LibraryCharacterOut(BaseModel):
    id: str
    key: str
    name: str
    description: str
    visual_prompt: str
    negative_prompt: str
    character_info: str
    body_outfit_prompt: str
    voice_id: str | None
    voice_hints: dict[str, str]
    lora_ids: list[str]
    template_key: str | None
    images: list[LibraryImageOut]
    source_project_id: str | None
    created_at: datetime
    updated_at: datetime


class LibraryCharacterUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    visual_prompt: str | None = None
    negative_prompt: str | None = None
    character_info: str | None = None
    body_outfit_prompt: str | None = None
    voice_hints: VoiceHintsIn | None = None
    voice_id: str | None = Field(default=None, max_length=64)


class ImportBody(BaseModel):
    """Cast a library character into the project, or copy a character from another project."""

    model_config = ConfigDict(extra="forbid")

    library_character_id: str | None = None
    ingredient_id: str | None = None

    @model_validator(mode="after")
    def _one(self) -> ImportBody:
        if bool(self.library_character_id) == bool(self.ingredient_id):
            raise ValueError("Give exactly one of library_character_id and ingredient_id")
        return self


class SaveToLibraryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    link: bool = Field(default=True, description="Link the ingredient to the new entry (sync status)")


# --- mentions & references ---------------------------------------------------------------------------
ResolveMode = Literal["keyframe", "t2i", "image_edit", "t2v", "ref2v", "i2v", "flf2v", "extend"]


class ResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str | None = Field(default=None, description="Omit (or the Playground) → the Cast library")
    text: str = Field(max_length=20000)
    mode: ResolveMode = "keyframe"


class MentionOut(BaseModel):
    key: str
    text: str
    start: int
    end: int
    known: bool
    name: str | None = None
    kind: str | None = None


class ResolveOut(BaseModel):
    resolved_text: str
    ingredient_ids: list[str]
    mentions: list[MentionOut]
    errors: list[str]


class RefOut(BaseModel):
    asset_id: str
    ingredient_id: str | None
    name: str
    kind: str
    view: str
    label: str


class RefOptionImage(BaseModel):
    asset_id: str
    view: str


class RefOption(BaseModel):
    ingredient_id: str
    name: str
    kind: str
    images: list[RefOptionImage]


class ReferencesOut(BaseModel):
    automatic: list[RefOut]
    selected: list[RefOut]
    overridden: bool
    max: int
    options: list[RefOption]
