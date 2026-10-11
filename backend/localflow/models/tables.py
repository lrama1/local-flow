"""SQLModel tables (Section 6.1). Later milestones add tables via new Alembic revisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, UniqueConstraint
from sqlmodel import Field

from localflow.models.base import IdMixin, TimestampMixin, UTCDateTime, utcnow


class Project(IdMixin, TimestampMixin, table=True):
    __tablename__ = "projects"

    name: str
    description: str = ""
    aspect_ratio: str = "16:9"
    fps: int = 24
    default_video_model_id: str | None = None
    default_image_model_id: str | None = None
    draft_mode: bool = False
    narrator_voice_id: str | None = None
    style_notes: str = ""
    archived: bool = False


class Asset(IdMixin, TimestampMixin, table=True):
    __tablename__ = "assets"

    project_id: str = Field(foreign_key="projects.id", index=True, ondelete="CASCADE")
    type: str = Field(index=True)  # image | video | audio
    path: str  # relative to the project dir
    thumb_path: str | None = None  # relative to the project dir
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    has_audio: bool = False
    source: str = "generated"  # generated | uploaded | extracted | export | imported
    parent_asset_ids_json: list[str] = Field(default_factory=list, sa_type=JSON)
    generation_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    tags_json: list[str] = Field(default_factory=list, sa_type=JSON)
    favorite: bool = False


class Collection(IdMixin, TimestampMixin, table=True):
    """A user-defined group of a project's assets (Section 13)."""

    __tablename__ = "collections"

    project_id: str = Field(foreign_key="projects.id", index=True, ondelete="CASCADE")
    name: str
    description: str = ""


class CollectionAsset(IdMixin, TimestampMixin, table=True):
    __tablename__ = "collection_assets"
    __table_args__ = (UniqueConstraint("collection_id", "asset_id"),)

    collection_id: str = Field(foreign_key="collections.id", index=True, ondelete="CASCADE")
    asset_id: str = Field(foreign_key="assets.id", index=True, ondelete="CASCADE")


class Job(IdMixin, TimestampMixin, table=True):
    __tablename__ = "jobs"

    project_id: str | None = Field(default=None, index=True)
    type: str = Field(index=True)
    status: str = Field(default="queued", index=True)  # queued|running|completed|failed|cancelled
    priority: int = 0
    params_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    result_json: dict[str, Any] | None = Field(default=None, sa_type=JSON)
    error: str | None = None
    progress: float = 0.0
    progress_message: str | None = None
    started_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    finished_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    target_ref_json: dict[str, Any] | None = Field(default=None, sa_type=JSON)


class Story(IdMixin, TimestampMixin, table=True):
    """One row per revision; the highest `revision` per project is the active story (Section 6.1)."""

    __tablename__ = "stories"
    __table_args__ = (UniqueConstraint("project_id", "revision"),)

    project_id: str = Field(foreign_key="projects.id", index=True, ondelete="CASCADE")
    revision: int
    step: str  # refine | bible | breakdown | edit | restore
    status: str = "draft"  # draft | refined | broken_down
    raw_idea: str = ""
    options_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    cast_keys_json: list[str] = Field(default_factory=list, sa_type=JSON)
    refined_story: str = ""  # prose story (RefinedStory.story), denormalized for search/display
    refined_json: dict[str, Any] | None = Field(default=None, sa_type=JSON)  # RefinedStory
    bible_json: dict[str, Any] | None = Field(default=None, sa_type=JSON)  # StoryBible
    breakdown_json: dict[str, Any] | None = Field(default=None, sa_type=JSON)  # Breakdown as applied
    llm_provider: str | None = None
    llm_model: str | None = None
    template_versions_json: dict[str, str] = Field(default_factory=dict, sa_type=JSON)
    edited_fields_json: list[str] = Field(
        default_factory=list, sa_type=JSON
    )  # manual edits since last LLM run
    warnings_json: list[str] = Field(default_factory=list, sa_type=JSON)
    usage_json: dict[str, Any] | None = Field(default=None, sa_type=JSON)


class Ingredient(IdMixin, TimestampMixin, table=True):
    __tablename__ = "ingredients"
    __table_args__ = (UniqueConstraint("project_id", "key"),)

    project_id: str = Field(foreign_key="projects.id", index=True, ondelete="CASCADE")
    kind: str  # character | location | prop | style
    key: str
    name: str
    description: str = ""
    visual_prompt: str = ""
    negative_prompt: str = ""
    voice_id: str | None = None
    primary_asset_id: str | None = None
    lora_ids_json: list[str] = Field(default_factory=list, sa_type=JSON)
    locked: bool = False
    origin: str = "bible"  # bible | user | library | import
    character_info: str | None = None
    body_outfit_prompt: str | None = None
    template_key: str | None = None
    library_character_id: str | None = None
    library_synced_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    # M3.5: a character being built in the wizard. Drafts are hidden from the story pipeline, keyframes and
    # mentions, and their `key` stays editable until they are saved.
    draft: bool = Field(default=False, sa_column_kwargs={"server_default": "0"})
    voice_hints_json: dict[str, str] = Field(
        default_factory=dict, sa_type=JSON, sa_column_kwargs={"server_default": "{}"}
    )


class IngredientAsset(IdMixin, TimestampMixin, table=True):
    """The chosen image for each extra view of an ingredient (Section 11.7). The primary image stays on
    `ingredients.primary_asset_id`; candidates are plain assets whose target_ref names the view."""

    __tablename__ = "ingredient_assets"
    __table_args__ = (UniqueConstraint("ingredient_id", "view"),)

    ingredient_id: str = Field(foreign_key="ingredients.id", index=True, ondelete="CASCADE")
    asset_id: str = Field(foreign_key="assets.id", index=True, ondelete="CASCADE")
    view: str  # full_body | three_view | side | back | face_closeup


class CharacterTemplate(IdMixin, TimestampMixin, table=True):
    """User-saved character templates; the built-ins come from `services/characters/templates.yaml`."""

    __tablename__ = "character_templates"

    key: str = Field(unique=True, index=True)
    name: str
    category: str = "custom"
    visual_prompt: str = ""
    negative_prompt: str = ""
    character_info: str = ""
    body_outfit_prompt: str = ""
    voice_hints_json: dict[str, str] = Field(default_factory=dict, sa_type=JSON)
    thumbnail_path: str | None = None  # relative to the data dir


class LibraryCharacter(IdMixin, TimestampMixin, table=True):
    """Global Cast library entry (Section 11.7.2). Images live in `library/characters/<id>/`."""

    __tablename__ = "library_characters"

    key: str = Field(unique=True, index=True)
    name: str
    description: str = ""
    visual_prompt: str = ""
    negative_prompt: str = ""
    character_info: str = ""
    body_outfit_prompt: str = ""
    voice_id: str | None = None
    voice_hints_json: dict[str, str] = Field(default_factory=dict, sa_type=JSON)
    lora_ids_json: list[str] = Field(default_factory=list, sa_type=JSON)
    template_key: str | None = None
    images_json: list[dict[str, Any]] = Field(default_factory=list, sa_type=JSON)  # {file, view, is_primary}
    source_project_id: str | None = None


class Scene(IdMixin, TimestampMixin, table=True):
    __tablename__ = "scenes"

    project_id: str = Field(foreign_key="projects.id", index=True, ondelete="CASCADE")
    story_id: str | None = None
    index: int
    title: str = ""
    summary: str = ""
    location_ingredient_id: str | None = None


class Shot(IdMixin, TimestampMixin, table=True):
    __tablename__ = "shots"

    scene_id: str = Field(foreign_key="scenes.id", index=True, ondelete="CASCADE")
    index: int
    duration_s: float = 4.0
    shot_type: str = "medium"
    camera_move: str = "static"
    video_prompt: str = ""
    negative_prompt: str = ""
    keyframe_prompt: str = ""
    end_keyframe_prompt: str | None = None
    generation_mode: str = "i2v"
    ingredient_ids_json: list[str] = Field(default_factory=list, sa_type=JSON)
    dialogue_json: list[dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    narration: str = ""  # M8: the film's voice-over for this shot (Section 12.3)
    sfx_prompt: str | None = None
    music_cue: str | None = None
    needs_extend: bool = False
    extend_segments: int | None = None
    continuity_notes: str = ""
    status: str = "planned"  # planned | keyframe_ready | video_ready | approved
    selected_keyframe_asset_id: str | None = None
    selected_end_keyframe_asset_id: str | None = None
    selected_video_asset_id: str | None = None
    selected_audio_asset_ids_json: list[str] = Field(default_factory=list, sa_type=JSON)
    # M3.5: user-chosen keyframe references; None = automatic selection by shot type (Section 11.7).
    reference_asset_ids_json: list[str] | None = Field(default=None, sa_type=JSON)


class Timeline(IdMixin, TimestampMixin, table=True):
    """One per project (Section 15.1). `tracks_json` holds the whole timeline: fps, size and tracks."""

    __tablename__ = "timelines"

    project_id: str = Field(foreign_key="projects.id", unique=True, index=True, ondelete="CASCADE")
    tracks_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)


class Voice(IdMixin, TimestampMixin, table=True):
    """TTS voices (Section 12.2). Built-ins are synced from the installed TTS adapters; custom voices are a
    built-in base voice plus adapter params (e.g. a Kokoro blend). Global, like the Cast library."""

    __tablename__ = "voices"
    __table_args__ = (UniqueConstraint("adapter_key", "kind", "native_id"),)

    kind: str = Field(index=True)  # builtin | custom
    adapter_key: str = Field(index=True)
    native_id: str | None = None  # builtin: the adapter's voice name (e.g. "af_heart"); custom: None
    base_voice_id: str | None = None  # custom: the builtin `voices.id` it starts from
    name: str
    description: str = ""
    language: str = ""  # BCP-47-ish, e.g. "en-US"
    gender: str = ""  # female | male | ""
    params_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    meta_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)  # builtin: quality grade, traits
    preview_path: str | None = None  # relative to the data dir: library/voices/<id>/preview.wav
    preview_text: str | None = None


class LlmUsage(IdMixin, table=True):
    __tablename__ = "llm_usage"

    provider: str
    model: str
    purpose: str = Field(index=True)
    project_id: str | None = Field(default=None, index=True)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    attempts: int = 1
    ok: bool = True
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=utcnow, sa_type=UTCDateTime, nullable=False, index=True)


class ModelEntry(IdMixin, TimestampMixin, table=True):
    """Registry of catalog, Hugging Face and Civitai models (Section 9.1).

    `files_json` lists download parts; it never contains tokens or signed URLs (Section 9.6)."""

    __tablename__ = "model_entries"

    catalog_key: str | None = Field(default=None, unique=True, index=True)
    source: str = Field(index=True)  # hf | civitai | local
    source_ref: str = Field(index=True)  # repo id or "<modelId>:<versionId>"
    name: str
    description: str = ""
    kind: str = "checkpoint"  # checkpoint | lora | vae | text_encoder | audio
    capabilities_json: list[str] = Field(default_factory=list, sa_type=JSON)
    adapter_key: str | None = None
    base_model: str | None = None
    license: str | None = None
    license_url: str | None = None
    nsfw: bool = False
    files_json: list[dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    local_path: str | None = None  # relative to the data dir
    size_bytes: int | None = None  # on-disk size when installed, else the expected download size
    status: str = Field(default="available", index=True)  # available | downloading | installed | error
    error: str | None = None
    default_params_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    vram_profiles_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    gated: bool = False
    access: str | None = None  # Section 9.3 access state, None = not checked
    access_message: str | None = None
    access_checked_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    trigger_words_json: list[str] = Field(default_factory=list, sa_type=JSON)
    preview_url: str | None = None
    requires_pickle: bool = False
    experimental: bool = False
    notes: str = ""
    meta_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)  # source-specific extras
    license_acknowledged_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    installed_at: datetime | None = Field(default=None, sa_type=UTCDateTime)


class Setting(TimestampMixin, table=True):
    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value_json: Any = Field(default=None, sa_type=JSON, nullable=True)
