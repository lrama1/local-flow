"""Ingredients, scenes, and shots API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from localflow.models import Ingredient, Scene, Shot
from localflow.schemas.story import CameraMove, GenerationMode, IngredientKind, ShotStatus, ShotType

# Section 11.7: reference views of a character. "primary" is `ingredients.primary_asset_id`.
CharacterView = Literal["primary", "full_body", "three_view", "side", "back", "face_closeup"]
ExtraView = Literal["full_body", "three_view", "side", "back", "face_closeup"]
EXTRA_VIEWS: tuple[ExtraView, ...] = ("full_body", "three_view", "side", "back", "face_closeup")
LibraryStatus = Literal["synced", "update_available", "missing"]


class VoiceHintsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str = Field(default="", max_length=40)
    age: str = Field(default="", max_length=40)
    tone: str = Field(default="", max_length=200)


class IngredientOut(BaseModel):
    id: str
    project_id: str
    kind: IngredientKind
    key: str
    name: str
    description: str
    visual_prompt: str
    negative_prompt: str
    voice_id: str | None
    primary_asset_id: str | None
    lora_ids: list[str]
    locked: bool
    origin: str
    character_info: str | None
    body_outfit_prompt: str | None = None
    template_key: str | None = None
    voice_hints: dict[str, str] = Field(default_factory=dict)
    draft: bool = False
    views: dict[str, str] = Field(
        default_factory=dict, description="view -> chosen image asset id (includes 'primary' when set)"
    )
    library_character_id: str | None = None
    library_synced_at: datetime | None = None
    library_status: LibraryStatus | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(
        cls,
        i: Ingredient,
        views: dict[str, str] | None = None,
        library_status: LibraryStatus | None = None,
    ) -> IngredientOut:
        all_views = dict(views or {})
        if i.primary_asset_id:
            all_views = {"primary": i.primary_asset_id, **all_views}
        return cls(
            id=i.id,
            project_id=i.project_id,
            kind=i.kind,
            key=i.key,
            name=i.name,
            description=i.description,
            visual_prompt=i.visual_prompt,
            negative_prompt=i.negative_prompt,
            voice_id=i.voice_id,
            primary_asset_id=i.primary_asset_id,
            lora_ids=i.lora_ids_json,
            locked=i.locked,
            origin=i.origin,
            character_info=i.character_info,
            body_outfit_prompt=i.body_outfit_prompt,
            template_key=i.template_key,
            voice_hints=dict(i.voice_hints_json or {}),
            draft=i.draft,
            views=all_views,
            library_character_id=i.library_character_id,
            library_synced_at=i.library_synced_at,
            library_status=library_status,
            created_at=i.created_at,
            updated_at=i.updated_at,
        )


class IngredientCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: IngredientKind
    name: str = Field(min_length=1, max_length=200)
    key: str | None = Field(default=None, max_length=64, description="Defaults to a slug of the name")
    description: str = ""
    visual_prompt: str = ""
    negative_prompt: str = ""
    character_info: str | None = None
    body_outfit_prompt: str | None = None
    voice_hints: VoiceHintsIn | None = None
    locked: bool = True


class IngredientUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    key: str | None = Field(
        default=None, min_length=1, max_length=64, description="Only while the character is a draft"
    )
    draft: Literal[False] | None = Field(default=None, description="false saves a wizard draft")
    description: str | None = None
    visual_prompt: str | None = None
    negative_prompt: str | None = None
    character_info: str | None = None
    body_outfit_prompt: str | None = None
    template_key: str | None = None
    voice_hints: VoiceHintsIn | None = None
    voice_id: str | None = Field(default=None, max_length=64)
    locked: bool | None = None
    primary_asset_id: str | None = Field(
        default=None, description="An image asset of this project; null clears"
    )


class DialogueLine(BaseModel):
    character_ingredient_id: str | None
    line: str
    emotion: str = ""


class ShotOut(BaseModel):
    id: str
    scene_id: str
    index: int
    duration_s: float
    shot_type: ShotType
    camera_move: CameraMove
    video_prompt: str
    negative_prompt: str
    keyframe_prompt: str
    end_keyframe_prompt: str | None
    generation_mode: GenerationMode
    ingredient_ids: list[str]
    dialogue: list[DialogueLine]
    narration: str
    sfx_prompt: str | None
    music_cue: str | None
    needs_extend: bool
    extend_segments: int | None
    continuity_notes: str
    status: ShotStatus
    selected_keyframe_asset_id: str | None
    selected_end_keyframe_asset_id: str | None
    selected_video_asset_id: str | None
    selected_audio_asset_ids: list[str]
    reference_asset_ids: list[str] | None = Field(
        default=None, description="User-chosen keyframe references; null = automatic by shot type"
    )
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, s: Shot) -> ShotOut:
        return cls(
            id=s.id,
            scene_id=s.scene_id,
            index=s.index,
            duration_s=s.duration_s,
            shot_type=s.shot_type,
            camera_move=s.camera_move,
            video_prompt=s.video_prompt,
            negative_prompt=s.negative_prompt,
            keyframe_prompt=s.keyframe_prompt,
            end_keyframe_prompt=s.end_keyframe_prompt,
            generation_mode=s.generation_mode,
            ingredient_ids=s.ingredient_ids_json,
            dialogue=[DialogueLine(**d) for d in s.dialogue_json],
            narration=s.narration or "",
            sfx_prompt=s.sfx_prompt,
            music_cue=s.music_cue,
            needs_extend=s.needs_extend,
            extend_segments=s.extend_segments,
            continuity_notes=s.continuity_notes,
            status=s.status,
            selected_keyframe_asset_id=s.selected_keyframe_asset_id,
            selected_end_keyframe_asset_id=s.selected_end_keyframe_asset_id,
            selected_video_asset_id=s.selected_video_asset_id,
            selected_audio_asset_ids=s.selected_audio_asset_ids_json,
            reference_asset_ids=s.reference_asset_ids_json,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )


class SceneOut(BaseModel):
    id: str
    project_id: str
    story_id: str | None
    index: int
    title: str
    summary: str
    location_ingredient_id: str | None
    shots: list[ShotOut]

    @classmethod
    def of(cls, sc: Scene, shots: list[Shot]) -> SceneOut:
        return cls(
            id=sc.id,
            project_id=sc.project_id,
            story_id=sc.story_id,
            index=sc.index,
            title=sc.title,
            summary=sc.summary,
            location_ingredient_id=sc.location_ingredient_id,
            shots=[ShotOut.of(s) for s in sorted(shots, key=lambda x: x.index)],
        )


class SceneCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="New scene", min_length=1)
    summary: str = ""
    location_ingredient_id: str | None = None
    index: int | None = Field(default=None, ge=0, description="Insert position; defaults to the end")


class SceneUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1)
    summary: str | None = None
    location_ingredient_id: str | None = None


class ShotFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_s: float | None = Field(default=None, gt=0, le=600)
    shot_type: ShotType | None = None
    camera_move: CameraMove | None = None
    video_prompt: str | None = None
    negative_prompt: str | None = None
    keyframe_prompt: str | None = None
    end_keyframe_prompt: str | None = None
    generation_mode: GenerationMode | None = None
    ingredient_ids: list[str] | None = None
    dialogue: list[DialogueLine] | None = None
    narration: str | None = Field(default=None, max_length=2000, description="Voice-over for this shot")
    sfx_prompt: str | None = None
    music_cue: str | None = None
    needs_extend: bool | None = None
    extend_segments: int | None = Field(default=None, ge=1, le=20)
    continuity_notes: str | None = None
    status: ShotStatus | None = None
    selected_keyframe_asset_id: str | None = None
    selected_end_keyframe_asset_id: str | None = None
    selected_video_asset_id: str | None = None
    selected_audio_asset_ids: list[str] | None = Field(
        default=None, max_length=8, description="Audio of this project placed with the shot (dialogue, SFX)"
    )
    reference_asset_ids: list[str] | None = Field(
        default=None, max_length=3, description="Keyframe reference override; null restores automatic"
    )


class ShotCreate(ShotFields):
    index: int | None = Field(default=None, ge=0, description="Insert position; defaults to the end")


class ShotUpdate(ShotFields):
    pass


class Reorder(BaseModel):
    ids: list[str] = Field(min_length=1)


class BoardOut(BaseModel):
    scenes: list[SceneOut]
    total_duration_s: float
    shot_count: int
    max_clip_seconds: float
    extra: dict[str, Any] = Field(default_factory=dict)
