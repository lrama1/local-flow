"""Voices and audio generation (Sections 12, 16): built-in/custom voices, voice design, previews, per-shot
dialogue and SFX, and music."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from localflow.schemas.voices import (
    AuditionBody,
    PreviewBody,
    PreviewOut,
    VoiceCreate,
    VoiceDesignBody,
    VoiceDesignOut,
    VoiceOut,
    VoiceSuggestion,
    VoiceUpdate,
)
from localflow.services.audiogen import (
    AudioPlanOut,
    AudioSlot,
    MusicGenBody,
    MusicGenOut,
    ShotAudioBody,
    ShotAudioOut,
    ShotAudioParams,
)
from localflow.services.container import Services, services

router = APIRouter(prefix="/api")


# --- voices (Section 12.2) ---------------------------------------------------------------------------
@router.get("/voices", response_model=list[VoiceOut], tags=["voices"])
def list_voices(
    adapter_key: str | None = None,
    available: bool = Query(default=False, description="Only voices whose TTS model is installed"),
    svc: Services = Depends(services),
) -> list[VoiceOut]:
    """Custom voices first, then built-ins by model, language and quality."""
    return svc.voices.list_voices(adapter_key, available_only=available)


@router.post("/voices", response_model=VoiceOut, status_code=201, tags=["voices"])
def create_voice(body: VoiceCreate, svc: Services = Depends(services)) -> VoiceOut:
    return svc.voices.create(body)


@router.get("/voices/suggest", response_model=VoiceSuggestion, tags=["voices"])
def suggest_voice(
    gender: str = "",
    age: str = "",
    tone: str = "",
    language: str = "en",
    svc: Services = Depends(services),
) -> VoiceSuggestion:
    """The best available built-in voice for a character's voice hints."""
    return svc.voices.suggest({"gender": gender, "age": age, "tone": tone}, language=language)


@router.post("/voices/design", response_model=VoiceDesignOut, tags=["voices"])
async def design_voice(body: VoiceDesignBody, svc: Services = Depends(services)) -> VoiceDesignOut:
    """Suggested custom-voice params for a text description (LLM purpose `voice_design`)."""
    return await svc.voices.design(body)


@router.post("/voices/audition", response_model=PreviewOut, status_code=201, tags=["voices"])
def audition_voice(body: AuditionBody, svc: Services = Depends(services)) -> PreviewOut:
    """Speaks a sample line with unsaved settings; play the job's audio asset when it completes."""
    return svc.voices.audition(body)


@router.get("/voices/{voice_id}", response_model=VoiceOut, tags=["voices"])
def get_voice(voice_id: str, svc: Services = Depends(services)) -> VoiceOut:
    return svc.voices.get(voice_id)


@router.patch("/voices/{voice_id}", response_model=VoiceOut, tags=["voices"])
def update_voice(voice_id: str, body: VoiceUpdate, svc: Services = Depends(services)) -> VoiceOut:
    """Custom voices only. Changing the sound drops the cached preview."""
    return svc.voices.update(voice_id, body)


@router.delete("/voices/{voice_id}", status_code=204, tags=["voices"])
def delete_voice(voice_id: str, svc: Services = Depends(services)) -> Response:
    """Custom voices only; characters using it go back to "no voice"."""
    svc.voices.delete(voice_id)
    return Response(status_code=204)


@router.post("/voices/{voice_id}/preview", response_model=PreviewOut, tags=["voices"])
def preview_voice(voice_id: str, body: PreviewBody, svc: Services = Depends(services)) -> PreviewOut:
    """Returns the cached sample's URL, or queues a TTS job that caches it (`voices.updated` when done)."""
    return svc.voices.preview(voice_id, body)


@router.get("/voices/{voice_id}/preview.wav", tags=["voices"], response_class=FileResponse)
def voice_preview_file(voice_id: str, svc: Services = Depends(services)) -> FileResponse:
    return FileResponse(
        svc.voices.preview_file(voice_id),
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=86400"},
    )


# --- shot audio (Section 12.1) -------------------------------------------------------------------------
@router.post("/shots/{shot_id}/audio-plan", response_model=AudioPlanOut, tags=["shots"])
def audio_plan(shot_id: str, body: ShotAudioParams, svc: Services = Depends(services)) -> AudioPlanOut:
    """Who says what with which voice, whether it fits the clip, and the SFX prompt, without queuing."""
    return svc.audiogen.plan(shot_id, body)


@router.post("/shots/{shot_id}/generate-audio", response_model=ShotAudioOut, status_code=201, tags=["shots"])
def generate_audio(shot_id: str, body: ShotAudioBody, svc: Services = Depends(services)) -> ShotAudioOut:
    """Dialogue (one clip-length TTS track) and/or SFX (video-to-audio on the selected take). Each result
    becomes the shot's selected audio for its part."""
    return svc.audiogen.generate(shot_id, body)


class SelectAudioBody(BaseModel):
    asset_id: str
    slot: AudioSlot | None = Field(default=None, description="For uploaded audio: dialogue or sfx")


class SelectAudioOut(BaseModel):
    selected_audio_asset_ids: list[str]


@router.post("/shots/{shot_id}/select-audio", response_model=SelectAudioOut, tags=["shots"])
def select_audio(shot_id: str, body: SelectAudioBody, svc: Services = Depends(services)) -> SelectAudioOut:
    """Makes an audio candidate the shot's dialogue or SFX (replacing the previous one of that part)."""
    return SelectAudioOut(selected_audio_asset_ids=svc.audiogen.select(shot_id, body.asset_id, body.slot))


@router.post(
    "/projects/{project_id}/generate-music", response_model=MusicGenOut, status_code=201, tags=["projects"]
)
def generate_music(project_id: str, body: MusicGenBody, svc: Services = Depends(services)) -> MusicGenOut:
    """Music for the film or one scene from the shots' music cues; placed on the Music track when done."""
    return svc.audiogen.generate_music(project_id, body)
