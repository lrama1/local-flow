"""Narration (Sections 12.3, 16): the plan, LLM writing, per-shot voicing and re-sync."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from localflow.schemas.narration import (
    NarrationGenerateBody,
    NarrationGenerateOut,
    NarrationPlanOut,
    NarrationWriteBody,
    NarrationWriteOut,
)
from localflow.schemas.timeline import TimelineOut
from localflow.services.container import Services, services

router = APIRouter(prefix="/api", tags=["narration"])


@router.get("/projects/{project_id}/narration", response_model=NarrationPlanOut)
def narration_plan(
    project_id: str, speed: float = 1.0, svc: Services = Depends(services)
) -> NarrationPlanOut:
    """One row per shot: its line, voiced status, where it starts and whether it overruns the next line."""
    return svc.narration.plan(project_id, speed)


@router.post("/projects/{project_id}/narration/write", response_model=NarrationWriteOut)
async def write_narration(
    project_id: str, body: NarrationWriteBody | None = None, svc: Services = Depends(services)
) -> NarrationWriteOut:
    """The LLM writes a line per shot (only shots without narration unless `overwrite`)."""
    return await svc.narration.write(project_id, body or NarrationWriteBody())


@router.post(
    "/projects/{project_id}/narration/generate", response_model=NarrationGenerateOut, status_code=201
)
def generate_narration(
    project_id: str, body: NarrationGenerateBody | None = None, svc: Services = Depends(services)
) -> NarrationGenerateOut:
    """Voices the lines with the narrator voice: one TTS job per shot; each take lands on the timeline."""
    return svc.narration.generate(project_id, body or NarrationGenerateBody())


@router.post("/projects/{project_id}/narration/resync", response_model=TimelineOut)
def resync_narration(project_id: str, svc: Services = Depends(services)) -> TimelineOut:
    """Moves every narration clip back to its shot's clip start (plus the lead)."""
    return svc.timeline.resync_narration(project_id)
