"""Timeline & export (Sections 15, 16)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from localflow.schemas.core import JobOut
from localflow.schemas.timeline import (
    AutoAssembleBody,
    AutoAssembleOut,
    ExportBody,
    TimelineData,
    TimelineOut,
)
from localflow.services.container import Services, services

router = APIRouter(prefix="/api", tags=["timeline"])


@router.get("/projects/{project_id}/timeline", response_model=TimelineOut)
def get_timeline(project_id: str, svc: Services = Depends(services)) -> TimelineOut:
    """The normalized timeline (an empty default before the first save or auto-assemble)."""
    return svc.timeline.get(project_id)


@router.put("/projects/{project_id}/timeline", response_model=TimelineOut)
def put_timeline(project_id: str, body: TimelineData, svc: Services = Depends(services)) -> TimelineOut:
    """Replaces the timeline. Video clip starts are recomputed from their order, trims and crossfades;
    `notes` says what else was adjusted."""
    return svc.timeline.put(project_id, body)


@router.post("/projects/{project_id}/timeline/auto-assemble", response_model=AutoAssembleOut)
def auto_assemble(
    project_id: str, body: AutoAssembleBody | None = None, svc: Services = Depends(services)
) -> AutoAssembleOut:
    return svc.timeline.auto_assemble(project_id, body or AutoAssembleBody())


@router.post("/projects/{project_id}/export", response_model=JobOut, status_code=201)
def export(project_id: str, body: ExportBody | None = None, svc: Services = Depends(services)) -> JobOut:
    """Queues an export job; the MP4 becomes an asset with `source="export"`."""
    return JobOut.of(svc.exports.submit(project_id, body or ExportBody()))
