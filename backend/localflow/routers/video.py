"""Shot video generation, extend, and generate-all (Sections 11.2-11.6, 16)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from localflow.services.container import Services, services
from localflow.services.videogen import (
    CAMERA_FRAGMENTS,
    ExtendShotBody,
    GenerateAllBody,
    GenerateAllOut,
    VideoBody,
    VideoOut,
    VideoPlanOut,
    camera_fragment,
)

router = APIRouter(prefix="/api", tags=["shots"])


@router.post("/shots/{shot_id}/generate-video", response_model=VideoOut, status_code=201)
def generate_video(shot_id: str, body: VideoBody, svc: Services = Depends(services)) -> VideoOut:
    """One job per variation. The first take is selected automatically; `extend_segments` extends it."""
    return svc.videogen.generate(shot_id, body)


@router.post("/shots/{shot_id}/video-plan", response_model=VideoPlanOut)
def video_plan(shot_id: str, body: VideoBody, svc: Services = Depends(services)) -> VideoPlanOut:
    """What `generate-video` would do (mode, model, size, frames, prompt), without queuing anything."""
    return svc.videogen.plan(shot_id, body)


@router.post("/shots/{shot_id}/extend", response_model=VideoOut, status_code=201)
def extend_shot(shot_id: str, body: ExtendShotBody, svc: Services = Depends(services)) -> VideoOut:
    return svc.videogen.extend(shot_id, body)


@router.post("/projects/{project_id}/generate-all", response_model=GenerateAllOut, status_code=201)
def generate_all(project_id: str, body: GenerateAllBody, svc: Services = Depends(services)) -> GenerateAllOut:
    return svc.videogen.generate_all(project_id, body)


@router.get("/camera-moves", response_model=dict[str, dict[str, str]])
def camera_moves() -> dict[str, dict[str, str]]:
    """Prompt fragment per camera move and intensity (Section 11.5), for tooltips."""
    return {m: {i: camera_fragment(m, i) for i in ("subtle", "normal", "strong")} for m in CAMERA_FRAGMENTS}
