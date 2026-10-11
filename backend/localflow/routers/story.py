from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from localflow.routers.cancel import run_cancellable
from localflow.schemas.review import ReviewBody, ShotReviewOut
from localflow.schemas.story import (
    BibleBody,
    BreakdownBody,
    RefineBody,
    RestoreBody,
    RevisionSummary,
    RewriteBody,
    ShotRewriteResult,
    StoryOut,
    StoryPatch,
    StoryStepResult,
)
from localflow.services.container import Services, services

router = APIRouter(prefix="/api", tags=["story"])


@router.get("/projects/{project_id}/story", response_model=StoryOut)
def get_story(project_id: str, svc: Services = Depends(services)) -> StoryOut:
    svc.projects.get(project_id)
    return svc.story.to_out(svc.story.latest(project_id), project_id)


@router.get("/projects/{project_id}/story/revisions", response_model=list[RevisionSummary])
def list_revisions(project_id: str, svc: Services = Depends(services)) -> list[RevisionSummary]:
    return svc.story.revisions(project_id)


@router.get("/projects/{project_id}/story/revisions/{revision}", response_model=StoryOut)
def get_revision(project_id: str, revision: int, svc: Services = Depends(services)) -> StoryOut:
    return svc.story.to_out(svc.story.get_revision(project_id, revision), project_id)


@router.post("/projects/{project_id}/story/refine", response_model=StoryStepResult)
async def refine(
    project_id: str, body: RefineBody, request: Request, svc: Services = Depends(services)
) -> StoryStepResult:
    return await run_cancellable(request, svc.story.refine(project_id, body))


@router.post("/projects/{project_id}/story/bible", response_model=StoryStepResult)
async def bible(
    project_id: str, request: Request, body: BibleBody | None = None, svc: Services = Depends(services)
) -> StoryStepResult:
    return await run_cancellable(request, svc.story.bible(project_id, body or BibleBody()))


@router.post("/projects/{project_id}/story/breakdown", response_model=StoryStepResult)
async def breakdown(
    project_id: str, body: BreakdownBody, request: Request, svc: Services = Depends(services)
) -> StoryStepResult:
    return await run_cancellable(request, svc.story.breakdown(project_id, body))


@router.patch("/projects/{project_id}/story", response_model=StoryStepResult)
def patch_story(project_id: str, body: StoryPatch, svc: Services = Depends(services)) -> StoryStepResult:
    return svc.story.patch(project_id, body)


@router.post("/projects/{project_id}/story/restore", response_model=StoryOut)
def restore_revision(project_id: str, body: RestoreBody, svc: Services = Depends(services)) -> StoryOut:
    return svc.story.restore(project_id, body.revision)


@router.post("/shots/{shot_id}/rewrite-prompt", response_model=ShotRewriteResult, tags=["shots"])
async def rewrite_prompt(
    shot_id: str, body: RewriteBody, request: Request, svc: Services = Depends(services)
) -> ShotRewriteResult:
    return await run_cancellable(request, svc.story.rewrite_shot(shot_id, body))


@router.post("/shots/{shot_id}/review", response_model=ShotReviewOut, tags=["shots"])
async def review_shot(
    shot_id: str, request: Request, body: ReviewBody | None = None, svc: Services = Depends(services)
) -> ShotReviewOut:
    """Vision review: does the keyframe (default) or `asset_id` match the shot's reference images?"""
    return await run_cancellable(request, svc.review.review_shot(shot_id, body or ReviewBody()))
