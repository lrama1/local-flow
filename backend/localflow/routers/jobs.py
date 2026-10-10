from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from localflow.errors import not_found
from localflow.schemas.core import JobOut, JobReorder, JobStatus, Page
from localflow.services.container import Services, services

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=Page[JobOut])
def list_jobs(
    status: list[JobStatus] | None = Query(default=None),
    project_id: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    svc: Services = Depends(services),
) -> Page[JobOut]:
    rows, total = svc.jobs.list_jobs(
        statuses=list(status) if status else None, project_id=project_id, limit=limit, offset=offset
    )
    return Page[JobOut](items=[JobOut.of(j) for j in rows], total=total, offset=offset, limit=limit)


@router.post("/reorder", response_model=list[JobOut])
def reorder_jobs(body: JobReorder, svc: Services = Depends(services)) -> list[JobOut]:
    return [JobOut.of(j) for j in svc.jobs.reorder(body.job_ids)]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.jobs.get(job_id))


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.jobs.cancel(job_id))


@router.post("/{job_id}/retry", response_model=JobOut, status_code=201)
def retry_job(job_id: str, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.jobs.retry(job_id))


@router.get("/{job_id}/preview")
def job_preview(job_id: str, svc: Services = Depends(services)) -> FileResponse:
    path = svc.supervisor.preview_path(job_id)
    if path is None or not path.is_file():
        raise not_found("Preview", job_id)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
