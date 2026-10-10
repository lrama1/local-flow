from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from localflow.schemas.core import ProjectCreate, ProjectOut, ProjectUpdate
from localflow.services.container import Services, services

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _out(svc: Services, project_id: str) -> ProjectOut:
    p = svc.projects.get(project_id)
    cover = svc.assets.latest_thumb(p.id)
    return ProjectOut.of(p, f"/api/thumbs/{p.id}/{cover.id}" if cover else None)


@router.get("", response_model=list[ProjectOut])
def list_projects(include_archived: bool = False, svc: Services = Depends(services)) -> list[ProjectOut]:
    out = []
    for p in svc.projects.list_projects(include_archived):
        cover = svc.assets.latest_thumb(p.id)
        out.append(ProjectOut.of(p, f"/api/thumbs/{p.id}/{cover.id}" if cover else None))
    return out


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, svc: Services = Depends(services)) -> ProjectOut:
    return ProjectOut.of(svc.projects.create(body))


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, svc: Services = Depends(services)) -> ProjectOut:
    return _out(svc, project_id)


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(project_id: str, body: ProjectUpdate, svc: Services = Depends(services)) -> ProjectOut:
    svc.projects.update(project_id, body)
    return _out(svc, project_id)


@router.delete("/{project_id}", status_code=204, response_class=Response)
def delete_project(project_id: str, svc: Services = Depends(services)) -> Response:
    svc.projects.delete(project_id)
    svc.bus.publish("project.deleted", project_id=project_id)
    return Response(status_code=204)
