from __future__ import annotations

from fastapi import APIRouter, Depends

from localflow.schemas.core import AdapterOut, PlaygroundInfo, SystemStatus
from localflow.services.container import Services, services
from localflow.services.system import system_status

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/system/status", response_model=SystemStatus)
def get_status(svc: Services = Depends(services)) -> SystemStatus:
    return system_status(svc.settings, svc.supervisor.info())


@router.get("/models/adapters", response_model=list[AdapterOut], tags=["models"])
def list_adapters(svc: Services = Depends(services)) -> list[AdapterOut]:
    return [AdapterOut(**a) for a in (svc.supervisor.adapters or [])]


@router.get("/playground", response_model=PlaygroundInfo, tags=["generate"])
def playground(svc: Services = Depends(services)) -> PlaygroundInfo:
    return PlaygroundInfo(project_id=svc.projects.ensure_playground().id)
