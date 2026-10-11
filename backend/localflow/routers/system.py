from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response

from localflow.schemas.core import AdapterOut, PlaygroundInfo, SystemStatus
from localflow.schemas.diagnostics import DiagnosticsOut, LogTail
from localflow.services import diagnostics as diag
from localflow.services.container import Services, services
from localflow.services.system import system_status

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/system/diagnostics", response_model=DiagnosticsOut)
def get_diagnostics(refresh: bool = False, svc: Services = Depends(services)) -> DiagnosticsOut:
    """Checks (data-dir write test, ffmpeg, GPU, CUDA, worker, secret storage), versions, storage, logs.
    Folder sizes are cached for a minute; `refresh` recounts."""
    return diag.diagnostics(svc, refresh)


@router.get("/system/logs/{name}", response_model=LogTail)
def get_log(
    name: str, lines: int = Query(default=200, ge=1, le=5000), svc: Services = Depends(services)
) -> LogTail:
    return diag.log_tail(svc.settings.logs_dir, name, lines)


@router.get("/system/diagnostics/bundle", response_class=Response)
def diagnostics_bundle(svc: Services = Depends(services)) -> Response:
    """A redacted zip (diagnostics, status, config.toml, recent logs) to attach to a bug report."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Response(
        diag.bundle(svc),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="localflow-diagnostics-{stamp}.zip"'},
    )


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
