"""Model registry endpoints (Section 16, "Models"). `/models/adapters` lives in the system router."""

from __future__ import annotations

from typing import Literal, get_args

from fastapi import APIRouter, Depends, Query, Request

from localflow.routers.cancel import run_cancellable
from localflow.schemas.models import (
    CIVITAI_SORTS,
    CIVITAI_TYPES,
    CivitaiInstallBody,
    CivitaiModelOut,
    CivitaiSearchOut,
    DeleteModelOut,
    HfInspectBody,
    HfInspectOut,
    HfInstallBody,
    ModelEntryOut,
    ModelInstallOut,
    ModelOptions,
    ModelSource,
    ModelStatus,
)
from localflow.services.container import Services, services

router = APIRouter(prefix="/api/models", tags=["models"])

CivitaiType = Literal["Checkpoint", "LORA", "LoCon", "DoRA", "VAE"]
CivitaiSort = Literal["Highest Rated", "Most Downloaded", "Newest"]
assert set(CIVITAI_TYPES) == set(get_args(CivitaiType)) and set(CIVITAI_SORTS) == set(get_args(CivitaiSort))


@router.get("", response_model=list[ModelEntryOut])
def list_models(
    status: ModelStatus | None = None,
    source: ModelSource | None = None,
    capability: str | None = None,
    catalog: bool | None = Query(default=None, description="true = catalog only, false = user-added only"),
    svc: Services = Depends(services),
) -> list[ModelEntryOut]:
    rows = svc.registry.list_entries(status=status, source=source, capability=capability, catalog=catalog)
    return [svc.registry.to_out(e) for e in rows]


@router.get("/options", response_model=ModelOptions)
def model_options(svc: Services = Depends(services)) -> ModelOptions:
    return svc.registry.options()


# --- Hugging Face --------------------------------------------------------------------------------
@router.post("/hf/inspect", response_model=HfInspectOut)
async def hf_inspect(
    body: HfInspectBody, request: Request, svc: Services = Depends(services)
) -> HfInspectOut:
    return await run_cancellable(request, svc.registry.hf_inspect(body.repo_id, body.revision))


@router.post("/hf/install", response_model=ModelInstallOut, status_code=201)
async def hf_install(body: HfInstallBody, svc: Services = Depends(services)) -> ModelInstallOut:
    entry, job = await svc.registry.hf_install(body)
    return ModelInstallOut(entry=svc.registry.to_out(entry), job_id=job.id)


# --- Civitai ---------------------------------------------------------------------------------------
@router.get("/civitai/search", response_model=CivitaiSearchOut)
async def civitai_search(
    request: Request,
    query: str | None = Query(default=None, max_length=200),
    types: list[CivitaiType] = Query(default_factory=list),
    base_models: list[str] = Query(default_factory=list, max_length=10),
    sort: CivitaiSort | None = None,
    cursor: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
    nsfw: bool = False,
    svc: Services = Depends(services),
) -> CivitaiSearchOut:
    return await run_cancellable(
        request,
        svc.registry.civitai_search(
            query=query,
            types=list(types),
            base_models=list(base_models),
            sort=sort,
            cursor=cursor,
            limit=limit,
            nsfw=nsfw,
        ),
    )


@router.get("/civitai/{model_id}", response_model=CivitaiModelOut)
async def civitai_model(
    model_id: int, request: Request, svc: Services = Depends(services)
) -> CivitaiModelOut:
    return await run_cancellable(request, svc.registry.civitai_model(model_id))


@router.post("/civitai/install", response_model=ModelInstallOut, status_code=201)
async def civitai_install(body: CivitaiInstallBody, svc: Services = Depends(services)) -> ModelInstallOut:
    entry, job = await svc.registry.civitai_install(body)
    return ModelInstallOut(entry=svc.registry.to_out(entry), job_id=job.id)


# --- entries ---------------------------------------------------------------------------------------
@router.get("/{entry_id}", response_model=ModelEntryOut)
def get_model(entry_id: str, svc: Services = Depends(services)) -> ModelEntryOut:
    return svc.registry.to_out(svc.registry.get(entry_id))


@router.delete("/{entry_id}", response_model=DeleteModelOut)
def delete_model(entry_id: str, svc: Services = Depends(services)) -> DeleteModelOut:
    return svc.registry.delete(entry_id)


@router.post("/{entry_id}/install", response_model=ModelInstallOut, status_code=201)
async def install_model(entry_id: str, svc: Services = Depends(services)) -> ModelInstallOut:
    entry, job = await svc.registry.install(entry_id)
    return ModelInstallOut(entry=svc.registry.to_out(entry), job_id=job.id)


@router.post("/{entry_id}/check-access", response_model=ModelEntryOut)
async def check_access(entry_id: str, svc: Services = Depends(services)) -> ModelEntryOut:
    return svc.registry.to_out(await svc.registry.check_access(entry_id))


@router.post("/{entry_id}/acknowledge-license", response_model=ModelEntryOut)
def acknowledge_license(entry_id: str, svc: Services = Depends(services)) -> ModelEntryOut:
    return svc.registry.to_out(svc.registry.acknowledge_license(entry_id))
