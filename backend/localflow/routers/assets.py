from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse

from localflow.errors import AppError, not_found
from localflow.schemas.core import AssetOut, AssetType, AssetUpdate, Page
from localflow.services.container import Services, services

router = APIRouter(prefix="/api", tags=["assets"])

_IMMUTABLE = {"Cache-Control": "private, max-age=31536000, immutable"}


@router.get("/projects/{project_id}/assets", response_model=Page[AssetOut])
def list_assets(
    project_id: str,
    type: AssetType | None = None,
    favorite: bool | None = None,
    tag: str | None = None,
    q: str | None = None,
    ingredient_id: str | None = Query(default=None, description="Candidates generated for this ingredient"),
    view: str | None = Query(
        default=None, description="With ingredient_id: primary | full_body | three_view | side | back | …"
    ),
    shot_id: str | None = Query(default=None, description="Candidates generated for this shot"),
    slot: str | None = Query(default=None, description="With shot_id: keyframe | end_keyframe | video"),
    source: str | None = Query(
        default=None, description="generated | uploaded | extracted | export | imported"
    ),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    svc: Services = Depends(services),
) -> Page[AssetOut]:
    svc.projects.get(project_id)
    target = {
        k: v
        for k, v in (("ingredient_id", ingredient_id), ("view", view), ("shot_id", shot_id), ("slot", slot))
        if v
    }
    rows, total = svc.assets.list_assets(
        project_id,
        type_=type,
        favorite=favorite,
        tag=tag,
        q=q,
        target=target or None,
        source=source,
        offset=offset,
        limit=limit,
    )
    return Page[AssetOut](items=[AssetOut.of(a) for a in rows], total=total, offset=offset, limit=limit)


@router.get("/assets/{asset_id}", response_model=AssetOut)
def get_asset(asset_id: str, svc: Services = Depends(services)) -> AssetOut:
    return AssetOut.of(svc.assets.get(asset_id))


@router.patch("/assets/{asset_id}", response_model=AssetOut)
def update_asset(asset_id: str, body: AssetUpdate, svc: Services = Depends(services)) -> AssetOut:
    return AssetOut.of(svc.assets.update(asset_id, tags=body.tags, favorite=body.favorite))


@router.delete("/assets/{asset_id}", status_code=204, response_class=Response)
def delete_asset(asset_id: str, svc: Services = Depends(services)) -> Response:
    project_id, changed = svc.assets.delete(asset_id)
    svc.bus.publish("asset.deleted", asset_id=asset_id)
    for view in sorted(changed):
        svc.bus.publish(f"{view}.updated", project_id=project_id)
    return Response(status_code=204)


@router.get("/files/{project_id}/{asset_id}", tags=["files"])
def get_file(project_id: str, asset_id: str, svc: Services = Depends(services)) -> FileResponse:
    """Serves an asset by id (DB lookup only, Range supported for video seeking)."""
    asset = svc.assets.get(asset_id)
    if asset.project_id != project_id:
        raise not_found("Asset", asset_id)
    path = svc.assets.file_path(asset)
    if not path.is_file():
        raise AppError(410, "asset_file_missing", "Asset file is missing on disk")
    return FileResponse(path, headers=_IMMUTABLE, content_disposition_type="inline")


@router.get("/thumbs/{project_id}/{asset_id}", tags=["files"])
def get_thumb(project_id: str, asset_id: str, svc: Services = Depends(services)) -> FileResponse:
    asset = svc.assets.get(asset_id)
    if asset.project_id != project_id:
        raise not_found("Asset", asset_id)
    path = svc.assets.thumb_file(asset)
    if path is None or not path.is_file():
        raise not_found("Thumbnail", asset_id)
    return FileResponse(path, media_type="image/jpeg", headers=_IMMUTABLE)
