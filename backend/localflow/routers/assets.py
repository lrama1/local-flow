from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse

from localflow.errors import AppError, bad_request, not_found
from localflow.schemas.collections import (
    AssetFacets,
    BulkAssetsBody,
    BulkAssetsOut,
    CollectionAssetsBody,
    CollectionCreate,
    CollectionOut,
    CollectionUpdate,
    ExtractFrameBody,
)
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
    collection: str | None = Query(default=None, description="Assets in this collection"),
    model: str | None = Query(default=None, description="Made with this model (entry id or name)"),
    scene_id: str | None = Query(default=None, description="Made for a shot of this scene"),
    featuring: str | None = Query(
        default=None, description="Ingredient id: its candidates, prompts mentioning it, shots it's in"
    ),
    sort: Literal["newest", "oldest"] = "newest",
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
        collection_id=collection,
        model=model,
        scene_id=scene_id,
        featuring=featuring,
        oldest_first=sort == "oldest",
        offset=offset,
        limit=limit,
    )
    return Page[AssetOut](items=[AssetOut.of(a) for a in rows], total=total, offset=offset, limit=limit)


@router.get("/projects/{project_id}/assets/facets", response_model=AssetFacets)
def asset_facets(project_id: str, svc: Services = Depends(services)) -> AssetFacets:
    """Filter values for the asset library, with counts."""
    svc.projects.get(project_id)
    return AssetFacets.model_validate(svc.assets.facets(project_id))


@router.post("/projects/{project_id}/assets/bulk", response_model=BulkAssetsOut)
def bulk_assets(project_id: str, body: BulkAssetsBody, svc: Services = Depends(services)) -> BulkAssetsOut:
    """Tag, untag, (un)favorite or delete several of the project's assets."""
    assets = [svc.assets.get(i) for i in dict.fromkeys(body.asset_ids)]
    if any(a.project_id != project_id for a in assets):
        raise bad_request("invalid_asset", "Every asset must belong to this project.")
    if body.delete:
        changed: set[str] = set()
        for a in assets:
            _pid, views = svc.assets.delete(a.id)
            changed |= views
            svc.bus.publish("asset.deleted", asset_id=a.id)
        for v in sorted(changed):
            svc.bus.publish(f"{v}.updated", project_id=project_id)
        svc.bus.publish("collections.updated", project_id=project_id)
        return BulkAssetsOut(updated=0, deleted=len(assets))
    add = {t.strip() for t in body.add_tags if t.strip()}
    remove = {t.strip() for t in body.remove_tags}
    for a in assets:
        tags = None
        if add or remove:
            tags = sorted((set(a.tags_json) | add) - remove)
        svc.assets.update(a.id, tags=tags, favorite=body.favorite)
    svc.bus.publish("assets.updated", project_id=project_id)
    return BulkAssetsOut(updated=len(assets), deleted=0)


@router.post("/assets/{asset_id}/extract-frame", response_model=AssetOut, status_code=201)
async def extract_frame(asset_id: str, body: ExtractFrameBody, svc: Services = Depends(services)) -> AssetOut:
    """A still of a video at `time_s` (snapped to its frame grid), as a new "extracted" image asset."""
    asset = await asyncio.to_thread(svc.assets.extract_frame, asset_id, body.time_s)
    out = AssetOut.of(asset)
    svc.bus.publish("asset.created", asset=out.model_dump(mode="json"))
    return out


@router.get("/assets/{asset_id}/collections", response_model=list[str])
def asset_collections(asset_id: str, svc: Services = Depends(services)) -> list[str]:
    """Ids of the collections this asset is in."""
    svc.assets.get(asset_id)
    return svc.collections.of_asset(asset_id)


# --- collections (Section 13) -------------------------------------------------------------------------
@router.get("/projects/{project_id}/collections", response_model=list[CollectionOut], tags=["collections"])
def list_collections(project_id: str, svc: Services = Depends(services)) -> list[CollectionOut]:
    return svc.collections.for_project(project_id)


@router.post(
    "/projects/{project_id}/collections", response_model=CollectionOut, status_code=201, tags=["collections"]
)
def create_collection(
    project_id: str, body: CollectionCreate, svc: Services = Depends(services)
) -> CollectionOut:
    return svc.collections.create(project_id, body)


@router.patch("/collections/{collection_id}", response_model=CollectionOut, tags=["collections"])
def update_collection(
    collection_id: str, body: CollectionUpdate, svc: Services = Depends(services)
) -> CollectionOut:
    return svc.collections.update(collection_id, body)


@router.delete("/collections/{collection_id}", status_code=204, response_class=Response, tags=["collections"])
def delete_collection(collection_id: str, svc: Services = Depends(services)) -> Response:
    """Deletes the collection; its assets stay."""
    svc.collections.delete(collection_id)
    return Response(status_code=204)


@router.post("/collections/{collection_id}/assets", response_model=CollectionOut, tags=["collections"])
def add_to_collection(
    collection_id: str, body: CollectionAssetsBody, svc: Services = Depends(services)
) -> CollectionOut:
    return svc.collections.add_assets(collection_id, body.asset_ids)


@router.delete("/collections/{collection_id}/assets", response_model=CollectionOut, tags=["collections"])
def remove_from_collection(
    collection_id: str, body: CollectionAssetsBody, svc: Services = Depends(services)
) -> CollectionOut:
    return svc.collections.remove_assets(collection_id, body.asset_ids)


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
