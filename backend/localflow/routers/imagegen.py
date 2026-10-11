"""Ingredient images, keyframes, uploads (Sections 11.1-11.2, 16)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict

from localflow.errors import AppError
from localflow.models import new_id
from localflow.schemas.board import CharacterView, IngredientOut, IngredientUpdate, ShotOut, ShotUpdate
from localflow.schemas.core import AssetOut, JobOut
from localflow.services.container import Services, services
from localflow.services.imagegen import IngredientGenerateBody, KeyframeBody, KeyframeOut

router = APIRouter(prefix="/api")

_CHUNK = 1 << 20


class PrimaryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str | None


class SelectBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: Literal["keyframe", "end_keyframe", "video"]
    asset_id: str | None


async def _save_upload(svc: Services, project_id: str, file: UploadFile, generation: dict | None = None):  # type: ignore[no-untyped-def]
    limit = svc.settings.max_upload_mb * 1024 * 1024
    tmp = svc.settings.tmp_dir / "uploads" / f"{new_id()}.upload"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with tmp.open("wb") as fh:
            while chunk := await file.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise AppError(
                        413, "upload_too_large", f"Uploads are limited to {svc.settings.max_upload_mb} MB."
                    )
                fh.write(chunk)
        if size == 0:
            raise AppError(400, "empty_upload", "The uploaded file is empty.")
        # Probing, transcoding and thumbnails can take a while: keep them off the event loop.
        return await asyncio.to_thread(
            svc.assets.ingest_upload, project_id, tmp, file.filename or "upload", generation=generation
        )
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


@router.post(
    "/projects/{project_id}/assets/upload", response_model=AssetOut, status_code=201, tags=["assets"]
)
async def upload_asset(
    project_id: str, file: UploadFile = File(...), svc: Services = Depends(services)
) -> AssetOut:
    svc.projects.get(project_id)
    asset = await _save_upload(svc, project_id, file)
    svc.bus.publish("asset.created", asset=AssetOut.of(asset).model_dump(mode="json"))
    return AssetOut.of(asset)


# --- ingredients -------------------------------------------------------------------------------------
@router.post(
    "/ingredients/{ingredient_id}/generate", response_model=JobOut, status_code=201, tags=["ingredients"]
)
def generate_ingredient(
    ingredient_id: str, body: IngredientGenerateBody, svc: Services = Depends(services)
) -> JobOut:
    return JobOut.of(svc.imagegen.generate_ingredient(ingredient_id, body))


@router.patch("/ingredients/{ingredient_id}/primary", response_model=IngredientOut, tags=["ingredients"])
def set_primary(ingredient_id: str, body: PrimaryBody, svc: Services = Depends(services)) -> IngredientOut:
    ing = svc.board.update_ingredient(ingredient_id, IngredientUpdate(primary_asset_id=body.asset_id))
    return svc.board.out(ing)


@router.post(
    "/ingredients/{ingredient_id}/assets", response_model=AssetOut, status_code=201, tags=["ingredients"]
)
async def upload_ingredient_image(
    ingredient_id: str,
    file: UploadFile = File(...),
    set_primary: bool = Form(default=True, description="Select it for `view` (default: the primary)"),
    view: CharacterView = Form(default="primary"),
    svc: Services = Depends(services),
) -> AssetOut:
    ing = svc.board.get_ingredient(ingredient_id)
    if view != "primary" and ing.kind != "character":
        raise AppError(400, "invalid_view", "Only characters have extra views.")
    target: dict[str, str] = {"kind": "ingredient", "ingredient_id": ing.id}
    if view != "primary":
        target["view"] = view
    asset = await _save_upload(svc, ing.project_id, file, {"target_ref": target})
    if asset.type != "image":
        svc.assets.delete(asset.id)
        raise AppError(415, "unsupported_media", "Ingredient references must be images.")
    if set_primary:
        svc.characters.set_view(ing.id, view, asset.id)
    else:
        svc.bus.publish("ingredients.updated", project_id=ing.project_id)
    svc.bus.publish("asset.created", asset=AssetOut.of(asset).model_dump(mode="json"))
    return AssetOut.of(asset)


# --- shots -------------------------------------------------------------------------------------------
@router.post(
    "/shots/{shot_id}/generate-keyframe", response_model=KeyframeOut, status_code=201, tags=["shots"]
)
def generate_keyframe(shot_id: str, body: KeyframeBody, svc: Services = Depends(services)) -> KeyframeOut:
    return svc.imagegen.compose_keyframe(shot_id, body)


@router.post("/shots/{shot_id}/select", response_model=ShotOut, tags=["shots"])
def select_for_shot(shot_id: str, body: SelectBody, svc: Services = Depends(services)) -> ShotOut:
    field = {
        "keyframe": "selected_keyframe_asset_id",
        "end_keyframe": "selected_end_keyframe_asset_id",
        "video": "selected_video_asset_id",
    }[body.slot]
    return ShotOut.of(svc.board.update_shot(shot_id, ShotUpdate.model_validate({field: body.asset_id})))
