from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from localflow.models import Shot
from localflow.schemas.board import (
    BoardOut,
    IngredientCreate,
    IngredientOut,
    IngredientUpdate,
    Reorder,
    SceneCreate,
    SceneOut,
    SceneUpdate,
    ShotCreate,
    ShotOut,
    ShotUpdate,
)
from localflow.services.container import Services, services

router = APIRouter(prefix="/api")


# --- ingredients ---------------------------------------------------------------------------------
@router.get("/projects/{project_id}/ingredients", response_model=list[IngredientOut], tags=["ingredients"])
def list_ingredients(
    project_id: str, include_drafts: bool = False, svc: Services = Depends(services)
) -> list[IngredientOut]:
    return svc.board.outs(svc.board.list_ingredients(project_id, include_drafts=include_drafts))


@router.post(
    "/projects/{project_id}/ingredients", response_model=IngredientOut, status_code=201, tags=["ingredients"]
)
def create_ingredient(
    project_id: str, body: IngredientCreate, svc: Services = Depends(services)
) -> IngredientOut:
    return svc.board.out(svc.board.create_ingredient(project_id, body))


@router.get("/ingredients/{ingredient_id}", response_model=IngredientOut, tags=["ingredients"])
def get_ingredient(ingredient_id: str, svc: Services = Depends(services)) -> IngredientOut:
    return svc.board.out(svc.board.get_ingredient(ingredient_id))


@router.patch("/ingredients/{ingredient_id}", response_model=IngredientOut, tags=["ingredients"])
def update_ingredient(
    ingredient_id: str, body: IngredientUpdate, svc: Services = Depends(services)
) -> IngredientOut:
    return svc.board.out(svc.board.update_ingredient(ingredient_id, body))


@router.delete("/ingredients/{ingredient_id}", status_code=204, response_class=Response, tags=["ingredients"])
def delete_ingredient(ingredient_id: str, svc: Services = Depends(services)) -> Response:
    svc.characters.delete_ingredient(ingredient_id)
    return Response(status_code=204)


# --- scenes --------------------------------------------------------------------------------------
@router.get("/projects/{project_id}/scenes", response_model=BoardOut, tags=["shots"])
def get_board(project_id: str, svc: Services = Depends(services)) -> BoardOut:
    return svc.board.board(project_id)


@router.post("/projects/{project_id}/scenes", response_model=SceneOut, status_code=201, tags=["shots"])
def create_scene(project_id: str, body: SceneCreate, svc: Services = Depends(services)) -> SceneOut:
    return SceneOut.of(svc.board.create_scene(project_id, body), [])


@router.post(
    "/projects/{project_id}/scenes/reorder", status_code=204, response_class=Response, tags=["shots"]
)
def reorder_scenes(project_id: str, body: Reorder, svc: Services = Depends(services)) -> Response:
    svc.board.reorder_scenes(project_id, body)
    return Response(status_code=204)


@router.patch("/scenes/{scene_id}", response_model=SceneOut, tags=["shots"])
def update_scene(scene_id: str, body: SceneUpdate, svc: Services = Depends(services)) -> SceneOut:
    scene = svc.board.update_scene(scene_id, body)
    board = svc.board.board(scene.project_id)
    return next(sc for sc in board.scenes if sc.id == scene.id)


@router.delete("/scenes/{scene_id}", status_code=204, response_class=Response, tags=["shots"])
def delete_scene(scene_id: str, svc: Services = Depends(services)) -> Response:
    svc.board.delete_scene(scene_id)
    return Response(status_code=204)


# --- shots ---------------------------------------------------------------------------------------
@router.post("/scenes/{scene_id}/shots", response_model=ShotOut, status_code=201, tags=["shots"])
def create_shot(scene_id: str, body: ShotCreate, svc: Services = Depends(services)) -> ShotOut:
    return ShotOut.of(svc.board.create_shot(scene_id, body))


@router.post("/scenes/{scene_id}/shots/reorder", status_code=204, response_class=Response, tags=["shots"])
def reorder_shots(scene_id: str, body: Reorder, svc: Services = Depends(services)) -> Response:
    svc.board.reorder_shots(scene_id, body)
    return Response(status_code=204)


@router.get("/shots/{shot_id}", response_model=ShotOut, tags=["shots"])
def get_shot(shot_id: str, svc: Services = Depends(services)) -> ShotOut:
    shot: Shot = svc.board.get_shot(shot_id)
    return ShotOut.of(shot)


@router.patch("/shots/{shot_id}", response_model=ShotOut, tags=["shots"])
def update_shot(shot_id: str, body: ShotUpdate, svc: Services = Depends(services)) -> ShotOut:
    return ShotOut.of(svc.board.update_shot(shot_id, body))


@router.delete("/shots/{shot_id}", status_code=204, response_class=Response, tags=["shots"])
def delete_shot(shot_id: str, svc: Services = Depends(services)) -> Response:
    svc.board.delete_shot(shot_id)
    return Response(status_code=204)
