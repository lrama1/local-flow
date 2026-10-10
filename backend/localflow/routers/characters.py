"""Character creator API (Sections 11.7, 16): templates, wizard, views, Cast library, mentions."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query, Response
from fastapi.responses import FileResponse

from localflow.errors import bad_request
from localflow.schemas.board import CharacterView, IngredientOut
from localflow.schemas.characters import (
    CharacterCapabilities,
    CleanReferenceBody,
    DescribeBody,
    DescribeOut,
    DraftCreate,
    ExpandBody,
    ExpandOut,
    ImportBody,
    LibraryCharacterOut,
    LibraryCharacterUpdate,
    ReferencesOut,
    ResolveBody,
    ResolveOut,
    SaveToLibraryBody,
    SetViewBody,
    TemplateCreate,
    TemplateOut,
    TemplateUpdate,
    ViewsBody,
    ViewsOut,
)
from localflow.schemas.core import JobOut
from localflow.services.container import Services, services

router = APIRouter(prefix="/api")

_IMMUTABLE = {"Cache-Control": "private, max-age=31536000, immutable"}
_DAY = {"Cache-Control": "private, max-age=86400"}


# --- templates (Section 11.7.1) ----------------------------------------------------------------------
@router.get("/character-templates", response_model=list[TemplateOut], tags=["characters"])
def list_templates(svc: Services = Depends(services)) -> list[TemplateOut]:
    return svc.templates.list()


@router.post("/character-templates", response_model=TemplateOut, status_code=201, tags=["characters"])
def create_template(body: TemplateCreate, svc: Services = Depends(services)) -> TemplateOut:
    source, primary = None, None
    if body.from_ingredient_id:
        source = svc.board.get_ingredient(body.from_ingredient_id)
        if source.kind != "character":
            raise bad_request("not_a_character", "Only characters can be saved as templates.")
        if source.primary_asset_id:
            primary = svc.assets.file_path(svc.assets.get(source.primary_asset_id))
    return svc.templates.create(body, source, primary)


@router.patch("/character-templates/{template_id}", response_model=TemplateOut, tags=["characters"])
def update_template(template_id: str, body: TemplateUpdate, svc: Services = Depends(services)) -> TemplateOut:
    return svc.templates.update(template_id, body)


@router.delete(
    "/character-templates/{template_id}", status_code=204, response_class=Response, tags=["characters"]
)
def delete_template(template_id: str, svc: Services = Depends(services)) -> Response:
    svc.templates.delete(template_id)
    return Response(status_code=204)


@router.get("/character-templates/{key}/thumbnail", tags=["characters"])
def template_thumbnail(key: str, v: str | None = None, svc: Services = Depends(services)) -> FileResponse:
    return FileResponse(svc.templates.thumbnail(key), headers=_IMMUTABLE if v else _DAY)


# --- creation wizard (Section 11.7) ------------------------------------------------------------------
@router.get("/characters/capabilities", response_model=CharacterCapabilities, tags=["characters"])
def capabilities(project_id: str | None = None, svc: Services = Depends(services)) -> CharacterCapabilities:
    return svc.characters.capabilities(project_id)


@router.post(
    "/projects/{project_id}/characters", response_model=IngredientOut, status_code=201, tags=["characters"]
)
def create_draft(project_id: str, body: DraftCreate, svc: Services = Depends(services)) -> IngredientOut:
    """Starts the wizard: a draft character (hidden from the story pipeline until saved with
    `PATCH /ingredients/{id}` `{draft: false}`)."""
    return svc.board.out(svc.characters.create_draft(project_id, body))


@router.post("/characters/expand", response_model=ExpandOut, tags=["characters"])
async def expand(
    body: ExpandBody, project_id: str | None = None, svc: Services = Depends(services)
) -> ExpandOut:
    return await svc.characters.expand(body, project_id)


@router.post("/characters/describe", response_model=DescribeOut, tags=["characters"])
async def describe(body: DescribeBody, svc: Services = Depends(services)) -> DescribeOut:
    return await svc.characters.describe(body)


# --- views -------------------------------------------------------------------------------------------
@router.post(
    "/ingredients/{ingredient_id}/generate-views",
    response_model=ViewsOut,
    status_code=201,
    tags=["characters"],
)
def generate_views(ingredient_id: str, body: ViewsBody, svc: Services = Depends(services)) -> ViewsOut:
    return svc.characters.generate_views(ingredient_id, body)


@router.post(
    "/ingredients/{ingredient_id}/clean-reference",
    response_model=JobOut,
    status_code=201,
    tags=["characters"],
)
def clean_reference(
    ingredient_id: str, body: CleanReferenceBody, svc: Services = Depends(services)
) -> JobOut:
    return JobOut.of(svc.characters.clean_reference(ingredient_id, body))


@router.put("/ingredients/{ingredient_id}/views/{view}", response_model=IngredientOut, tags=["characters"])
def set_view(
    ingredient_id: str, view: CharacterView, body: SetViewBody, svc: Services = Depends(services)
) -> IngredientOut:
    """Chooses the image for one view (`primary`, `full_body`, …); `asset_id: null` clears it."""
    return svc.board.out(svc.characters.set_view(ingredient_id, view, body.asset_id))


# --- Cast library (Section 11.7.2) -------------------------------------------------------------------
@router.get("/library/characters", response_model=list[LibraryCharacterOut], tags=["library"])
def list_library(q: str | None = None, svc: Services = Depends(services)) -> list[LibraryCharacterOut]:
    return [svc.library.out(r) for r in svc.library.search(q)]


@router.get("/library/characters/{library_id}", response_model=LibraryCharacterOut, tags=["library"])
def get_library(library_id: str, svc: Services = Depends(services)) -> LibraryCharacterOut:
    return svc.library.out(svc.library.get(library_id))


@router.patch("/library/characters/{library_id}", response_model=LibraryCharacterOut, tags=["library"])
def update_library(
    library_id: str, body: LibraryCharacterUpdate, svc: Services = Depends(services)
) -> LibraryCharacterOut:
    return svc.library.out(svc.library.update(library_id, body))


@router.delete("/library/characters/{library_id}", status_code=204, response_class=Response, tags=["library"])
def delete_library(library_id: str, svc: Services = Depends(services)) -> Response:
    svc.library.delete(library_id)
    return Response(status_code=204)


@router.get("/library/characters/{library_id}/images/{n}", tags=["library"])
def library_image(
    library_id: str, n: int, thumb: bool = False, svc: Services = Depends(services)
) -> FileResponse:
    return FileResponse(svc.library.image_path(library_id, n, thumb), headers=_IMMUTABLE)


@router.post(
    "/ingredients/{ingredient_id}/save-to-library",
    response_model=LibraryCharacterOut,
    status_code=201,
    tags=["library"],
)
def save_to_library(
    ingredient_id: str,
    body: SaveToLibraryBody = Body(default_factory=SaveToLibraryBody),
    svc: Services = Depends(services),
) -> LibraryCharacterOut:
    return svc.library.out(svc.library.save(ingredient_id, link=body.link))


@router.post(
    "/ingredients/{ingredient_id}/push-to-library", response_model=LibraryCharacterOut, tags=["library"]
)
def push_to_library(ingredient_id: str, svc: Services = Depends(services)) -> LibraryCharacterOut:
    return svc.library.out(svc.library.push(ingredient_id))


@router.post(
    "/ingredients/{ingredient_id}/update-from-library", response_model=IngredientOut, tags=["library"]
)
def update_from_library(ingredient_id: str, svc: Services = Depends(services)) -> IngredientOut:
    return svc.board.out(svc.library.update_from_library(ingredient_id))


@router.post(
    "/projects/{project_id}/ingredients/import",
    response_model=IngredientOut,
    status_code=201,
    tags=["library"],
)
def import_character(project_id: str, body: ImportBody, svc: Services = Depends(services)) -> IngredientOut:
    """Cast a library character into the project, or copy a character from another project."""
    svc.projects.get(project_id)
    if body.library_character_id:
        ing = svc.library.cast(project_id, body.library_character_id)
    else:
        ing = svc.library.import_from_project(project_id, str(body.ingredient_id))
    return svc.board.out(ing)


# --- mentions & references (Sections 11.7, 11.7.3) ---------------------------------------------------
@router.post("/prompts/resolve", response_model=ResolveOut, tags=["characters"])
def resolve_prompt(body: ResolveBody, svc: Services = Depends(services)) -> ResolveOut:
    return svc.characters.resolve(body)


@router.get("/shots/{shot_id}/references", response_model=ReferencesOut, tags=["shots"])
def shot_references(
    shot_id: str,
    shot_type: str | None = None,
    ingredient_ids: list[str] | None = Query(default=None, description="Unsaved panel selection"),
    svc: Services = Depends(services),
) -> ReferencesOut:
    return svc.imagegen.references_for(shot_id, shot_type, ingredient_ids)
