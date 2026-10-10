"""Session-level helpers for character views and library status (shared by several services)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sqlmodel import Session, col, select

from localflow.errors import bad_request
from localflow.models import Asset, Ingredient, IngredientAsset, LibraryCharacter, utcnow
from localflow.schemas.board import EXTRA_VIEWS, LibraryStatus
from localflow.services.characters.mentions import Mentionable


def views_for(s: Session, ingredient_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    """ingredient id -> {extra view -> asset id} (the primary image is on the ingredient row)."""
    ids = list(dict.fromkeys(ingredient_ids))
    out: dict[str, dict[str, str]] = {i: {} for i in ids}
    if not ids:
        return out
    for row in s.exec(select(IngredientAsset).where(col(IngredientAsset.ingredient_id).in_(ids))).all():
        out.setdefault(row.ingredient_id, {})[row.view] = row.asset_id
    return out


def all_views(ing: Ingredient, extra: dict[str, str]) -> dict[str, str]:
    out = {"primary": ing.primary_asset_id} if ing.primary_asset_id else {}
    return {**out, **extra}


def check_image(s: Session, project_id: str, asset_id: str) -> Asset:
    a = s.get(Asset, asset_id)
    if a is None or a.project_id != project_id:
        raise bad_request("invalid_asset", f"Asset '{asset_id}' isn't in this project.")
    if a.type != "image":
        raise bad_request("wrong_asset_type", f"Asset '{asset_id}' is a {a.type}, expected an image.")
    return a


def set_view(s: Session, ing: Ingredient, view: str, asset_id: str | None) -> None:
    """Chooses (or clears, with None) the image for one view. Caller commits."""
    if asset_id is not None:
        check_image(s, ing.project_id, asset_id)
    if view == "primary":
        ing.primary_asset_id = asset_id
    elif view in EXTRA_VIEWS:
        row = s.exec(
            select(IngredientAsset).where(
                IngredientAsset.ingredient_id == ing.id, IngredientAsset.view == view
            )
        ).first()
        if asset_id is None:
            if row is not None:
                s.delete(row)
        elif row is None:
            s.add(IngredientAsset(ingredient_id=ing.id, asset_id=asset_id, view=view))
        else:
            row.asset_id = asset_id
            row.updated_at = utcnow()
            s.add(row)
    else:
        raise bad_request("invalid_view", f"Unknown view '{view}'.")
    ing.updated_at = utcnow()
    s.add(ing)


def project_mentionables(
    s: Session, project_id: str, exclude_id: str | None = None
) -> dict[str, Mentionable]:
    """Ingredients a prompt in this project can @mention: everything saved except the style."""
    rows = s.exec(
        select(Ingredient).where(
            Ingredient.project_id == project_id,
            Ingredient.draft == False,  # noqa: E712
            Ingredient.kind != "style",
        )
    ).all()
    return {
        i.key: Mentionable(i.id, i.key, i.name, i.visual_prompt, i.kind) for i in rows if i.id != exclude_id
    }


def library_mentionables(s: Session) -> dict[str, Mentionable]:
    """Playground prompts mention Cast library characters (Section 11.7.3)."""
    return {
        r.key: Mentionable(r.id, r.key, r.name, r.visual_prompt, "character")
        for r in s.exec(select(LibraryCharacter)).all()
    }


def library_statuses(s: Session, ings: Sequence[Ingredient]) -> dict[str, LibraryStatus]:
    linked = [i for i in ings if i.library_character_id]
    if not linked:
        return {}
    ids = {str(i.library_character_id) for i in linked}
    libs = {
        lc.id: lc for lc in s.exec(select(LibraryCharacter).where(col(LibraryCharacter.id).in_(ids))).all()
    }
    out: dict[str, LibraryStatus] = {}
    for i in linked:
        lib = libs.get(str(i.library_character_id))
        if lib is None:
            out[i.id] = "missing"
        elif i.library_synced_at is None or lib.updated_at != i.library_synced_at:
            out[i.id] = "update_available"
        else:
            out[i.id] = "synced"
    return out
