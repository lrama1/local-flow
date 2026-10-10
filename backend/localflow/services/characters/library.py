"""Cast library (Section 11.7.2): global characters, copied to and from projects as snapshots.

- Save to library: copies the character's fields, primary image and views into `library/characters/<id>/`.
- Cast into project: copies them back as a new ingredient (`origin="library"`); images become project
  assets with `source="imported"`.
- Copies never sync by themselves. A project copy shows "update available" when the library entry is newer
  than its `library_synced_at`; "Update from library" and "Push to library" copy explicitly.
- Import from another project copies a character and its images the same way (`origin="import"`).
- Key collisions get a numeric suffix (`mara-2`).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlmodel import Session, col, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, bad_request, conflict, not_found
from localflow.models import Asset, Ingredient, LibraryCharacter, new_id, utcnow
from localflow.schemas.board import EXTRA_VIEWS
from localflow.schemas.characters import LibraryCharacterOut, LibraryCharacterUpdate, LibraryImageOut
from localflow.schemas.story import slugify
from localflow.services.assets import AssetService
from localflow.services.board import unique_key
from localflow.services.characters import store
from localflow.services.characters.templates import square_thumb
from localflow.services.events import EventBus
from localflow.services.voice_store import check_voice

_FIELDS = (
    "name",
    "description",
    "visual_prompt",
    "negative_prompt",
    "character_info",
    "body_outfit_prompt",
    "voice_id",
    "template_key",
)


class LibraryService:
    def __init__(self, settings: Settings, db: Database, bus: EventBus, assets: AssetService) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.assets = assets

    def dir(self, library_id: str) -> Path:
        return self.settings.library_dir / "characters" / library_id

    # --- reads -----------------------------------------------------------------------------------
    def get(self, library_id: str) -> LibraryCharacter:
        with self.db.session() as s:
            row = s.get(LibraryCharacter, library_id)
        if row is None:
            raise not_found("Library character", library_id)
        return row

    def search(self, q: str | None = None) -> list[LibraryCharacter]:
        with self.db.session() as s:
            rows = list(s.exec(select(LibraryCharacter)).all())
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in r.name.lower() or ql in r.key or ql in r.visual_prompt.lower()]
        return sorted(rows, key=lambda r: r.name.lower())

    def out(self, row: LibraryCharacter) -> LibraryCharacterOut:
        v = int(row.updated_at.timestamp())
        base = f"/api/library/characters/{row.id}/images"
        return LibraryCharacterOut(
            id=row.id,
            key=row.key,
            name=row.name,
            description=row.description,
            visual_prompt=row.visual_prompt,
            negative_prompt=row.negative_prompt,
            character_info=row.character_info,
            body_outfit_prompt=row.body_outfit_prompt,
            voice_id=row.voice_id,
            voice_hints=dict(row.voice_hints_json),
            lora_ids=list(row.lora_ids_json),
            template_key=row.template_key,
            images=[
                LibraryImageOut(
                    n=n,
                    view=str(img.get("view") or "primary"),
                    is_primary=bool(img.get("is_primary")),
                    url=f"{base}/{n}?v={v}",
                    thumb_url=f"{base}/{n}?thumb=1&v={v}",
                )
                for n, img in enumerate(row.images_json, 1)
            ],
            source_project_id=row.source_project_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def image_path(self, library_id: str, n: int, thumb: bool = False) -> Path:
        row = self.get(library_id)
        if not 1 <= n <= len(row.images_json):
            raise not_found("Library image", f"{library_id}/{n}")
        img = row.images_json[n - 1]
        name = str(img.get("thumb") if thumb else img.get("file"))
        path = (self.dir(library_id) / name).resolve()
        if not path.is_relative_to(self.dir(library_id).resolve()) or not path.is_file():
            raise not_found("Library image", f"{library_id}/{n}")
        return path

    # --- library CRUD ----------------------------------------------------------------------------
    def update(self, library_id: str, body: LibraryCharacterUpdate) -> LibraryCharacter:
        with self.db.session() as s:
            row = s.get(LibraryCharacter, library_id)
            if row is None:
                raise not_found("Library character", library_id)
            for k, v in body.model_dump(exclude_unset=True).items():
                if k == "voice_hints":
                    row.voice_hints_json = dict(v or {})
                elif k == "voice_id":
                    check_voice(s, v)
                    row.voice_id = v
                elif v is not None:
                    setattr(row, k, v)
            row.updated_at = utcnow()
            s.add(row)
            s.commit()
            s.refresh(row)
        self.bus.publish("library.updated", library_character_id=library_id)
        return row

    def delete(self, library_id: str) -> None:
        with self.db.session() as s:
            row = s.get(LibraryCharacter, library_id)
            if row is None:
                raise not_found("Library character", library_id)
            s.delete(row)
            s.commit()
        shutil.rmtree(self.dir(library_id), ignore_errors=True)
        self.bus.publish("library.updated", library_character_id=library_id)

    # --- project -> library ----------------------------------------------------------------------
    def save(self, ingredient_id: str, link: bool = True) -> LibraryCharacter:
        """Save to library. Drafts can be saved too (the library wizard), but are never linked."""
        with self.db.session() as s:
            ing = self._character(s, ingredient_id)
            if ing.library_character_id and s.get(LibraryCharacter, ing.library_character_id):
                raise conflict(
                    "already_in_library",
                    f"{ing.name} is already in the Cast library. Use Push to library to update it.",
                )
            images = self._ingredient_images(s, ing)
            taken = {r.key for r in s.exec(select(LibraryCharacter)).all()}
            base = slugify(ing.name if ing.draft else ing.key)
            row = LibraryCharacter(
                key=unique_key(base, taken), name=ing.name, source_project_id=ing.project_id
            )
            _copy_fields(ing, row)
            row.images_json = self._write_images(row.id, images)
            s.add(row)
            if link and not ing.draft:
                ing.library_character_id, ing.library_synced_at = row.id, row.updated_at
                s.add(ing)
            s.commit()
            s.refresh(row)
            project_id = ing.project_id
        self.bus.publish("library.updated", library_character_id=row.id)
        self.bus.publish("ingredients.updated", project_id=project_id)
        return row

    def push(self, ingredient_id: str) -> LibraryCharacter:
        """Overwrites the linked library entry with the project copy (the UI confirms first)."""
        with self.db.session() as s:
            ing = self._character(s, ingredient_id)
            row = s.get(LibraryCharacter, ing.library_character_id) if ing.library_character_id else None
            if row is None:
                raise conflict("not_in_library", f"{ing.name} isn't linked to a Cast library character.")
            images = self._ingredient_images(s, ing)
            _copy_fields(ing, row)
            old = row.images_json
            row.images_json = self._write_images(row.id, images)
            _remove_unused(self.dir(row.id), old, row.images_json)
            row.updated_at = utcnow()
            ing.library_synced_at = row.updated_at
            s.add(row)
            s.add(ing)
            s.commit()
            s.refresh(row)
            project_id = ing.project_id
        self.bus.publish("library.updated", library_character_id=row.id)
        self.bus.publish("ingredients.updated", project_id=project_id)
        return row

    # --- library -> project ----------------------------------------------------------------------
    def cast(self, project_id: str, library_id: str) -> Ingredient:
        lib = self.get(library_id)
        with self.db.session() as s:
            taken = {i.key for i in s.exec(select(Ingredient).where(Ingredient.project_id == project_id))}
            ing = Ingredient(
                project_id=project_id,
                kind="character",
                key=unique_key(lib.key, taken),
                name=lib.name,
                origin="library",
                locked=True,
                library_character_id=lib.id,
                library_synced_at=lib.updated_at,
            )
            _copy_fields(lib, ing)
            s.add(ing)
            s.commit()
            s.refresh(ing)
        self._import_library_images(ing, lib)
        self.bus.publish("ingredients.updated", project_id=project_id)
        return self._reload(ing.id)

    def update_from_library(self, ingredient_id: str) -> Ingredient:
        with self.db.session() as s:
            ing = self._character(s, ingredient_id)
            lib = s.get(LibraryCharacter, ing.library_character_id) if ing.library_character_id else None
            if lib is None:
                raise conflict("not_in_library", f"{ing.name} isn't linked to a Cast library character.")
            _copy_fields(lib, ing)
            ing.library_synced_at = lib.updated_at
            ing.updated_at = utcnow()
            s.add(ing)
            s.commit()
            s.refresh(ing)
        self._import_library_images(ing, lib, replace=True)
        self.bus.publish("ingredients.updated", project_id=ing.project_id)
        return self._reload(ing.id)

    def import_from_project(self, project_id: str, source_ingredient_id: str) -> Ingredient:
        with self.db.session() as s:
            src = self._character(s, source_ingredient_id)
            if src.draft:
                raise bad_request("draft_character", "Save the character before importing it.")
            images = [(view, a) for view, a, _ in self._ingredient_images(s, src)]
            taken = {i.key for i in s.exec(select(Ingredient).where(Ingredient.project_id == project_id))}
            ing = Ingredient(
                project_id=project_id,
                kind="character",
                key=unique_key(src.key, taken),
                name=src.name,
                origin="import",
                locked=True,
                lora_ids_json=list(src.lora_ids_json),
                voice_hints_json=dict(src.voice_hints_json),
            )
            _copy_fields(src, ing)
            s.add(ing)
            s.commit()
            s.refresh(ing)
        chosen: dict[str, str] = {}
        for view, asset in images:
            copy = self._import_file(
                ing,
                self.assets.file_path(asset),
                view,
                {"project_id": src.project_id, "ingredient_id": src.id, "asset_id": asset.id},
            )
            chosen[view] = copy.id
        self._choose(ing.id, chosen)
        self.bus.publish("ingredients.updated", project_id=project_id)
        return self._reload(ing.id)

    # --- helpers ---------------------------------------------------------------------------------
    def _character(self, s: Session, ingredient_id: str) -> Ingredient:
        ing = s.get(Ingredient, ingredient_id)
        if ing is None:
            raise not_found("Ingredient", ingredient_id)
        if ing.kind != "character":
            raise bad_request("not_a_character", "Only characters can be shared through the Cast library.")
        return ing

    def _reload(self, ingredient_id: str) -> Ingredient:
        with self.db.session() as s:
            ing = s.get(Ingredient, ingredient_id)
        assert ing is not None
        return ing

    def _ingredient_images(self, s: Session, ing: Ingredient) -> list[tuple[str, Asset, Path]]:
        """(view, asset, file) for the primary and each chosen view, primary first."""
        views = store.all_views(ing, store.views_for(s, [ing.id]).get(ing.id, {}))
        order = ["primary", *EXTRA_VIEWS]
        ids = [views[v] for v in order if v in views]
        assets = {a.id: a for a in s.exec(select(Asset).where(col(Asset.id).in_(ids))).all()} if ids else {}
        out = []
        for view in order:
            asset = assets.get(views.get(view, ""))
            if asset is not None:
                path = self.assets.file_path(asset)
                if path.is_file():
                    out.append((view, asset, path))
        return out

    def _write_images(self, library_id: str, images: list[tuple[str, Asset, Path]]) -> list[dict[str, Any]]:
        """Copies images into the library dir under fresh names (`<token>-<n>.png` + a thumbnail)."""
        folder = self.dir(library_id)
        folder.mkdir(parents=True, exist_ok=True)
        token = new_id()[-6:].lower()
        out: list[dict[str, Any]] = []
        for n, (view, _asset, path) in enumerate(images, 1):
            name = f"{token}-{n}{path.suffix.lower() or '.png'}"
            thumb = f"{token}-{n}.thumb.jpg"
            shutil.copy2(path, folder / name)
            square_thumb(folder / name, folder / thumb)
            out.append({"file": name, "thumb": thumb, "view": view, "is_primary": view == "primary"})
        return out

    def _import_library_images(self, ing: Ingredient, lib: LibraryCharacter, replace: bool = False) -> None:
        chosen: dict[str, str] = {}
        for img in lib.images_json:
            src = self.dir(lib.id) / str(img.get("file"))
            if not src.is_file():
                continue
            view = str(img.get("view") or "primary")
            asset = self._import_file(
                ing, src, view, {"library_character_id": lib.id, "file": img.get("file")}
            )
            chosen[view] = asset.id
        self._choose(ing.id, chosen, clear_missing=replace)

    def _import_file(self, ing: Ingredient, src: Path, view: str, origin: dict[str, Any]) -> Asset:
        if not src.is_file():
            raise AppError(404, "image_missing", f"An image of {ing.name} is missing on disk.")
        tmp = self.settings.tmp_dir / "imports" / f"{new_id()}{src.suffix.lower() or '.png'}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tmp)
        target: dict[str, str] = {"kind": "ingredient", "ingredient_id": ing.id}
        if view != "primary":
            target["view"] = view
        return self.assets.ingest_file(
            ing.project_id,
            tmp,
            "image",
            source="imported",
            generation={"imported_from": origin, "target_ref": target},
        )

    def _choose(self, ingredient_id: str, chosen: dict[str, str], clear_missing: bool = False) -> None:
        with self.db.session() as s:
            ing = s.get(Ingredient, ingredient_id)
            assert ing is not None
            for view in ["primary", *EXTRA_VIEWS]:
                if view in chosen:
                    store.set_view(s, ing, view, chosen[view])
                elif clear_missing:
                    store.set_view(s, ing, view, None)
            s.commit()


def _copy_fields(src: Any, dst: Any) -> None:
    """Character fields between an ingredient and a library entry (either direction)."""
    for f in _FIELDS:
        value = getattr(src, f, None)
        if isinstance(dst, LibraryCharacter) and value is None and f != "voice_id" and f != "template_key":
            value = ""  # library text columns aren't nullable
        setattr(dst, f, value)
    dst.voice_hints_json = dict(getattr(src, "voice_hints_json", {}) or {})
    dst.lora_ids_json = list(getattr(src, "lora_ids_json", []) or [])


def _remove_unused(folder: Path, old: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
    keep = {str(i.get(k)) for i in new for k in ("file", "thumb")}
    for img in old:
        for k in ("file", "thumb"):
            name = str(img.get(k) or "")
            if name and name not in keep:
                (folder / name).unlink(missing_ok=True)
