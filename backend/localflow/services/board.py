"""Ingredients, scenes, and shots CRUD + ordering (Sections 6.1, 16)."""

from __future__ import annotations

from typing import Any

from sqlmodel import Session, col, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import bad_request, conflict, not_found
from localflow.models import Asset, Ingredient, Project, Scene, Shot, utcnow
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
    ShotUpdate,
)
from localflow.schemas.story import slugify
from localflow.services.characters import mentions, store
from localflow.services.events import EventBus
from localflow.services.voice_store import check_voice

_KIND_ORDER = {"character": 0, "location": 1, "prop": 2, "style": 3}
_PROMPT_FIELDS = ("keyframe_prompt", "end_keyframe_prompt", "video_prompt")
DRAFT_KEY_PREFIX = "draft-"


def derived_status(shot: Shot) -> str:
    """Status implied by what's selected: a take → video_ready (approval is kept), else a keyframe →
    keyframe_ready, else planned."""
    if shot.selected_video_asset_id:
        return "approved" if shot.status == "approved" else "video_ready"
    return "keyframe_ready" if shot.selected_keyframe_asset_id else "planned"


def unique_key(base: str, taken: set[str]) -> str:
    """`mara`, then `mara-2`, `mara-3`… (Section 11.7.2 key collisions)."""
    key, n = base, 2
    while key in taken:
        key = f"{base[:44]}-{n}"
        n += 1
    return key


class BoardService:
    def __init__(self, settings: Settings, db: Database, bus: EventBus) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus

    def _project(self, s: Session, project_id: str) -> Project:
        p = s.get(Project, project_id)
        if p is None:
            raise not_found("Project", project_id)
        return p

    # --- ingredients -----------------------------------------------------------------------------
    def list_ingredients(self, project_id: str, include_drafts: bool = False) -> list[Ingredient]:
        with self.db.session() as s:
            self._project(s, project_id)
            stmt = select(Ingredient).where(Ingredient.project_id == project_id)
            if not include_drafts:
                stmt = stmt.where(Ingredient.draft == False)  # noqa: E712 (SQL expression)
            rows = list(s.exec(stmt).all())
        return sorted(rows, key=lambda i: (_KIND_ORDER.get(i.kind, 9), i.created_at, i.name.lower()))

    def outs(self, ings: list[Ingredient]) -> list[IngredientOut]:
        """API shape with each ingredient's chosen views and Cast library sync status."""
        with self.db.session() as s:
            views = store.views_for(s, [i.id for i in ings])
            statuses = store.library_statuses(s, ings)
        return [IngredientOut.of(i, views.get(i.id), statuses.get(i.id)) for i in ings]

    def out(self, ing: Ingredient) -> IngredientOut:
        return self.outs([ing])[0]

    def get_ingredient(self, ingredient_id: str) -> Ingredient:
        with self.db.session() as s:
            ing = s.get(Ingredient, ingredient_id)
            if ing is None:
                raise not_found("Ingredient", ingredient_id)
            return ing

    def create_ingredient(self, project_id: str, body: IngredientCreate) -> Ingredient:
        key = slugify(body.key or body.name)
        with self.db.session() as s:
            self._project(s, project_id)
            exists = s.exec(
                select(Ingredient).where(Ingredient.project_id == project_id, Ingredient.key == key)
            ).first()
            if exists:
                raise conflict("key_taken", f"An ingredient with key '{key}' already exists in this project.")
            ing = Ingredient(
                project_id=project_id,
                kind=body.kind,
                key=key,
                name=body.name,
                description=body.description,
                visual_prompt=body.visual_prompt,
                negative_prompt=body.negative_prompt,
                character_info=body.character_info if body.kind == "character" else None,
                body_outfit_prompt=body.body_outfit_prompt if body.kind == "character" else None,
                voice_hints_json=body.voice_hints.model_dump() if body.voice_hints else {},
                locked=body.locked,
                origin="user",
            )
            s.add(ing)
            s.commit()
            s.refresh(ing)
        self.bus.publish("ingredients.updated", project_id=project_id)
        return ing

    def update_ingredient(self, ingredient_id: str, body: IngredientUpdate) -> Ingredient:
        _CHARACTER_ONLY = {"character_info", "body_outfit_prompt", "voice_hints", "voice_id", "template_key"}
        with self.db.session() as s:
            ing = s.get(Ingredient, ingredient_id)
            if ing is None:
                raise not_found("Ingredient", ingredient_id)
            data = body.model_dump(exclude_unset=True)
            if "key" in data:
                self._change_key(s, ing, str(data.pop("key")))
            finish_draft = data.pop("draft", None) is False and ing.draft
            for k, v in data.items():
                if k in _CHARACTER_ONLY and ing.kind != "character":
                    continue
                if k == "primary_asset_id" and v is not None:
                    store.check_image(s, ing.project_id, v)
                if k == "voice_id":
                    check_voice(s, v)
                if k == "voice_hints":
                    ing.voice_hints_json = dict(v or {})
                    continue
                setattr(ing, k, v)
            if finish_draft:
                if ing.key.startswith(DRAFT_KEY_PREFIX):
                    self._change_key(s, ing, ing.name)
                ing.draft, ing.locked = False, True  # Section 11.7 step 5: origin="user", locked
            ing.updated_at = utcnow()
            s.add(ing)
            s.commit()
            s.refresh(ing)
        self.bus.publish("ingredients.updated", project_id=ing.project_id)
        return ing

    def _change_key(self, s: Session, ing: Ingredient, wanted: str) -> None:
        """Keys are editable until a wizard draft is saved, then immutable (Section 11.7 step 3)."""
        key = slugify(wanted)
        if key == ing.key:
            return
        if not ing.draft:
            raise conflict("key_immutable", "An ingredient's key can't change once it's saved.")
        if key.startswith(DRAFT_KEY_PREFIX):
            raise bad_request("invalid_key", f"Keys can't start with '{DRAFT_KEY_PREFIX}'.")
        taken = s.exec(
            select(Ingredient).where(
                Ingredient.project_id == ing.project_id, Ingredient.key == key, Ingredient.id != ing.id
            )
        ).first()
        if taken:
            raise conflict("key_taken", f"An ingredient with key '{key}' already exists in this project.")
        ing.key = key

    def delete_ingredient(self, ingredient_id: str) -> None:
        with self.db.session() as s:
            ing = s.get(Ingredient, ingredient_id)
            if ing is None:
                raise not_found("Ingredient", ingredient_id)
            project_id = ing.project_id
            scenes = list(s.exec(select(Scene).where(Scene.project_id == project_id)).all())
            scene_ids = [sc.id for sc in scenes]
            for sc in scenes:
                if sc.location_ingredient_id == ingredient_id:
                    sc.location_ingredient_id = None
                    s.add(sc)
            if scene_ids:
                for shot in s.exec(select(Shot).where(col(Shot.scene_id).in_(scene_ids))).all():
                    if ingredient_id in shot.ingredient_ids_json:
                        shot.ingredient_ids_json = [i for i in shot.ingredient_ids_json if i != ingredient_id]
                        s.add(shot)
                    if any(d.get("character_ingredient_id") == ingredient_id for d in shot.dialogue_json):
                        shot.dialogue_json = [
                            {**d, "character_ingredient_id": None}
                            if d.get("character_ingredient_id") == ingredient_id
                            else d
                            for d in shot.dialogue_json
                        ]
                        s.add(shot)
            s.delete(ing)
            s.commit()
        self.bus.publish("ingredients.updated", project_id=project_id)
        self.bus.publish("board.updated", project_id=project_id)

    # --- board -----------------------------------------------------------------------------------
    def board(self, project_id: str) -> BoardOut:
        with self.db.session() as s:
            self._project(s, project_id)
            scenes = list(
                s.exec(select(Scene).where(Scene.project_id == project_id).order_by(col(Scene.index))).all()
            )
            ids = [sc.id for sc in scenes]
            shots = list(s.exec(select(Shot).where(col(Shot.scene_id).in_(ids))).all()) if ids else []
        by_scene: dict[str, list[Shot]] = {}
        for sh in shots:
            by_scene.setdefault(sh.scene_id, []).append(sh)
        out = [SceneOut.of(sc, by_scene.get(sc.id, [])) for sc in scenes]
        return BoardOut(
            scenes=out,
            total_duration_s=round(sum(sh.duration_s * (1 + (sh.extend_segments or 0)) for sh in shots), 3),
            shot_count=len(shots),
            max_clip_seconds=self.settings.default_max_clip_seconds,
        )

    def _changed(self, project_id: str) -> None:
        self.bus.publish("board.updated", project_id=project_id)

    def create_scene(self, project_id: str, body: SceneCreate) -> Scene:
        with self.db.session() as s:
            self._project(s, project_id)
            self._check_location(s, project_id, body.location_ingredient_id)
            scenes = list(
                s.exec(select(Scene).where(Scene.project_id == project_id).order_by(col(Scene.index)))
            )
            pos = len(scenes) if body.index is None else min(body.index, len(scenes))
            scene = Scene(
                project_id=project_id,
                index=pos,
                title=body.title,
                summary=body.summary,
                location_ingredient_id=body.location_ingredient_id,
            )
            scenes.insert(pos, scene)
            for i, sc in enumerate(scenes):
                sc.index = i
                s.add(sc)
            s.commit()
            s.refresh(scene)
        self._changed(project_id)
        return scene

    def update_scene(self, scene_id: str, body: SceneUpdate) -> Scene:
        with self.db.session() as s:
            scene = s.get(Scene, scene_id)
            if scene is None:
                raise not_found("Scene", scene_id)
            data = body.model_dump(exclude_unset=True)
            if "location_ingredient_id" in data:
                self._check_location(s, scene.project_id, data["location_ingredient_id"])
            for k, v in data.items():
                setattr(scene, k, v)
            scene.updated_at = utcnow()
            s.add(scene)
            s.commit()
            s.refresh(scene)
        self._changed(scene.project_id)
        return scene

    def delete_scene(self, scene_id: str) -> None:
        with self.db.session() as s:
            scene = s.get(Scene, scene_id)
            if scene is None:
                raise not_found("Scene", scene_id)
            project_id = scene.project_id
            s.delete(scene)
            s.flush()
            rest = s.exec(
                select(Scene).where(Scene.project_id == project_id).order_by(col(Scene.index))
            ).all()
            for i, sc in enumerate(rest):
                sc.index = i
                s.add(sc)
            s.commit()
        self._changed(project_id)

    def reorder_scenes(self, project_id: str, body: Reorder) -> None:
        with self.db.session() as s:
            scenes = {sc.id: sc for sc in s.exec(select(Scene).where(Scene.project_id == project_id))}
            if set(body.ids) != set(scenes) or len(body.ids) != len(scenes):
                raise bad_request("invalid_order", "Provide every scene id of the project exactly once.")
            for i, sid in enumerate(body.ids):
                scenes[sid].index = i
                s.add(scenes[sid])
            s.commit()
        self._changed(project_id)

    def _scene(self, s: Session, scene_id: str) -> Scene:
        scene = s.get(Scene, scene_id)
        if scene is None:
            raise not_found("Scene", scene_id)
        return scene

    def create_shot(self, scene_id: str, body: ShotCreate) -> Shot:
        with self.db.session() as s:
            scene = self._scene(s, scene_id)
            data = self._shot_fields(
                s, scene.project_id, body.model_dump(exclude_unset=True, exclude={"index"})
            )
            shots = list(s.exec(select(Shot).where(Shot.scene_id == scene_id).order_by(col(Shot.index))))
            pos = len(shots) if body.index is None else min(body.index, len(shots))
            data.setdefault("keyframe_prompt", "")
            data.setdefault("video_prompt", "")
            shot = Shot(scene_id=scene_id, index=pos, **data)
            self._add_mentioned(s, scene.project_id, shot)
            shots.insert(pos, shot)
            for i, sh in enumerate(shots):
                sh.index = i
                s.add(sh)
            s.commit()
            s.refresh(shot)
        self._changed(scene.project_id)
        return shot

    def get_shot(self, shot_id: str) -> Shot:
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                raise not_found("Shot", shot_id)
            return shot

    def update_shot(self, shot_id: str, body: ShotUpdate) -> Shot:
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                raise not_found("Shot", shot_id)
            scene = self._scene(s, shot.scene_id)
            data = body.model_dump(exclude_unset=True)
            for k, v in self._shot_fields(s, scene.project_id, data).items():
                setattr(shot, k, v)
            if any(k in data for k in (*_PROMPT_FIELDS, "ingredient_ids")):
                self._add_mentioned(s, scene.project_id, shot)
            if (
                "selected_keyframe_asset_id" in data or "selected_video_asset_id" in data
            ) and "status" not in data:
                # Picking a keyframe or a take advances the shot; clearing it goes back (Section 11.2).
                shot.status = derived_status(shot)
            shot.updated_at = utcnow()
            s.add(shot)
            s.commit()
            s.refresh(shot)
        self._changed(scene.project_id)
        return shot

    def delete_shot(self, shot_id: str) -> None:
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                raise not_found("Shot", shot_id)
            scene = self._scene(s, shot.scene_id)
            s.delete(shot)
            s.flush()
            for i, sh in enumerate(
                s.exec(select(Shot).where(Shot.scene_id == scene.id).order_by(col(Shot.index)))
            ):
                sh.index = i
                s.add(sh)
            s.commit()
        self._changed(scene.project_id)

    def reorder_shots(self, scene_id: str, body: Reorder) -> None:
        """Sets the scene's shot order. Ids may include shots from other scenes (they move here)."""
        with self.db.session() as s:
            scene = self._scene(s, scene_id)
            current = {sh.id for sh in s.exec(select(Shot).where(Shot.scene_id == scene_id))}
            if len(set(body.ids)) != len(body.ids):
                raise bad_request("invalid_order", "Duplicate shot ids.")
            missing = current - set(body.ids)
            if missing:
                raise bad_request("invalid_order", "Include every shot currently in the scene.")
            project_scene_ids = {
                sc.id for sc in s.exec(select(Scene).where(Scene.project_id == scene.project_id))
            }
            shots = {sh.id: sh for sh in s.exec(select(Shot).where(col(Shot.id).in_(body.ids)))}
            if set(shots) != set(body.ids) or any(
                sh.scene_id not in project_scene_ids for sh in shots.values()
            ):
                raise bad_request("invalid_order", "Unknown shot id, or shot from another project.")
            sources = {sh.scene_id for sh in shots.values()} - {scene_id}
            for i, sid in enumerate(body.ids):
                shots[sid].scene_id, shots[sid].index = scene_id, i
                s.add(shots[sid])
            s.flush()
            for src in sources:
                for i, sh in enumerate(
                    s.exec(select(Shot).where(Shot.scene_id == src).order_by(col(Shot.index)))
                ):
                    sh.index = i
                    s.add(sh)
            s.commit()
        self._changed(scene.project_id)

    # --- validation ------------------------------------------------------------------------------
    def _check_location(self, s: Session, project_id: str, ingredient_id: str | None) -> None:
        if ingredient_id is None:
            return
        ing = s.get(Ingredient, ingredient_id)
        if ing is None or ing.project_id != project_id or ing.kind != "location":
            raise bad_request(
                "invalid_location", "location_ingredient_id must be a location in this project."
            )

    def _shot_fields(self, s: Session, project_id: str, data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        project_ings = None

        def ingredient_ids() -> set[str]:
            nonlocal project_ings
            if project_ings is None:
                project_ings = {
                    i.id for i in s.exec(select(Ingredient).where(Ingredient.project_id == project_id))
                }
            return project_ings

        for k, v in data.items():
            if k == "ingredient_ids":
                bad = [i for i in v if i not in ingredient_ids()]
                if bad:
                    raise bad_request("invalid_ingredient", f"Unknown ingredient id(s): {', '.join(bad)}")
                out["ingredient_ids_json"] = list(dict.fromkeys(v))
            elif k == "dialogue":
                speakers = [d["character_ingredient_id"] for d in v if d.get("character_ingredient_id")]
                bad = [i for i in speakers if i not in ingredient_ids()]
                if bad:
                    raise bad_request("invalid_ingredient", f"Unknown speaker id(s): {', '.join(bad)}")
                out["dialogue_json"] = v
            elif k == "narration":
                out["narration"] = (v or "").strip()
            elif k in ("selected_keyframe_asset_id", "selected_end_keyframe_asset_id"):
                if v is not None:
                    store.check_image(s, project_id, v)
                out[k] = v
            elif k == "selected_video_asset_id":
                if v is not None:
                    a = s.get(Asset, v)
                    if a is None or a.project_id != project_id or a.type != "video":
                        raise bad_request(
                            "invalid_asset", "selected_video_asset_id must be a video in this project."
                        )
                out[k] = v
            elif k == "selected_audio_asset_ids":
                v = list(dict.fromkeys(v or []))
                for asset_id in v:
                    a = s.get(Asset, asset_id)
                    if a is None or a.project_id != project_id or a.type != "audio":
                        raise bad_request(
                            "invalid_asset", "selected_audio_asset_ids must be audio in this project."
                        )
                out["selected_audio_asset_ids_json"] = v
            elif k == "reference_asset_ids":
                if v is not None:
                    v = list(dict.fromkeys(v))
                    for asset_id in v:
                        store.check_image(s, project_id, asset_id)
                out["reference_asset_ids_json"] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def _add_mentioned(s: Session, project_id: str, shot: Shot) -> None:
        """`@key` mentions in a shot's prompts add those ingredients to the shot (Section 11.7.3). Unknown
        keys are left as text here; generation endpoints reject them with a 422."""
        keys = {m.key for f in _PROMPT_FIELDS for m in mentions.parse(getattr(shot, f) or "")}
        if not keys:
            return
        rows = s.exec(
            select(Ingredient).where(
                Ingredient.project_id == project_id,
                col(Ingredient.key).in_(keys),
                Ingredient.draft == False,  # noqa: E712
                Ingredient.kind != "style",
            )
        ).all()
        missing = [i.id for i in rows if i.id not in shot.ingredient_ids_json]
        if missing:
            shot.ingredient_ids_json = [*shot.ingredient_ids_json, *missing]
