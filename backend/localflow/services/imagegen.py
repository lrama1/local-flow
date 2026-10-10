"""Ingredient images and keyframe composition (Sections 11.1-11.2).

Both build ordinary `t2i` / `image_edit` jobs with a `target_ref`, so candidates are plain assets that can be
listed with `GET /projects/{id}/assets?ingredient_id=` or `?shot_id=&slot=`."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import select

from localflow.db import Database
from localflow.errors import bad_request, not_found
from localflow.models import Asset, Ingredient, Job, Project, Scene, Shot
from localflow.schemas.characters import ReferencesOut, RefOption, RefOptionImage, RefOut
from localflow.schemas.core import JobOut
from localflow.schemas.generate import ImageEditBody, T2IBody
from localflow.services.characters import mentions, store
from localflow.services.characters.mentions import Mentionable
from localflow.services.characters.references import (
    MAX_REFERENCES,
    VIEW_LABELS,
    RefImage,
    select_references,
)
from localflow.services.events import EventBus
from localflow.services.jobs.generation import GenerationService
from localflow.services.registry.service import RegistryService
from localflow.worker.adapters.base import LoraRef

log = logging.getLogger("localflow.imagegen")

STYLE_KEY = "style"
_KIND_SUFFIX = {
    "character": "character reference portrait, head and shoulders, facing the camera, neutral expression, "
    "plain light grey studio background, soft even lighting",
    "location": "establishing shot of the location, wide angle, no people",
    "prop": "single object, centered product shot, plain background",
    "style": "style reference frame",
}
_REF_ORDER = {"character": 0, "prop": 1, "location": 2}
KEYFRAME_SIZES = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "21:9": (1344, 576),
}


class ImageGenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0, le=50)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    loras: list[LoraRef] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class IngredientGenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=4, ge=1, le=8)
    model_entry_id: str | None = None
    params: ImageGenParams = Field(default_factory=ImageGenParams)


class KeyframeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    which: Literal["start", "end"] = "start"
    count: int = Field(default=2, ge=1, le=8)
    mode: Literal["auto", "edit", "t2i"] = Field(
        default="auto",
        description="auto = compose from ingredient references when an edit model is installed",
    )
    model_entry_id: str | None = None
    params: ImageGenParams = Field(default_factory=ImageGenParams)


class KeyframeOut(BaseModel):
    job: JobOut
    mode: Literal["edit", "t2i"]
    prompt: str
    reference_asset_ids: list[str]
    reference_names: list[str]
    references: list[RefOut] = Field(default_factory=list)
    note: str | None = None


@dataclass
class ShotContext:
    shot: Shot
    scene: Scene
    project: Project
    rows: dict[str, Ingredient]  # the project's saved (non-draft) ingredients
    views: dict[str, dict[str, str]]
    lookup: dict[str, Mentionable]


def keyframe_size(aspect_ratio: str) -> tuple[int, int]:
    if aspect_ratio in KEYFRAME_SIZES:
        return KEYFRAME_SIZES[aspect_ratio]
    try:
        w, h = (float(x) for x in aspect_ratio.split(":"))
    except ValueError:
        return KEYFRAME_SIZES["16:9"]
    area = 1280 * 720
    width = math.sqrt(area * w / h)
    return round(width / 16) * 16, round(area / width / 16) * 16


def ingredient_prompt(ing: Ingredient, style: str, visual: str | None = None) -> str:
    """`visual` overrides the ingredient's visual prompt (e.g. with its @mentions resolved).

    A character's outfit goes into its portrait too, so the primary image (which every view is edited
    from) already wears what the full-body views will show."""
    base = (visual if visual is not None else ing.visual_prompt) or ing.description or ing.name
    outfit = (ing.body_outfit_prompt or "").strip().rstrip(".")
    lead = worn = ""
    if ing.kind == "character" and outfit:
        # The framing leads, so a long outfit list (down to the boots) doesn't become a full-body shot.
        lead, worn = "Close-up head and shoulders portrait", f"wearing {outfit}"
    parts = [lead, base.rstrip(". "), worn, _KIND_SUFFIX.get(ing.kind, ""), style.rstrip(". ")]
    return ". ".join(p for p in parts if p) + "."


def _positions(n: int) -> list[str]:
    return {1: ["reference"], 2: ["left", "right"], 3: ["left", "middle", "right"]}.get(
        n, [f"#{i + 1}" for i in range(n)]
    )


def edit_prompt(base: str, names: list[str], style: str, people: list[str] | None = None) -> str:
    """Kontext-style instruction for composing a frame from a reference image/sheet (DECISIONS M3).

    `names` label each reference; `people[i]` is who reference i shows. Several views of one character
    (full body + portrait) are one person, and must not come out as twins (DECISIONS M3.5)."""
    people = people or names
    if len(names) == 1:
        keep = f"Keep {names[0]}'s identity, face, hair and outfit exactly as in the reference image."
    else:
        listed = ", ".join(f"{n} ({p})" for n, p in zip(names, _positions(len(names)), strict=True))
        distinct = list(dict.fromkeys(people))
        same = "".join(
            f" The images of {p} are different views of the same character; show {p} only once."
            for p in distinct
            if people.count(p) > 1
        )
        if len(distinct) == 1:
            keep = (
                f"The reference sheet shows {listed}.{same} Keep {distinct[0]}'s identity, face, hair and "
                "outfit exactly as shown."
            )
        else:
            keep = (
                f"The reference sheet shows {listed}.{same} Keep each one's identity, face, hair and outfit "
                "exactly as shown, and place them together in one scene."
            )
    tail = " Create a single new cinematic frame, not a collage."
    style_part = f" Style: {style.rstrip('. ')}." if style else ""
    return f"{base.rstrip('. ')}. {keep}{tail}{style_part}"


def t2i_keyframe_prompt(base: str, ingredients: list[Ingredient], style: str) -> str:
    parts = [base.rstrip(". ")]
    for i in ingredients:
        if i.visual_prompt:
            parts.append(f"{i.name}: {i.visual_prompt.rstrip('. ')}")
    if style:
        parts.append(style.rstrip(". "))
    return ". ".join(parts)[:1500] + "."


class ImageGenService:
    def __init__(
        self, db: Database, generation: GenerationService, registry: RegistryService, bus: EventBus
    ) -> None:
        self.db = db
        self.generation = generation
        self.registry = registry
        self.bus = bus

    def _style(self, project_id: str) -> str:
        with self.db.session() as s:
            st = s.exec(
                select(Ingredient).where(Ingredient.project_id == project_id, Ingredient.key == STYLE_KEY)
            ).first()
        return (st.visual_prompt if st else "") or ""

    # --- ingredients (Section 11.1) ----------------------------------------------------------------
    def generate_ingredient(self, ingredient_id: str, body: IngredientGenerateBody) -> Job:
        with self.db.session() as s:
            ing = s.get(Ingredient, ingredient_id)
            if ing is None:
                raise not_found("Ingredient", ingredient_id)
            lookup = store.project_mentionables(s, ing.project_id, exclude_id=ing.id)
        resolved = mentions.require_known(mentions.resolve(ing.visual_prompt, lookup, "full"))
        style = "" if ing.kind == "style" else self._style(ing.project_id)
        p = body.params
        loras = list(p.loras) or [LoraRef(model_entry_id=i) for i in ing.lora_ids_json]
        prompt = ingredient_prompt(ing, style, resolved.text)
        req = T2IBody(
            project_id=ing.project_id,
            prompt=prompt,
            prompt_raw=ingredient_prompt(ing, style) if resolved.changed else prompt,
            mentions=resolved.keys,
            negative_prompt=ing.negative_prompt,
            width=p.width or 1024,
            height=p.height or 1024,
            num_images=body.count,
            model_entry_id=body.model_entry_id,
            steps=p.steps,
            guidance_scale=p.guidance_scale,
            seed=p.seed,
            loras=loras,
            extra=p.extra,
            target_ref={"kind": "ingredient", "ingredient_id": ing.id},
        )
        return self.generation.submit("t2i", req)

    # --- keyframes (Section 11.2) ------------------------------------------------------------------
    def shot_context(self, shot_id: str) -> ShotContext:
        return self._shot_context(shot_id)

    def style(self, project_id: str) -> str:
        return self._style(project_id)

    def _shot_context(self, shot_id: str) -> ShotContext:
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                raise not_found("Shot", shot_id)
            scene = s.get(Scene, shot.scene_id)
            assert scene is not None
            project = s.get(Project, scene.project_id)
            assert project is not None
            rows = {
                i.id: i
                for i in s.exec(
                    select(Ingredient).where(
                        Ingredient.project_id == project.id,
                        Ingredient.draft == False,  # noqa: E712
                    )
                ).all()
            }
            views = store.views_for(s, rows)
            lookup = store.project_mentionables(s, project.id)
        return ShotContext(shot, scene, project, rows, views, lookup)

    def _shot_ingredients(self, ctx: ShotContext, ids: list[str]) -> list[Ingredient]:
        ids = list(ids)
        loc = ctx.scene.location_ingredient_id
        if loc and loc not in ids:
            ids.append(loc)
        return [ctx.rows[i] for i in dict.fromkeys(ids) if i in ctx.rows and ctx.rows[i].kind != "style"]

    def _references(self, ctx: ShotContext, ingredients: list[Ingredient], shot_type: str) -> list[RefImage]:
        """The shot's override if set (Section 11.7 "The user can override references"), else by shot type."""
        override = ctx.shot.reference_asset_ids_json
        if not override:
            return select_references(shot_type, ingredients, ctx.views)
        owner: dict[str, tuple[Ingredient, str]] = {}
        for ing in ctx.rows.values():
            for view, asset_id in store.all_views(ing, ctx.views.get(ing.id, {})).items():
                owner.setdefault(asset_id, (ing, view))
        counts: dict[str, int] = {}
        for asset_id in override:
            if asset_id in owner:
                counts[owner[asset_id][0].id] = counts.get(owner[asset_id][0].id, 0) + 1
        out = []
        for n, asset_id in enumerate(override[:MAX_REFERENCES], 1):
            if asset_id in owner:
                ing, view = owner[asset_id]
                label = f"{ing.name} ({VIEW_LABELS.get(view, view)})" if counts[ing.id] > 1 else ing.name
                out.append(RefImage(asset_id, ing.id, ing.name, ing.kind, view, label))
            else:
                out.append(RefImage(asset_id, None, f"reference {n}", "other", "other", f"reference {n}"))
        return out

    def references_for(
        self, shot_id: str, shot_type: str | None = None, ingredient_ids: list[str] | None = None
    ) -> ReferencesOut:
        """What a keyframe would use, for the shot panel (which may pass unsaved type/ingredients)."""
        ctx = self._shot_context(shot_id)
        ids = list(ctx.shot.ingredient_ids_json if ingredient_ids is None else ingredient_ids)
        for f in (ctx.shot.keyframe_prompt, ctx.shot.end_keyframe_prompt or ""):
            ids += mentions.resolve(f, ctx.lookup).ingredient_ids
        ingredients = self._shot_ingredients(ctx, ids)
        stype = shot_type or ctx.shot.shot_type
        auto = select_references(stype, ingredients, ctx.views)
        selected = self._references(ctx, ingredients, stype)
        options = [
            RefOption(
                ingredient_id=i.id,
                name=i.name,
                kind=i.kind,
                images=[
                    RefOptionImage(asset_id=a, view=v)
                    for v, a in store.all_views(i, ctx.views.get(i.id, {})).items()
                ],
            )
            for i in ingredients
        ]
        return ReferencesOut(
            automatic=[RefOut(**r.__dict__) for r in auto],
            selected=[RefOut(**r.__dict__) for r in selected],
            overridden=bool(ctx.shot.reference_asset_ids_json),
            max=MAX_REFERENCES,
            options=[o for o in options if o.images],
        )

    def compose_keyframe(
        self, shot_id: str, body: KeyframeBody, *, chain: dict[str, Any] | None = None
    ) -> KeyframeOut:
        ctx = self._shot_context(shot_id)
        shot, project = ctx.shot, ctx.project
        slot = "keyframe" if body.which == "start" else "end_keyframe"
        raw = (shot.end_keyframe_prompt if body.which == "end" else shot.keyframe_prompt) or shot.video_prompt
        if not raw.strip():
            raise bad_request("empty_prompt", "The shot has no keyframe prompt yet. Write one first.")
        resolved = mentions.require_known(mentions.resolve(raw, ctx.lookup, "full"))
        base = resolved.text
        ingredients = self._shot_ingredients(ctx, [*shot.ingredient_ids_json, *resolved.ingredient_ids])
        style = self._style(project.id)
        refs = self._references(ctx, ingredients, shot.shot_type)
        p = body.params
        width, height = keyframe_size(project.aspect_ratio)
        width, height = p.width or width, p.height or height
        target = {"kind": "shot", "shot_id": shot.id, "slot": slot}
        note: str | None = None
        use_edit = body.mode != "t2i" and bool(refs)
        edit_model_id = body.model_entry_id
        if use_edit and body.model_entry_id is None:
            usable = self.generation.model_adapters("image_edit")
            edit_model = self.registry.default_entry(
                "image_edit", adapter_key=None, usable=usable, preferred=[project.default_image_model_id]
            )
            if edit_model is None:
                if body.mode == "edit":
                    raise bad_request("no_edit_model", "No image-edit model is installed (Models → Catalog).")
                use_edit = False
                note = (
                    "No image-edit model is installed, so the keyframe uses text-to-image without references."
                )
            else:
                edit_model_id = edit_model.id
        elif body.mode == "edit" and not refs:
            raise bad_request(
                "no_references", "None of the shot's ingredients has a primary image to compose from."
            )
        elif not refs and body.mode == "auto" and ingredients:
            note = "The shot's ingredients have no primary images yet, so the keyframe uses text-to-image."
        if use_edit:
            names = [r.label for r in refs]
            prompt = edit_prompt(base, names, style, [r.name for r in refs])
            req: T2IBody | ImageEditBody = ImageEditBody(
                project_id=project.id,
                prompt=prompt,
                prompt_raw=raw,
                mentions=resolved.keys,
                reference_asset_ids=[r.asset_id for r in refs],
                width=width,
                height=height,
                num_images=body.count,
                model_entry_id=edit_model_id,
                steps=p.steps,
                guidance_scale=p.guidance_scale,
                seed=p.seed,
                loras=list(p.loras),
                extra=p.extra,
                target_ref=target,
            )
            job = self.generation.submit("image_edit", req, chain=chain)
            return KeyframeOut(
                job=JobOut.of(job),
                mode="edit",
                prompt=prompt,
                reference_asset_ids=[r.asset_id for r in refs],
                reference_names=names,
                references=[RefOut(**r.__dict__) for r in refs],
                note=note,
            )
        described = [i for i in ingredients if i.id not in resolved.ingredient_ids]  # mentions already did
        prompt = t2i_keyframe_prompt(base, described, style)
        t2i_model = body.model_entry_id
        if t2i_model and "t2i" not in (self.registry.get(t2i_model).capabilities_json or []):
            t2i_model = None  # the chosen model was an edit model; use the default text-to-image model
        req = T2IBody(
            project_id=project.id,
            prompt=prompt,
            prompt_raw=raw,
            mentions=resolved.keys,
            negative_prompt=shot.negative_prompt,
            width=width,
            height=height,
            num_images=body.count,
            model_entry_id=t2i_model,
            steps=p.steps,
            guidance_scale=p.guidance_scale,
            seed=p.seed,
            loras=list(p.loras),
            extra=p.extra,
            target_ref=target,
        )
        job = self.generation.submit("t2i", req, chain=chain)
        return KeyframeOut(
            job=JobOut.of(job),
            mode="t2i",
            prompt=prompt,
            reference_asset_ids=[],
            reference_names=[],
            note=note,
        )

    # --- linking outputs (Section 14.2) ---------------------------------------------------------------
    def on_job_assets(self, job: Job, assets: list[Asset]) -> None:
        """Called after a job's outputs are ingested; links them to the job's target_ref."""
        ref = job.target_ref_json or {}
        images = [a for a in assets if a.type == "image"]
        if ref.get("kind") == "ingredient" and images:
            view = str(ref.get("view") or "primary")
            with self.db.session() as s:
                ing = s.get(Ingredient, ref.get("ingredient_id"))
                if ing is None:
                    return
                current = store.all_views(ing, store.views_for(s, [ing.id]).get(ing.id, {}))
                if view not in current:
                    # Convenience: the first candidate is chosen until the user picks another.
                    store.set_view(s, ing, view, images[0].id)
                    s.commit()
                project_id = ing.project_id
            self.bus.publish("ingredients.updated", project_id=project_id)
        elif ref.get("kind") == "shot" and job.project_id:
            self.bus.publish("board.updated", project_id=job.project_id)
