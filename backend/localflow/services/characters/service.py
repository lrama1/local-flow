"""Character creator workflows (Section 11.7): drafts, AI expand/describe, extra views, mentions."""

from __future__ import annotations

import base64
import io
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from PIL import Image
from sqlmodel import select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, bad_request, conflict, not_found
from localflow.models import Ingredient, Job, Project, new_id, utcnow
from localflow.schemas.characters import (
    CharacterCapabilities,
    CharacterDescription,
    CharacterExpansion,
    CleanReferenceBody,
    DescribeBody,
    DescribeOut,
    DraftCreate,
    ExpandBody,
    ExpandOut,
    MentionOut,
    ResolveBody,
    ResolveOut,
    ViewsBody,
    ViewsOut,
)
from localflow.schemas.core import JobOut
from localflow.schemas.generate import ImageEditBody, T2IBody
from localflow.schemas.story import UsageOut
from localflow.services.assets import AssetService
from localflow.services.board import DRAFT_KEY_PREFIX, BoardService
from localflow.services.characters import mentions, store
from localflow.services.characters.mentions import MODE_FOR_KIND, Mentionable, Resolved
from localflow.services.characters.templates import TemplateService
from localflow.services.events import EventBus
from localflow.services.imagegen import STYLE_KEY
from localflow.services.jobs.generation import GenerationService
from localflow.services.llm.service import LLMService
from localflow.services.llm.templates import render
from localflow.services.llm.types import LLMError, StructuredResult
from localflow.services.projects import ProjectService
from localflow.services.registry.service import RegistryService

log = logging.getLogger("localflow.characters")

DRAFT_MAX_AGE = timedelta(hours=24)
_DESCRIBE_MAX_SIDE = 1024

# Kontext-style instructions for extra views (the primary image is the reference). {outfit} is the
# character's body/outfit prompt, or "the same outfit".
VIEW_INSTRUCTIONS: dict[str, str] = {
    "full_body": (
        "Show the same person from the reference in a full-body shot, head to toe, standing in a relaxed "
        "neutral pose and facing the camera, wearing {outfit}. Keep the face, hair, skin tone and build "
        "exactly the same. Plain light grey studio background, soft even lighting."
    ),
    "three_view": (
        "Create a character turnaround reference sheet of the same person from the reference with exactly "
        "three full-body figures side by side: on the left facing the camera (front view), in the middle "
        "turned sideways in profile (side view), on the right facing away (back view). Only these three "
        "figures, standing in a neutral pose and wearing {outfit}. Keep the face, hair and build exactly the "
        "same in every view. Plain light grey studio background, even lighting."
    ),
    "side": (
        "Show the same person from the reference in a full-body side profile view, facing left, standing "
        "straight and wearing {outfit}. Keep the face, hair and build exactly the same. Plain light grey "
        "studio background, soft even lighting."
    ),
    "back": (
        "Show the same person from the reference seen from behind, a full-body back view, standing straight "
        "and wearing {outfit}. Keep the hair and build exactly the same. Plain light grey studio background, "
        "soft even lighting."
    ),
    "face_closeup": (
        "Show an extreme close-up portrait of the same person's face from the reference, looking at the "
        "camera with a neutral expression, the face filling the frame. Keep every facial feature exactly the "
        "same. Soft even lighting, plain background."
    ),
}
# Text-to-image fallbacks (no edit model): the same framing, described from the prompts.
VIEW_T2I_SUFFIX: dict[str, str] = {
    "full_body": "full-body shot, head to toe, standing in a relaxed neutral pose, facing the camera",
    "three_view": (
        "character turnaround reference sheet with exactly three full-body figures side by side: front view, "
        "side profile view, back view, neutral pose"
    ),
    "side": "full-body side profile view facing left, standing straight",
    "back": "full-body back view, seen from behind, standing straight",
    "face_closeup": "extreme close-up portrait of the face, neutral expression, looking at the camera",
}
# Output size per view (Kontext's preferred resolutions around 1 MP).
VIEW_SIZES: dict[str, tuple[int, int]] = {
    "full_body": (832, 1248),
    "three_view": (1392, 752),  # wider sheets invite extra figures
    "side": (832, 1248),
    "back": (832, 1248),
    "face_closeup": (1024, 1024),
}
CLEAN_INSTRUCTION = (
    "Turn this picture into a clean character reference portrait of the same person: head and shoulders, "
    "facing the camera, neutral expression, plain light grey studio background, soft even lighting. Keep the "
    "face, hair, skin tone and features exactly the same."
)


class CharacterService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        bus: EventBus,
        assets: AssetService,
        board: BoardService,
        generation: GenerationService,
        registry: RegistryService,
        llm: LLMService,
        projects: ProjectService,
        templates: TemplateService,
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.assets = assets
        self.board = board
        self.generation = generation
        self.registry = registry
        self.llm = llm
        self.projects = projects
        self.templates = templates

    # --- drafts (wizard) -------------------------------------------------------------------------
    def create_draft(self, project_id: str, body: DraftCreate) -> Ingredient:
        self.projects.get(project_id)
        f = body.fields
        tpl = self.templates.find(body.template_key) if body.template_key else None

        def pick(name: str, default: str = "") -> str:
            value = getattr(f, name, None) if f else None
            if value is not None:
                return str(value)
            return str(getattr(tpl, name, default) or default) if tpl else default

        hints = f.voice_hints.model_dump() if f and f.voice_hints else dict(tpl.voice_hints if tpl else {})
        ing = Ingredient(
            project_id=project_id,
            kind="character",
            key=f"{DRAFT_KEY_PREFIX}{new_id()[-10:].lower()}",
            name=pick("name", "New character"),
            visual_prompt=pick("visual_prompt"),
            negative_prompt=pick("negative_prompt"),
            character_info=pick("character_info") or None,
            body_outfit_prompt=pick("body_outfit_prompt") or None,
            voice_hints_json=hints,
            template_key=tpl.key if tpl else None,
            origin="user",
            locked=True,
            draft=True,
        )
        with self.db.session() as s:
            s.add(ing)
            s.commit()
            s.refresh(ing)
        return ing

    def delete_ingredient(self, ingredient_id: str) -> None:
        """Deleting a draft (cancelling the wizard) also deletes the images generated for it."""
        ing = self.board.get_ingredient(ingredient_id)
        if ing.draft:
            rows, _ = self.assets.list_assets(ing.project_id, target={"ingredient_id": ing.id}, limit=10_000)
            for a in rows:
                self.assets.delete(a.id)
        self.board.delete_ingredient(ingredient_id)

    def purge_stale_drafts(self) -> int:
        """Abandoned wizard drafts are removed at startup after a day."""
        cutoff = utcnow() - DRAFT_MAX_AGE
        with self.db.session() as s:
            stale = [
                i.id
                for i in s.exec(select(Ingredient).where(Ingredient.draft == True)).all()  # noqa: E712
                if i.updated_at < cutoff
            ]
        for ingredient_id in stale:
            try:
                self.delete_ingredient(ingredient_id)
            except AppError:
                continue
        if stale:
            log.info("removed %d abandoned character draft(s)", len(stale))
        return len(stale)

    # --- AI helpers ------------------------------------------------------------------------------
    async def expand(self, body: ExpandBody, project_id: str | None = None) -> ExpandOut:
        tpl = self.templates.find(body.template_key) if body.template_key else None
        system, user, _ = render(
            "character_expand",
            {
                "description": body.description,
                "template": tpl.model_dump(exclude={"id", "thumbnail_url", "builtin"}) if tpl else None,
                "schema": CharacterExpansion.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="character_expand",
            project_id=project_id,
            choice=body.llm,
            system=system,
            user=user,
            schema=CharacterExpansion,
        )
        v = res.value
        return ExpandOut(
            name=v.name,
            visual_prompt=v.visual_prompt,
            body_outfit_prompt=v.body_outfit_prompt,
            negative_prompt=v.negative_prompt,
            character_info=v.character_info,
            voice_hints=v.voice.model_dump(),
            usage=_usage_out(res),
        )

    async def describe(self, body: DescribeBody) -> DescribeOut:
        images, project_id = [], None
        for asset_id in body.asset_ids:
            a = self.assets.get(asset_id)
            if a.type != "image":
                raise bad_request("wrong_asset_type", f"Asset '{asset_id}' isn't an image.")
            images.append(_data_url(self.assets.file_path(a)))
            project_id = a.project_id
        system, user, _ = render(
            "character_describe",
            {
                "image_count": len(images),
                "name": body.name,
                "schema": CharacterDescription.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="character_describe",
            project_id=project_id,
            choice=body.llm,
            system=system,
            user=user,
            schema=CharacterDescription,
            images=images,
            temperature=0.3,
        )
        return DescribeOut(
            visual_prompt=res.value.visual_prompt,
            body_outfit_prompt=res.value.body_outfit_prompt,
            usage=_usage_out(res),
        )

    def capabilities(self, project_id: str | None = None) -> CharacterCapabilities:
        expand_ok, expand_reason = True, None
        try:
            self.llm.resolve("character_expand")
        except (AppError, LLMError) as e:
            expand_ok, expand_reason = False, e.message
        describe_ok, describe_model, describe_reason = self.llm.vision("character_describe")
        edit = self._edit_model(project_id)
        return CharacterCapabilities(
            expand=expand_ok,
            expand_reason=expand_reason,
            describe=describe_ok,
            describe_model=describe_model,
            describe_reason=describe_reason,
            edit_model=edit is not None,
            edit_model_name=edit[1] if edit else None,
            voices=False,  # built-in and custom voices arrive with the TTS adapters (M6)
        )

    # --- views (Section 11.7 step 2) -------------------------------------------------------------
    def _edit_model(self, project_id: str | None, explicit: str | None = None) -> tuple[str, str] | None:
        """(model entry id, name) of the image-edit model to use, or None if none is installed."""
        if explicit:
            chosen = self.registry.get(explicit)
            return chosen.id, chosen.name
        preferred: list[str | None] = []
        if project_id:
            with self.db.session() as s:
                p = s.get(Project, project_id)
                preferred = [p.default_image_model_id] if p else []
        usable = self.generation.model_adapters("image_edit")
        entry = self.registry.default_entry(
            "image_edit", adapter_key=None, usable=usable, preferred=preferred
        )
        return (entry.id, entry.name) if entry is not None else None

    def _style(self, project_id: str) -> str:
        with self.db.session() as s:
            st = s.exec(
                select(Ingredient).where(Ingredient.project_id == project_id, Ingredient.key == STYLE_KEY)
            ).first()
        return (st.visual_prompt if st else "") or ""

    def _character(self, ingredient_id: str) -> Ingredient:
        ing = self.board.get_ingredient(ingredient_id)
        if ing.kind != "character":
            raise bad_request("not_a_character", "Only characters have extra views.")
        return ing

    def set_view(self, ingredient_id: str, view: str, asset_id: str | None) -> Ingredient:
        with self.db.session() as s:
            ing = s.get(Ingredient, ingredient_id)
            if ing is None:
                raise not_found("Ingredient", ingredient_id)
            if view != "primary" and ing.kind != "character":
                raise bad_request("not_a_character", "Only characters have extra views.")
            store.set_view(s, ing, view, asset_id)
            s.commit()
            s.refresh(ing)
        self.bus.publish("ingredients.updated", project_id=ing.project_id)
        return ing

    def generate_views(self, ingredient_id: str, body: ViewsBody) -> ViewsOut:
        ing = self._character(ingredient_id)
        if not ing.primary_asset_id:
            raise conflict("no_primary", "Pick or upload the character's primary image first.")
        if body.body_outfit_prompt is not None and body.body_outfit_prompt != (ing.body_outfit_prompt or ""):
            with self.db.session() as s:
                row = s.get(Ingredient, ing.id)
                assert row is not None
                row.body_outfit_prompt = body.body_outfit_prompt or None
                row.updated_at = utcnow()
                s.add(row)
                s.commit()
                s.refresh(row)
                ing = row
        outfit = (ing.body_outfit_prompt or "").strip().rstrip(".") or "the same outfit as in the reference"
        style = self._style(ing.project_id).rstrip(". ")
        edit = self._edit_model(ing.project_id, body.model_entry_id)
        note = None
        if edit is None:
            note = (
                "No image-edit model is installed, so the views are generated from text and may not match "
                "the primary image closely (lower consistency). Install FLUX Kontext from Models → Catalog."
            )
        jobs: list[Job] = []
        views = list(dict.fromkeys(body.views))
        for view in views:
            width, height = VIEW_SIZES[view]
            target = {"kind": "ingredient", "ingredient_id": ing.id, "view": view}
            if edit is not None:
                prompt = VIEW_INSTRUCTIONS[view].format(outfit=outfit)
                if style:
                    prompt += f" Style: {style}."
                req: ImageEditBody | T2IBody = ImageEditBody(
                    project_id=ing.project_id,
                    prompt=prompt,
                    prompt_raw=prompt,
                    reference_asset_ids=[ing.primary_asset_id],
                    width=width,
                    height=height,
                    num_images=body.count,
                    model_entry_id=edit[0],
                    seed=body.seed,
                    target_ref=target,
                )
                jobs.append(self.generation.submit("image_edit", req))
            else:
                parts = [
                    (ing.visual_prompt or ing.name).rstrip(". "),
                    f"wearing {outfit}" if ing.body_outfit_prompt else "",
                    VIEW_T2I_SUFFIX[view],
                    "plain light grey studio background, soft even lighting",
                    style,
                ]
                prompt = ". ".join(p for p in parts if p) + "."
                req = T2IBody(
                    project_id=ing.project_id,
                    prompt=prompt,
                    prompt_raw=prompt,
                    negative_prompt=ing.negative_prompt,
                    width=width,
                    height=height,
                    num_images=body.count,
                    seed=body.seed,
                    target_ref=target,
                )
                jobs.append(self.generation.submit("t2i", req))
        self.bus.publish("ingredients.updated", project_id=ing.project_id)
        return ViewsOut(
            jobs=[JobOut.of(j) for j in jobs], views=views, mode="t2i" if edit is None else "edit", note=note
        )

    def clean_reference(self, ingredient_id: str, body: CleanReferenceBody) -> Job:
        """ImageEdit turns an uploaded photo into a neutral, plain-background primary reference."""
        ing = self._character(ingredient_id)
        asset = self.assets.get(body.asset_id)
        if asset.project_id != ing.project_id or asset.type != "image":
            raise bad_request("invalid_asset", "Pick an image of this project.")
        edit = self._edit_model(ing.project_id, body.model_entry_id)
        if edit is None:
            raise conflict(
                "no_edit_model", "Clean reference needs an image-edit model (install FLUX Kontext in Models)."
            )
        req = ImageEditBody(
            project_id=ing.project_id,
            prompt=CLEAN_INSTRUCTION,
            prompt_raw=CLEAN_INSTRUCTION,
            reference_asset_ids=[asset.id],
            width=1024,
            height=1024,
            num_images=body.count,
            model_entry_id=edit[0],
            target_ref={"kind": "ingredient", "ingredient_id": ing.id},
        )
        return self.generation.submit("image_edit", req)

    # --- mentions (Section 11.7.3) ---------------------------------------------------------------
    def _lookup(self, project_id: str | None) -> tuple[dict[str, Mentionable], bool]:
        """(lookup, from_library). The Playground (or no project) resolves against the Cast library."""
        playground = self.projects.ensure_playground().id
        with self.db.session() as s:
            if not project_id or project_id == playground:
                return store.library_mentionables(s), True
            if s.get(Project, project_id) is None:
                raise not_found("Project", project_id)
            return store.project_mentionables(s, project_id), False

    def resolve(self, body: ResolveBody) -> ResolveOut:
        lookup, _ = self._lookup(body.project_id)
        res = mentions.resolve(body.text, lookup, MODE_FOR_KIND[body.mode])
        found = []
        for m in mentions.parse(body.text):
            item = lookup.get(m.key)
            found.append(
                MentionOut(
                    key=m.key,
                    text=m.text,
                    start=m.start,
                    end=m.end,
                    known=item is not None,
                    name=item.name if item else None,
                    kind=item.kind if item else None,
                )
            )
        return ResolveOut(
            resolved_text=res.text,
            ingredient_ids=res.ingredient_ids,
            mentions=found,
            errors=[f"Unknown mention {m.text}" for m in res.unknown],
        )

    def resolve_for_generation(self, kind: str, project_id: str, text: str) -> Resolved | None:
        """`GenerationService` hook for prompts sent to `POST /generate/{kind}` (Playground, API)."""
        mode = MODE_FOR_KIND.get(kind)
        if mode is None:
            return None
        lookup, _ = self._lookup(project_id)
        res = mentions.require_known(mentions.resolve(text, lookup, mode))
        return res if res.changed else None


def _data_url(path: Path) -> str:
    """A downscaled JPEG data URL (vision models don't need full resolution; it also caps cost)."""
    with Image.open(path) as im:
        img = im.convert("RGB")
        img.thumbnail((_DESCRIBE_MAX_SIDE, _DESCRIBE_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _usage_out(res: StructuredResult[Any]) -> UsageOut:
    return UsageOut(
        provider=res.provider,
        model=res.model,
        prompt_tokens=res.usage.prompt_tokens,
        completion_tokens=res.usage.completion_tokens,
        cost_usd=res.usage.cost_usd,
        attempts=res.attempts,
    )
