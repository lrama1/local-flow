"""Story pipeline (Section 8.3): refine → bible → breakdown, with revisions, dry runs, and prompt rewrites."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import col, func, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, bad_request, conflict, not_found
from localflow.models import Ingredient, Project, Scene, Shot, Story, Voice, utcnow
from localflow.schemas.story import (
    BibleBody,
    Breakdown,
    BreakdownBody,
    CastMember,
    RefineBody,
    RefinedStory,
    RenderConstraints,
    RevisionSummary,
    RewriteBody,
    ShotPrompts,
    ShotRewriteResult,
    StoryBible,
    StoryOptions,
    StoryOut,
    StoryPatch,
    StoryStepResult,
    UsageOut,
)
from localflow.services.events import EventBus
from localflow.services.llm.service import LLMService
from localflow.services.llm.templates import render
from localflow.services.llm.types import StructuredResult
from localflow.services.story.rules import (
    BreakdownContext,
    autofix_breakdown,
    bible_errors,
    breakdown_errors,
    breakdown_warnings,
)
from localflow.services.voice_store import voices_by_id

log = logging.getLogger("localflow.story")

STYLE_KEY = "style"
PROMPT_STYLE_GUIDE = (
    "Describe subject, action, setting, and lighting in 1-3 plain sentences. Avoid lists and on-screen text."
)


class StoryService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        llm: LLMService,
        bus: EventBus,
        video_profile: Callable[[str | None], tuple[str, dict[str, Any]] | None] | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.llm = llm
        self.bus = bus
        # project.default_video_model_id -> (model name, render profile) from the registry (Section 8.2)
        self.video_profile = video_profile or (lambda _id: None)

    # --- reads -----------------------------------------------------------------------------------
    def _project(self, project_id: str) -> Project:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None:
                raise not_found("Project", project_id)
            return p

    def latest(self, project_id: str) -> Story | None:
        with self.db.session() as s:
            return s.exec(
                select(Story)
                .where(Story.project_id == project_id)
                .order_by(col(Story.revision).desc())
                .limit(1)
            ).first()

    def revisions(self, project_id: str) -> list[RevisionSummary]:
        self._project(project_id)
        with self.db.session() as s:
            rows = s.exec(
                select(Story).where(Story.project_id == project_id).order_by(col(Story.revision).desc())
            ).all()
        return [
            RevisionSummary(
                revision=r.revision,
                step=r.step,
                status=r.status,
                llm_model=r.llm_model,
                title=(r.refined_json or {}).get("title"),
                created_at=r.created_at,
            )
            for r in rows
        ]

    def get_revision(self, project_id: str, revision: int) -> Story:
        with self.db.session() as s:
            row = s.exec(
                select(Story).where(Story.project_id == project_id, Story.revision == revision)
            ).first()
        if row is None:
            raise not_found("Story revision", str(revision))
        return row

    def to_out(self, story: Story | None, project_id: str) -> StoryOut:
        if story is None:
            return StoryOut(
                id=None,
                project_id=project_id,
                revision=None,
                step="none",
                status="draft",
                raw_idea="",
                options=StoryOptions(),
                cast_keys=[],
                refined=None,
                bible=None,
                has_breakdown=False,
                llm_provider=None,
                llm_model=None,
                template_versions={},
                edited_fields=[],
                warnings=[],
                usage=None,
                created_at=None,
            )
        return StoryOut(
            id=story.id if story.revision else None,
            project_id=story.project_id,
            revision=story.revision or None,
            step=story.step,
            status=story.status,
            raw_idea=story.raw_idea,
            options=StoryOptions.model_validate(story.options_json or {}),
            cast_keys=list(story.cast_keys_json),
            refined=RefinedStory.model_validate(story.refined_json) if story.refined_json else None,
            bible=StoryBible.model_validate(story.bible_json) if story.bible_json else None,
            has_breakdown=story.breakdown_json is not None,
            llm_provider=story.llm_provider,
            llm_model=story.llm_model,
            template_versions=dict(story.template_versions_json),
            edited_fields=list(story.edited_fields_json),
            warnings=list(story.warnings_json),
            usage=story.usage_json,
            created_at=story.created_at if story.revision else None,
        )

    # --- context blocks --------------------------------------------------------------------------
    def render_constraints(self, project: Project) -> RenderConstraints:
        # From the project's default video model when set (Section 8.2); otherwise the profile of the default
        # catalog model (Wan 2.2 TI2V-5B: 5 s clips, T2V + I2V, no audio, no FLF).
        profile = self.video_profile(project.default_video_model_id)
        if profile is not None:
            name, render = profile
            modes = [
                m for m in render.get("supported_modes") or ["t2v", "i2v"] if m in ("t2v", "i2v", "flf2v")
            ]
            return RenderConstraints(
                video_model=name,
                max_clip_seconds=float(
                    render.get("max_clip_seconds") or self.settings.default_max_clip_seconds
                ),
                fps=project.fps,
                supported_modes=modes or ["t2v"],
                supports_native_audio=bool(render.get("supports_native_audio")),
                supports_first_last_frame="flf2v" in (render.get("supported_modes") or []),
                prompt_style_guide=PROMPT_STYLE_GUIDE,
                aspect_ratio=project.aspect_ratio,
            )
        return RenderConstraints(
            video_model="default (Wan 2.2 TI2V-5B profile)",
            max_clip_seconds=self.settings.default_max_clip_seconds,
            fps=project.fps,
            supported_modes=["t2v", "i2v"],
            supports_native_audio=False,
            supports_first_last_frame=False,
            prompt_style_guide=PROMPT_STYLE_GUIDE,
            aspect_ratio=project.aspect_ratio,
        )

    def ingredients(self, project_id: str) -> list[Ingredient]:
        """Saved ingredients; character-wizard drafts never reach the story pipeline."""
        with self.db.session() as s:
            return list(
                s.exec(
                    select(Ingredient).where(
                        Ingredient.project_id == project_id,
                        Ingredient.draft == False,  # noqa: E712
                    )
                ).all()
            )

    def cast_block(self, ingredients: list[Ingredient], keys: list[str] | None = None) -> list[CastMember]:
        chars = [i for i in ingredients if i.kind == "character"]
        if keys is not None:
            by_key = {i.key: i for i in chars}
            missing = [k for k in keys if k not in by_key]
            if missing:
                raise bad_request("unknown_cast", f"Unknown character key(s): {', '.join(missing)}")
            chars = [by_key[k] for k in keys]
        with self.db.session() as s:
            voices = voices_by_id(s, {i.voice_id for i in chars if i.voice_id})
        return [
            CastMember(
                key=i.key,
                name=i.name,
                visual_prompt=i.visual_prompt,
                character_info=i.character_info,
                voice=_voice(i, voices.get(i.voice_id or "")),
            )
            for i in chars
        ]

    # --- steps -----------------------------------------------------------------------------------
    async def refine(self, project_id: str, body: RefineBody) -> StoryStepResult:
        project = self._project(project_id)
        base = self.latest(project_id)
        must_appear = self.cast_block(self.ingredients(project_id), body.cast_keys) if body.cast_keys else []
        system, user, version = render(
            "refine",
            {
                "rc": self.render_constraints(project),
                "options": body.options,
                "raw_idea": body.raw_idea,
                "must_appear": [c.model_dump(exclude_none=True) for c in must_appear],
                "schema": RefinedStory.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="refine",
            project_id=project_id,
            choice=body.llm,
            system=system,
            user=user,
            schema=RefinedStory,
        )
        warnings = []
        if base and base.bible_json:
            warnings.append("The story changed: rebuild the bible and the shot breakdown to match.")
        changes: dict[str, Any] = {
            "raw_idea": body.raw_idea,
            "options_json": body.options.model_dump(),
            "cast_keys_json": body.cast_keys,
            "refined_json": res.value.model_dump(),
            "refined_story": res.value.story,
            "llm_provider": res.provider,
            "llm_model": res.model,
            "usage_json": _usage_dict(res),
            "warnings_json": warnings,
        }
        return self._finish_step(
            project_id, base, "refine", version, changes, res, warnings, body.dry_run, "refined"
        )

    async def bible(self, project_id: str, body: BibleBody) -> StoryStepResult:
        self._project(project_id)
        base = self.latest(project_id)
        if base is None or not base.refined_json:
            raise conflict("story_not_refined", "Refine the story before building the bible.")
        ingredients = self.ingredients(project_id)
        existing_cast = [
            c.model_dump(exclude_none=True)
            for c in self.cast_block(
                [i for i in ingredients if i.kind == "character" and (i.origin != "bible" or i.locked)]
            )
        ]
        locked_keys = sorted(i.key for i in ingredients if i.locked and i.kind in ("location", "prop"))
        system, user, version = render(
            "bible",
            {
                "story": base.refined_json,
                "existing_cast": existing_cast,
                "locked_keys": locked_keys,
                "schema": StoryBible.model_json_schema(),
            },
        )
        cast_keys = [c["key"] for c in existing_cast]
        res = await self.llm.structured(
            purpose="bible",
            project_id=project_id,
            choice=body.llm,
            system=system,
            user=user,
            schema=StoryBible,
            validate=lambda b: bible_errors(b, cast_keys),
        )
        bible = _dedupe_bible_keys(res.value)
        warnings = [f"Note: {e}" for e in res.remaining_issues if "missing" in e]
        if base.breakdown_json:
            warnings.append("The bible changed: re-run the breakdown so shots use the new elements.")
        changes = {
            "bible_json": bible.model_dump(),
            "llm_provider": res.provider,
            "llm_model": res.model,
            "usage_json": _usage_dict(res),
            "warnings_json": warnings,
        }
        result = self._finish_step(
            project_id, base, "bible", version, changes, res, warnings, body.dry_run, "bible"
        )
        if not body.dry_run:
            result.ingredients = self.upsert_ingredients(project_id, bible)
            self.bus.publish("ingredients.updated", project_id=project_id)
        return result

    async def breakdown(self, project_id: str, body: BreakdownBody) -> StoryStepResult:
        project = self._project(project_id)
        base = self.latest(project_id)
        if base is None or not base.refined_json:
            raise conflict("story_not_refined", "Refine the story before breaking it down.")
        if not base.bible_json:
            raise conflict("bible_missing", "Build the bible before breaking the story down into shots.")
        if self._scene_count(project_id) and not body.replace_existing:
            raise conflict(
                "shots_exist", "This project already has scenes and shots. Confirm to replace them."
            )
        refined = RefinedStory.model_validate(base.refined_json)
        options = StoryOptions.model_validate(base.options_json or {})
        target = body.target_length_seconds or options.target_length_seconds or refined.estimated_runtime_s
        rc = self.render_constraints(project)
        ingredients = self.ingredients(project_id)
        bible = StoryBible.model_validate(base.bible_json)
        ctx = BreakdownContext(
            rc=rc,
            character_keys={i.key for i in ingredients if i.kind == "character"},
            location_keys={i.key for i in ingredients if i.kind == "location"},
            other_keys={i.key for i in ingredients if i.kind in ("prop", "style")},
            target_length_s=float(target),
        )
        system, user, version = render(
            "breakdown",
            {
                "rc": rc,
                "allow_extend": ctx.allow_extend,
                "target_length": target,
                "shot_estimate": max(1, round(target / (rc.max_clip_seconds * 0.85))),
                "story": refined,
                "bible": _bible_for_prompt(bible, ingredients),
                "cast": [c.model_dump(exclude_none=True) for c in self.cast_block(ingredients)],
                "schema": Breakdown.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="breakdown",
            project_id=project_id,
            choice=body.llm,
            system=system,
            user=user,
            schema=Breakdown,
            validate=lambda bd: breakdown_errors(bd, ctx),
        )
        bd = res.value
        fixes: list[str] = []
        if res.remaining_issues:
            bd, fixes = autofix_breakdown(bd, ctx)
        warnings = breakdown_warnings(bd, ctx) + [f"Auto-fixed: {f}" for f in fixes]
        changes = {
            "options_json": {**options.model_dump(), "target_length_seconds": target},
            "breakdown_json": bd.model_dump(),
            "llm_provider": res.provider,
            "llm_model": res.model,
            "usage_json": _usage_dict(res),
            "warnings_json": warnings,
        }
        result = self._finish_step(
            project_id, base, "breakdown", version, changes, res, warnings, False, None
        )
        assert result.story.id is not None
        scenes, shots = self.apply_breakdown(project_id, result.story.id, bd)
        result.scenes, result.shots = scenes, shots
        self.bus.publish("board.updated", project_id=project_id)
        return result

    def patch(self, project_id: str, patch: StoryPatch) -> StoryStepResult:
        self._project(project_id)
        base = self.latest(project_id)
        changes: dict[str, Any] = {}
        edited = set(base.edited_fields_json if base else [])
        if patch.raw_idea is not None:
            changes["raw_idea"] = patch.raw_idea
        if patch.options is not None:
            changes["options_json"] = patch.options.model_dump()
        if patch.cast_keys is not None:
            self.cast_block(self.ingredients(project_id), patch.cast_keys)  # validates keys
            changes["cast_keys_json"] = patch.cast_keys
        if patch.refined is not None:
            changes["refined_json"] = patch.refined.model_dump()
            changes["refined_story"] = patch.refined.story
            (edited.discard if patch.applied_from == "refine" else edited.add)("refined")
        if patch.bible is not None:
            if base is None or not (base.refined_json or patch.refined):
                raise conflict("story_not_refined", "Refine the story before editing the bible.")
            bible = _dedupe_bible_keys(patch.bible)
            changes["bible_json"] = bible.model_dump()
            (edited.discard if patch.applied_from == "bible" else edited.add)("bible")
        if not changes:
            raise bad_request("empty_patch", "Nothing to change.")
        step = patch.applied_from or "edit"
        if patch.applied_from and patch.llm:
            changes |= {"llm_provider": patch.llm.provider, "llm_model": patch.llm.model}
        changes["edited_fields_json"] = sorted(edited)
        changes["warnings_json"] = []
        story = self._new_revision(project_id, base, step, changes)
        counts = None
        if patch.bible is not None:
            counts = self.upsert_ingredients(project_id, StoryBible.model_validate(story.bible_json))
            self.bus.publish("ingredients.updated", project_id=project_id)
        self.bus.publish("story.updated", project_id=project_id, revision=story.revision)
        return StoryStepResult(
            story=self.to_out(story, project_id), saved=True, usage=None, warnings=[], ingredients=counts
        )

    def restore(self, project_id: str, revision: int) -> StoryOut:
        old = self.get_revision(project_id, revision)
        base = self.latest(project_id)
        changes = {
            "raw_idea": old.raw_idea,
            "options_json": old.options_json,
            "cast_keys_json": old.cast_keys_json,
            "refined_json": old.refined_json,
            "refined_story": old.refined_story,
            "bible_json": old.bible_json,
            "llm_provider": old.llm_provider,
            "llm_model": old.llm_model,
            "edited_fields_json": old.edited_fields_json,
            "warnings_json": [f"Restored from revision {revision}. Scenes and shots were not changed."],
        }
        story = self._new_revision(project_id, base, "restore", changes)
        if old.bible_json:
            self.upsert_ingredients(project_id, StoryBible.model_validate(old.bible_json))
            self.bus.publish("ingredients.updated", project_id=project_id)
        self.bus.publish("story.updated", project_id=project_id, revision=story.revision)
        return self.to_out(story, project_id)

    async def rewrite_shot(self, shot_id: str, body: RewriteBody) -> ShotRewriteResult:
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                raise not_found("Shot", shot_id)
            scene = s.get(Scene, shot.scene_id)
            assert scene is not None
        project = self._project(scene.project_id)
        story = self.latest(project.id)
        ingredients = self.ingredients(project.id)
        by_id = {i.id: i for i in ingredients}
        in_shot = [by_id[i] for i in shot.ingredient_ids_json if i in by_id]
        location = by_id.get(scene.location_ingredient_id or "")
        style = (story.bible_json or {}).get("style", {}) if story else {}
        rc = self.render_constraints(project)
        shot_ctx = {
            "duration_s": shot.duration_s,
            "shot_type": shot.shot_type,
            "camera_move": shot.camera_move,
            "generation_mode": shot.generation_mode,
            "keyframe_prompt": shot.keyframe_prompt,
            "end_keyframe_prompt": shot.end_keyframe_prompt,
            "video_prompt": shot.video_prompt,
            "negative_prompt": shot.negative_prompt,
            "visible": [{"key": i.key, "name": i.name, "visual_prompt": i.visual_prompt} for i in in_shot],
        }
        scene_ctx = {
            "title": scene.title,
            "summary": scene.summary,
            "location": {"name": location.name, "visual_prompt": location.visual_prompt}
            if location
            else None,
        }
        cast = self.cast_block([i for i in in_shot if i.kind == "character"])
        system, user, _ = render(
            "prompt_rewrite",
            {
                "rc": rc,
                "instruction": body.instruction,
                "shot": shot_ctx,
                "scene": scene_ctx,
                "style": style,
                "cast": [c.model_dump(exclude_none=True) for c in cast if c.character_info],
                "schema": ShotPrompts.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="prompt_rewrite",
            project_id=project.id,
            choice=body.llm,
            system=system,
            user=user,
            schema=ShotPrompts,
            temperature=0.5,
        )
        proposal = res.value
        if shot.generation_mode != "flf2v":
            proposal.end_keyframe_prompt = None
        return ShotRewriteResult(shot_id=shot_id, proposal=proposal, usage=_usage_out(res))

    # --- writes ----------------------------------------------------------------------------------
    def _finish_step(
        self,
        project_id: str,
        base: Story | None,
        step: str,
        version: str,
        changes: dict[str, Any],
        res: StructuredResult[Any],
        warnings: list[str],
        dry_run: bool,
        edited_field: str | None,
    ) -> StoryStepResult:
        edited = set(base.edited_fields_json if base else [])
        if edited_field:
            edited.discard(edited_field)
        changes = {**changes, "edited_fields_json": sorted(edited)}
        changes["template_versions_json"] = {**(base.template_versions_json if base else {}), step: version}
        if dry_run:
            preview = self._build_revision(project_id, base, step, changes, revision=0)
            return StoryStepResult(
                story=self.to_out(preview, project_id), saved=False, usage=_usage_out(res), warnings=warnings
            )
        story = self._new_revision(project_id, base, step, changes)
        self.bus.publish("story.updated", project_id=project_id, revision=story.revision)
        return StoryStepResult(
            story=self.to_out(story, project_id), saved=True, usage=_usage_out(res), warnings=warnings
        )

    def _build_revision(
        self, project_id: str, base: Story | None, step: str, changes: dict[str, Any], revision: int
    ) -> Story:
        carried = {}
        if base is not None:
            carried = base.model_dump(
                exclude={"id", "project_id", "revision", "step", "status", "created_at", "updated_at"}
            )
            carried["usage_json"] = None
        data = {**carried, **changes}
        story = Story(project_id=project_id, revision=revision, step=step, **data)
        story.status = "broken_down" if story.breakdown_json else "refined" if story.refined_json else "draft"
        if not story.refined_story and story.refined_json:
            story.refined_story = story.refined_json.get("story", "")
        return story

    def _new_revision(self, project_id: str, base: Story | None, step: str, changes: dict[str, Any]) -> Story:
        for _ in range(3):
            with self.db.session() as s:
                top = (
                    s.exec(select(func.max(Story.revision)).where(Story.project_id == project_id)).one() or 0
                )
                story = self._build_revision(project_id, base, step, changes, int(top) + 1)
                s.add(story)
                try:
                    s.commit()
                except IntegrityError:  # concurrent step wrote the same revision number
                    s.rollback()
                    continue
                s.refresh(story)
                return story
        raise AppError(409, "revision_conflict", "Another change was saved at the same time; try again.")

    def upsert_ingredients(self, project_id: str, bible: StoryBible) -> dict[str, int]:
        """Create/update ingredients from the bible by key; never touch locked ones (Section 8.3)."""
        counts = {"created": 0, "updated": 0, "locked": 0, "removed": 0}
        items: list[tuple[str, str, str, str, str, str]] = []
        voices = {c.key: c.voice.model_dump() for c in bible.characters}
        for c in bible.characters:
            items.append(("character", c.key, c.name, c.description, c.visual_prompt, ""))
        for loc in bible.locations:
            items.append(("location", loc.key, loc.name, "", loc.visual_prompt, ""))
        for p in bible.props:
            items.append(("prop", p.key, p.name, "", p.visual_prompt, ""))
        st = bible.style
        style_prompt = ", ".join(x for x in (st.look, st.palette, st.lighting, st.lens, st.era) if x)
        if style_prompt or st.negative:
            items.append(("style", STYLE_KEY, "Visual style", "", style_prompt, st.negative))
        new_keys = {it[1] for it in items}
        with self.db.session() as s:
            existing = {
                i.key: i for i in s.exec(select(Ingredient).where(Ingredient.project_id == project_id))
            }
            for kind, key, name, description, visual, negative in items:
                ing = existing.get(key)
                if ing is None:
                    s.add(
                        Ingredient(
                            project_id=project_id,
                            kind=kind,
                            key=key,
                            name=name,
                            description=description,
                            visual_prompt=visual,
                            negative_prompt=negative,
                            origin="bible",
                            voice_hints_json=_hints(voices.get(key)),
                        )
                    )
                    counts["created"] += 1
                elif ing.locked or ing.draft or (ing.kind == "character" and ing.origin != "bible"):
                    # Section 8.5: user/library/imported characters keep their key and look as they are.
                    if ing.kind == "character" and not ing.voice_hints_json and _hints(voices.get(key)):
                        ing.voice_hints_json = _hints(voices.get(key))
                        s.add(ing)
                    counts["locked"] += 1
                else:
                    ing.kind, ing.name, ing.visual_prompt = kind, name, visual
                    ing.description = description or ing.description
                    ing.negative_prompt = negative or ing.negative_prompt
                    if kind == "character" and _hints(voices.get(key)):
                        ing.voice_hints_json = _hints(voices.get(key))
                    ing.updated_at = utcnow()
                    s.add(ing)
                    counts["updated"] += 1
            for ing in existing.values():
                if (
                    ing.origin == "bible"
                    and not ing.locked
                    and ing.key not in new_keys
                    and not ing.primary_asset_id
                ):
                    s.delete(ing)
                    counts["removed"] += 1
            s.commit()
        return counts

    def _scene_count(self, project_id: str) -> int:
        with self.db.session() as s:
            return int(
                s.exec(select(func.count()).select_from(Scene).where(Scene.project_id == project_id)).one()
            )

    def apply_breakdown(self, project_id: str, story_id: str, bd: Breakdown) -> tuple[int, int]:
        ingredients = {i.key: i for i in self.ingredients(project_id)}
        shots_total = 0
        with self.db.session() as s:
            for old in s.exec(select(Scene).where(Scene.project_id == project_id)).all():
                s.delete(old)  # shots cascade (FK ON DELETE CASCADE)
            s.flush()
            for si, spec in enumerate(bd.scenes):
                loc = ingredients.get(spec.location_key or "")
                scene = Scene(
                    project_id=project_id,
                    story_id=story_id,
                    index=si,
                    title=spec.title,
                    summary=spec.summary,
                    location_ingredient_id=loc.id if loc and loc.kind == "location" else None,
                )
                s.add(scene)
                s.flush()
                for hi, sh in enumerate(spec.shots):
                    s.add(
                        Shot(
                            scene_id=scene.id,
                            index=hi,
                            duration_s=sh.duration_s,
                            shot_type=sh.shot_type,
                            camera_move=sh.camera_move,
                            video_prompt=sh.video_prompt,
                            negative_prompt=sh.negative_prompt,
                            keyframe_prompt=sh.keyframe_prompt,
                            end_keyframe_prompt=sh.end_keyframe_prompt,
                            generation_mode=sh.generation_mode,
                            ingredient_ids_json=[
                                ingredients[k].id for k in sh.ingredient_keys if k in ingredients
                            ],
                            dialogue_json=[
                                {
                                    "character_ingredient_id": ingredients[d.character_key].id
                                    if d.character_key in ingredients
                                    else None,
                                    "line": d.line,
                                    "emotion": d.emotion,
                                }
                                for d in sh.dialogue
                            ],
                            sfx_prompt=sh.sfx_prompt,
                            music_cue=sh.music_cue,
                            needs_extend=sh.needs_extend,
                            extend_segments=sh.extend_segments,
                            continuity_notes=sh.continuity_notes,
                        )
                    )
                    shots_total += 1
            s.commit()
        return len(bd.scenes), shots_total


# --- helpers -----------------------------------------------------------------------------------
def _hints(voice: dict[str, str] | None) -> dict[str, str]:
    return {k: v for k, v in (voice or {}).items() if v}


def _voice(ing: Ingredient, voice: Voice | None = None) -> dict[str, str] | None:
    """Cast-block voice (Section 8.5): the assigned voice's name and description, else the voice hints."""
    hints = _hints(ing.voice_hints_json)
    if voice is not None:
        traits = voice.description or str((voice.meta_json or {}).get("traits") or "")
        parts = [p for p in (voice.gender, traits) if p]
        return {"name": voice.name, "description": ", ".join(parts) or ", ".join(hints.values())}
    if not hints:
        return None
    description = ", ".join(hints[k] for k in ("gender", "age", "tone") if k in hints)
    return {"name": "", "description": description}


def _usage_dict(res: StructuredResult[Any]) -> dict[str, Any]:
    return _usage_out(res).model_dump()


def _usage_out(res: StructuredResult[Any]) -> UsageOut:
    return UsageOut(
        provider=res.provider,
        model=res.model,
        prompt_tokens=res.usage.prompt_tokens,
        completion_tokens=res.usage.completion_tokens,
        cost_usd=res.usage.cost_usd,
        attempts=res.attempts,
    )


usage_out = _usage_out  # for the other LLM steps (narration)


def _dedupe_bible_keys(bible: StoryBible) -> StoryBible:
    out = bible.model_copy(deep=True)
    seen: set[str] = {STYLE_KEY}
    for item in (*out.characters, *out.locations, *out.props):
        key, n = item.key, 2
        while key in seen:
            key = f"{item.key}-{n}"
            n += 1
        item.key = key
        seen.add(key)
    return out


def _bible_for_prompt(bible: StoryBible, ingredients: list[Ingredient]) -> dict[str, Any]:
    """The bible as the LLM should see it now: style from the bible, elements from current ingredient rows
    (so edits in the Ingredients tab, and user-created characters, are respected)."""

    def rows(kind: str) -> list[dict[str, str]]:
        return [
            {"key": i.key, "name": i.name, "visual_prompt": i.visual_prompt}
            for i in sorted(ingredients, key=lambda x: x.created_at)
            if i.kind == kind
        ]

    return {
        "style": bible.style.model_dump(),
        "characters": rows("character"),
        "locations": rows("location"),
        "props": rows("prop"),
    }
