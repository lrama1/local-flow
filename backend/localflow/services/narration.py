"""Narration (Section 12.3): a film-level voice-over whose text lives on the shots (written by the LLM or by
hand) and whose voice is made per shot with the project's narrator voice, then placed on the Timeline's
Narration track by the shot audio machinery (`AudioGenService.on_job_assets`).

A line is *stale* when its selected take voiced different text than the shot says now. Placement, overflow
into the next line and following the picture are the timeline's rules (`timeline.model`)."""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Any

from sqlmodel import col, select

from localflow.db import Database
from localflow.errors import AppError, bad_request, conflict, not_found
from localflow.models import Asset, Ingredient, Project, Scene, Shot
from localflow.schemas.board import ShotUpdate
from localflow.schemas.core import JobOut
from localflow.schemas.generate import TTSBody
from localflow.schemas.narration import (
    NarratedLine,
    NarrationGenerateBody,
    NarrationGenerateOut,
    NarrationLines,
    NarrationPlanOut,
    NarrationShotOut,
    NarrationStatus,
    NarrationWriteBody,
    NarrationWriteOut,
    NarratorOut,
    SkippedLine,
)
from localflow.schemas.timeline import TimelineData
from localflow.services.audiogen import audio_slot, estimate_line_s
from localflow.services.board import BoardService
from localflow.services.jobs.generation import FAKE, GenerationService
from localflow.services.llm.service import LLMService
from localflow.services.llm.templates import render
from localflow.services.registry.service import RegistryService
from localflow.services.story.service import StoryService, usage_out
from localflow.services.timeline.model import NARRATION_LEAD_S, NARRATION_TRACK, duration_s
from localflow.services.timeline.service import TimelineService
from localflow.services.voices import VoiceService

log = logging.getLogger("localflow.narration")

WORDS_PER_S = 2.3  # what the LLM is asked for; speech runs a little faster (audiogen.WORDS_PER_S)
TONE_HINT = "calm, warm, clear narrator"
_QUOTES = re.compile(r"^[\"'\u201c\u201d\u2018\u2019]+|[\"'\u201c\u201d\u2018\u2019]+$")


def budget_words(duration_s_: float) -> int:
    """About 2.3 words per second of the shot, less a beat of silence."""
    return max(0, math.floor((duration_s_ - 0.6) * WORDS_PER_S))


def clean_line(text: str) -> str:
    """The LLM's line without wrapping quotes or stray whitespace."""
    return _QUOTES.sub("", " ".join(text.split())).strip()


def narration_text(asset: Asset | None) -> str:
    """The text a narration take voiced (recorded in the job's request)."""
    return str(((asset.generation_json or {}).get("params") or {}).get("text") or "").strip() if asset else ""


class NarrationService:
    def __init__(
        self,
        db: Database,
        llm: LLMService,
        story: StoryService,
        board: BoardService,
        voices: VoiceService,
        generation: GenerationService,
        registry: RegistryService,
        timeline: TimelineService,
    ) -> None:
        self.db = db
        self.llm = llm
        self.story = story
        self.board = board
        self.voices = voices
        self.generation = generation
        self.registry = registry
        self.timeline = timeline

    # --- reads -------------------------------------------------------------------------------------
    def _project(self, project_id: str) -> Project:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None:
                raise not_found("Project", project_id)
            return p

    def _ordered(self, project_id: str) -> list[tuple[Scene, Shot, str]]:
        """(scene, shot, "scene.shot" number) in film order."""
        with self.db.session() as s:
            scenes = s.exec(
                select(Scene).where(Scene.project_id == project_id).order_by(col(Scene.index))
            ).all()
            out: list[tuple[Scene, Shot, str]] = []
            for si, sc in enumerate(scenes):
                shots = s.exec(select(Shot).where(Shot.scene_id == sc.id).order_by(col(Shot.index))).all()
                out.extend((sc, sh, f"{si + 1}.{hi + 1}") for hi, sh in enumerate(shots))
        return out

    def narrator(self, project: Project) -> tuple[NarratorOut, str | None]:
        """The project's narrator voice, else a suggested one (calm and warm), else none."""
        if project.narrator_voice_id:
            try:
                v = self.voices.get(project.narrator_voice_id)
            except AppError:
                v = None
            if v is not None and v.available:
                return NarratorOut(voice_id=v.id, voice_name=v.name, source="project"), v.id
        sug = self.voices.suggest({"tone": TONE_HINT})
        if sug.voice is not None:
            return NarratorOut(
                voice_id=sug.voice.id, voice_name=sug.voice.name, source="suggested"
            ), sug.voice.id
        return NarratorOut(voice_id=None, voice_name=None, source="none"), None

    def _takes(self, shots: list[Shot]) -> dict[str, Asset]:
        """Each shot's selected narration take."""
        ids = [a for sh in shots for a in sh.selected_audio_asset_ids_json]
        if not ids:
            return {}
        with self.db.session() as s:
            assets = {a.id: a for a in s.exec(select(Asset).where(col(Asset.id).in_(ids)))}
        out: dict[str, Asset] = {}
        for sh in shots:
            for aid in sh.selected_audio_asset_ids_json:
                a = assets.get(aid)
                if a is not None and audio_slot(a) == "narration":
                    out[sh.id] = a
        return out

    def plan(self, project_id: str, speed: float = 1.0, tts_entry_id: str | None = None) -> NarrationPlanOut:
        project = self._project(project_id)
        ordered = self._ordered(project_id)
        takes = self._takes([sh for _, sh, _ in ordered])
        tl: TimelineData = self.timeline.load(project_id)
        clip_starts = {c.shot_id: c.start_s for c in tl.video.clips if c.shot_id}
        placed = {c.shot_id: c for t in tl.audio if t.id == NARRATION_TRACK for c in t.clips if c.shot_id}
        narrator, voice_id = self.narrator(project)
        notes: list[str] = []
        if narrator.source == "suggested":
            notes.append(f"No narrator voice is chosen; using {narrator.voice_name} (suggested).")
        elif narrator.source == "none":
            notes.append("No text-to-speech model is installed. Install Kokoro from the Models page.")

        rows: list[NarrationShotOut] = []
        for sc, sh, number in ordered:
            text = (sh.narration or "").strip()
            take = takes.get(sh.id)
            status: NarrationStatus = "empty"
            if text and take is None:
                status = "text"
            elif text:
                status = "voiced" if narration_text(take) == text else "stale"
            estimated = round(estimate_line_s(text, speed), 2) if text else 0.0
            clip_start = clip_starts.get(sh.id)
            clip = placed.get(sh.id)
            start = (
                clip.start_s
                if clip
                else (round(clip_start + NARRATION_LEAD_S, 3) if clip_start is not None else None)
            )
            rows.append(
                NarrationShotOut(
                    shot_id=sh.id,
                    number=number,
                    scene_title=sc.title,
                    duration_s=sh.duration_s,
                    text=text,
                    words=len(text.split()),
                    budget_words=budget_words(sh.duration_s),
                    estimated_s=estimated,
                    status=status,
                    selected_asset_id=take.id if take else None,
                    asset_duration_s=take.duration_s if take else None,
                    clip_start_s=clip_start,
                    start_s=start,
                    overflow_s=0.0,
                )
            )
        # Overflow: how far a line runs past the start of the next narrated line (or the film's end).
        total = duration_s(tl)
        spoken = [r for r in rows if r.text and r.start_s is not None]
        spoken.sort(key=lambda r: r.start_s or 0.0)
        for i, r in enumerate(spoken):
            limit = spoken[i + 1].start_s if i + 1 < len(spoken) else (total or None)
            length = r.asset_duration_s if r.status == "voiced" and r.asset_duration_s else r.estimated_s
            if limit is not None and r.start_s is not None:
                r.overflow_s = round(max(0.0, r.start_s + length - limit), 2)
        over = [r.number for r in rows if r.overflow_s > 0.05]
        if over:
            notes.append(
                f"Shot{'s' if len(over) > 1 else ''} {', '.join(over)} run into the next narrated line: "
                "shorten the line, speed it up, or lengthen the shot."
            )
        model = None
        if voice_id is not None:
            adapter, _, _, _ = self.voices.native(voice_id)
            entry = (
                self.registry.require_usable(tts_entry_id, "tts")
                if tts_entry_id
                else self.registry.default_entry(
                    "tts", adapter_key=adapter, usable=self.generation.model_adapters("tts"), preferred=[]
                )
            )
            model = entry.name if entry else ("Fake engine" if adapter == FAKE else None)
        return NarrationPlanOut(
            narrator=narrator,
            tts_model=model,
            speed=speed,
            shots=rows,
            total_words=sum(r.words for r in rows),
            notes=notes,
        )

    # --- writing (LLM) -----------------------------------------------------------------------------
    async def write(self, project_id: str, body: NarrationWriteBody) -> NarrationWriteOut:
        project = self._project(project_id)
        ordered = self._ordered(project_id)
        if not ordered:
            raise bad_request("no_shots", "Break the story into shots first (or add shots by hand).")
        wanted = set(body.shot_ids) if body.shot_ids else None
        if wanted is not None:
            known = {sh.id for _, sh, _ in ordered}
            if wanted - known:
                raise bad_request("invalid_shot", "shot_ids must be shots of this project.")
        targets = {
            sh.id
            for _, sh, _ in ordered
            if (wanted is None or sh.id in wanted) and (body.overwrite or not (sh.narration or "").strip())
        }
        if not targets:
            raise bad_request(
                "nothing_to_write",
                "Every shot already has narration. Turn on 'rewrite' to replace it, or clear a line first.",
            )
        story = self.story.latest(project_id)
        refined = (story.refined_json or {}) if story else {}
        with self.db.session() as s:
            ings = list(s.exec(select(Ingredient).where(Ingredient.project_id == project_id)))
        by_id = {i.id: i for i in ings}
        cast = [
            {"name": i.name, "description": (i.description or i.visual_prompt or "")[:200]}
            for i in ings
            if i.kind == "character" and not i.draft
        ]
        shots_ctx: list[dict[str, Any]] = []
        for n, (sc, sh, _) in enumerate(ordered, start=1):
            dialogue = [
                {
                    "speaker": (
                        by_id[d["character_ingredient_id"]].name
                        if d.get("character_ingredient_id") in by_id
                        else "Narrator"
                    ),
                    "line": d.get("line", ""),
                }
                for d in sh.dialogue_json
                if str(d.get("line") or "").strip()
            ]
            shots_ctx.append(
                {
                    "n": n,
                    "scene": sc.title,
                    "scene_summary": sc.summary,
                    "duration_s": sh.duration_s,
                    "action": (sh.video_prompt or sh.keyframe_prompt or "")[:300],
                    "dialogue": dialogue,
                    "narration": (sh.narration or "").strip() if sh.id not in targets else "",
                    "write": sh.id in targets,
                    "budget_words": budget_words(sh.duration_s),
                }
            )
        system, user, _ = render(
            "narration",
            {
                "style": (body.style or "").strip(),
                "tone": refined.get("tone", ""),
                "story": {k: refined[k] for k in ("title", "logline", "synopsis", "story") if refined.get(k)}
                or None,
                "cast": cast,
                "shots": shots_ctx,
                "schema": NarrationLines.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="narration",
            project_id=project.id,
            choice=body.llm,
            system=system,
            user=user,
            schema=NarrationLines,
            temperature=0.7,
        )
        by_n = {n: (sc, sh, number) for n, (sc, sh, number) in enumerate(ordered, start=1)}
        updated: list[NarratedLine] = []
        warnings: list[str] = []
        for line in res.value.lines:
            hit = by_n.get(line.n)
            if hit is None or hit[1].id not in targets:
                continue
            _, sh, number = hit
            text = clean_line(line.text)
            if not text:
                continue
            words, budget = len(text.split()), budget_words(sh.duration_s)
            if words > budget + max(2, budget // 3):
                warnings.append(
                    f"Shot {number}: {words} words for a {sh.duration_s:g} s shot (budget {budget})."
                )
            self.board.update_shot(sh.id, ShotUpdate(narration=text))
            updated.append(
                NarratedLine(shot_id=sh.id, number=number, text=text, words=words, budget_words=budget)
            )
            targets.discard(sh.id)
        return NarrationWriteOut(
            updated=updated,
            skipped=len(ordered) - len(updated),
            usage=usage_out(res),
            warnings=warnings,
        )

    # --- voicing -----------------------------------------------------------------------------------
    def generate(self, project_id: str, body: NarrationGenerateBody) -> NarrationGenerateOut:
        project = self._project(project_id)
        plan = self.plan(project_id, body.speed, body.tts_model_entry_id)
        if plan.narrator.voice_id is None:
            raise conflict(
                "no_tts_model", "No text-to-speech model is installed. Install Kokoro from the Models page."
            )
        known = {r.shot_id for r in plan.shots}
        if body.shot_ids and set(body.shot_ids) - known:
            raise bad_request("invalid_shot", "shot_ids must be shots of this project.")
        wanted = set(body.shot_ids) if body.shot_ids else None
        jobs: list[JobOut] = []
        skipped: list[SkippedLine] = []
        if not any(r.text for r in plan.shots if wanted is None or r.shot_id in wanted):
            raise bad_request("no_narration", "No shot has narration yet. Write it or type it in first.")
        seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)
        for r in plan.shots:
            if wanted is not None and r.shot_id not in wanted:
                continue
            if not r.text:
                skipped.append(SkippedLine(shot_id=r.shot_id, reason="No narration text."))
                continue
            if body.only_missing and r.status == "voiced":
                skipped.append(SkippedLine(shot_id=r.shot_id, reason="Already voiced."))
                continue
            req = TTSBody(
                project_id=project.id,
                model_entry_id=body.tts_model_entry_id,
                text=r.text,
                voice_id=plan.narrator.voice_id,
                speed=body.speed,
                seed=seed,
                target_ref={"kind": "shot", "shot_id": r.shot_id, "slot": "narration"},
            )
            jobs.append(JobOut.of(self.generation.submit("tts", req)))
        return NarrationGenerateOut(jobs=jobs, skipped=skipped, plan=plan)
