"""Shot video generation (Sections 11.2-11.6): image-to-video from the selected keyframe, first/last frame,
text-to-video, extend, variations, and "generate all".

Takes are ordinary video jobs with `target_ref = {kind: shot, shot_id, slot: "video"}`, so they're listed with
`GET /projects/{id}/assets?shot_id=&slot=video`. Follow-up work (extend the new take, queue the video once an
auto-approved keyframe exists) is stored as `params.chain` on the job and run by `on_job_assets`."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import col, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, bad_request, not_found
from localflow.models import Asset, Job, ModelEntry, Project, Scene, Shot
from localflow.schemas.board import ShotUpdate
from localflow.schemas.core import JobOut
from localflow.schemas.generate import ExtendBody, FLF2VBody, I2VBody, T2VBody
from localflow.services.board import BoardService
from localflow.services.characters import mentions
from localflow.services.events import EventBus
from localflow.services.imagegen import ImageGenService, KeyframeBody, ShotContext, keyframe_size
from localflow.services.jobs.generation import GenerationService
from localflow.services.registry.service import RegistryService
from localflow.worker.adapters.base import LoraRef

log = logging.getLogger("localflow.videogen")

VideoKind = Literal["t2v", "i2v", "flf2v"]
VideoMode = Literal["auto", "t2v", "i2v", "flf2v"]
CameraIntensity = Literal["subtle", "normal", "strong"]

# --- camera presets (Section 11.5) -----------------------------------------------------------------
# Wan responds well to plain cinematography language; `{adv}` carries the intensity slider.
CAMERA_FRAGMENTS: dict[str, str] = {
    "static": "Static camera, a locked-off shot with no camera movement",
    "pan_left": "The camera pans {adv} to the left",
    "pan_right": "The camera pans {adv} to the right",
    "tilt_up": "The camera tilts {adv} upward",
    "tilt_down": "The camera tilts {adv} downward",
    "dolly_in": "The camera dollies in {adv} toward the subject",
    "dolly_out": "The camera dollies out {adv}, pulling away from the subject",
    "truck_left": "The camera trucks {adv} to the left, moving sideways alongside the subject",
    "truck_right": "The camera trucks {adv} to the right, moving sideways alongside the subject",
    "orbit": "The camera orbits {adv} around the subject",
    "handheld": "Handheld camera with {shake} shake",
    "crane_up": "The camera cranes up {adv}, rising above the scene",
    "crane_down": "The camera cranes down {adv}, descending toward the subject",
    "zoom_in": "The camera zooms in {adv} on the subject",
    "zoom_out": "The camera zooms out {adv} to reveal the scene",
}
_ADVERBS = {"subtle": "slowly and gently", "normal": "smoothly", "strong": "quickly and dramatically"}
_SHAKE = {"subtle": "slight, gentle", "normal": "natural", "strong": "strong, energetic"}
# A safeguard (DECISIONS M4): one 720p "static camera" take drifted into a handheld follow shot.
CAMERA_NEGATIVES: dict[str, str] = {"static": "camera movement, panning, zooming, camera shake"}
CONTINUE_SUFFIX = "The action continues seamlessly from the previous moment"


def camera_fragment(move: str, intensity: str = "normal") -> str:
    template = CAMERA_FRAGMENTS.get(move, "")
    return template.format(adv=_ADVERBS.get(intensity, "smoothly"), shake=_SHAKE.get(intensity, "natural"))


def video_negative(shot: Shot) -> str:
    """The shot's negative prompt plus what its camera move rules out."""
    parts = ((shot.negative_prompt or "").strip(), CAMERA_NEGATIVES.get(shot.camera_move, ""))
    return ", ".join(x for x in parts if x)


def snap_frames(frames: float, max_frames: int) -> int:
    """Wan needs 4k+1 frames (its VAE compresses time 4x); at least 5, at most the model's clip length."""
    top = 4 * ((max_frames - 1) // 4) + 1
    k = max(1, round((frames - 1) / 4))
    return max(5, min(4 * k + 1, top))


def video_size(defaults: dict[str, Any], profile: str, aspect_ratio: str) -> tuple[int, int]:
    """The model's native size for the project's aspect ratio and the VRAM profile (catalog `sizes`)."""
    sizes = defaults.get("sizes") or {}
    table = sizes.get(profile) or sizes.get("balanced") or {}
    if aspect_ratio in table:
        w, h = table[aspect_ratio]
        return int(w), int(h)
    ref = table.get("16:9") or [1280, 720]
    area = int(ref[0]) * int(ref[1])
    kw, kh = keyframe_size(aspect_ratio)
    width = math.sqrt(area * kw / kh)
    return round(width / 32) * 32, round(area / width / 32) * 32


# --- request/response bodies -------------------------------------------------------------------------
class VideoGenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0, le=50)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    duration_s: float | None = Field(default=None, gt=0, le=60, description="Defaults to the shot's duration")
    loras: list[LoraRef] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class VideoBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: VideoMode = Field(
        default="auto",
        description="auto = the shot's generation mode; text-to-video when its keyframes are missing",
    )
    model_entry_id: str | None = None
    variations: int = Field(default=1, ge=1, le=4, description="Takes with consecutive seeds")
    camera_intensity: CameraIntensity = "normal"
    extend_segments: int = Field(default=0, ge=0, le=4, description="Extend each new take this many times")
    params: VideoGenParams = Field(default_factory=VideoGenParams)


class ExtendShotBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str | None = Field(default=None, description="Defaults to the shot's selected take")
    segments: int = Field(default=1, ge=1, le=4)
    prompt: str | None = Field(default=None, description="Defaults to the shot's video prompt, continuing")
    model_entry_id: str | None = None
    camera_intensity: CameraIntensity = "normal"
    crossfade_frames: int = Field(default=0, ge=0, le=8)
    params: VideoGenParams = Field(default_factory=VideoGenParams)


class VideoPlanOut(BaseModel):
    mode: VideoKind
    prompt: str
    model_entry_id: str | None
    model_name: str | None
    width: int
    height: int
    fps: int
    num_frames: int
    duration_s: float
    max_clip_seconds: float
    note: str | None = None


class VideoOut(VideoPlanOut):
    jobs: list[JobOut]
    extend_segments: int = 0


class GenerateAllBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyframes: bool = True
    videos: bool = True
    auto_approve_keyframes: bool | None = Field(
        default=None,
        description="Use the first keyframe and queue the video right away (default: config.toml)",
    )
    extend: bool = Field(default=True, description="Extend shots the breakdown marked as needing it")
    camera_intensity: CameraIntensity = "normal"


class SkippedShot(BaseModel):
    shot_id: str
    reason: str


class GenerateAllOut(BaseModel):
    keyframe_jobs: int
    video_jobs: int
    skipped: list[SkippedShot]


@dataclass
class _Plan:
    kind: VideoKind
    entry: ModelEntry | None
    width: int
    height: int
    fps: int
    num_frames: int
    max_frames: int
    inputs: dict[str, str]  # request field -> asset id
    note: str | None


class VideoGenService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        generation: GenerationService,
        registry: RegistryService,
        imagegen: ImageGenService,
        board: BoardService,
        bus: EventBus,
    ) -> None:
        self.settings = settings
        self.db = db
        self.generation = generation
        self.registry = registry
        self.imagegen = imagegen
        self.board = board
        self.bus = bus

    # --- planning ----------------------------------------------------------------------------------
    def _mode(self, shot: Shot, mode: VideoMode) -> tuple[VideoKind, dict[str, str], str | None]:
        start, end = shot.selected_keyframe_asset_id, shot.selected_end_keyframe_asset_id
        if mode == "t2v":
            return "t2v", {}, None
        if mode == "i2v":
            if not start:
                raise bad_request(
                    "no_keyframe", "Select a keyframe first (Keyframe section), or use text-to-video."
                )
            return "i2v", {"image_asset_id": start}, None
        if mode == "flf2v":
            if not (start and end):
                raise bad_request(
                    "no_keyframe", "First/last-frame video needs both a first and a last keyframe."
                )
            return "flf2v", {"first_asset_id": start, "last_asset_id": end}, None
        wanted = shot.generation_mode
        if wanted == "flf2v" and start and end:
            return "flf2v", {"first_asset_id": start, "last_asset_id": end}, None
        if wanted in ("i2v", "flf2v", "ref2v") and start:
            note = (
                "No last keyframe is selected, so this take animates the first keyframe."
                if wanted == "flf2v"
                else None
            )
            return "i2v", {"image_asset_id": start}, note
        if wanted == "t2v":
            return "t2v", {}, None
        return (
            "t2v",
            {},
            "No keyframe is selected, so this take is text-to-video (less consistent with the cast).",
        )

    def _model(self, kind: str, entry_id: str | None, project: Project) -> ModelEntry | None:
        if entry_id:
            return self.registry.require_usable(entry_id, kind)
        return self.registry.default_entry(
            kind,
            adapter_key=None,
            usable=self.generation.model_adapters(kind),
            preferred=[project.default_video_model_id],
        )

    def _plan(
        self,
        kind: VideoKind,
        entry: ModelEntry | None,
        project: Project,
        params: VideoGenParams,
        duration_s: float,
        inputs: dict[str, str],
        note: str | None,
    ) -> _Plan:
        defaults = dict(entry.default_params_json or {}) if entry else {}
        fps = int(defaults.get("fps") or project.fps)
        max_frames = int(
            defaults.get("max_frames") or round(self.settings.default_max_clip_seconds * fps) + 1
        )
        width, height = video_size(defaults, self.settings.vram_profile, project.aspect_ratio)
        width, height = params.width or width, params.height or height
        frames = snap_frames(duration_s * fps, max_frames)
        return _Plan(kind, entry, width, height, fps, frames, max_frames, inputs, note)

    def _prompt(
        self,
        ctx: ShotContext,
        kind: VideoKind,
        intensity: str,
        text: str | None = None,
        *,
        extend: bool = False,
    ) -> mentions.Resolved:
        raw = text if text is not None else (ctx.shot.video_prompt or ctx.shot.keyframe_prompt)
        if not raw.strip():
            raise bad_request("empty_prompt", "The shot has no video prompt yet. Write one first.")
        # The keyframe already shows who's who, so image-conditioned modes only need names (Section 11.7.3).
        resolved = mentions.require_known(
            mentions.resolve(raw, ctx.lookup, "full" if kind == "t2v" else "name")
        )
        parts = [resolved.text.rstrip(". ")]
        if kind == "t2v":
            for ing in self.imagegen._shot_ingredients(ctx, ctx.shot.ingredient_ids_json):
                if ing.id not in resolved.ingredient_ids and ing.visual_prompt and ing.kind != "location":
                    parts.append(f"{ing.name}: {ing.visual_prompt.rstrip('. ')}")
        if extend:
            parts.append(CONTINUE_SUFFIX)
        fragment = camera_fragment(ctx.shot.camera_move, intensity)
        if fragment:
            parts.append(fragment)
        if kind == "t2v":
            style = self.imagegen.style(ctx.project.id)
            if style:
                parts.append(style.rstrip(". "))
        resolved.text = ". ".join(p for p in parts if p)[:2000] + "."
        return resolved

    def plan(self, shot_id: str, body: VideoBody) -> VideoPlanOut:
        ctx = self.imagegen.shot_context(shot_id)
        plan, resolved = self._prepare(ctx, body)
        return self._plan_out(plan, resolved.text)

    def _prepare(self, ctx: ShotContext, body: VideoBody) -> tuple[_Plan, mentions.Resolved]:
        kind, inputs, note = self._mode(ctx.shot, body.mode)
        entry = self._model(kind, body.model_entry_id, ctx.project)
        duration = body.params.duration_s or ctx.shot.duration_s
        plan = self._plan(kind, entry, ctx.project, body.params, duration, inputs, note)
        clip_s = (plan.max_frames - 1) / plan.fps
        if ctx.shot.duration_s > clip_s + 0.05 and body.extend_segments == 0:
            longer = (
                f"This shot is {ctx.shot.duration_s:g} s, but the model makes clips of up to {clip_s:g} s: "
                "use Extend (or Generate + extend) for the rest."
            )
            plan.note = f"{plan.note} {longer}" if plan.note else longer
        return plan, self._prompt(ctx, kind, body.camera_intensity)

    def _plan_out(self, plan: _Plan, prompt: str) -> VideoPlanOut:
        return VideoPlanOut(
            mode=plan.kind,
            prompt=prompt,
            model_entry_id=plan.entry.id if plan.entry else None,
            model_name=plan.entry.name if plan.entry else None,
            width=plan.width,
            height=plan.height,
            fps=plan.fps,
            num_frames=plan.num_frames,
            duration_s=round(plan.num_frames / plan.fps, 3),
            max_clip_seconds=round((plan.max_frames - 1) / plan.fps, 3),
            note=plan.note,
        )

    # --- generation --------------------------------------------------------------------------------
    def generate(self, shot_id: str, body: VideoBody) -> VideoOut:
        ctx = self.imagegen.shot_context(shot_id)
        plan, resolved = self._prepare(ctx, body)
        p = body.params
        target = {"kind": "shot", "shot_id": ctx.shot.id, "slot": "video"}
        chain: dict[str, Any] | None = None
        if body.extend_segments:
            chain = {"extend": {"segments": body.extend_segments, "camera_intensity": body.camera_intensity}}
        seed = p.seed if p.seed is not None else random.randint(0, 2**32 - 1)
        common: dict[str, Any] = {
            "project_id": ctx.project.id,
            "prompt": resolved.text,
            "prompt_raw": resolved.raw,
            "mentions": resolved.keys,
            "negative_prompt": video_negative(ctx.shot),
            "width": plan.width,
            "height": plan.height,
            "num_frames": plan.num_frames,
            "fps": plan.fps,
            "model_entry_id": plan.entry.id if plan.entry else None,
            "steps": p.steps,
            "guidance_scale": p.guidance_scale,
            "loras": [lo.model_dump() for lo in p.loras],
            "extra": p.extra,
            "target_ref": target,
            **plan.inputs,
        }
        bodies: dict[str, Any] = {"t2v": T2VBody, "i2v": I2VBody, "flf2v": FLF2VBody}
        body_cls = bodies[plan.kind]
        jobs = []
        for i in range(body.variations):
            req = body_cls.model_validate({**common, "seed": (seed + i) % (2**32)})
            jobs.append(self.generation.submit(plan.kind, req, chain=chain))
        out = self._plan_out(plan, resolved.text)
        return VideoOut(
            **out.model_dump(), jobs=[JobOut.of(j) for j in jobs], extend_segments=body.extend_segments
        )

    def extend(self, shot_id: str, body: ExtendShotBody) -> VideoOut:
        ctx = self.imagegen.shot_context(shot_id)
        source_id = body.asset_id or ctx.shot.selected_video_asset_id
        if not source_id:
            raise bad_request("no_video", "Select a take to extend first.")
        with self.db.session() as s:
            source = s.get(Asset, source_id)
        if source is None or source.project_id != ctx.project.id or source.type != "video":
            raise bad_request("invalid_asset", "Only videos in this project can be extended.")
        entry = self._model("extend", body.model_entry_id, ctx.project)
        defaults = dict(entry.default_params_json or {}) if entry else {}
        fps = int(source.fps or defaults.get("fps") or ctx.project.fps)
        max_frames = int(
            defaults.get("max_frames") or round(self.settings.default_max_clip_seconds * fps) + 1
        )
        p = body.params
        seg_s = p.duration_s or (max_frames - 1) / fps
        frames = snap_frames(seg_s * fps, max_frames)
        width, height = video_size(defaults, self.settings.vram_profile, ctx.project.aspect_ratio)
        if source.width and source.height:
            width, height = source.width, source.height  # continuations must match the take
        resolved = self._prompt(ctx, "i2v", body.camera_intensity, body.prompt, extend=True)
        req = ExtendBody(
            project_id=ctx.project.id,
            prompt=resolved.text,
            prompt_raw=resolved.raw,
            mentions=resolved.keys,
            negative_prompt=video_negative(ctx.shot),
            video_asset_id=source.id,
            segments=body.segments,
            crossfade_frames=body.crossfade_frames,
            width=p.width or width,
            height=p.height or height,
            num_frames=frames,
            fps=fps,
            model_entry_id=entry.id if entry else None,
            seed=p.seed,
            steps=p.steps,
            guidance_scale=p.guidance_scale,
            loras=list(p.loras),
            extra=p.extra,
            target_ref={"kind": "shot", "shot_id": ctx.shot.id, "slot": "video"},
        )
        job = self.generation.submit("extend", req)
        total = (source.duration_s or 0) + body.segments * (frames - 1) / fps
        return VideoOut(
            mode="i2v",
            prompt=resolved.text,
            model_entry_id=entry.id if entry else None,
            model_name=entry.name if entry else None,
            width=req.width,
            height=req.height,
            fps=fps,
            num_frames=frames,
            duration_s=round(total, 3),
            max_clip_seconds=round((max_frames - 1) / fps, 3),
            jobs=[JobOut.of(job)],
            extend_segments=body.segments,
        )

    # --- generate all (Section 11.6) ---------------------------------------------------------------
    def _busy_shots(self, project_id: str) -> set[str]:
        with self.db.session() as s:
            jobs = s.exec(
                select(Job).where(Job.project_id == project_id, col(Job.status).in_(["queued", "running"]))
            ).all()
        return {
            str((j.target_ref_json or {}).get("shot_id"))
            for j in jobs
            if (j.target_ref_json or {}).get("shot_id")
        }

    def generate_all(self, project_id: str, body: GenerateAllBody) -> GenerateAllOut:
        auto = (
            self.settings.auto_approve_keyframes
            if body.auto_approve_keyframes is None
            else body.auto_approve_keyframes
        )
        with self.db.session() as s:
            if s.get(Project, project_id) is None:
                raise not_found("Project", project_id)
            scenes = s.exec(
                select(Scene).where(Scene.project_id == project_id).order_by(col(Scene.index))
            ).all()
            order = {sc.id: sc.index for sc in scenes}
            shots = sorted(
                s.exec(select(Shot).where(col(Shot.scene_id).in_(list(order)))).all(),
                key=lambda sh: (order[sh.scene_id], sh.index),
            )
        busy = self._busy_shots(project_id)
        keyframe_jobs = video_jobs = 0
        skipped: list[SkippedShot] = []
        for shot in shots:
            if shot.id in busy:
                skipped.append(SkippedShot(shot_id=shot.id, reason="Already generating."))
                continue
            if shot.selected_video_asset_id:
                skipped.append(SkippedShot(shot_id=shot.id, reason="Already has a take."))
                continue
            segments = (shot.extend_segments or 1) if (body.extend and shot.needs_extend) else 0
            video = VideoBody(camera_intensity=body.camera_intensity, extend_segments=min(segments, 4))
            try:
                missing = []
                if shot.generation_mode in ("i2v", "flf2v", "ref2v") and not shot.selected_keyframe_asset_id:
                    missing.append("start")
                if shot.generation_mode == "flf2v" and not shot.selected_end_keyframe_asset_id:
                    missing.append("end")
                if missing:
                    if not body.keyframes:
                        skipped.append(SkippedShot(shot_id=shot.id, reason="No keyframe selected."))
                        continue
                    for n, which in enumerate(missing):
                        chain: dict[str, Any] | None = None
                        if auto:
                            chain = {"select": which}
                            if body.videos and n == len(missing) - 1:
                                chain["video"] = video.model_dump(mode="json")
                        self.imagegen.compose_keyframe(
                            shot.id, KeyframeBody(which=which, count=1 if auto else 2), chain=chain
                        )
                        keyframe_jobs += 1
                    if not auto:
                        skipped.append(
                            SkippedShot(shot_id=shot.id, reason="Pick a keyframe, then generate the video.")
                        )
                    continue
                if body.videos:
                    self.generate(shot.id, video)
                    video_jobs += 1
            except AppError as e:
                skipped.append(SkippedShot(shot_id=shot.id, reason=e.message))
        return GenerateAllOut(keyframe_jobs=keyframe_jobs, video_jobs=video_jobs, skipped=skipped)

    # --- linking outputs ---------------------------------------------------------------------------
    def on_job_assets(self, job: Job, assets: list[Asset]) -> None:
        """Selects new takes and runs chained work. Errors are logged: the job itself succeeded."""
        ref = job.target_ref_json or {}
        if ref.get("kind") != "shot" or not ref.get("shot_id"):
            return
        shot_id = str(ref["shot_id"])
        chain = (job.params_json or {}).get("chain") or {}
        try:
            if ref.get("slot") == "video":
                self._on_take(job, shot_id, [a for a in assets if a.type == "video"], chain)
            elif chain.get("select"):
                self._on_keyframe(shot_id, [a for a in assets if a.type == "image"], chain)
        except AppError as e:
            log.warning("follow-up for job %s failed: %s", job.id, e.message, extra={"job_id": job.id})

    def _on_take(self, job: Job, shot_id: str, videos: list[Asset], chain: dict[str, Any]) -> None:
        if not videos:
            return
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
        if shot is None:
            return
        parents = videos[0].parent_asset_ids_json or []
        # The first take is selected automatically; an extension replaces the take it extended.
        if shot.selected_video_asset_id is None or (
            job.type == "extend" and shot.selected_video_asset_id in parents
        ):
            self.board.update_shot(shot_id, ShotUpdate(selected_video_asset_id=videos[0].id))
        ext = chain.get("extend")
        if ext and ext.get("segments"):
            self.extend(
                shot_id,
                ExtendShotBody(
                    asset_id=videos[0].id,
                    segments=int(ext["segments"]),
                    camera_intensity=ext.get("camera_intensity", "normal"),
                ),
            )

    def _on_keyframe(self, shot_id: str, images: list[Asset], chain: dict[str, Any]) -> None:
        if not images:
            return
        which = chain.get("select")
        field = "selected_end_keyframe_asset_id" if which == "end" else "selected_keyframe_asset_id"
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
        if shot is None:
            return
        if getattr(shot, field) is None:
            self.board.update_shot(shot_id, ShotUpdate.model_validate({field: images[0].id}))
        if chain.get("video"):
            self.generate(shot_id, VideoBody.model_validate(chain["video"]))
