"""Shot audio and music (Section 12.1): dialogue as one clip-length TTS track per shot, SFX from the selected
take (video-to-audio), and music per scene or for the whole film.

Dialogue and SFX are ordinary audio jobs with `target_ref = {kind: shot, shot_id, slot: dialogue|sfx}`.
A finished one becomes the shot's selected audio for its slot (`selected_audio_asset_ids`) and, if the
timeline already holds the shot's take, replaces the shot's clip on the Dialogue/SFX track."""

from __future__ import annotations

import logging
import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import col, select

from localflow.db import Database
from localflow.errors import AppError, bad_request, conflict, not_found
from localflow.models import Asset, Ingredient, Job, ModelEntry, Project, Scene, Shot
from localflow.schemas.board import ShotUpdate
from localflow.schemas.core import JobOut
from localflow.schemas.generate import MusicBody, TTSBody, V2ABody
from localflow.services.board import BoardService
from localflow.services.events import EventBus
from localflow.services.jobs.generation import FAKE, GenerationService
from localflow.services.registry.service import RegistryService
from localflow.services.timeline.service import TimelineService
from localflow.services.voice_store import voices_by_id
from localflow.services.voices import VoiceService
from localflow.worker.adapters.base import SpeechLine

log = logging.getLogger("localflow.audiogen")

AudioSlot = Literal["dialogue", "sfx"]
ALL_PARTS: tuple[AudioSlot, ...] = ("dialogue", "sfx")
SLOT_TRACK = {"dialogue": "a_dialogue", "sfx": "a_sfx", "narration": "a_narration"}
JOB_SLOT = {"tts": "dialogue", "v2a": "sfx"}
WORDS_PER_S = 2.6  # ~155 wpm, typical for TTS at speed 1
LEAD_S, GAP_S = 0.3, 0.35
# Kokoro has no emotion control, so emotions only change the pace a little (DECISIONS M6).
EMOTION_SPEED = {
    "excited": 1.08,
    "angry": 1.06,
    "urgent": 1.1,
    "panicked": 1.12,
    "nervous": 1.05,
    "happy": 1.03,
    "sad": 0.92,
    "tired": 0.9,
    "somber": 0.92,
    "whisper": 0.95,
    "whispering": 0.95,
    "calm": 0.97,
}
SFX_NEGATIVE = ["speech", "talking", "voice", "music"]


def emotion_speed(emotion: str) -> float:
    words = emotion.lower().replace(",", " ").split()
    return next((EMOTION_SPEED[w] for w in words if w in EMOTION_SPEED), 1.0)


def estimate_line_s(text: str, speed: float) -> float:
    return max(0.6, len(text.split()) / WORDS_PER_S / speed)


def sfx_negative(prompt: str) -> str:
    low = prompt.lower()
    return ", ".join(t for t in SFX_NEGATIVE if t not in low)


# --- schemas -----------------------------------------------------------------------------------------
class ShotAudioParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speed: float = Field(default=1.0, ge=0.7, le=1.4, description="Dialogue pace (on top of each voice's)")
    fit: Literal["speed", "none"] = Field(
        default="speed", description="speed: speed lines up (<= max_speed) to fit the clip"
    )
    max_speed: float = Field(default=1.15, ge=1.0, le=1.5)
    tts_model_entry_id: str | None = None
    sfx_prompt: str | None = Field(default=None, max_length=1000, description="Defaults to the shot's")
    sfx_negative_prompt: str | None = Field(default=None, max_length=1000)
    sfx_steps: int | None = Field(default=None, ge=1, le=200)
    v2a_model_entry_id: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class ShotAudioBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[AudioSlot] = Field(default_factory=lambda: list(ALL_PARTS), min_length=1)
    params: ShotAudioParams = Field(default_factory=ShotAudioParams)


class DialoguePlanLine(BaseModel):
    index: int
    character_ingredient_id: str | None
    speaker: str
    line: str
    emotion: str
    voice_id: str | None
    voice_name: str | None
    voice_source: Literal["character", "suggested", "none"]
    speed: float
    estimated_s: float


class AudioPlanOut(BaseModel):
    clip_s: float = Field(description="The length the audio is made for: the selected take, else the shot")
    has_take: bool
    lines: list[DialoguePlanLine]
    speech_s: float = Field(description="Estimated dialogue length incl. lead-in and pauses")
    fit_speed: float = Field(description="Speed-up needed to fit the clip (1 = fits)")
    fits: bool = Field(description="Fits within max_speed")
    tts_model: str | None
    sfx_prompt: str
    v2a_model: str | None
    selected: dict[str, str | None]
    notes: list[str]


class ShotAudioOut(AudioPlanOut):
    jobs: list[JobOut]


class MusicGenBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["film", "scene"] = "film"
    scene_id: str | None = None
    prompt: str | None = Field(default=None, max_length=1000, description="Defaults to the shots' music cues")
    duration_s: float | None = Field(default=None, gt=0, le=600)
    model_entry_id: str | None = None
    place: bool = Field(default=True, description="Add it to the timeline's Music track when it's done")
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class MusicGenOut(BaseModel):
    job: JobOut
    prompt: str
    duration_s: float
    model_name: str | None
    notes: list[str]


class AudioGenService:
    def __init__(
        self,
        db: Database,
        generation: GenerationService,
        registry: RegistryService,
        board: BoardService,
        voices: VoiceService,
        timeline: TimelineService,
        bus: EventBus,
    ) -> None:
        self.db = db
        self.generation = generation
        self.registry = registry
        self.board = board
        self.voices = voices
        self.timeline = timeline
        self.bus = bus

    # --- helpers -----------------------------------------------------------------------------------
    def _shot(self, shot_id: str) -> tuple[Shot, Project, Asset | None]:
        with self.db.session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                raise not_found("Shot", shot_id)
            scene = s.get(Scene, shot.scene_id)
            project = s.get(Project, scene.project_id) if scene else None
            if project is None:
                raise not_found("Shot", shot_id)
            take = s.get(Asset, shot.selected_video_asset_id) if shot.selected_video_asset_id else None
        return shot, project, take

    def _model(self, kind: str, entry_id: str | None, adapter_key: str | None = None) -> ModelEntry | None:
        if entry_id:
            return self.registry.require_usable(entry_id, kind)
        usable = self.generation.model_adapters(kind)
        return self.registry.default_entry(kind, adapter_key=adapter_key, usable=usable, preferred=[])

    def _real(self, kind: str) -> bool:
        usable = self.generation.usable_adapters(kind)
        return usable is not None and FAKE not in usable

    def _selected(self, shot: Shot) -> dict[str, str | None]:
        out: dict[str, str | None] = {"dialogue": None, "sfx": None}
        with self.db.session() as s:
            for aid in shot.selected_audio_asset_ids_json:
                a = s.get(Asset, aid)
                slot = audio_slot(a) if a else None
                if slot in out and out[slot] is None:
                    out[slot] = aid
        return out

    # --- planning ----------------------------------------------------------------------------------
    def _lines(
        self, shot: Shot, project: Project, params: ShotAudioParams, notes: list[str]
    ) -> list[DialoguePlanLine]:
        with self.db.session() as s:
            ings = {i.id: i for i in s.exec(select(Ingredient).where(Ingredient.project_id == project.id))}
        assigned = voices_by_id_safe(self.db, {i.voice_id for i in ings.values() if i.voice_id})
        usable = self.voices.usable_adapters()
        taken = {i.voice_id for i in ings.values() if i.voice_id}
        # speaker -> (voice id, voice name, source, the voice's own pace for estimates)
        chosen: dict[str | None, tuple[str | None, str | None, str, float]] = {}
        out: list[DialoguePlanLine] = []
        for n, d in enumerate(shot.dialogue_json):
            text = str(d.get("line") or "").strip()
            if not text:
                continue
            cid = d.get("character_ingredient_id")
            ing = ings.get(cid) if cid else None
            speaker = ing.name if ing else "Narrator"
            if cid not in chosen:
                voice = assigned.get(ing.voice_id) if ing and ing.voice_id else None
                if voice is not None and voice.adapter_key in usable:
                    pace = float((voice.params_json or {}).get("speed") or 1.0)
                    chosen[cid] = (voice.id, voice.name, "character", pace)
                else:
                    hints = dict(ing.voice_hints_json or {}) if ing else {"tone": "calm, clear narrator"}
                    sug = self.voices.suggest(hints, exclude=taken)
                    if sug.voice is None:
                        sug = self.voices.suggest(hints)  # every voice is taken: reuse one
                    if sug.voice is not None:
                        taken.add(sug.voice.id)
                        chosen[cid] = (sug.voice.id, sug.voice.name, "suggested", 1.0)
                        why = (
                            "has no voice yet"
                            if not (ing and ing.voice_id)
                            else "has a voice whose TTS model isn't installed"
                        )
                        tip = " Pick a voice in the Cast tab." if ing else ""
                        notes.append(f"{speaker} {why}; using {sug.voice.name} (suggested).{tip}")
                    else:
                        chosen[cid] = (None, None, "none", 1.0)
            vid, vname, source, pace = chosen[cid]
            emotion = str(d.get("emotion") or "")
            speed = round(params.speed * emotion_speed(emotion), 3)
            out.append(
                DialoguePlanLine(
                    index=n,
                    character_ingredient_id=cid,
                    speaker=speaker,
                    line=text,
                    emotion=emotion,
                    voice_id=vid,
                    voice_name=vname,
                    voice_source=source,
                    speed=speed,
                    estimated_s=round(estimate_line_s(text, speed * pace), 2),
                )
            )
        return out

    def plan(self, shot_id: str, params: ShotAudioParams | None = None) -> AudioPlanOut:
        params = params or ShotAudioParams()
        shot, project, take = self._shot(shot_id)
        notes: list[str] = []
        clip_s = float(take.duration_s) if take and take.duration_s else float(shot.duration_s)
        if take is None:
            notes.append(
                f"No take is selected: dialogue is timed to the planned {clip_s:g} s, "
                "and sound effects need a take."
            )
        lines = self._lines(shot, project, params, notes)
        speech = sum(ln.estimated_s for ln in lines)
        speech_total = speech + (LEAD_S + GAP_S * (len(lines) - 1) if lines else 0.0)
        room = clip_s - 0.1 - 0.15 * max(0, len(lines) - 1)
        fit_speed = 1.0 if not lines or speech <= room else (speech / room if room > 0 else 9.99)
        max_speed = params.max_speed if params.fit == "speed" else 1.0
        fits = fit_speed <= max_speed + 1e-6
        if lines and not fits:
            notes.append(
                f"The dialogue (~{speech_total:.1f} s) is longer than the clip ({clip_s:g} s), even sped up "
                f"{max_speed:g}x: extend the shot, or shorten the lines."
            )
        elif lines and fit_speed > 1.0:
            notes.append(f"The dialogue will be sped up about {fit_speed:.2f}x to fit the clip.")
        tts = None
        if lines and self._real("tts"):
            adapters = {self.voices.native(ln.voice_id)[0] for ln in lines if ln.voice_id}
            tts = self._model("tts", params.tts_model_entry_id, next(iter(adapters), None))
        v2a = self._model("v2a", params.v2a_model_entry_id) if self._real("v2a") else None
        return AudioPlanOut(
            clip_s=round(clip_s, 3),
            has_take=take is not None,
            lines=lines,
            speech_s=round(speech_total, 2),
            fit_speed=round(fit_speed, 3),
            fits=fits,
            tts_model=tts.name if tts else ("Fake engine" if lines and not self._real("tts") else None),
            sfx_prompt=(params.sfx_prompt if params.sfx_prompt is not None else shot.sfx_prompt) or "",
            v2a_model=v2a.name if v2a else (None if self._real("v2a") else "Fake engine"),
            selected=self._selected(shot),
            notes=notes,
        )

    # --- generation --------------------------------------------------------------------------------
    def generate(self, shot_id: str, body: ShotAudioBody) -> ShotAudioOut:
        p = body.params
        plan = self.plan(shot_id, p)
        shot, project, take = self._shot(shot_id)
        seed = p.seed if p.seed is not None else random.randint(0, 2**32 - 1)
        jobs: list[Job] = []
        notes = list(plan.notes)
        if "dialogue" in body.parts:
            if not plan.lines:
                raise bad_request(
                    "no_dialogue", "This shot has no dialogue lines. Add some in the shot panel."
                )
            if any(ln.voice_id is None for ln in plan.lines):
                raise conflict(
                    "no_tts_model",
                    "No text-to-speech model is installed. Install Kokoro from the Models page.",
                )
            natives = {ln.voice_id: self.voices.native(str(ln.voice_id)) for ln in plan.lines}
            adapters = {v[0] for v in natives.values()}
            if len(adapters) > 1:
                raise conflict(
                    "mixed_voices",
                    "This shot's voices belong to different TTS models. "
                    "Give every speaker a voice from one model.",
                )
            lines = [
                SpeechLine(
                    text=ln.line,
                    voice_id=natives[ln.voice_id][1],
                    voice_params=natives[ln.voice_id][2],
                    speed=ln.speed,
                    key=str(ln.index),
                )
                for ln in plan.lines
            ]
            req = TTSBody(
                project_id=project.id,
                adapter_key=next(iter(adapters)) if not p.tts_model_entry_id else None,
                model_entry_id=p.tts_model_entry_id,
                lines=lines,
                duration_s=plan.clip_s,
                lead_s=LEAD_S,
                gap_s=GAP_S,
                max_speed=p.max_speed if p.fit == "speed" else 1.0,
                seed=seed,
                target_ref={"kind": "shot", "shot_id": shot.id, "slot": "dialogue"},
            )
            jobs.append(self.generation.submit("tts", req))
        if "sfx" in body.parts:
            if take is None:
                raise bad_request(
                    "no_video", "Sound effects are made from the shot's take: select a take first."
                )
            prompt = plan.sfx_prompt.strip()
            if not prompt:
                notes.append("No SFX prompt: the sound follows the picture alone.")
            v2a = V2ABody(
                project_id=project.id,
                video_asset_id=take.id,
                prompt=prompt,
                negative_prompt=p.sfx_negative_prompt
                if p.sfx_negative_prompt is not None
                else sfx_negative(prompt),
                steps=p.sfx_steps,
                seed=seed,
                model_entry_id=p.v2a_model_entry_id,
                target_ref={"kind": "shot", "shot_id": shot.id, "slot": "sfx"},
            )
            jobs.append(self.generation.submit("v2a", v2a))
        return ShotAudioOut(**{**plan.model_dump(), "notes": notes}, jobs=[JobOut.of(j) for j in jobs])

    # --- music ---------------------------------------------------------------------------------------
    def _music_limit(self, entry: ModelEntry | None) -> float | None:
        key = entry.adapter_key if entry else FAKE
        for a in self.generation.adapters() or []:
            if a.get("adapter_key") == key and a.get("max_duration_s"):
                return float(a["max_duration_s"])
        return None

    def generate_music(self, project_id: str, body: MusicGenBody) -> MusicGenOut:
        notes: list[str] = []
        with self.db.session() as s:
            project = s.get(Project, project_id)
            if project is None:
                raise not_found("Project", project_id)
            scenes = s.exec(
                select(Scene).where(Scene.project_id == project_id).order_by(col(Scene.index))
            ).all()
            if body.scope == "scene":
                if not body.scene_id or body.scene_id not in {sc.id for sc in scenes}:
                    raise bad_request("invalid_scene", "Pick a scene of this project.")
                scene_ids = [body.scene_id]
            else:
                scene_ids = [sc.id for sc in scenes]
            order = {sid: i for i, sid in enumerate(scene_ids)}
            shots = sorted(
                s.exec(select(Shot).where(col(Shot.scene_id).in_(scene_ids))).all(),
                key=lambda sh: (order[sh.scene_id], sh.index),
            )
            takes = {
                a.id: a
                for a in s.exec(
                    select(Asset).where(
                        col(Asset.id).in_(
                            [sh.selected_video_asset_id for sh in shots if sh.selected_video_asset_id]
                        )
                    )
                )
            }
        cues = list(
            dict.fromkeys(sh.music_cue.strip() for sh in shots if sh.music_cue and sh.music_cue.strip())
        )
        prompt = (body.prompt or "").strip() or "; ".join(cues[:4])
        if not prompt:
            prompt = "Cinematic instrumental underscore, gentle and evocative"
            notes.append(
                "No music cues on the board, so this is a generic underscore. Describe what you want."
            )
        if project.style_notes and not body.prompt:
            prompt = f"{prompt}. Mood: {project.style_notes.strip().rstrip('.')}"[:1000]
        duration = body.duration_s
        if duration is None:
            if body.scope == "film":
                tl = self.timeline.get(project_id)
                duration = tl.duration_s or None
            if not duration:
                duration = (
                    sum(
                        (takes[sh.selected_video_asset_id].duration_s or sh.duration_s)
                        if sh.selected_video_asset_id in takes
                        else sh.duration_s
                        for sh in shots
                    )
                    or 30.0
                )
        entry = self._model("music", body.model_entry_id) if self._real("music") else None
        if self._real("music") and entry is None:
            raise conflict(
                "no_model_installed", "No music model is installed. Install MusicGen from the Models page."
            )
        limit = self._music_limit(entry)
        if limit and duration > limit:
            notes.append(
                f"{entry.name if entry else 'The model'} makes up to {limit:g} s of music, "
                f"so it's {limit:g} s."
            )
            duration = limit
        duration = round(min(float(duration), 600.0), 2)
        req = MusicBody(
            project_id=project_id,
            prompt=prompt,
            duration_s=duration,
            seed=body.seed,
            model_entry_id=entry.id if entry else None,
            target_ref={
                "kind": "music",
                "project_id": project_id,
                "scope": body.scope,
                "scene_id": body.scene_id,
                "place": body.place,
            },
        )
        job = self.generation.submit("music", req)
        return MusicGenOut(
            job=JobOut.of(job),
            prompt=prompt,
            duration_s=duration,
            model_name=entry.name if entry else None,
            notes=notes,
        )

    # --- linking outputs -----------------------------------------------------------------------------
    def on_job_assets(self, job: Job, assets: list[Asset]) -> None:
        ref = job.target_ref_json or {}
        audio = next((a for a in assets if a.type == "audio"), None)
        if audio is None:
            return
        try:
            if ref.get("kind") == "shot" and ref.get("slot") in SLOT_TRACK and ref.get("shot_id"):
                self._select(str(ref["shot_id"]), str(ref["slot"]), audio)
            elif ref.get("kind") == "music" and ref.get("place") and job.project_id:
                start = 0.0
                if ref.get("scope") == "scene" and ref.get("scene_id"):
                    start = self.timeline.scene_start(job.project_id, str(ref["scene_id"])) or 0.0
                self.timeline.place_audio(job.project_id, audio, "a_music", start)
        except AppError as e:
            log.warning("follow-up for job %s failed: %s", job.id, e.message, extra={"job_id": job.id})

    def select(self, shot_id: str, asset_id: str, slot: AudioSlot | None = None) -> list[str]:
        """Makes `asset_id` the shot's audio for its slot (by the job that made it, else `slot`)."""
        with self.db.session() as s:
            a = s.get(Asset, asset_id)
        if a is None or a.type != "audio":
            raise bad_request("invalid_asset", "Pick an audio asset.")
        chosen = audio_slot(a) or slot
        if chosen is None:
            raise bad_request("unknown_slot", "Say whether this audio is dialogue or sound effects.")
        return self._select(shot_id, chosen, a)

    def _select(self, shot_id: str, slot: str, audio: Asset) -> list[str]:
        shot, project, _ = self._shot(shot_id)
        keep: list[str] = []
        with self.db.session() as s:
            for aid in shot.selected_audio_asset_ids_json:
                a = s.get(Asset, aid)
                if a is not None and audio_slot(a) != slot and aid != audio.id:
                    keep.append(aid)
        ids = [*keep, audio.id]
        self.board.update_shot(shot_id, ShotUpdate(selected_audio_asset_ids=ids))
        self.timeline.sync_shot_audio(project.id, shot_id, audio, SLOT_TRACK[slot])
        return ids


def audio_slot(a: Asset) -> str | None:
    gen = a.generation_json or {}
    slot = (gen.get("target_ref") or {}).get("slot")
    if slot in SLOT_TRACK:
        return str(slot)
    return JOB_SLOT.get(str(gen.get("job_type")))


def voices_by_id_safe(db: Database, ids: set[str | None]) -> dict[str, Any]:
    with db.session() as s:
        return voices_by_id(s, {i for i in ids if i})
