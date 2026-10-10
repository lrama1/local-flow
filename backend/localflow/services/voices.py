"""Voices (Section 12.2): built-ins synced from the TTS adapters, custom voices, suggestions, LLM voice
design, and cached previews in `library/voices/<id>/preview.wav`.

Custom voices are a built-in base voice plus adapter params (for Kokoro and the FakeEngine: a blend of up
to three built-in voices and a speaking rate, `VoiceParams`). Characters store a `voices.id`; TTS jobs get
the adapter-native voice and params (`resolve_tts`)."""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlmodel import Session, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, bad_request, conflict, not_found
from localflow.models import Asset, Ingredient, Job, LibraryCharacter, Voice, utcnow
from localflow.schemas.core import JobOut
from localflow.schemas.generate import TTSBody
from localflow.schemas.story import UsageOut
from localflow.schemas.voices import (
    AuditionBody,
    PreviewBody,
    PreviewOut,
    VoiceCreate,
    VoiceDesign,
    VoiceDesignBody,
    VoiceDesignOut,
    VoiceOut,
    VoiceSuggestion,
    VoiceUpdate,
)
from localflow.services.assets import AssetService
from localflow.services.events import EventBus
from localflow.services.jobs.generation import FAKE, GenerationService
from localflow.services.llm.service import LLMService
from localflow.services.llm.templates import render
from localflow.services.registry.service import RegistryService
from localflow.worker.adapters.speech import VoiceParams

log = logging.getLogger("localflow.voices")

SAMPLE_TEXT = {
    "en": "Hi there! This is how I sound when I'm telling your story.",
    "es": "¡Hola! Así sueno cuando cuento tu historia.",
    "fr": "Bonjour ! Voici ma voix quand je raconte votre histoire.",
    "hi": "नमस्ते! जब मैं आपकी कहानी सुनाता हूँ, तो मेरी आवाज़ ऐसी लगती है।",
    "it": "Ciao! Ecco come suono quando racconto la tua storia.",
    "pt": "Olá! É assim que eu soo quando conto a sua história.",
}
_QUALITY = {"A": 3.0, "A-": 2.6, "B+": 2.3, "B": 2.0, "B-": 1.8, "C+": 1.4, "C": 1.2, "C-": 1.0}
_FEMALE = {"female", "woman", "women", "feminine", "girl", "she", "her", "lady", "mother", "grandmother"}
_MALE = {"male", "man", "men", "masculine", "boy", "he", "him", "gentleman", "father", "grandfather"}
_WORD = re.compile(r"[a-z]+")


def sample_text(language: str) -> str:
    return SAMPLE_TEXT.get(language.split("-")[0].lower(), SAMPLE_TEXT["en"])


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def hint_gender(text: str) -> str:
    w = _words(text)
    female, male = bool(w & _FEMALE), bool(w & _MALE)
    return "female" if female and not male else "male" if male and not female else ""


class VoiceService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        bus: EventBus,
        generation: GenerationService,
        registry: RegistryService,
        assets: AssetService,
        llm: LLMService,
        adapters: Callable[[], list[dict[str, Any]] | None],
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.generation = generation
        self.registry = registry
        self.assets = assets
        self.llm = llm
        self._adapters = adapters
        self._synced: tuple[Any, ...] | None = None

    # --- built-ins ---------------------------------------------------------------------------------
    def _tts_adapters(self) -> list[dict[str, Any]]:
        return [
            a
            for a in self._adapters() or []
            if a.get("available") and "tts" in (a.get("capabilities") or []) and a.get("voices")
        ]

    def usable_adapters(self) -> set[str]:
        """TTS adapters the worker can run now with an installed model (the FakeEngine needs none)."""
        out = set()
        for a in self._tts_adapters():
            key = str(a["adapter_key"])
            if key == FAKE or self.registry.default_entry("tts", adapter_key=key, usable=None, preferred=[]):
                out.add(key)
        return out

    def sync(self) -> None:
        """Upserts the built-in voices of every usable TTS adapter. Voices of adapters that went away are
        kept (characters may reference them) and listed as unavailable."""
        adapters = [a for a in self._tts_adapters() if str(a["adapter_key"]) in self.usable_adapters()]
        fingerprint = tuple(
            (a["adapter_key"], a.get("adapter_version"), len(a.get("voices") or [])) for a in adapters
        )
        if fingerprint == self._synced:
            return
        changed = False
        with self.db.session() as s:
            for a in adapters:
                key = str(a["adapter_key"])
                existing = {
                    v.native_id: v
                    for v in s.exec(select(Voice).where(Voice.adapter_key == key, Voice.kind == "builtin"))
                }
                for raw in a.get("voices") or []:
                    old = existing.get(raw["native_id"])
                    v = old or Voice(
                        kind="builtin", adapter_key=key, native_id=raw["native_id"], name=raw["name"]
                    )
                    fields = {
                        "name": raw["name"],
                        "description": raw.get("traits", ""),
                        "language": raw.get("language", ""),
                        "gender": raw.get("gender", ""),
                        "meta_json": {"quality": raw.get("quality", ""), "traits": raw.get("traits", "")},
                    }
                    if old is None or any(getattr(v, k) != val for k, val in fields.items()):
                        for k, val in fields.items():
                            setattr(v, k, val)
                        v.updated_at = utcnow()
                        s.add(v)
                        changed = True
            s.commit()
        self._synced = fingerprint
        if changed:
            log.info("synced built-in voices for %s", ", ".join(a["adapter_key"] for a in adapters) or "-")
            self.bus.publish("voices.updated")

    # --- reads -------------------------------------------------------------------------------------
    def _out(self, v: Voice, usable: set[str], names: dict[str, str]) -> VoiceOut:
        meta = v.meta_json or {}
        preview = self.settings.data_dir / v.preview_path if v.preview_path else None
        return VoiceOut(
            id=v.id,
            kind=v.kind,
            adapter_key=v.adapter_key,
            native_id=v.native_id,
            base_voice_id=v.base_voice_id,
            base_voice_name=names.get(v.base_voice_id or ""),
            name=v.name,
            description=v.description,
            language=v.language,
            gender=v.gender,
            params=v.params_json or {},
            quality=str(meta.get("quality") or ""),
            traits=str(meta.get("traits") or ""),
            available=v.adapter_key in usable,
            preview_url=(
                f"/api/voices/{v.id}/preview.wav?v={int(v.updated_at.timestamp())}"
                if preview is not None and preview.is_file()
                else None
            ),
            preview_text=v.preview_text,
            created_at=v.created_at,
            updated_at=v.updated_at,
        )

    def list_voices(self, adapter_key: str | None = None, available_only: bool = False) -> list[VoiceOut]:
        self.sync()
        usable = self.usable_adapters()
        with self.db.session() as s:
            q = select(Voice)
            if adapter_key:
                q = q.where(Voice.adapter_key == adapter_key)
            rows = list(s.exec(q))
        names = {v.id: v.name for v in rows if v.kind == "builtin"}
        rank = {"custom": 0, "builtin": 1}
        rows.sort(
            key=lambda v: (
                rank.get(v.kind, 2),
                v.adapter_key,
                v.language,
                -_QUALITY.get(str((v.meta_json or {}).get("quality") or ""), 0.0),
                v.name.lower(),
            )
        )
        out = [self._out(v, usable, names) for v in rows]
        return [v for v in out if v.available] if available_only else out

    def _get(self, s: Session, voice_id: str) -> Voice:
        v = s.get(Voice, voice_id)
        if v is None:
            raise not_found("Voice", voice_id)
        return v

    def get(self, voice_id: str) -> VoiceOut:
        with self.db.session() as s:
            v = self._get(s, voice_id)
            base = s.get(Voice, v.base_voice_id) if v.base_voice_id else None
        return self._out(v, self.usable_adapters(), {base.id: base.name} if base else {})

    # --- custom voices -----------------------------------------------------------------------------
    def _base(self, s: Session, base_voice_id: str) -> Voice:
        base = s.get(Voice, base_voice_id)
        if base is None or base.kind != "builtin":
            raise bad_request("invalid_voice", "base_voice_id must be a built-in voice.")
        return base

    def _schema(self, adapter_key: str) -> dict[str, Any] | None:
        for a in self._adapters() or []:
            if a.get("adapter_key") == adapter_key:
                return a.get("voice_params_schema") or {}
        return None

    def validate_params(self, s: Session, base: Voice, params: dict[str, Any]) -> dict[str, Any]:
        """Normalized params for `base`'s adapter. Blend voices must be built-ins of the same adapter and
        language family."""
        schema = self._schema(base.adapter_key)
        if schema == {}:
            if params:
                raise bad_request("invalid_voice_params", "This TTS model has no adjustable voice settings.")
            return {}
        try:
            p = VoiceParams.model_validate(params or {})
        except ValidationError as e:
            raise bad_request(
                "invalid_voice_params", f"Invalid voice settings: {e.errors()[0]['msg']}"
            ) from e
        family = base.language.split("-")[0]
        allowed = {
            v.native_id
            for v in s.exec(
                select(Voice).where(Voice.adapter_key == base.adapter_key, Voice.kind == "builtin")
            )
            if v.language.split("-")[0] == family
        }
        bad = [m.voice for m in p.mix if m.voice not in allowed]
        if bad:
            raise bad_request(
                "invalid_voice_params",
                f"Can't blend {', '.join(bad)}: blend voices must be built-in voices in the same language.",
            )
        p.mix = [m for m in p.mix if m.voice != base.native_id]
        return p.model_dump(mode="json")

    def create(self, body: VoiceCreate) -> VoiceOut:
        with self.db.session() as s:
            base = self._base(s, body.base_voice_id)
            params = self.validate_params(s, base, body.params)
            v = Voice(
                kind="custom",
                adapter_key=base.adapter_key,
                base_voice_id=base.id,
                name=body.name.strip(),
                description=body.description.strip(),
                language=base.language,
                gender=base.gender,
                params_json=params,
            )
            s.add(v)
            s.commit()
            s.refresh(v)
            vid = v.id
        self.bus.publish("voices.updated")
        return self.get(vid)

    def update(self, voice_id: str, body: VoiceUpdate) -> VoiceOut:
        data = body.model_dump(exclude_unset=True)
        with self.db.session() as s:
            v = self._get(s, voice_id)
            if v.kind != "custom":
                raise conflict(
                    "builtin_voice", "Built-in voices can't be edited. Make a custom voice from it."
                )
            base = self._base(s, data.get("base_voice_id") or v.base_voice_id or "")
            sound_changed = False
            if "base_voice_id" in data and base.id != v.base_voice_id:
                v.base_voice_id, v.adapter_key = base.id, base.adapter_key
                v.language, v.gender = base.language, base.gender
                sound_changed = True
            if "params" in data or sound_changed:
                params = self.validate_params(s, base, data.get("params", v.params_json) or {})
                sound_changed = sound_changed or params != (v.params_json or {})
                v.params_json = params
            if data.get("name"):
                v.name = data["name"].strip()
            if data.get("description") is not None:
                v.description = data["description"].strip()
            if sound_changed:
                self._drop_preview(v)
            v.updated_at = utcnow()
            s.add(v)
            s.commit()
        self.bus.publish("voices.updated")
        return self.get(voice_id)

    def delete(self, voice_id: str) -> None:
        """Deletes a custom voice; characters using it go back to "no voice"."""
        with self.db.session() as s:
            v = self._get(s, voice_id)
            if v.kind != "custom":
                raise conflict(
                    "builtin_voice", "Built-in voices come with their TTS model and can't be deleted."
                )
            projects, characters = set(), set()
            for ing in s.exec(select(Ingredient).where(Ingredient.voice_id == voice_id)):
                ing.voice_id = None
                s.add(ing)
                projects.add(ing.project_id)
            for ch in s.exec(select(LibraryCharacter).where(LibraryCharacter.voice_id == voice_id)):
                ch.voice_id = None
                s.add(ch)
                characters.add(ch.id)
            s.delete(v)
            s.commit()
        shutil.rmtree(self._dir(voice_id), ignore_errors=True)
        for pid in projects:
            self.bus.publish("ingredients.updated", project_id=pid)
        for cid in characters:
            self.bus.publish("library.updated", library_character_id=cid)
        self.bus.publish("voices.updated")

    # --- suggestions -------------------------------------------------------------------------------
    def suggest(
        self, hints: dict[str, str], language: str = "en", exclude: set[str] | None = None
    ) -> VoiceSuggestion:
        """The best available built-in voice for a character's voice hints (gender, age, tone)."""
        text = " ".join(str(v) for v in hints.values() if v)
        gender = hint_gender(str(hints.get("gender") or "")) or hint_gender(text)
        words = _words(text)
        british = bool(words & {"british", "english", "uk", "london", "posh"})
        american = bool(words & {"american", "us", "usa"})
        best: tuple[float, VoiceOut] | None = None
        for v in self.list_voices(available_only=True):
            if v.kind != "builtin" or v.language.split("-")[0] != language or v.id in (exclude or set()):
                continue
            score = _QUALITY.get(v.quality, 0.0)
            if gender:
                score += 5 if v.gender == gender else -5
            if british:
                score += 3 if v.language == "en-GB" else 0
            elif american or language == "en":
                score += 1 if v.language == "en-US" else 0
            score += 1.5 * len(words & _words(v.traits))
            if best is None or score > best[0]:
                best = (score, v)
        if best is None:
            return VoiceSuggestion(voice=None, reason="No TTS model is installed.")
        v = best[1]
        why = ", ".join(x for x in (gender, v.traits) if x)
        return VoiceSuggestion(voice=v, reason=f"{v.name}: {why}" if why else v.name)

    # --- LLM voice design --------------------------------------------------------------------------
    async def design(self, body: VoiceDesignBody, project_id: str | None = None) -> VoiceDesignOut:
        with self.db.session() as s:
            base = self._base(s, body.base_voice_id)
            family = base.language.split("-")[0]
            candidates = [
                v
                for v in s.exec(
                    select(Voice).where(Voice.adapter_key == base.adapter_key, Voice.kind == "builtin")
                )
                if v.language.split("-")[0] == family and v.id != base.id
            ]
        if self._schema(base.adapter_key) == {}:
            raise conflict("no_voice_params", "This TTS model has no adjustable voice settings.")

        def card(v: Voice) -> dict[str, Any]:
            meta = v.meta_json or {}
            return {
                "native_id": v.native_id,
                "name": v.name,
                "gender": v.gender,
                "language": v.language,
                "quality": meta.get("quality", ""),
                "traits": meta.get("traits", ""),
            }

        system, user, _ = render(
            "voice_design",
            {
                "description": body.description,
                "base": card(base),
                "voices": [card(v) for v in candidates],
                "schema": VoiceDesign.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="voice_design",
            project_id=project_id,
            choice=body.llm,
            system=system,
            user=user,
            schema=VoiceDesign,
            temperature=0.4,
        )
        d = res.value
        warnings: list[str] = []
        known = {v.native_id for v in candidates}
        mix = []
        for m in d.mix:
            if m.voice in known:
                mix.append(m)
            else:
                warnings.append(f"Ignored unknown voice '{m.voice}' suggested by the model.")
        params = VoiceParams(mix=mix, base_weight=d.base_weight, speed=d.speed).model_dump(mode="json")
        usage = UsageOut(
            provider=res.provider,
            model=res.model,
            prompt_tokens=res.usage.prompt_tokens,
            completion_tokens=res.usage.completion_tokens,
            cost_usd=res.usage.cost_usd,
            attempts=res.attempts,
        )
        return VoiceDesignOut(params=params, notes=d.notes, warnings=warnings, usage=usage)

    # --- resolving for TTS jobs ---------------------------------------------------------------------
    def native(self, voice_id: str) -> tuple[str, str, dict[str, Any], str]:
        """(adapter_key, native voice, params, language) for a LocalFlow voice id."""
        with self.db.session() as s:
            v = self._get(s, voice_id)
            if v.kind == "builtin":
                return v.adapter_key, str(v.native_id), {}, v.language
            base = s.get(Voice, v.base_voice_id) if v.base_voice_id else None
            if base is None:
                raise conflict("voice_broken", f"The base voice of '{v.name}' is gone. Pick another one.")
            return v.adapter_key, str(base.native_id), dict(v.params_json or {}), v.language

    def resolve_tts(self, body: TTSBody, request: dict[str, Any]) -> None:
        """`POST /generate/tts` hook: LocalFlow voice ids -> native voices + params, and the adapter to use.
        Ids that aren't voices are passed through as native voice names."""
        adapters: set[str] = set()

        def swap(target: dict[str, Any]) -> None:
            vid = target.get("voice_id")
            if not vid:
                return
            with self.db.session() as s:
                if s.get(Voice, vid) is None:
                    return
            adapter, native, params, _ = self.native(str(vid))
            target["voice_id"] = native
            if params and not target.get("voice_params"):
                target["voice_params"] = params
            adapters.add(adapter)

        swap(request)
        for line in request.get("lines") or []:
            swap(line)
        if len(adapters) > 1:
            raise bad_request("mixed_voices", "These voices belong to different TTS models; use one model.")
        if adapters and body.adapter_key is None and not request.get("model_entry_id"):
            body.adapter_key = adapters.pop()

    # --- previews ----------------------------------------------------------------------------------
    def _dir(self, voice_id: str) -> Path:
        return self.settings.library_dir / "voices" / voice_id

    def _drop_preview(self, v: Voice) -> None:
        if v.preview_path:
            (self.settings.data_dir / v.preview_path).unlink(missing_ok=True)
        v.preview_path, v.preview_text = None, None

    def preview_file(self, voice_id: str) -> Path:
        with self.db.session() as s:
            v = self._get(s, voice_id)
        path = self.settings.data_dir / v.preview_path if v.preview_path else None
        if path is None or not path.is_file():
            raise not_found("Voice preview", voice_id)
        return path

    def _submit(
        self,
        voice_id: str | None,
        adapter: str,
        native: str,
        params: dict[str, Any],
        text: str,
        ref: dict[str, Any],
    ) -> Job:
        body = TTSBody(text=text, voice_id=native, voice_params=params, target_ref=ref)
        if adapter in self.usable_adapters():
            body.adapter_key = adapter
        elif voice_id is not None:
            raise conflict("voice_unavailable", "This voice's TTS model isn't installed (Models page).")
        return self.generation.submit("tts", body)

    def preview(self, voice_id: str, body: PreviewBody) -> PreviewOut:
        adapter, native, params, language = self.native(voice_id)
        text = (body.text or "").strip() or sample_text(language)
        with self.db.session() as s:
            v = self._get(s, voice_id)
            cached = (
                v.preview_path
                and v.preview_text == text
                and (self.settings.data_dir / v.preview_path).is_file()
            )
        if cached and not body.refresh:
            return PreviewOut(url=self.get(voice_id).preview_url)
        ref = {"kind": "voice_preview", "voice_id": voice_id, "text": text}
        job = self._submit(voice_id, adapter, native, params, text, ref)
        return PreviewOut(job=JobOut.of(job))

    def audition(self, body: AuditionBody) -> PreviewOut:
        with self.db.session() as s:
            base = self._base(s, body.base_voice_id)
            params = self.validate_params(s, base, body.params)
            adapter, native, language = base.adapter_key, str(base.native_id), base.language
        text = (body.text or "").strip() or sample_text(language)
        ref = {"kind": "voice_audition", "base_voice_id": body.base_voice_id}
        return PreviewOut(job=JobOut.of(self._submit(body.base_voice_id, adapter, native, params, text, ref)))

    def on_job_assets(self, job: Job, assets: list[Asset]) -> None:
        """Caches a finished preview as the voice's sample."""
        ref = job.target_ref_json or {}
        if ref.get("kind") != "voice_preview":
            return
        audio = next((a for a in assets if a.type == "audio"), None)
        if audio is None:
            return
        voice_id = str(ref.get("voice_id"))
        try:
            src = self.assets.file_path(audio)
            dst = self._dir(voice_id) / "preview.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            with self.db.session() as s:
                v = s.get(Voice, voice_id)
                if v is None:
                    return
                v.preview_path = dst.relative_to(self.settings.data_dir).as_posix()
                v.preview_text = str(ref.get("text") or "")
                v.updated_at = utcnow()
                s.add(v)
                s.commit()
        except (OSError, AppError) as e:
            log.warning("caching the preview of voice %s failed: %s", voice_id, e)
            return
        self.bus.publish("voices.updated")
