"""Kokoro-82M text-to-speech (Section 12.1), loaded from the catalog files (no downloads at run time).

Layout (catalog part "base"): `config.json`, `kokoro-v1_0.pth` and `voices/<voice>.pt` (one style pack per
built-in voice). Custom voices are weighted averages of style packs (`VoiceParams`), which is how Kokoro's
own "voice_a,voice_b" mixing works. Output is 24 kHz mono."""

from __future__ import annotations

import logging
import random
from array import array
from pathlib import Path
from typing import Any

import torch

from localflow.worker.adapters.base import AdapterInfo
from localflow.worker.adapters.kokoro_voices import KOKORO_VOICES, LANG_BY_PREFIX
from localflow.worker.adapters.speech import SpeechAdapter, VoiceParams
from localflow.worker.context import JobContext
from localflow.worker.messages import WorkerConfig
from localflow.worker.spec import ModelSpec

log = logging.getLogger("localflow.worker.kokoro")

REPO_ID = "hexgrad/Kokoro-82M"


class KokoroAdapter(SpeechAdapter):
    requires_model = True
    sample_rate = 24000
    builtin_voices = KOKORO_VOICES
    info = AdapterInfo(
        adapter_key="kokoro",
        adapter_version="1.0.0",
        name="Kokoro TTS",
        capabilities=["tts"],
        supported_base_models=["kokoro"],
        vram_profiles={"low": {}, "balanced": {}, "quality": {}},
        heavy=False,
        voices=[v.to_dict() for v in KOKORO_VOICES],
        voice_params_schema=VoiceParams.model_json_schema(),
    )

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.model: Any = None
        self.voice_dir: Path | None = None
        self._pipelines: dict[str, Any] = {}
        self._packs: dict[str, torch.Tensor] = {}

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        from kokoro import KModel

        assert model is not None
        base = model.part("base", "main")
        if base is None:
            raise RuntimeError(f"{model.name}: missing the Kokoro files (reinstall it from the Models page).")
        config, weights = base.path / "config.json", base.path / "kokoro-v1_0.pth"
        if not weights.is_file():
            raise RuntimeError(
                f"{model.name}: {weights.name} is missing (reinstall it from the Models page)."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = KModel(repo_id=REPO_ID, config=str(config), model=str(weights)).to(device).eval()
        self.voice_dir = base.path / "voices"
        return {"device": device}

    def unload(self) -> None:
        self.model = None
        self._pipelines.clear()
        self._packs.clear()

    def _pipeline(self, lang_code: str) -> Any:
        """One G2P pipeline per language, sharing the loaded model."""
        from kokoro import KPipeline

        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=self.model)
        return self._pipelines[lang_code]

    def _pack(self, voice: str) -> torch.Tensor:
        if voice not in self._packs:
            assert self.voice_dir is not None
            path = self.voice_dir / f"{voice}.pt"
            if voice[:1] not in LANG_BY_PREFIX or not path.is_file():
                raise RuntimeError(f"Unknown Kokoro voice '{voice}'.")
            self._packs[voice] = torch.load(path, map_location="cpu", weights_only=True).float()
        return self._packs[voice]

    def voice_pack(self, voice: str, params: VoiceParams) -> torch.Tensor:
        """The (blended) style pack, on the CPU as float32 (what KPipeline expects)."""
        weights = params.weights(voice)
        if len(weights) == 1:
            return self._pack(voice)
        return sum((self._pack(v) * w for v, w in weights), torch.zeros_like(self._pack(voice)))

    @torch.inference_mode()
    def speak(self, text: str, voice: str, params: VoiceParams, speed: float, ctx: JobContext) -> array:
        # Kokoro is deterministic except for its noise source, which the seed pins.
        torch.manual_seed(ctx.seed % (2**32))
        random.seed(ctx.seed)
        pipeline = self._pipeline(voice[0])
        pack = self.voice_pack(voice, params)
        chunks = []
        for result in pipeline(text, voice=pack, speed=speed, split_pattern=r"\n+"):
            ctx.check_cancelled()
            if result.audio is not None:
                chunks.append(result.audio.float().cpu())
        if not chunks:
            raise RuntimeError("Kokoro produced no audio for this text (is it empty after cleanup?).")
        audio = torch.cat(chunks).clamp(-1, 1).numpy()
        return array("f", audio.astype("float32").tobytes())
