"""MusicGen text-to-music (Section 12.1) through transformers: 32 kHz mono, up to 30 s per job."""

from __future__ import annotations

import logging
import wave
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from localflow.worker.adapters.base import Adapter, AdapterInfo, AudioResult, MusicRequest
from localflow.worker.context import JobContext
from localflow.worker.messages import WorkerConfig
from localflow.worker.spec import ModelSpec

log = logging.getLogger("localflow.worker.musicgen")

MAX_S = 30.0


class MusicGenExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guidance_scale: float = Field(default=3.0, ge=1.0, le=10.0)
    temperature: float = Field(default=1.0, gt=0, le=2.0)


class _Progress:
    """transformers streamer: token-level progress and cooperative cancel."""

    def __init__(self, ctx: JobContext, total: int) -> None:
        self.ctx, self.total, self.n = ctx, max(1, total), -1  # the first put() is the prompt

    def put(self, value: Any) -> None:
        self.ctx.check_cancelled()
        self.n += 1
        if self.n > 0 and self.n % 25 == 0:
            self.ctx.report_progress(
                0.05 + 0.9 * min(1.0, self.n / self.total), f"{self.n / 50:.0f} s composed"
            )

    def end(self) -> None:
        pass


class MusicGenAdapter(Adapter):
    requires_model = True
    info = AdapterInfo(
        adapter_key="musicgen",
        adapter_version="1.0.0",
        name="MusicGen",
        capabilities=["music"],
        supported_base_models=["musicgen"],
        param_schema=MusicGenExtra.model_json_schema(),
        vram_profiles={"low": {}, "balanced": {}, "quality": {}},
        max_duration_s=MAX_S,
    )

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        assert model is not None
        base = model.part("base", "main")
        if base is None:
            raise RuntimeError(f"{model.name}: missing files (reinstall it from the Models page).")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(base.path)
        # fp32: the small model is ~1.2 GB, and half precision is known to produce silence/NaNs in places.
        self.model = MusicgenForConditionalGeneration.from_pretrained(base.path, dtype=torch.float32)
        self.model.to(device).eval()
        return {"device": device}

    def unload(self) -> None:
        self.model = self.processor = None

    @torch.inference_mode()
    def generate_music(self, req: MusicRequest, ctx: JobContext) -> AudioResult:
        extra = MusicGenExtra.model_validate(req.extra)
        duration = min(req.duration_s, MAX_S)
        cfg = self.model.config.audio_encoder
        rate, frame_rate = int(cfg.sampling_rate), int(getattr(cfg, "frame_rate", 50) or 50)
        tokens = int(np.ceil(duration * frame_rate)) + 3  # + the delay pattern's codebook offsets
        torch.manual_seed(ctx.seed % (2**32))
        inputs = self.processor(text=[req.prompt or "instrumental music"], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        ctx.report_progress(0.05, "Composing…", force=True)
        audio = self.model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=extra.guidance_scale,
            temperature=extra.temperature,
            max_new_tokens=tokens,
            streamer=_Progress(ctx, tokens),
        )
        wav = audio[0, 0].float().cpu().numpy()
        n = round(duration * rate)
        wav = np.pad(wav[:n], (0, max(0, n - len(wav))))
        fade = min(len(wav), int(0.5 * rate))  # no click at the end of a cut-off phrase
        if fade:
            wav[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        peak = float(np.abs(wav).max() or 1.0)
        wav = wav / max(peak, 1.0)
        path = ctx.output_dir / "music.wav"
        pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        ctx.report_progress(1.0, "Done", force=True)
        return AudioResult(str(path), n / rate, rate, ctx.seed)
