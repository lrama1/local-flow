"""FakeEngine adapter: deterministic placeholder outputs (Pillow images, ffmpeg test patterns, tones).

Lets the API, job queue, and UI be developed and tested on any machine (Section 0.4)."""

from __future__ import annotations

import os
import random
import textwrap
import time
from array import array
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from localflow.services.media.ffmpeg import probe, run_ffmpeg
from localflow.worker.adapters.base import (
    GPU_JOB_TYPES,
    AdapterInfo,
    AudioResult,
    ExtendRequest,
    FLF2VRequest,
    I2VRequest,
    ImageEditRequest,
    ImageResult,
    MusicRequest,
    Ref2VRequest,
    T2IRequest,
    T2VRequest,
    TTSRequest,
    V2ARequest,
    VideoParams,
    VideoResult,
)
from localflow.worker.adapters.speech import BuiltinVoice, SpeechAdapter, VoiceParams, tone
from localflow.worker.context import JobContext


class FakeExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=12, ge=1, le=200, description="Simulated denoising steps")
    step_delay_s: float | None = Field(default=None, ge=0, le=5, description="Seconds per simulated step")
    fail: str | None = Field(default=None, description="Fail the job with this message (testing)")
    crash: bool = Field(default=False, description="Hard-exit the worker process (testing crash recovery)")
    hang_s: float = Field(
        default=0, ge=0, le=600, description="Block this long ignoring cancel (testing hard cancel)"
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _colors(seed: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    rng = random.Random(seed)
    a = tuple(rng.randint(20, 120) for _ in range(3))
    b = tuple(rng.randint(120, 255) for _ in range(3))
    return a, b  # type: ignore[return-value]


def placeholder_image(
    width: int, height: int, seed: int, title: str, prompt: str, progress: float = 1.0
) -> Image.Image:
    c1, c2 = _colors(seed)
    grad = Image.linear_gradient("L").rotate(seed % 360, expand=False).resize((width, height))
    img = ImageOps.colorize(grad, c1, c2).convert("RGB")
    if progress < 1.0:
        noise = Image.effect_noise((width, height), 90).convert("RGB")
        img = Image.blend(noise, img, max(0.0, min(1.0, progress)))
    draw = ImageDraw.Draw(img)
    big = _font(max(14, width // 18))
    small = _font(max(10, width // 40))
    draw.text((width * 0.05, height * 0.06), title, fill=(255, 255, 255), font=big)
    lines = textwrap.wrap(prompt or "(no prompt)", width=max(20, width // max(8, width // 60)))[:8]
    y = height * 0.06 + max(14, width // 18) * 1.6
    for line in lines:
        draw.text((width * 0.05, y), line, fill=(240, 240, 240), font=small)
        y += max(10, width // 40) * 1.4
    draw.text(
        (width * 0.05, height * 0.9), f"seed {seed} · LocalFlow FakeEngine", fill=(255, 255, 255), font=small
    )
    return img


FAKE_VOICES: tuple[BuiltinVoice, ...] = (
    BuiltinVoice("fake_aria", "Aria", "en-US", "female", "A", "warm, friendly, clear"),
    BuiltinVoice("fake_ben", "Ben", "en-US", "male", "A", "calm, deep, steady"),
    BuiltinVoice("fake_cleo", "Cleo", "en-US", "female", "B", "bright, energetic, young"),
    BuiltinVoice("fake_dev", "Dev", "en-US", "male", "B", "relaxed, conversational"),
    BuiltinVoice("fake_emma", "Emma", "en-GB", "female", "B", "crisp, composed, mature"),
    BuiltinVoice("fake_finn", "Finn", "en-GB", "male", "C", "gravelly, older, gruff"),
)
_FAKE_PITCH = {
    "fake_aria": 220.0,
    "fake_ben": 110.0,
    "fake_cleo": 260.0,
    "fake_dev": 135.0,
    "fake_emma": 200.0,
    "fake_finn": 95.0,
}


class FakeAdapter(SpeechAdapter):
    builtin_voices = FAKE_VOICES
    info = AdapterInfo(
        adapter_key="fake",
        adapter_version="1.1.0",
        name="Fake engine (placeholder outputs)",
        capabilities=list(GPU_JOB_TYPES),
        supported_base_models=["fake"],
        param_schema=FakeExtra.model_json_schema(),
        vram_profiles={"low": {}, "balanced": {}, "quality": {}},
        supports_lora=True,
        supports_native_audio=True,
        heavy=False,
        voices=[v.to_dict() for v in FAKE_VOICES],
        voice_params_schema=VoiceParams.model_json_schema(),
    )

    def __init__(self, ffmpeg: str | None, ffprobe: str | None, default_step_delay_s: float) -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.default_step_delay_s = default_step_delay_s

    # --- helpers ---------------------------------------------------------------------------------
    def _extra(self, raw: dict[str, Any]) -> FakeExtra:
        return FakeExtra.model_validate(raw)

    def _simulate(
        self,
        ctx: JobContext,
        extra: FakeExtra,
        *,
        weight: float,
        preview: tuple[int, int, str, str] | None = None,
        offset: float = 0.0,
    ) -> None:
        delay = self.default_step_delay_s if extra.step_delay_s is None else extra.step_delay_s
        for i in range(extra.steps):
            ctx.check_cancelled()
            if i == 1 or extra.steps == 1:
                if extra.crash:
                    os._exit(3)
                if extra.fail:
                    raise RuntimeError(extra.fail)
                if extra.hang_s:
                    time.sleep(extra.hang_s)  # like a long CUDA kernel: cancel can't interrupt it
            time.sleep(delay)
            frac = (i + 1) / extra.steps
            ctx.report_progress(offset + weight * frac, f"Step {i + 1}/{extra.steps}")
            if preview and (i % 3 == 2 or i == extra.steps - 1):
                w, h, title, prompt = preview
                pw, ph = _fit(w, h, 384)
                ctx.report_preview(placeholder_image(pw, ph, ctx.seed, title, prompt, progress=frac))
        ctx.check_cancelled()

    def _encode(self, ctx: JobContext, args: list[str], total_s: float, offset: float) -> None:
        run_ffmpeg(
            self.ffmpeg,
            args,
            total_s=total_s,
            cancel_event=ctx.cancel_event,
            on_progress=lambda f: ctx.report_progress(offset + (1 - offset) * f, "Encoding"),
        )

    def _video_out(self, ctx: JobContext, name: str = "video.mp4") -> Path:
        return ctx.output_dir / name

    def _video_result(
        self, path: Path, req: VideoParams, ctx: JobContext, frames: int, has_audio: bool
    ) -> VideoResult:
        return VideoResult(
            path=str(path),
            width=_even(req.width),
            height=_even(req.height),
            fps=float(req.fps),
            num_frames=frames,
            has_audio=has_audio,
            seed=ctx.seed,
        )

    def _from_still(self, ctx: JobContext, req: VideoParams, image_path: str, title: str) -> VideoResult:
        extra = self._extra(req.extra)
        w, h = _even(req.width), _even(req.height)
        self._simulate(ctx, extra, weight=0.7, preview=(w, h, title, req.prompt))
        frames = req.resolved_frames()
        dur = frames / req.fps
        out = self._video_out(ctx)
        vf = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            f"zoompan=z='1+0.15*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={req.fps},"
            "format=yuv420p"
        )
        args = ["-loop", "1", "-framerate", str(req.fps), "-t", f"{dur:.3f}", "-i", image_path]
        args += _audio_input(req.with_audio, dur, ctx.seed)
        args += ["-vf", vf, "-frames:v", str(frames), *_h264(), *_aac(req.with_audio), str(out)]
        self._encode(ctx, args, dur, 0.7)
        return self._video_result(out, req, ctx, frames, req.with_audio)

    # --- capabilities ----------------------------------------------------------------------------
    def generate_t2i(self, req: T2IRequest, ctx: JobContext) -> list[ImageResult]:
        extra = self._extra(req.extra)
        self._simulate(ctx, extra, weight=1.0, preview=(req.width, req.height, "Text to image", req.prompt))
        results = []
        for i in range(req.num_images):
            seed = ctx.seed + i
            img = placeholder_image(req.width, req.height, seed, "Text to image", req.prompt)
            path = ctx.output_dir / f"image_{i}.png"
            img.save(path, "PNG")
            results.append(ImageResult(str(path), req.width, req.height, seed))
        return results

    def edit(self, req: ImageEditRequest, ctx: JobContext) -> list[ImageResult]:
        extra = self._extra(req.extra)
        self._simulate(ctx, extra, weight=1.0, preview=(req.width, req.height, "Image edit", req.prompt))
        results = []
        for i in range(req.num_images):
            seed = ctx.seed + i
            img = placeholder_image(req.width, req.height, seed, "Image edit", req.prompt)
            tile = max(48, min(req.width, req.height) // 4)
            x = int(req.width * 0.05)
            for ref in req.reference_image_paths[:4]:
                with Image.open(ref) as r:
                    t = ImageOps.fit(r.convert("RGB"), (tile, tile))
                img.paste(t, (x, int(req.height * 0.62)))
                x += tile + 8
            path = ctx.output_dir / f"image_{i}.png"
            img.save(path, "PNG")
            results.append(ImageResult(str(path), req.width, req.height, seed))
        return results

    def generate_t2v(self, req: T2VRequest, ctx: JobContext) -> VideoResult:
        extra = self._extra(req.extra)
        w, h = _even(req.width), _even(req.height)
        self._simulate(ctx, extra, weight=0.7, preview=(w, h, "Text to video", req.prompt))
        frames = req.resolved_frames()
        dur = frames / req.fps
        out = self._video_out(ctx)
        args = ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={req.fps}:duration={dur:.3f}"]
        args += _audio_input(req.with_audio, dur, ctx.seed)
        args += ["-vf", f"hue=h={ctx.seed % 360},format=yuv420p", "-frames:v", str(frames)]
        args += [*_h264(), *_aac(req.with_audio), str(out)]
        self._encode(ctx, args, dur, 0.7)
        return self._video_result(out, req, ctx, frames, req.with_audio)

    def generate_i2v(self, req: I2VRequest, ctx: JobContext) -> VideoResult:
        return self._from_still(ctx, req, req.image_path, "Image to video")

    def generate_ref2v(self, req: Ref2VRequest, ctx: JobContext) -> VideoResult:
        return self._from_still(ctx, req, req.reference_image_paths[0], "Reference to video")

    def generate_flf2v(self, req: FLF2VRequest, ctx: JobContext) -> VideoResult:
        extra = self._extra(req.extra)
        w, h = _even(req.width), _even(req.height)
        self._simulate(ctx, extra, weight=0.7, preview=(w, h, "First/last frame", req.prompt))
        frames = req.resolved_frames()
        dur = frames / req.fps
        fade = dur * 0.5
        seg = (dur + fade) / 2
        out = self._video_out(ctx)
        norm = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps={req.fps}"
        args = [
            "-loop",
            "1",
            "-t",
            f"{seg:.3f}",
            "-i",
            req.first_image_path,
            "-loop",
            "1",
            "-t",
            f"{seg:.3f}",
            "-i",
            req.last_image_path,
        ]
        args += _audio_input(req.with_audio, dur, ctx.seed)
        args += [
            "-filter_complex",
            f"[0:v]{norm}[a];[1:v]{norm}[b];"
            f"[a][b]xfade=transition=fade:duration={fade:.3f}:offset={seg - fade:.3f},format=yuv420p[v]",
            "-map",
            "[v]",
            *(["-map", "2:a"] if req.with_audio else []),
            "-frames:v",
            str(frames),
            *_h264(),
            *_aac(req.with_audio),
            str(out),
        ]
        self._encode(ctx, args, dur, 0.7)
        return self._video_result(out, req, ctx, frames, req.with_audio)

    def extend(self, req: ExtendRequest, ctx: JobContext) -> VideoResult:
        extra = self._extra(req.extra)
        w, h = _even(req.width), _even(req.height)
        self._simulate(ctx, extra, weight=0.6, preview=(w, h, "Extend", req.prompt))
        src = probe(self.ffprobe, Path(req.video_path))
        src_dur = float(src.get("duration_s") or 0)
        frames = req.resolved_frames() * req.segments
        add = frames / req.fps
        total = src_dur + add
        out = self._video_out(ctx)
        norm = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            f"setsar=1,fps={req.fps},format=yuv420p"
        )
        args = [
            "-i",
            req.video_path,
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={w}x{h}:rate={req.fps}:duration={add:.3f}",
            "-filter_complex",
            f"[0:v]{norm}[a];[1:v]hue=h={ctx.seed % 360},{norm}[b];[a][b]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            *_h264(),
            str(out),
        ]
        self._encode(ctx, args, total, 0.6)
        info = probe(self.ffprobe, out)
        return VideoResult(
            path=str(out),
            width=w,
            height=h,
            fps=float(req.fps),
            num_frames=int(info.get("num_frames") or round(total * req.fps)),
            has_audio=False,
            seed=ctx.seed,
        )

    def synthesize(self, req: TTSRequest, ctx: JobContext) -> AudioResult:
        extra = self._extra(req.extra)
        self._simulate(ctx, extra, weight=0.3)
        return SpeechAdapter.synthesize(self, req, ctx)

    def speak(self, text: str, voice: str, params: VoiceParams, speed: float, ctx: JobContext) -> array:
        """~2.5 words/s, pitched by voice (and blend); the line's "pitch" follows the seed slightly."""
        words = max(1, len(text.split()))
        weights = params.weights(voice)
        freq = sum(_FAKE_PITCH.get(v, 180.0) * w for v, w in weights) + ctx.seed % 7
        return tone(max(0.6, words / 2.5 / speed), freq, self.sample_rate)

    def generate_v2a(self, req: V2ARequest, ctx: JobContext) -> AudioResult:
        extra = self._extra(req.extra)
        self._simulate(ctx, extra, weight=0.8)
        dur = float(probe(self.ffprobe, Path(req.video_path)).get("duration_s") or 4.0)
        out = ctx.output_dir / "sfx.wav"
        args = [
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=color=brown:sample_rate=48000:duration={dur:.3f}:seed={ctx.seed}",
            "-af",
            "volume=0.3",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
        self._encode(ctx, args, dur, 0.8)
        return AudioResult(str(out), dur, 48000, ctx.seed)

    def generate_music(self, req: MusicRequest, ctx: JobContext) -> AudioResult:
        extra = self._extra(req.extra)
        self._simulate(ctx, extra, weight=0.6)
        dur = req.duration_s
        root = 196 + (ctx.seed % 5) * 22
        out = ctx.output_dir / "music.wav"
        args = []
        for ratio in (1.0, 1.26, 1.5):
            args += [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={root * ratio:.2f}:sample_rate=44100:duration={dur:.3f}",
            ]
        args += [
            "-filter_complex",
            "[0:a][1:a][2:a]amix=inputs=3,volume=0.6,tremolo=f=2:d=0.3[a]",
            "-map",
            "[a]",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
        self._encode(ctx, args, dur, 0.6)
        return AudioResult(str(out), dur, 44100, ctx.seed)


def _even(v: int) -> int:
    return v - (v % 2)


def _fit(w: int, h: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(w, h))
    return max(16, int(w * scale)), max(16, int(h * scale))


def _h264() -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


def _audio_input(enabled: bool, dur: float, seed: int) -> list[str]:
    if not enabled:
        return []
    return ["-f", "lavfi", "-i", f"sine=frequency={220 + seed % 220}:sample_rate=48000:duration={dur:.3f}"]


def _aac(enabled: bool) -> list[str]:
    return ["-c:a", "aac", "-b:a", "128k", "-shortest"] if enabled else ["-an"]
