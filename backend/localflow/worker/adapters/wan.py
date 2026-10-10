"""Wan 2.1 / 2.2 video: text-to-video, image-to-video, first/last frame, and extend (Section 10.4).

Layout (catalog parts):
- `transformer`: one GGUF (Wan 2.2 TI2V-5B, Wan 2.1 FLF2V-14B), or `transformers`: the HighNoise and LowNoise
  GGUF experts of Wan 2.2 A14B.
- `base`: model_index.json (`expand_timesteps`, `boundary_ratio`), scheduler, VAE, transformer config(s), and
  for Wan 2.1 the CLIP image encoder.
- `text` (or `base`): the UMT5 text encoder + tokenizer. Every Wan entry shares the same files.

Memory on a 16 GB GPU with little free system RAM (DECISIONS M4):
- UMT5 is loaded in fp8 only to encode a new prompt; embeddings are cached on the CPU. When it doesn't fit
  next to the transformer, the transformer is dropped first and reloaded from the GGUF memmap (~3 s for 5 GB).
- Wan 2.2 A14B's experts take turns: the high-noise expert runs the first steps and is swapped for the
  low-noise expert at the pipeline's timestep boundary, so only one 9.7 GB expert is ever resident.
- The bf16 VAE (1.4 GB) waits on the CPU. It moves to the GPU only to encode the conditioning image and to
  decode, tiled, after denoising (`output_type="latent"`). If the decode won't fit next to the transformer,
  the transformer is dropped first. Windows spills into system RAM instead of raising OOM, which is many times
  slower, so this is checked up front.
- TI2V-5B's per-token timestep embeddings are computed per distinct timestep (`wan_ti2v`), about 3 GB less
  at 720p.
- Every transformer block and VAE decoder call checks the job's cancel flag, so a cancel lands within a
  fraction of a denoising step instead of after it."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from localflow.services.media.ffmpeg import concat_args, encode_frames, extract_frame, probe, run_ffmpeg
from localflow.worker.adapters import wan_ti2v
from localflow.worker.adapters.base import (
    AdapterInfo,
    ExtendRequest,
    FLF2VRequest,
    I2VRequest,
    T2VRequest,
    VideoParams,
    VideoResult,
)
from localflow.worker.adapters.diffusion import (
    GB,
    DiffusionAdapter,
    PromptCache,
    StepReporter,
    cpu_generator,
    cuda_free_bytes,
    free_cuda,
    load_gguf_state_dict,
    load_t5_fp8,
    quiet_libraries,
)
from localflow.worker.context import JobContext
from localflow.worker.engine import is_oom
from localflow.worker.images import snap
from localflow.worker.messages import WorkerConfig
from localflow.worker.spec import ModelSpec

log = logging.getLogger("localflow.worker.wan")

UMT5_FP8_BYTES = 6.8 * GB  # fp8 Linear weights + the bf16 256k-token embedding
_MARGIN = 0.8 * GB
_PROMPTS = PromptCache(16)
# Progress budget: denoising, then VAE decode, then MP4 encoding.
_DENOISE_END, _DECODE_END = 0.9, 0.97
# The official Wan negative prompt (English translation). Discourages frozen, washed-out, deformed results.
DEFAULT_NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra "
    "fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)


def negative_prompt(prompt: str, negative: str, use_default: bool) -> str:
    """`negative` plus Wan's default, minus default terms the prompt asks for: a "static camera" shot
    must not also negate "static" (DECISIONS M4)."""
    text = prompt.lower()
    default = [
        t for t in DEFAULT_NEGATIVE.split(", ") if use_default and not re.search(rf"\b{re.escape(t)}\b", text)
    ]
    return ", ".join(x for x in (negative.strip(), ", ".join(default)) if x)


class WanExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_sequence_length: int = Field(default=512, ge=64, le=512, description="UMT5 prompt tokens")
    guidance_scale_2: float | None = Field(
        default=None, ge=0, le=20, description="Guidance for the low-noise expert (Wan 2.2 A14B)"
    )
    flow_shift: float | None = Field(
        default=None, ge=0.5, le=20, description="Scheduler shift (default: the model's, 3.0 at 480p)"
    )
    default_negative: bool = Field(default=True, description="Append Wan's recommended negative prompt")


@dataclass
class _Paths:
    base: Path
    text: Path
    single: Path | None  # one GGUF transformer
    high: Path | None  # Wan 2.2 A14B experts
    low: Path | None
    index: dict[str, Any] = field(default_factory=dict)

    @property
    def boundary_ratio(self) -> float | None:
        value = self.index.get("boundary_ratio")
        return float(value) if value is not None else None

    @property
    def expand_timesteps(self) -> bool:
        return bool(self.index.get("expand_timesteps"))


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop to the video's aspect ratio, so the pipeline's resize doesn't squash the keyframe."""
    return ImageOps.fit(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)


class WanAdapter(DiffusionAdapter):
    family = "wan"
    info = AdapterInfo(
        adapter_key="wan",
        adapter_version="1.0.0",
        name="Wan 2.1 / 2.2",
        capabilities=["t2v", "i2v", "flf2v", "extend"],
        supported_base_models=["wan2.2-5b", "wan2.2-a14b-i2v", "wan2.2-a14b-t2v", "wan2.1-flf2v-14b"],
        param_schema=WanExtra.model_json_schema(),
        vram_profiles={
            "quality": {"placement": "text encoder resident if it fits, else loaded per new prompt"},
            "balanced": {"placement": "text encoder loaded per new prompt"},
            "low": {"placement": "text encoder loaded per new prompt, smaller sizes (catalog)"},
        },
        supports_lora=False,
    )

    def __init__(self, config: WorkerConfig) -> None:
        super().__init__(config)
        self.p: _Paths | None = None
        self.profile = "balanced"
        self.vae: Any = None
        self.tokenizer: Any = None
        self.image_encoder: Any = None
        self.image_processor: Any = None
        self.text_encoder: Any = None
        self.transformer: Any = None
        self.expert: str | None = None  # "single" | "high" | "low"
        self._pipes: dict[str, Any] = {}
        self._ctx: JobContext | None = None

    # --- loading -----------------------------------------------------------------------------------
    def _resolve(self, spec: ModelSpec) -> _Paths:
        base = spec.part("base", "main")
        if base is None:
            raise RuntimeError(f"{spec.name}: missing the Wan base components (install a catalog Wan entry).")
        text = spec.part("text") or base
        single = high = low = None
        if (tr := spec.part("transformer")) is not None:
            single = tr.first(".gguf")
        if (trs := spec.part("transformers")) is not None:
            for f in trs.files:
                if f.lower().endswith(".gguf"):
                    if "highnoise" in f.lower():
                        high = trs.path / f
                    elif "lownoise" in f.lower():
                        low = trs.path / f
        if single is None and not (high and low):
            raise RuntimeError(f"{spec.name}: no GGUF transformer found (reinstall the model).")
        index_file = base.path / "model_index.json"
        index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.is_file() else {}
        return _Paths(base.path, text.path, single, high, low, index)

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        from diffusers import AutoencoderKLWan
        from transformers import AutoTokenizer

        assert model is not None
        quiet_libraries()
        p = self._resolve(model)
        self.spec, self.p, self.profile = model, p, profile
        ctx.report_progress(0.0, "Loading the Wan VAE…", force=True)
        self.vae = AutoencoderKLWan.from_pretrained(p.base / "vae", dtype=torch.bfloat16)  # parked on the CPU
        # Measured (Wan 2.2 VAE, 41 frames at 1280x704; DECISIONS M4): untiled decode spills into system RAM;
        # tiled fp32 256/192 took 30 s at a 2.1 GB peak, bf16 384/320 12.6 s at 2.0 GB, 47 dB PSNR against it
        # (same as fp32 with other tile sizes). Tiles are blended, so there are no seams.
        self.vae.enable_tiling(384, 384, 320, 320)
        self.vae.decoder.register_forward_pre_hook(self._cancel_hook)
        self.tokenizer = AutoTokenizer.from_pretrained(p.text / "tokenizer")
        if (p.base / "image_encoder").is_dir():
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self.image_encoder = CLIPVisionModel.from_pretrained(
                p.base / "image_encoder", dtype=torch.bfloat16, device_map="cuda"
            )
            self.image_processor = CLIPImageProcessor.from_pretrained(p.base / "image_processor")
        transformer_bytes = max(x.stat().st_size for x in (p.single, p.high, p.low) if x is not None)
        free = cuda_free_bytes()
        keep_text = profile == "quality" and free > transformer_bytes + UMT5_FP8_BYTES + 6 * GB
        self.mode = "resident" if keep_text else "swap"
        return {
            "mode": self.mode,
            "experts": 2 if p.high else 1,
            "transformer_gb": round(transformer_bytes / GB, 2),
            "free_gb_at_load": round(free / GB, 2),
        }

    def _cancel_hook(self, *_: Any) -> None:
        if self._ctx is not None:
            self._ctx.check_cancelled()

    def _drop_transformer(self) -> None:
        if self.transformer is None:
            return
        for pipe in self._pipes.values():
            pipe.transformer, pipe.transformer_2 = None, None
        self.transformer, self.expert = None, None
        free_cuda()

    def _load_expert(self, expert: str, ctx: JobContext) -> None:
        """Puts `expert` ("single", "high" or "low") on the GPU, dropping whichever transformer is there."""
        if self.expert == expert and self.transformer is not None:
            return
        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        assert self.p is not None
        self._drop_transformer()
        path = {"single": self.p.single, "high": self.p.high, "low": self.p.low}[expert]
        assert path is not None
        label = {"single": "transformer", "high": "high-noise expert", "low": "low-noise expert"}[expert]
        ctx.report_progress(0.03, f"Loading the {label}…", force=True)
        state = load_gguf_state_dict(path, "cuda")
        transformer = WanTransformer3DModel.from_single_file(
            state,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(self.p.base),
            subfolder="transformer_2" if expert == "low" else "transformer",
            dtype=torch.bfloat16,
            device="cuda",
        )
        del state
        free_cuda()
        for block in transformer.blocks:
            block.register_forward_pre_hook(self._cancel_hook)
        if self.p.expand_timesteps:
            wan_ti2v.apply(transformer)
        self.transformer, self.expert = transformer, expert

    def _vae_to(self, device: str) -> None:
        if self.vae is not None and self.vae.device.type != device:
            self.vae.to(device)
            if device == "cpu":
                free_cuda()

    def unload(self) -> None:
        self._pipes.clear()
        self.transformer = self.text_encoder = self.vae = self.image_encoder = None
        self.image_processor = self.tokenizer = None
        self.expert, self.p, self._ctx = None, None, None
        super().unload()

    # --- pipelines and prompts ---------------------------------------------------------------------
    def _scheduler(self, flow_shift: float | None) -> Any:
        import diffusers

        assert self.p is not None
        config = json.loads((self.p.base / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
        cls = getattr(diffusers, str(config.get("_class_name") or "UniPCMultistepScheduler"))
        if flow_shift is not None and "flow_shift" in config:
            config["flow_shift"] = flow_shift
        return cls.from_config(config)

    def _pipe(self, kind: str) -> Any:
        from diffusers import WanImageToVideoPipeline, WanPipeline

        assert self.p is not None
        key = "t2v" if kind == "t2v" else "i2v"
        pipe: Any = self._pipes.get(key)
        if pipe is None:
            common: dict[str, Any] = {
                "tokenizer": self.tokenizer,
                "text_encoder": None,
                "vae": self.vae,
                "scheduler": self._scheduler(None),
                "transformer": None,
                "transformer_2": None,
                "boundary_ratio": self.p.boundary_ratio,
                "expand_timesteps": self.p.expand_timesteps,
            }
            if key == "t2v":
                pipe = WanPipeline(**common)
            else:
                pipe = WanImageToVideoPipeline(
                    **common, image_processor=self.image_processor, image_encoder=self.image_encoder
                )
                pipe.prepare_latents = self._with_vae(pipe.prepare_latents)  # encodes the first/last frame
            pipe.set_progress_bar_config(disable=True)
            self._pipes[key] = pipe
        return pipe

    def _with_vae(self, fn: Any) -> Any:
        """Runs `fn` with the VAE on the GPU, then parks it again for denoising."""

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._vae_to("cuda")
            try:
                return fn(*args, **kwargs)
            finally:
                self._vae_to("cpu")

        return wrapped

    def _encode(self, prompt: str, negative: str, max_len: int, ctx: JobContext) -> tuple[Any, Any]:
        assert self.p is not None
        key = (str(self.p.text), prompt, negative, max_len)
        hit = _PROMPTS.get(key)
        if hit is not None:
            return hit[0].to("cuda"), hit[1].to("cuda")
        if self.text_encoder is None:
            if self.transformer is not None and cuda_free_bytes() < UMT5_FP8_BYTES + _MARGIN:
                self._drop_transformer()  # cheap to reload from the GGUF memmap
            ctx.report_progress(0.01, "Loading the text encoder…", force=True)
            self.text_encoder = load_t5_fp8(self.p.text / "text_encoder")
        ctx.report_progress(0.02, "Encoding the prompt…", force=True)
        pipe = self._pipe("t2v")
        pipe.text_encoder = self.text_encoder
        try:
            with torch.inference_mode():
                pe, npe = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=1,
                    max_sequence_length=max_len,
                    device=torch.device("cuda"),
                )
        finally:
            pipe.text_encoder = None
        _PROMPTS.put(key, (pe, npe))
        if self.mode != "resident":
            self.text_encoder = None
            free_cuda()
        return pe, npe

    # --- generation --------------------------------------------------------------------------------
    def _multiple(self) -> int:
        assert self.p is not None and self.vae is not None
        cfg = json.loads((self.p.base / "transformer" / "config.json").read_text(encoding="utf-8"))
        patch = (cfg.get("patch_size") or [1, 2, 2])[1]
        return int(self.vae.config.scale_factor_spatial) * int(patch)

    def _size(self, req: VideoParams) -> tuple[int, int]:
        m = self._multiple()
        return snap(req.width, m), snap(req.height, m)

    def _frames(self, req: VideoParams) -> int:
        n = req.resolved_frames(5.0)
        return max(5, 4 * round((n - 1) / 4) + 1)

    def _denoise(
        self,
        req: VideoParams,
        ctx: JobContext,
        *,
        kind: str,
        image: Image.Image | None = None,
        last: Image.Image | None = None,
        index: int = 0,
        count: int = 1,
    ) -> np.ndarray:
        """One clip → uint8 frames (T, H, W, 3)."""
        assert self.p is not None and self.spec is not None
        self._ctx = ctx
        extra = WanExtra.model_validate(req.extra or {})
        width, height = self._size(req)
        frames = self._frames(req)
        steps = req.steps or int(self.spec.default("steps", 40))
        guidance = (
            req.guidance_scale
            if req.guidance_scale is not None
            else float(self.spec.default("guidance_scale", 5.0))
        )
        negative = negative_prompt(req.prompt, req.negative_prompt, extra.default_negative)
        pe, npe = self._encode(req.prompt, negative, extra.max_sequence_length, ctx)
        self._load_expert("high" if self.p.high else "single", ctx)
        pipe = self._pipe(kind)
        pipe.transformer, pipe.transformer_2 = self.transformer, None
        if self.p.high:
            lazy = _LazyExpert(self, ctx, self.transformer.config, self.transformer.dtype)
            object.__setattr__(pipe, "transformer_2", lazy)  # not a registered module: skip config updates
        shift = extra.flow_shift
        if shift is None and width * height <= 832 * 480 * 1.05:
            shift = 3.0  # Wan's recommendation for 480p
        pipe.scheduler = self._scheduler(shift)
        reporter = StepReporter(
            ctx, index=index, count=count, steps=steps, start=0.04, end=_DENOISE_END, noun="Segment"
        )
        callback = _StepCallback(self, reporter, ctx)
        kwargs: dict[str, Any] = {
            "prompt_embeds": pe,
            "negative_prompt_embeds": npe,
            "height": height,
            "width": width,
            "num_frames": frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": cpu_generator((ctx.seed + index) % (2**32)),
            "output_type": "latent",
            "callback_on_step_end": callback,
            "max_sequence_length": extra.max_sequence_length,
        }
        if self.p.boundary_ratio is not None:
            kwargs["guidance_scale_2"] = (
                extra.guidance_scale_2 if extra.guidance_scale_2 is not None else guidance
            )
        if kind != "t2v":
            assert image is not None
            kwargs["image"] = _fit(image, width, height)
            if last is not None:
                kwargs["last_image"] = _fit(last, width, height)
        with torch.inference_mode():
            latents = pipe(**kwargs).frames
        ctx.report_progress(reporter.fraction(1.0), "Decoding the video…", force=True)
        return self._decode(latents)

    def _decode_bytes(self, latents: torch.Tensor) -> int:
        """VRAM for a tiled decode: the VAE, the output video, and tile activations (2.0 GB measured)."""
        vae = self.vae
        _, _, t, h, w = latents.shape
        frames = (t - 1) * int(vae.config.scale_factor_temporal) + 1
        pixels = frames * h * w * int(vae.config.scale_factor_spatial) ** 2
        weights = sum(x.numel() * x.element_size() for x in vae.parameters())
        return int(weights + pixels * 3 * vae.dtype.itemsize + 2.5 * GB)

    @torch.inference_mode()
    def _decode(self, latents: torch.Tensor) -> np.ndarray:
        vae = self.vae
        if self.transformer is not None and cuda_free_bytes() < self._decode_bytes(latents):
            self._drop_transformer()
        self._vae_to("cuda")
        try:
            z = latents.to("cuda", torch.float32)
            dim = vae.config.z_dim
            std = torch.tensor(vae.config.latents_std).view(1, dim, 1, 1, 1).to(z)
            mean = torch.tensor(vae.config.latents_mean).view(1, dim, 1, 1, 1).to(z)
            z = (z * std + mean).to(vae.dtype)
            del latents
            try:
                video = vae.decode(z, return_dict=False)[0][0]  # (C, T, H, W)
            except Exception as e:
                if not is_oom(e):
                    raise
                log.warning("VAE decode ran out of memory; dropping the transformer and retrying")
                self._drop_transformer()
                video = vae.decode(z, return_dict=False)[0][0]
            del z
            # To 8 bits in fp32, a few frames at a time: bf16 can't hold the levels exactly, and the whole
            # clip in fp32 would be 1.3 GB at 720p.
            _, count, height, width = video.shape
            out = np.empty((count, height, width, 3), np.uint8)
            for i in range(0, count, 16):
                chunk = video[:, i : i + 16].float().clamp_(-1, 1).add_(1).mul_(127.5).round_()
                out[i : i + chunk.shape[1]] = chunk.to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            del video
        finally:
            self._vae_to("cpu")
        return out

    def _write(self, frames: np.ndarray, fps: float, path: Path, ctx: JobContext) -> None:
        h, w = frames.shape[1], frames.shape[2]
        encode_frames(self.config.ffmpeg, frames, w, h, fps, path, cancel_event=ctx.cancel_event)

    def _clip(self, frames: np.ndarray, req: VideoParams, ctx: JobContext) -> VideoResult:
        path = ctx.output_dir / "video.mp4"
        self._write(frames, req.fps, path, ctx)
        return VideoResult(
            path=str(path),
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            fps=float(req.fps),
            num_frames=int(frames.shape[0]),
            has_audio=False,
            seed=ctx.seed,
        )

    def _needs_image_conditioning(self) -> None:
        assert self.p is not None and self.spec is not None
        if "t2v" in (self.spec.base_model or "") and not self.p.expand_timesteps:
            raise RuntimeError(f"{self.spec.name} is text-to-video only; pick an image-to-video model.")

    def generate_t2v(self, req: T2VRequest, ctx: JobContext) -> VideoResult:
        return self._clip(self._denoise(req, ctx, kind="t2v"), req, ctx)

    def generate_i2v(self, req: I2VRequest, ctx: JobContext) -> VideoResult:
        self._needs_image_conditioning()
        image = Image.open(req.image_path)
        return self._clip(self._denoise(req, ctx, kind="i2v", image=image), req, ctx)

    def generate_flf2v(self, req: FLF2VRequest, ctx: JobContext) -> VideoResult:
        self._needs_image_conditioning()
        assert self.p is not None and self.spec is not None
        if self.p.expand_timesteps:
            # TI2V-5B conditions on the first latent frame only; a last frame would be silently ignored.
            raise RuntimeError(
                f"{self.spec.name} can't use a last frame. Install Wan 2.1 FLF2V for first/last."
            )
        first, last = Image.open(req.first_image_path), Image.open(req.last_image_path)
        return self._clip(self._denoise(req, ctx, kind="flf2v", image=first, last=last), req, ctx)

    def extend(self, req: ExtendRequest, ctx: JobContext) -> VideoResult:
        """Last frame → image-to-video, `segments` times; each continuation drops its first frame (a repeat of
        the previous clip's last frame), then everything is joined in one encode (Section 11.4)."""
        self._needs_image_conditioning()
        src = Path(req.video_path)
        info = probe(self.config.ffprobe, src)
        seed_frame = ctx.output_dir / "last_frame.png"
        extract_frame(self.config.ffmpeg, src, seed_frame, last=True)
        image: Image.Image = Image.open(seed_frame).convert("RGB")
        width, height = self._size(req)
        clips, durations = [src], [float(info.get("duration_s") or 0.0)]
        for k in range(req.segments):
            frames = self._denoise(req, ctx, kind="i2v", image=image, index=k, count=req.segments)
            segment = ctx.output_dir / f"segment_{k}.mp4"
            self._write(frames[1:], req.fps, segment, ctx)
            clips.append(segment)
            durations.append((frames.shape[0] - 1) / req.fps)
            image = Image.fromarray(frames[-1])
        out = ctx.output_dir / "video.mp4"
        total = sum(durations)
        args = concat_args(
            clips,
            durations,
            out,
            width=width,
            height=height,
            fps=req.fps,
            crossfade_frames=req.crossfade_frames,
        )
        run_ffmpeg(
            self.config.ffmpeg,
            args,
            total_s=total,
            cancel_event=ctx.cancel_event,
            on_progress=lambda f: ctx.report_progress(_DECODE_END + (1 - _DECODE_END) * f, "Joining clips"),
        )
        joined = probe(self.config.ffprobe, out)
        return VideoResult(
            path=str(out),
            width=width,
            height=height,
            fps=float(req.fps),
            num_frames=int(joined.get("num_frames") or round(total * req.fps)),
            has_audio=False,
            seed=ctx.seed,
        )


class _StepCallback:
    """Progress + cancel each step, and the A14B expert hand-over: once the next timestep falls below the
    pipeline's boundary, the high-noise expert is released (see `_LazyExpert`)."""

    def __init__(self, adapter: WanAdapter, reporter: StepReporter, ctx: JobContext) -> None:
        self.adapter, self.reporter, self.ctx = adapter, reporter, ctx

    def __call__(self, pipe: Any, step: int, timestep: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = self.reporter(pipe, step, timestep, kwargs)
        p = self.adapter.p
        if p is None or p.boundary_ratio is None or self.adapter.expert != "high":
            return kwargs
        timesteps = pipe.scheduler.timesteps
        boundary = p.boundary_ratio * pipe.scheduler.config.num_train_timesteps
        if step + 1 < len(timesteps) and timesteps[step + 1] < boundary:
            object.__setattr__(pipe, "transformer", None)
            self.adapter.transformer, self.adapter.expert = None, None
        return kwargs


class _LazyExpert:
    """Stands in for the pipeline's `transformer_2` and loads the low-noise expert on first use.

    Loading it from the step callback instead would briefly hold both 9.7 GB experts: the pipeline's loop
    keeps the high-noise one in a local variable until the next step starts. By the time this is called,
    that reference is gone, so the high-noise expert's memory is reused for the low-noise one."""

    def __init__(self, adapter: WanAdapter, ctx: JobContext, config: Any, dtype: torch.dtype) -> None:
        self.adapter, self.ctx, self.config, self.dtype = adapter, ctx, config, dtype

    def _real(self) -> Any:
        if self.adapter.expert != "low":
            self.adapter._load_expert("low", self.ctx)
        return self.adapter.transformer

    def cache_context(self, name: str) -> Any:
        return self._real().cache_context(name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._real()(*args, **kwargs)
