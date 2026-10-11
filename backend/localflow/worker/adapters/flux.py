"""FLUX.1 [dev] text-to-image and FLUX.1 Kontext [dev] image editing (Section 10.4).

Layout (catalog): a GGUF transformer part + the FLUX.1-dev base (text encoders, tokenizers, VAE, configs).
Kontext adds a `kontext` part with its own transformer config and scheduler."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from localflow.worker.adapters.base import AdapterInfo, ImageEditRequest, ImageResult, T2IRequest
from localflow.worker.adapters.diffusion import (
    GB,
    DiffusionAdapter,
    PromptCache,
    StepReporter,
    cpu_generator,
    cuda_free_bytes,
    dir_bytes,
    free_cuda,
    latents_to_preview,
    load_gguf_state_dict,
    load_t5_fp8,
    quiet_libraries,
)
from localflow.worker.context import JobContext
from localflow.worker.images import reference_sheet, save_png, snap
from localflow.worker.messages import WorkerConfig
from localflow.worker.spec import LoraSpec, ModelSpec

log = logging.getLogger("localflow.worker.flux")

T5_FP8_BYTES = 4.7 * GB  # measured: T5-XXL encoder with fp8 Linear weights
# Measured peak reserved minus peak allocated while denoising: GGUF layers dequantize per forward, which
# fragments the caching allocator. Ignoring it put a "resident" 16 GB card 2 GB into Windows' sysmem
# fallback (no OOM, just slower and eating system RAM).
_ALLOCATOR_OVERHEAD = 2.5 * GB
_MARGIN = 0.4 * GB
# Shared by Flux and Kontext: both use the identical FLUX.1-dev text encoders (same files).
_PROMPTS = PromptCache(32)


class FluxExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_sequence_length: int = Field(default=512, ge=64, le=512, description="T5 prompt tokens")


class _FluxFamily(DiffusionAdapter):
    family = "flux"
    activation_bytes = 2.6 * GB  # 1 MP image, measured on FLUX.1-dev Q4_K_S

    def __init__(self, config: WorkerConfig) -> None:
        super().__init__(config)
        self.t5: Any = None
        self.transformer: Any = None
        self._paths: dict[str, Path] = {}

    # --- loading -----------------------------------------------------------------------------------
    def _pipeline_cls(self) -> Any:
        raise NotImplementedError

    def _resolve(self, spec: ModelSpec) -> dict[str, Path]:
        tr = spec.part("transformer")
        base = spec.part("base", "main")
        gguf = tr.first(".gguf") if tr else None
        if gguf is None or base is None:
            raise RuntimeError(
                f"{spec.name}: LocalFlow's Flux adapters need a GGUF transformer plus the FLUX.1-dev base "
                "components (install a catalog FLUX entry)."
            )
        cfg = spec.part("kontext") or base
        return {"gguf": gguf, "config": cfg.path, "base": base.path}

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

        assert model is not None
        quiet_libraries()
        paths = self._resolve(model)
        base = paths["base"]
        transformer_bytes = paths["gguf"].stat().st_size
        small = dir_bytes(base / "text_encoder") + dir_bytes(base / "vae")
        free = cuda_free_bytes()
        denoise = self.activation_bytes + _ALLOCATOR_OVERHEAD
        resident_need = transformer_bytes + T5_FP8_BYTES + small + denoise + _MARGIN
        swap_need = transformer_bytes + T5_FP8_BYTES + small + _MARGIN  # T5 and transformer, briefly
        if profile == "quality" or (profile == "balanced" and resident_need <= free):
            mode = "resident"
        elif swap_need <= free:
            mode = "swap"  # T5 only while encoding new prompts
        else:
            mode = "full_swap"  # transformer leaves the GPU while T5 encodes
        self.spec, self.mode, self._paths = model, mode, paths
        dtype = torch.bfloat16

        ctx.report_progress(0.0, "Loading text encoders…", force=True)
        clip = CLIPTextModel.from_pretrained(base / "text_encoder", dtype=dtype, device_map="cuda")
        vae = AutoencoderKL.from_pretrained(base / "vae", dtype=dtype, device_map="cuda")
        if profile == "low":
            vae.enable_tiling()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(paths["config"] / "scheduler")
        if mode == "resident":
            self.t5 = load_t5_fp8(base / "text_encoder_2")
        ctx.report_progress(0.0, "Loading transformer…", force=True)
        self._load_transformer()
        self.pipe = self._pipeline_cls()(
            scheduler=scheduler,
            vae=vae,
            text_encoder=clip,
            tokenizer=CLIPTokenizer.from_pretrained(base / "tokenizer"),
            text_encoder_2=self.t5,
            tokenizer_2=AutoTokenizer.from_pretrained(base / "tokenizer_2"),
            transformer=self.transformer,
        )
        self.pipe.set_progress_bar_config(disable=True)
        return {
            "mode": mode,
            "transformer_gb": round(transformer_bytes / GB, 2),
            "free_gb_at_load": round(free / GB, 2),
            "resident_need_gb": round(resident_need / GB, 2),
        }

    def _load_transformer(self, ctx: JobContext | None = None) -> None:
        from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

        state = load_gguf_state_dict(self._paths["gguf"], "cuda")
        self.transformer = FluxTransformer2DModel.from_single_file(
            state,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(self._paths["config"]),
            subfolder="transformer",
            dtype=torch.bfloat16,
            device="cuda",
        )
        del state
        free_cuda()
        if self.pipe is not None:
            self.pipe.transformer = self.transformer
            # LoRA adapters lived on the dropped transformer: load them again.
            specs = list(self._lora_specs)
            self._lora_names.clear()
            self._active = ()
            if specs and ctx is not None:
                self.apply_loras(specs, ctx)

    def unload(self) -> None:
        self.t5, self.transformer = None, None
        super().unload()

    # --- bundled LoRA (draft models) ---------------------------------------------------------------
    def _bundled_lora(self) -> LoraSpec | None:
        """A catalog entry's own `lora` part (e.g. Turbo-Alpha): always on, whatever the job asks for."""
        spec = self.spec
        part = spec.part("lora") if spec else None
        path = part.first(".safetensors") if part else None
        if spec is None or path is None:
            return None
        return LoraSpec(
            entry_id=f"bundled:{spec.entry_id}", name=path.parent.name or "bundled", path=str(path)
        )

    def apply_loras(self, loras: list[LoraSpec], ctx: JobContext) -> None:
        bundled = self._bundled_lora()
        if bundled is not None:
            loras = [bundled, *(lo for lo in loras if lo.entry_id != bundled.entry_id)]
        super().apply_loras(loras, ctx)

    # --- prompts -----------------------------------------------------------------------------------
    def _encode(self, prompt: str, max_len: int, ctx: JobContext) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(self._paths["base"]), prompt, max_len)
        hit = _PROMPTS.get(key)
        if hit is not None:
            return hit[0].to("cuda"), hit[1].to("cuda")
        pipe = self.pipe
        dropped = False
        if self.t5 is None:
            if self.mode == "full_swap" and self.transformer is not None:
                pipe.transformer, self.transformer, dropped = None, None, True
                free_cuda()
            ctx.report_progress(0.01, "Loading text encoder…", force=True)
            self.t5 = load_t5_fp8(self._paths["base"] / "text_encoder_2")
            pipe.text_encoder_2 = self.t5
        ctx.report_progress(0.02, "Encoding prompt…", force=True)
        with torch.inference_mode():
            pe, ppe, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device="cuda",
                num_images_per_prompt=1,
                max_sequence_length=max_len,
            )
        _PROMPTS.put(key, (pe, ppe))
        if self.mode != "resident":
            pipe.text_encoder_2, self.t5 = None, None
            free_cuda()
        if dropped:
            ctx.report_progress(0.02, "Reloading transformer…", force=True)
            self._load_transformer(ctx)
        return pe, ppe

    def _preview_fn(self, height: int, width: int) -> Any:
        pipe = self.pipe

        def preview(latents: torch.Tensor) -> Image.Image:
            unpacked = pipe._unpack_latents(latents[:1], height, width, pipe.vae_scale_factor)
            return latents_to_preview(unpacked, "flux")

        return preview

    def _defaults(self, req: Any) -> tuple[int, float, int]:
        assert self.spec is not None
        steps = req.steps or int(self.spec.default("steps", 28))
        guidance = (
            req.guidance_scale
            if req.guidance_scale is not None
            else float(self.spec.default("guidance_scale", 3.5))
        )
        extra = FluxExtra.model_validate(req.extra or {})
        return steps, guidance, extra.max_sequence_length


class FluxAdapter(_FluxFamily):
    info = AdapterInfo(
        adapter_key="flux",
        adapter_version="1.0.0",
        name="FLUX.1",
        capabilities=["t2i"],
        supported_base_models=["flux1"],
        param_schema=FluxExtra.model_json_schema(),
        vram_profiles={
            "quality": {"placement": "resident"},
            "balanced": {"placement": "resident if it fits, else T5 swapped per prompt"},
            "low": {"placement": "T5 swapped per prompt, tiled VAE"},
        },
        supports_lora=True,
    )

    def _pipeline_cls(self) -> Any:
        from diffusers import FluxPipeline

        return FluxPipeline

    def generate_t2i(self, req: T2IRequest, ctx: JobContext) -> list[ImageResult]:
        assert self.pipe is not None and self.spec is not None
        width, height = snap(req.width, 16), snap(req.height, 16)
        steps, guidance, max_len = self._defaults(req)
        pe, ppe = self._encode(req.prompt, max_len, ctx)
        results: list[ImageResult] = []
        for i in range(req.num_images):
            seed = (ctx.seed + i) % (2**32)
            with torch.inference_mode():
                out = self.pipe(
                    prompt_embeds=pe,
                    pooled_prompt_embeds=ppe,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=cpu_generator(seed),
                    callback_on_step_end=StepReporter(
                        ctx,
                        index=i,
                        count=req.num_images,
                        steps=steps,
                        preview=self._preview_fn(height, width),
                    ),
                )
            path = ctx.output_dir / f"image_{i:02d}.png"
            meta = self.png_meta(prompt=req.prompt, seed=seed, steps=steps, guidance_scale=guidance)
            save_png(out.images[0], path, meta)
            results.append(ImageResult(str(path), width, height, seed))
        return results


class FluxKontextAdapter(_FluxFamily):
    activation_bytes = 4.2 * GB  # reference image tokens double the sequence length
    info = AdapterInfo(
        adapter_key="flux_kontext",
        adapter_version="1.0.0",
        name="FLUX.1 Kontext",
        capabilities=["image_edit"],
        supported_base_models=["flux1-kontext"],
        param_schema=FluxExtra.model_json_schema(),
        vram_profiles={
            "quality": {"placement": "resident"},
            "balanced": {"placement": "resident if it fits, else T5 swapped per prompt"},
            "low": {"placement": "T5 swapped per prompt, tiled VAE"},
        },
        supports_lora=True,
    )

    def _pipeline_cls(self) -> Any:
        from diffusers import FluxKontextPipeline

        return FluxKontextPipeline

    def edit(self, req: ImageEditRequest, ctx: JobContext) -> list[ImageResult]:
        assert self.pipe is not None and self.spec is not None
        refs = [Image.open(p).convert("RGB") for p in req.reference_image_paths]
        # Kontext takes one reference image: several references become one side-by-side sheet.
        reference = reference_sheet(refs)
        width, height = snap(req.width, 16), snap(req.height, 16)
        steps, guidance, max_len = self._defaults(req)
        if req.guidance_scale is None:
            guidance = float(self.spec.default("guidance_scale", 2.5))
        pe, ppe = self._encode(req.prompt, max_len, ctx)
        results: list[ImageResult] = []
        for i in range(req.num_images):
            seed = (ctx.seed + i) % (2**32)
            with torch.inference_mode():
                out = self.pipe(
                    image=reference,
                    prompt_embeds=pe,
                    pooled_prompt_embeds=ppe,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=cpu_generator(seed),
                    max_area=width * height,
                    callback_on_step_end=StepReporter(
                        ctx,
                        index=i,
                        count=req.num_images,
                        steps=steps,
                        preview=self._preview_fn(height, width),
                    ),
                )
            path = ctx.output_dir / f"image_{i:02d}.png"
            meta = self.png_meta(
                prompt=req.prompt, seed=seed, steps=steps, guidance_scale=guidance, references=len(refs)
            )
            save_png(out.images[0], path, meta)
            results.append(ImageResult(str(path), width, height, seed))
        return results
