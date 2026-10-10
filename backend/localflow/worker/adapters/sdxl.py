"""SDXL text-to-image: the catalog base (diffusers layout, fp16) and Civitai SDXL/Pony/Illustrious
single-file checkpoints, with LoRAs (Section 10.4)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from localflow.worker.adapters.base import AdapterInfo, ImageResult, T2IRequest
from localflow.worker.adapters.diffusion import (
    DiffusionAdapter,
    StepReporter,
    cpu_generator,
    free_cuda,
    latents_to_preview,
    quiet_libraries,
)
from localflow.worker.context import JobContext
from localflow.worker.images import save_png, snap
from localflow.worker.spec import ModelSpec

log = logging.getLogger("localflow.worker.sdxl")

SDXL_HUB_CONFIG = "stabilityai/stable-diffusion-xl-base-1.0"
SCHEDULERS = ("euler", "euler_a", "dpmpp_2m", "dpmpp_2m_karras", "ddim")


class SdxlExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _make_scheduler(name: str, config: Any) -> Any:
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
    )

    if name == "euler":
        return EulerDiscreteScheduler.from_config(config)
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config)
    if name == "dpmpp_2m":
        return DPMSolverMultistepScheduler.from_config(config)
    if name == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
    if name == "ddim":
        return DDIMScheduler.from_config(config)
    raise ValueError(f"Unknown scheduler '{name}'. Choose one of: {', '.join(SCHEDULERS)}.")


class SdxlAdapter(DiffusionAdapter):
    family = "sdxl"
    info = AdapterInfo(
        adapter_key="sdxl",
        adapter_version="1.0.0",
        name="SDXL / Civitai checkpoints",
        capabilities=["t2i"],
        supported_base_models=["sdxl", "pony", "illustrious", "noobai"],
        param_schema=SdxlExtra.model_json_schema(),
        vram_profiles={
            "quality": {"placement": "resident"},
            "balanced": {"placement": "resident"},
            "low": {"placement": "resident, tiled/sliced VAE"},
        },
        supports_lora=True,
    )

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._base_scheduler_config: Any = None
        self._scheduler_name: str | None = None

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        assert model is not None
        quiet_libraries()
        dtype = torch.float16
        vae_dir = model.aux.get("vae_dir")
        vae_part = model.part("vae")
        if vae_part is not None:
            vae_dir = vae_part.local_dir
        vae = AutoencoderKL.from_pretrained(vae_dir, dtype=dtype) if vae_dir else None
        if model.single_file:
            config = model.aux.get("sdxl_config_dir") or SDXL_HUB_CONFIG
            kwargs: dict[str, Any] = {"dtype": dtype, "config": config}
            if vae is not None:
                kwargs["vae"] = vae
            ctx.report_progress(0.0, "Loading checkpoint…", force=True)
            pipe = StableDiffusionXLPipeline.from_single_file(model.single_file, **kwargs)
            pipe.to("cuda")
            source = "single_file"
        else:
            base = model.part("base", "main")
            if base is None:
                raise RuntimeError(f"{model.name}: no SDXL weights found in the model entry.")
            variant = "fp16" if any(".fp16." in f for f in base.files) else None
            kwargs = {"dtype": dtype, "variant": variant, "use_safetensors": True, "device_map": "cuda"}
            if vae is not None:
                kwargs["vae"] = vae
            pipe = StableDiffusionXLPipeline.from_pretrained(base.local_dir, **kwargs)
            source = "diffusers"
        if vae is not None:
            pipe.vae.to("cuda")
        if profile == "low":
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.spec, self.mode = pipe, model, "resident"
        self._base_scheduler_config = pipe.scheduler.config
        self._scheduler_name = None
        free_cuda()
        return {"mode": "resident", "source": source, "vae": Path(vae_dir).name if vae_dir else "built-in"}

    def _set_scheduler(self, name: str | None) -> None:
        name = name or "default"
        if name == self._scheduler_name:
            return
        if name == "default":
            from diffusers import EulerDiscreteScheduler

            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self._base_scheduler_config)
        else:
            self.pipe.scheduler = _make_scheduler(name, self._base_scheduler_config)
        self._scheduler_name = name

    def generate_t2i(self, req: T2IRequest, ctx: JobContext) -> list[ImageResult]:
        assert self.pipe is not None and self.spec is not None
        width, height = snap(req.width, 8), snap(req.height, 8)
        steps = req.steps or int(self.spec.default("steps", 30))
        guidance = (
            req.guidance_scale
            if req.guidance_scale is not None
            else float(self.spec.default("guidance_scale", 6.0))
        )
        self._set_scheduler(req.scheduler)

        def preview(latents: torch.Tensor) -> Any:
            return latents_to_preview(latents[:1], "sdxl")

        results: list[ImageResult] = []
        for i in range(req.num_images):
            seed = (ctx.seed + i) % (2**32)
            with torch.inference_mode():
                out = self.pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt or None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=cpu_generator(seed),
                    callback_on_step_end=StepReporter(
                        ctx, index=i, count=req.num_images, steps=steps, preview=preview
                    ),
                )
            path = ctx.output_dir / f"image_{i:02d}.png"
            meta = self.png_meta(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance,
                scheduler=self._scheduler_name,
            )
            save_png(out.images[0], path, meta)
            results.append(ImageResult(str(path), width, height, seed))
        return results
