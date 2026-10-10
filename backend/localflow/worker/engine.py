"""Engine: adapter registry + ModelCache (one heavy model resident at a time, Section 10.3)."""

from __future__ import annotations

import gc
import importlib.util
import logging
import sys
import time
from typing import Any

from localflow.worker.adapters.base import (
    CAPABILITY_METHODS,
    REQUEST_MODELS,
    Adapter,
    AudioResult,
    ImageResult,
    VideoResult,
)
from localflow.worker.context import JobContext
from localflow.worker.messages import JobRequest, OutputFile, WorkerConfig, WorkerEvent
from localflow.worker.spec import LoraSpec, ModelSpec

log = logging.getLogger("localflow.worker.engine")

LOWER_PROFILE = {"quality": "balanced", "balanced": "low"}
_REQUIRED_MODULES = ("torch", "diffusers", "transformers", "accelerate", "gguf")


def gpu_unavailable_reason() -> str | None:
    """Why real inference can't run here, or None. Imports torch (needed anyway for real adapters)."""
    for mod in _REQUIRED_MODULES:
        if importlib.util.find_spec(mod) is None:
            return f"'{mod}' is not installed (run `uv sync` in backend/)."
    import torch

    if not torch.cuda.is_available():
        return "PyTorch can't see a CUDA GPU (driver or CUDA build mismatch)."
    return None


def resolve_engine(requested: str) -> tuple[str, str | None]:
    """`auto` → real when an NVIDIA GPU and the GPU stack are present, else fake. Returns (engine, reason)."""
    if requested == "fake":
        return "fake", None
    reason = gpu_unavailable_reason()
    if requested == "auto":
        return ("real", None) if reason is None else ("fake", reason)
    return "real", reason


def discover_adapters(config: WorkerConfig, engine: str, reason: str | None) -> list[Adapter]:
    if engine == "fake":
        from localflow.worker.adapters.fake import FakeAdapter

        return [FakeAdapter(config.ffmpeg, config.ffprobe, config.fake_step_delay_s)]
    from localflow.worker.adapters.flux import FluxAdapter, FluxKontextAdapter
    from localflow.worker.adapters.kokoro import KokoroAdapter
    from localflow.worker.adapters.mmaudio import MMAudioAdapter
    from localflow.worker.adapters.musicgen import MusicGenAdapter
    from localflow.worker.adapters.sdxl import SdxlAdapter
    from localflow.worker.adapters.wan import WanAdapter

    adapters: list[Adapter] = [
        FluxAdapter(config),
        SdxlAdapter(config),
        FluxKontextAdapter(config),
        WanAdapter(config),
        KokoroAdapter(config),
        MMAudioAdapter(config),
        MusicGenAdapter(config),
    ]
    if reason:
        for a in adapters:
            a.info.available, a.info.unavailable_reason = False, reason
    return adapters


def vram_snapshot() -> dict[str, int] | None:
    torch = sys.modules.get("torch")  # never import torch just to measure
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    free, total = torch.cuda.mem_get_info()
    mb = 1 << 20
    return {
        "device_used_mb": (total - free) // mb,
        "total_mb": total // mb,
        "allocated_mb": torch.cuda.memory_allocated() // mb,
        "reserved_mb": torch.cuda.memory_reserved() // mb,
    }


def _free_accelerator_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


class ModelCache:
    """Holds at most one heavy model. Key = (adapter, model entry, VRAM profile); LoRAs are swapped on the
    resident model without reloading it (Section 10.3)."""

    def __init__(self, events: Any) -> None:
        self._events = events
        self.key: tuple[str, str | None, str] | None = None
        self.adapter: Adapter | None = None
        self.baseline: dict[str, int] | None = vram_snapshot()

    def ensure(self, adapter: Adapter, spec: ModelSpec | None, profile: str, ctx: JobContext) -> None:
        key = (adapter.info.adapter_key, spec.entry_id if spec else None, profile)
        if self.key == key:
            return
        self.unload()
        name = spec.name if spec else adapter.info.name
        ctx.report_progress(0.0, f"Loading {name}…", force=True)
        t0 = time.monotonic()
        try:
            details = adapter.load(spec, profile, ctx) or {}
        except BaseException:
            adapter.unload()
            _free_accelerator_memory()
            raise
        vram = vram_snapshot()
        self.key, self.adapter = key, adapter
        load_s = round(time.monotonic() - t0, 1)
        log.info(
            "loaded %s with %s (%s profile, %s) in %.1fs; VRAM %s",
            name,
            adapter.info.adapter_key,
            profile,
            details,
            load_s,
            vram,
            extra={"job_id": ctx.job_id},
        )
        self._events.put(
            WorkerEvent(
                "model_loaded",
                {
                    "adapter_key": key[0],
                    "model_entry_id": key[1],
                    "model_name": name,
                    "vram_profile": profile,
                    "load_s": load_s,
                    "details": details,
                    "vram": vram,
                },
            )
        )

    def unload(self) -> None:
        if self.adapter is None:
            return
        old = self.key
        self.adapter.unload()
        self.adapter, self.key = None, None
        _free_accelerator_memory()
        vram = vram_snapshot()
        log.info("unloaded %s; VRAM %s (baseline %s)", old, vram, self.baseline)
        self._events.put(
            WorkerEvent(
                "model_unloaded",
                {"adapter_key": old[0] if old else None, "vram": vram, "baseline": self.baseline},
            )
        )


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, config: WorkerConfig, events: Any) -> None:
        self.config = config
        self.engine, self.reason = resolve_engine(config.engine)
        if self.engine == "real" and self.reason is None:
            _init_torch()
        self.adapters = {a.info.adapter_key: a for a in discover_adapters(config, self.engine, self.reason)}
        self.cache = ModelCache(events)

    def describe(self) -> list[dict[str, Any]]:
        return [a.info.to_dict() for a in self.adapters.values()]

    def run(self, req: JobRequest, ctx: JobContext) -> tuple[list[OutputFile], dict[str, Any]]:
        adapter = self.adapters.get(req.adapter_key)
        if adapter is None:
            raise EngineError(f"Adapter '{req.adapter_key}' is not installed in the worker.")
        if not adapter.info.available:
            raise EngineError(
                f"Adapter '{req.adapter_key}' is unavailable: {adapter.info.unavailable_reason}"
            )
        if req.job_type not in adapter.info.capabilities:
            raise EngineError(f"Adapter '{req.adapter_key}' does not support '{req.job_type}'.")
        model = REQUEST_MODELS[req.job_type].model_validate(req.params)
        spec = ModelSpec.from_dict(req.model) if req.model else None
        if adapter.requires_model and spec is None:
            raise EngineError(f"'{req.adapter_key}' needs an installed model. Pick one on the Models page.")
        loras = [LoraSpec.from_dict(d) for d in req.loras]
        if loras and not adapter.info.supports_lora:
            raise EngineError(f"Adapter '{req.adapter_key}' doesn't support LoRAs.")
        method = getattr(adapter, CAPABILITY_METHODS[req.job_type])
        profile = req.vram_profile
        retried = False
        while True:
            try:
                self.cache.ensure(adapter, spec, profile, ctx)
                ctx.vram_profile = profile
                adapter.apply_loras(loras, ctx)
                result = method(model, ctx)
                break
            except Exception as e:
                if not is_oom(e):
                    raise
                log.warning("out of GPU memory (%s profile): %s", profile, e, extra={"job_id": ctx.job_id})
                self.cache.unload()
                lower = LOWER_PROFILE.get(profile)
                if retried or lower is None:
                    raise EngineError(oom_message(req, model, spec, profile)) from None
                ctx.report_progress(
                    0.0, f"Out of GPU memory — retrying with the '{lower}' VRAM profile", force=True
                )
                profile, retried = lower, True
        outputs = _to_outputs(result)
        meta: dict[str, Any] = {
            "adapter_key": adapter.info.adapter_key,
            "adapter_version": adapter.info.adapter_version,
            "vram_profile": profile,
        }
        if spec is not None:
            meta["model_name"] = spec.name
        if loras:
            meta["loras"] = [{"model_entry_id": lo.entry_id, "weight": lo.weight} for lo in loras]
        if isinstance(result, AudioResult) and result.meta:
            meta["output"] = result.meta
        return outputs, meta

    def shutdown(self) -> None:
        self.cache.unload()


def _init_torch() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.zeros(1, device="cuda")  # create the CUDA context now, so the VRAM baseline includes it


def oom_message(req: JobRequest, model: Any, spec: ModelSpec | None, profile: str) -> str:
    """Actionable OOM error (Section 19)."""
    name = spec.name if spec else req.adapter_key
    what = req.job_type
    w, h = getattr(model, "width", None), getattr(model, "height", None)
    frames = getattr(model, "num_frames", None)
    if w and h:
        n = getattr(model, "num_images", None)
        what = f"{n} image{'s' if n and n > 1 else ''} at {w}x{h}" if n else f"{w}x{h}"
        if frames:
            what = f"a {frames}-frame {w}x{h} video"
    things = "a shorter clip" if frames else "fewer images"
    hint = f"Try {things} or a smaller resolution"
    if profile != "low":
        hint = f"Try the 'low' VRAM profile (config.toml: vram_profile), {things}, or a smaller resolution"
    head = f"Out of GPU memory generating {what} with {name} ({profile} profile)."
    return f"{head} {hint}, or close other GPU apps."


def _to_outputs(result: Any) -> list[OutputFile]:
    items = result if isinstance(result, list) else [result]
    outputs: list[OutputFile] = []
    for r in items:
        if isinstance(r, ImageResult):
            outputs.append(OutputFile("image", r.path, width=r.width, height=r.height, seed=r.seed))
        elif isinstance(r, VideoResult):
            outputs.append(
                OutputFile(
                    "video",
                    r.path,
                    width=r.width,
                    height=r.height,
                    fps=r.fps,
                    num_frames=r.num_frames,
                    duration_s=r.num_frames / r.fps if r.fps else None,
                    has_audio=r.has_audio,
                    seed=r.seed,
                )
            )
        elif isinstance(r, AudioResult):
            outputs.append(OutputFile("audio", r.path, duration_s=r.duration_s, seed=r.seed))
        else:
            raise EngineError(f"Adapter returned an unsupported result type: {type(r).__name__}")
    return outputs
