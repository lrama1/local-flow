"""Shared helpers for diffusers-based adapters. Imports torch; only loaded by the real engine.

Memory strategy for a 16 GB GPU with little free system RAM (DECISIONS M3):
- Weights stream from safetensors/GGUF straight to the GPU, tensor by tensor, instead of being staged in RAM.
- The T5-XXL text encoder is stored in fp8 (computed in bf16): ~4.9 GB instead of ~11.5 GB.
- Prompt embeddings are cached, so the text encoder is only needed for new prompts."""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import types
import warnings
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from localflow.worker.adapters.base import Adapter
from localflow.worker.context import JobContext
from localflow.worker.messages import WorkerConfig
from localflow.worker.spec import LoraSpec, ModelSpec

log = logging.getLogger("localflow.worker.diffusion")

GB = 1e9


def quiet_libraries() -> None:
    """No tqdm bars from diffusers/transformers in the worker log (progress goes through JobContext)."""
    from diffusers.utils import logging as diffusers_logging
    from transformers.utils import logging as transformers_logging

    diffusers_logging.disable_progress_bar()
    transformers_logging.disable_progress_bar()


# Linear latent→RGB projections for cheap step previews (same idea as ComfyUI's latent previews).
LATENT_RGB: dict[str, tuple[list[list[float]], list[float]]] = {
    "sdxl": (
        [
            [0.3651, 0.4232, 0.4341],
            [-0.2533, -0.0042, 0.1068],
            [0.1076, 0.1111, -0.0362],
            [-0.3165, -0.2492, -0.2188],
        ],
        [0.1084, -0.0175, -0.0011],
    ),
    "flux": (
        [
            [-0.0346, 0.0244, 0.0681],
            [0.0034, 0.0210, 0.0687],
            [0.0275, -0.0668, -0.0433],
            [-0.0174, 0.0160, 0.0617],
            [0.0859, 0.0721, 0.0329],
            [0.0004, 0.0383, 0.0115],
            [0.0405, 0.0861, 0.0915],
            [-0.0236, -0.0185, -0.0259],
            [-0.0245, 0.0250, 0.1180],
            [0.1008, 0.0755, -0.0421],
            [-0.0515, 0.0201, 0.0011],
            [0.0428, -0.0012, -0.0036],
            [0.0817, 0.0765, 0.0749],
            [-0.1264, -0.0522, -0.1103],
            [-0.0280, -0.0881, -0.0499],
            [-0.1262, -0.0982, -0.0778],
        ],
        [-0.0329, -0.0718, -0.0851],
    ),
}


def latents_to_preview(latents: torch.Tensor, family: str, scale: int = 4) -> Image.Image:
    """(C, H, W) or (1, C, H, W) latents → small RGB image."""
    factors, bias = LATENT_RGB[family]
    lat = latents[0] if latents.dim() == 4 else latents
    lat = lat.detach().float()
    f = torch.tensor(factors, device=lat.device, dtype=lat.dtype)
    b = torch.tensor(bias, device=lat.device, dtype=lat.dtype)
    rgb = torch.einsum("chw,cr->rhw", lat, f) + b[:, None, None]
    arr = ((rgb.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.BILINEAR)


class StepReporter:
    """`callback_on_step_end`: progress, cooperative cancel, and periodic latent previews (Section 10.4)."""

    def __init__(
        self,
        ctx: JobContext,
        *,
        index: int,
        count: int,
        steps: int,
        preview: Callable[[torch.Tensor], Image.Image] | None = None,
        start: float = 0.02,
        end: float = 0.98,
        noun: str = "Image",
    ) -> None:
        self.ctx, self.index, self.count, self.steps = ctx, index, count, max(1, steps)
        self.preview = preview
        self.every = int(ctx.settings.get("preview_every_n_steps") or 0)
        self.start, self.end, self.noun = start, end, noun

    def fraction(self, done_in_item: float) -> float:
        return self.start + (self.end - self.start) * (self.index + done_in_item) / self.count

    def __call__(self, pipe: Any, step: int, timestep: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.ctx.check_cancelled()
        label = f"{self.noun} {self.index + 1}/{self.count} · " if self.count > 1 else ""
        self.ctx.report_progress(
            self.fraction((step + 1) / self.steps), f"{label}step {step + 1}/{self.steps}"
        )
        if self.preview and self.every and (step + 1) % self.every == 0 and step + 1 < self.steps:
            latents = kwargs.get("latents")
            if latents is not None:
                try:
                    self.ctx.report_preview(self.preview(latents))
                except Exception as e:  # previews are best effort
                    log.debug("preview failed: %s", e)
        return kwargs


def cpu_generator(seed: int) -> torch.Generator:
    """CPU generators make seeds portable across GPUs (Section 10.5)."""
    return torch.Generator(device="cpu").manual_seed(int(seed) % (2**32))


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def cuda_free_bytes() -> int:
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def dir_bytes(path: Path, pattern: str = "*.safetensors") -> int:
    return sum(p.stat().st_size for p in path.glob(pattern)) if path.is_dir() else 0


# --- streaming loaders -----------------------------------------------------------------------------
def _safetensors_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [model_dir / f for f in sorted(set(weight_map.values()))]
    return sorted(model_dir.glob("*.safetensors"))


def _t5_ff_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    # transformers' T5DenseGatedActDense casts activations to `wo.weight.dtype` (meant for fp32 `wo`);
    # with fp8 storage that would feed fp8 activations into the matmul. Compute in the input dtype instead.
    hidden_gelu = self.act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = self.dropout(hidden_gelu * hidden_linear)
    return self.wo(hidden_states)


def load_t5_fp8(model_dir: Path, device: str = "cuda") -> Any:
    """T5-XXL (Flux) or UMT5-XXL (Wan) encoder with Linear weights stored as float8_e4m3fn and computed in
    bf16.

    Tensors stream from safetensors straight to `device` and are cast one at a time, so neither system RAM
    nor VRAM ever holds the bf16 model (~9.5 GB for T5, ~11.4 GB for UMT5)."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers.hooks import apply_layerwise_casting
    from safetensors import safe_open
    from transformers import T5Config, T5EncoderModel, UMT5Config, UMT5EncoderModel

    arch = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("architectures") or []
    umt5 = any("UMT5" in a for a in arch)
    config_cls: Any = UMT5Config if umt5 else T5Config
    model_cls: Any = UMT5EncoderModel if umt5 else T5EncoderModel
    config = config_cls.from_pretrained(model_dir)
    with init_empty_weights():
        model = model_cls(config)
    linear_weights = {f"{name}.weight" for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    for file in _safetensors_files(model_dir):
        with safe_open(str(file), framework="pt", device=device) as f:
            for key in f.keys():  # noqa: SIM118 - safe_open isn't a dict
                if key not in expected:
                    continue
                t = f.get_tensor(key)
                dtype = torch.float8_e4m3fn if key in linear_weights else torch.bfloat16
                # `dtype=` is required: otherwise accelerate casts to the empty model's fp32.
                set_module_tensor_to_device(model, key, device, value=t.to(dtype), dtype=dtype)
                loaded.add(key)
                del t
    torch.cuda.empty_cache()
    # The checkpoint stores the shared embedding once; tie the encoder's copy to it.
    if "shared.weight" in loaded and "encoder.embed_tokens.weight" not in loaded:
        model.encoder.embed_tokens.weight = model.shared.weight
        loaded.add("encoder.embed_tokens.weight")
    missing = expected - loaded
    if missing:
        raise RuntimeError(f"T5 checkpoint is missing {len(missing)} tensors (e.g. {sorted(missing)[:3]})")
    for module in model.modules():
        if type(module).__name__ in ("T5DenseGatedActDense", "UMT5DenseGatedActDense"):
            module.forward = types.MethodType(_t5_ff_forward, module)
    # Upcast fp8 Linear weights to bf16 just for each layer's forward.
    apply_layerwise_casting(
        model,
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        skip_modules_pattern=("norm", "embed", "shared", "relative_attention_bias"),
    )
    model.eval()
    model.requires_grad_(False)
    return model


def load_gguf_state_dict(path: Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    """GGUF tensors read from the memory map and copied straight to `device` (diffusers' loader would first
    copy the whole file into system RAM)."""
    import gguf
    import numpy as np
    from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter

    reader = gguf.GGUFReader(str(path))
    out: dict[str, torch.Tensor] = {}
    plain = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)  # same split as diffusers
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", UserWarning
        )  # read-only memmap → non-writable tensor; we only copy it
        for tensor in reader.tensors:
            qtype = tensor.tensor_type
            is_quant = qtype not in plain
            if is_quant and qtype not in SUPPORTED_GGUF_QUANT_TYPES:
                raise ValueError(f"{tensor.name}: unsupported GGUF quantization {qtype!r}")
            t = torch.from_numpy(np.asarray(tensor.data)).to(device)
            out[tensor.name] = GGUFParameter(t, quant_type=qtype) if is_quant else t
    del reader
    return out


class PromptCache:
    """LRU of text-encoder outputs (kept on CPU; a Flux prompt is ~4 MB)."""

    def __init__(self, size: int = 32) -> None:
        self.size = size
        self._data: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, ...]] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> tuple[torch.Tensor, ...] | None:
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key: tuple[Any, ...], value: tuple[torch.Tensor, ...]) -> None:
        self._data[key] = tuple(v.detach().to("cpu") for v in value)
        self._data.move_to_end(key)
        while len(self._data) > self.size:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


class DiffusionAdapter(Adapter):
    """Common state for diffusers pipelines: the pipeline, placement mode, and LoRA bookkeeping."""

    requires_model = True
    family = "sdxl"

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.pipe: Any = None
        self.spec: ModelSpec | None = None
        self.mode: str | None = None
        self._lora_names: dict[str, str] = {}
        self._lora_specs: list[LoraSpec] = []
        self._active: tuple[tuple[str, float], ...] = ()

    def unload(self) -> None:
        pipe, self.pipe = self.pipe, None
        if pipe is not None:
            with contextlib.suppress(Exception):
                pipe.maybe_free_model_hooks()
            del pipe
        self.spec, self.mode = None, None
        self._lora_names.clear()
        self._lora_specs = []
        self._active = ()
        free_cuda()

    def lora_target(self) -> Any:
        return self.pipe

    def png_meta(self, **fields: Any) -> dict[str, Any]:
        """Generation metadata embedded in each PNG: enough to reproduce it (model, LoRAs, sampling)."""
        assert self.spec is not None
        meta: dict[str, Any] = {**fields, "model": self.spec.name}
        if self._lora_specs:
            meta["loras"] = [{"name": lo.name, "weight": lo.weight} for lo in self._lora_specs]
        return meta

    def apply_loras(self, loras: list[LoraSpec], ctx: JobContext) -> None:
        pipe = self.lora_target()
        if pipe is None:
            return
        want = tuple((lo.entry_id, round(lo.weight, 3)) for lo in loras)
        if want == self._active:
            return
        for lo in loras:
            if lo.entry_id not in self._lora_names:
                ctx.report_progress(0.0, f"Loading LoRA {lo.name}…", force=True)
                name = f"lora_{len(self._lora_names)}"
                pipe.load_lora_weights(lo.path, adapter_name=name)
                self._lora_names[lo.entry_id] = name
        if loras:
            pipe.enable_lora()
            pipe.set_adapters(
                [self._lora_names[lo.entry_id] for lo in loras], adapter_weights=[lo.weight for lo in loras]
            )
        elif self._lora_names:
            pipe.disable_lora()
        self._active = want
        self._lora_specs = list(loras)
        log.info("active LoRAs: %s", want or "none", extra={"job_id": ctx.job_id})
