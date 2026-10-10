"""Adapter metadata and base-model inference for Hugging Face repos and Civitai versions (Section 9.1/9.4)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdapterMeta:
    key: str
    name: str
    capabilities: tuple[str, ...]
    base_models: tuple[str, ...]
    supports_lora: bool = False
    milestone: str = "M3"


# Adapters the registry knows about. The GPU worker reports which are actually loadable (`/models/adapters`).
KNOWN_ADAPTERS: dict[str, AdapterMeta] = {
    a.key: a
    for a in (
        AdapterMeta("flux", "FLUX.1", ("t2i",), ("flux1",), supports_lora=True),
        AdapterMeta(
            "sdxl",
            "SDXL / Civitai checkpoints",
            ("t2i",),
            ("sdxl", "pony", "illustrious", "noobai"),
            supports_lora=True,
        ),
        AdapterMeta(
            "flux_kontext", "FLUX.1 Kontext", ("image_edit",), ("flux1-kontext",), supports_lora=True
        ),
        AdapterMeta(
            "wan",
            "Wan video",
            ("t2v", "i2v", "flf2v", "extend"),
            (
                "wan2.2-5b",
                "wan2.2-a14b-t2v",
                "wan2.2-a14b-i2v",
                "wan2.1-flf2v-14b",
                "wan2.1-14b-t2v",
                "wan2.1-14b-i2v",
                "wan2.1-1.3b",
            ),
            supports_lora=True,
            milestone="M4",
        ),
        AdapterMeta(
            "wan_vace", "Wan VACE", ("ref2v",), ("wan2.1-vace-1.3b", "wan2.1-vace-14b"), milestone="M7"
        ),
        AdapterMeta("kokoro", "Kokoro TTS", ("tts",), ("kokoro",), milestone="M6"),
        AdapterMeta("mmaudio", "MMAudio", ("v2a",), ("mmaudio",), milestone="M6"),
        AdapterMeta("ace_step", "ACE-Step", ("music",), ("ace-step",), milestone="M6"),
        AdapterMeta("musicgen", "MusicGen", ("music",), ("musicgen",), milestone="M6"),
    )
}


def adapter_for_base(base_model: str | None) -> str | None:
    if not base_model:
        return None
    for a in KNOWN_ADAPTERS.values():
        if base_model in a.base_models:
            return a.key
    return None


def capabilities_for(adapter_key: str | None, base_model: str | None) -> list[str]:
    a = KNOWN_ADAPTERS.get(adapter_key or "")
    if a is None:
        return []
    if a.key == "wan" and base_model:
        if base_model.endswith("-t2v"):
            return ["t2v"]
        if base_model.endswith("-i2v"):
            return ["i2v", "extend"]
        if "flf2v" in base_model:
            return ["flf2v"]
        if base_model == "wan2.2-5b":
            return ["t2v", "i2v", "extend"]
    return list(a.capabilities)


@dataclass
class Suggestion:
    kind: str = "checkpoint"
    base_model: str | None = None
    adapter_key: str | None = None
    capabilities: list[str] = field(default_factory=list)
    supported: bool = False
    reason: str = ""


# --- Civitai -------------------------------------------------------------------------------------
_CIVITAI_BASE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^sdxl"), "sdxl"),
    (re.compile(r"^pony"), "pony"),
    (re.compile(r"^illustrious"), "illustrious"),
    (re.compile(r"^noobai"), "noobai"),
    (re.compile(r"^flux\.1 kontext"), "flux1-kontext"),
    (re.compile(r"^flux\.1 (d|s|dev|schnell|krea)\b"), "flux1"),
    (re.compile(r"^wan video 2\.2 ti2v-5b"), "wan2.2-5b"),
    (re.compile(r"^wan video 2\.2 t2v-a14b"), "wan2.2-a14b-t2v"),
    (re.compile(r"^wan video 2\.2 i2v-a14b"), "wan2.2-a14b-i2v"),
    (re.compile(r"^wan video 14b i2v"), "wan2.1-14b-i2v"),
    (re.compile(r"^wan video 14b t2v"), "wan2.1-14b-t2v"),
    (re.compile(r"^wan video 1\.3b"), "wan2.1-1.3b"),
]

CIVITAI_KINDS = {
    "Checkpoint": "checkpoint",
    "LORA": "lora",
    "LoCon": "lora",
    "DoRA": "lora",
    "VAE": "vae",
}


def civitai_base_model(raw: str | None) -> str | None:
    s = (raw or "").strip().lower()
    for pat, base in _CIVITAI_BASE:
        if pat.search(s):
            return base
    return None


def suggest_civitai(model_type: str, raw_base: str | None) -> Suggestion:
    kind = CIVITAI_KINDS.get(model_type)
    base = civitai_base_model(raw_base)
    adapter = adapter_for_base(base)
    if kind is None:
        return Suggestion(kind="other", reason=f"Civitai type '{model_type}' isn't supported.")
    if adapter is None:
        return Suggestion(
            kind=kind,
            base_model=_slug(raw_base) if raw_base else None,
            reason=f"Base model '{raw_base}' isn't supported by any LocalFlow adapter (still downloadable).",
        )
    return Suggestion(
        kind=kind,
        base_model=base,
        adapter_key=adapter,
        capabilities=capabilities_for(adapter, base),
        supported=True,
        reason=f"Civitai base model '{raw_base}'",
    )


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")


# --- Hugging Face --------------------------------------------------------------------------------
_HF_PIPELINES: dict[str, tuple[str, str | None]] = {
    "FluxPipeline": ("flux1", "flux"),
    "FluxKontextPipeline": ("flux1-kontext", "flux_kontext"),
    "StableDiffusionXLPipeline": ("sdxl", "sdxl"),
    "WanPipeline": ("wan2.1-14b-t2v", "wan"),
    "WanImageToVideoPipeline": ("wan2.1-14b-i2v", "wan"),
    "WanVACEPipeline": ("wan2.1-vace-1.3b", "wan_vace"),
}
_HF_REPO_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"flux\.?1[-_.]?kontext", re.I), "flux1-kontext"),
    (re.compile(r"flux\.?1[-_.]?(dev|schnell|krea)", re.I), "flux1"),
    (re.compile(r"stable-diffusion-xl|sdxl", re.I), "sdxl"),
    (re.compile(r"wan2\.2[-_]ti2v[-_]5b", re.I), "wan2.2-5b"),
    (re.compile(r"wan2\.2[-_]t2v[-_]a14b", re.I), "wan2.2-a14b-t2v"),
    (re.compile(r"wan2\.2[-_]i2v[-_]a14b", re.I), "wan2.2-a14b-i2v"),
    (re.compile(r"wan2\.1[-_]flf2v", re.I), "wan2.1-flf2v-14b"),
    (re.compile(r"wan2\.1[-_]vace[-_]1\.3b", re.I), "wan2.1-vace-1.3b"),
    (re.compile(r"wan2\.1[-_]vace[-_]14b", re.I), "wan2.1-vace-14b"),
    (re.compile(r"wan2\.1[-_]t2v[-_]14b", re.I), "wan2.1-14b-t2v"),
    (re.compile(r"wan2\.1[-_]i2v[-_]14b", re.I), "wan2.1-14b-i2v"),
    (re.compile(r"wan2\.1[-_]t2v[-_]1\.3b", re.I), "wan2.1-1.3b"),
    (re.compile(r"kokoro", re.I), "kokoro"),
    (re.compile(r"mmaudio", re.I), "mmaudio"),
    (re.compile(r"ace-step", re.I), "ace-step"),
    (re.compile(r"musicgen", re.I), "musicgen"),
]


def suggest_hf(repo_id: str, info: dict[str, Any]) -> Suggestion:
    """`info`: {tags, pipeline_tag, library_name, base_models, files} from HF model metadata."""
    tags = [str(t) for t in info.get("tags") or []]
    files = [str(f) for f in info.get("files") or []]
    base: str | None = None
    adapter: str | None = None
    reason = ""
    for t in tags:
        if t.startswith("diffusers:") and t.split(":", 1)[1] in _HF_PIPELINES:
            base, adapter = _HF_PIPELINES[t.split(":", 1)[1]]
            reason = f"diffusers pipeline {t.split(':', 1)[1]}"
    for candidate in [repo_id, *[str(b) for b in info.get("base_models") or []]]:
        for pat, b in _HF_REPO_HINTS:
            if pat.search(candidate):
                base, reason = b, f"name/base model '{candidate}'"
                break
        else:
            continue
        break
    adapter = adapter_for_base(base) or adapter
    is_lora = "lora" in tags or any("lora" in f.lower() for f in files if f.endswith(".safetensors"))
    is_vae = "vae" in repo_id.lower() and not any(f == "model_index.json" for f in files)
    pipeline = info.get("pipeline_tag") or ""
    if is_lora:
        kind = "lora"
    elif is_vae:
        kind = "vae"
    elif pipeline in ("text-to-speech", "text-to-audio") or base in (
        "kokoro",
        "mmaudio",
        "ace-step",
        "musicgen",
    ):
        kind = "audio"
    else:
        kind = "checkpoint"
    return Suggestion(
        kind=kind,
        base_model=base,
        adapter_key=adapter,
        capabilities=capabilities_for(adapter, base) if kind in ("checkpoint", "lora", "audio") else [],
        supported=adapter is not None,
        reason=reason or "no known base model; choose the adapter manually",
    )
