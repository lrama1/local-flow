"""Messages exchanged between the API process and the GPU worker over multiprocessing queues.

Plain dataclasses only (picklable, no torch)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JobEventKind = Literal["started", "progress", "preview", "completed", "failed", "cancelled"]
WorkerEventKind = Literal["ready", "model_loaded", "model_unloaded", "log"]


@dataclass(frozen=True)
class WorkerConfig:
    data_dir: str
    ffmpeg: str | None
    ffprobe: str | None
    engine: str
    vram_profile: str
    fake_step_delay_s: float
    log_level: str = "INFO"
    hf_home: str | None = None
    preview_every_n_steps: int = 4


@dataclass(frozen=True)
class JobRequest:
    job_id: str
    job_type: str
    adapter_key: str
    params: dict[str, Any]
    output_dir: str
    preview_dir: str
    seed: int
    vram_profile: str
    model_entry_id: str | None = None
    model: dict[str, Any] | None = None  # ModelSpec.to_dict(), resolved by the API
    loras: list[dict[str, Any]] = field(default_factory=list)  # LoraSpec.to_dict() each


@dataclass(frozen=True)
class CancelRequest:
    job_id: str


@dataclass(frozen=True)
class ShutdownRequest:
    pass


@dataclass(frozen=True)
class OutputFile:
    type: Literal["image", "video", "audio"]
    path: str
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    num_frames: int | None = None
    has_audio: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class JobEvent:
    job_id: str
    kind: JobEventKind
    fraction: float | None = None
    message: str | None = None
    preview_path: str | None = None
    outputs: list[OutputFile] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class WorkerEvent:
    kind: WorkerEventKind
    data: dict[str, Any] = field(default_factory=dict)
