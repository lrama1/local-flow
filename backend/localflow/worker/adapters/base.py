"""Capability interfaces, request/result types, and adapter metadata (Section 10).

This module must stay importable without torch: the API process uses the request models for validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from localflow.worker.context import JobContext
    from localflow.worker.spec import LoraSpec, ModelSpec

Capability = Literal["t2i", "image_edit", "t2v", "i2v", "flf2v", "ref2v", "extend", "tts", "v2a", "music"]
GPU_JOB_TYPES: tuple[Capability, ...] = (
    "t2i",
    "image_edit",
    "t2v",
    "i2v",
    "flf2v",
    "ref2v",
    "extend",
    "tts",
    "v2a",
    "music",
)


# --- Requests (Section 10.2) -------------------------------------------------------------------
class LoraRef(BaseModel):
    model_entry_id: str
    weight: float = 1.0


class BaseGenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_entry_id: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, ge=1, le=500)
    guidance_scale: float | None = Field(default=None, ge=0, le=50)
    loras: list[LoraRef] = Field(default_factory=list)
    scheduler: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ImageParams(BaseGenRequest):
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    num_images: int = Field(default=1, ge=1, le=8)


class VideoParams(BaseGenRequest):
    width: int = Field(default=1280, ge=64, le=4096)
    height: int = Field(default=720, ge=64, le=4096)
    num_frames: int | None = Field(default=None, ge=1, le=2000)
    duration_s: float | None = Field(default=None, gt=0, le=120)
    fps: int = Field(default=24, ge=1, le=120)
    with_audio: bool = False

    def resolved_frames(self, default_duration_s: float = 4.0) -> int:
        if self.num_frames:
            return self.num_frames
        return max(1, round((self.duration_s or default_duration_s) * self.fps))


class T2IRequest(ImageParams):
    pass


class ImageEditRequest(ImageParams):
    reference_image_paths: list[str] = Field(min_length=1)


class T2VRequest(VideoParams):
    pass


class I2VRequest(VideoParams):
    image_path: str


class FLF2VRequest(VideoParams):
    first_image_path: str
    last_image_path: str


class Ref2VRequest(VideoParams):
    reference_image_paths: list[str] = Field(min_length=1)


class ExtendParams(VideoParams):
    """Extend (Section 11.4): `num_frames`/`duration_s` is the length of each continuation."""

    segments: int = Field(default=1, ge=1, le=8, description="Continuations to append, one after another")
    crossfade_frames: int = Field(
        default=0, ge=0, le=8, description="Blend this many frames at each seam (0 = cut)"
    )


class ExtendRequest(ExtendParams):
    video_path: str


class TTSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_entry_id: str | None = None
    text: str = Field(min_length=1, max_length=5000)
    voice_id: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    extra: dict[str, Any] = Field(default_factory=dict)


class V2AParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_entry_id: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, ge=1, le=500)
    extra: dict[str, Any] = Field(default_factory=dict)


class V2ARequest(V2AParams):
    video_path: str


class MusicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_entry_id: str | None = None
    prompt: str = ""
    duration_s: float = Field(default=30.0, gt=0, le=600)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, ge=1, le=500)
    extra: dict[str, Any] = Field(default_factory=dict)


REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "t2i": T2IRequest,
    "image_edit": ImageEditRequest,
    "t2v": T2VRequest,
    "i2v": I2VRequest,
    "flf2v": FLF2VRequest,
    "ref2v": Ref2VRequest,
    "extend": ExtendRequest,
    "tts": TTSRequest,
    "v2a": V2ARequest,
    "music": MusicRequest,
}


# --- Results -----------------------------------------------------------------------------------
@dataclass
class ImageResult:
    path: str
    width: int
    height: int
    seed: int


@dataclass
class VideoResult:
    path: str
    width: int
    height: int
    fps: float
    num_frames: int
    has_audio: bool
    seed: int


@dataclass
class AudioResult:
    path: str
    duration_s: float
    sample_rate: int
    seed: int


# --- Capability interfaces (Section 10.1) ------------------------------------------------------
@runtime_checkable
class TextToImage(Protocol):
    def generate_t2i(self, req: T2IRequest, ctx: JobContext) -> list[ImageResult]: ...


@runtime_checkable
class ImageEdit(Protocol):
    def edit(self, req: ImageEditRequest, ctx: JobContext) -> list[ImageResult]: ...


@runtime_checkable
class TextToVideo(Protocol):
    def generate_t2v(self, req: T2VRequest, ctx: JobContext) -> VideoResult: ...


@runtime_checkable
class ImageToVideo(Protocol):
    def generate_i2v(self, req: I2VRequest, ctx: JobContext) -> VideoResult: ...


@runtime_checkable
class FirstLastFrameToVideo(Protocol):
    def generate_flf2v(self, req: FLF2VRequest, ctx: JobContext) -> VideoResult: ...


@runtime_checkable
class ReferenceToVideo(Protocol):
    def generate_ref2v(self, req: Ref2VRequest, ctx: JobContext) -> VideoResult: ...


@runtime_checkable
class VideoExtend(Protocol):
    def extend(self, req: ExtendRequest, ctx: JobContext) -> VideoResult: ...


@runtime_checkable
class TextToSpeech(Protocol):
    def synthesize(self, req: TTSRequest, ctx: JobContext) -> AudioResult: ...


@runtime_checkable
class VideoToAudio(Protocol):
    def generate_v2a(self, req: V2ARequest, ctx: JobContext) -> AudioResult: ...


@runtime_checkable
class TextToMusic(Protocol):
    def generate_music(self, req: MusicRequest, ctx: JobContext) -> AudioResult: ...


# Capability -> adapter method name. A single adapter class may implement several capabilities.
CAPABILITY_METHODS: dict[str, str] = {
    "t2i": "generate_t2i",
    "image_edit": "edit",
    "t2v": "generate_t2v",
    "i2v": "generate_i2v",
    "flf2v": "generate_flf2v",
    "ref2v": "generate_ref2v",
    "extend": "extend",
    "tts": "synthesize",
    "v2a": "generate_v2a",
    "music": "generate_music",
}


# --- Adapter metadata --------------------------------------------------------------------------
@dataclass
class AdapterInfo:
    adapter_key: str
    adapter_version: str
    name: str
    capabilities: list[str]
    supported_base_models: list[str] = field(default_factory=list)
    param_schema: dict[str, Any] = field(default_factory=dict)
    vram_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    supports_lora: bool = False
    supports_native_audio: bool = False
    available: bool = True
    unavailable_reason: str | None = None
    heavy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class Adapter:
    """Base class for adapters. Subclasses set `info` and implement capability methods."""

    info: ClassVar[AdapterInfo]
    requires_model: ClassVar[bool] = False  # real adapters need a resolved ModelSpec for every job

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        """Load weights for `model` under a VRAM profile. Called by the ModelCache before a job runs.
        Returns load details (e.g. placement mode) for logs/status."""
        return {}

    def apply_loras(self, loras: list[LoraSpec], ctx: JobContext) -> None:
        """Activate exactly these LoRAs (loading new ones) without reloading the base model."""

    def unload(self) -> None:
        """Release all memory held by this adapter."""
