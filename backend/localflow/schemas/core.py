"""Request/response schemas for projects, assets, jobs, and system endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from localflow.models import Asset, Job, Project

AspectRatio = Literal["16:9", "9:16", "1:1"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
AssetType = Literal["image", "video", "audio"]


class Page[T](BaseModel):
    items: list[T]
    total: int
    offset: int
    limit: int


# --- Projects ----------------------------------------------------------------------------------
class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    aspect_ratio: AspectRatio = "16:9"
    fps: int = Field(default=24, ge=1, le=120)
    default_video_model_id: str | None = None
    default_image_model_id: str | None = None
    draft_mode: bool = False
    narrator_voice_id: str | None = None
    style_notes: str = ""


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    aspect_ratio: AspectRatio | None = None
    fps: int | None = Field(default=None, ge=1, le=120)
    default_video_model_id: str | None = None
    default_image_model_id: str | None = None
    draft_mode: bool | None = None
    narrator_voice_id: str | None = None
    style_notes: str | None = None
    archived: bool | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    aspect_ratio: str
    fps: int
    default_video_model_id: str | None
    default_image_model_id: str | None
    draft_mode: bool
    narrator_voice_id: str | None
    style_notes: str
    archived: bool
    created_at: datetime
    updated_at: datetime
    cover_thumb_url: str | None = None

    @classmethod
    def of(cls, p: Project, cover_thumb_url: str | None = None) -> ProjectOut:
        return cls(**p.model_dump(), cover_thumb_url=cover_thumb_url)


# --- Assets ------------------------------------------------------------------------------------
class AssetOut(BaseModel):
    id: str
    project_id: str
    type: AssetType
    width: int | None
    height: int | None
    duration_s: float | None
    fps: float | None
    has_audio: bool
    source: str
    parent_asset_ids: list[str]
    generation: dict[str, Any]
    tags: list[str]
    favorite: bool
    created_at: datetime
    updated_at: datetime
    url: str
    thumb_url: str | None

    @classmethod
    def of(cls, a: Asset) -> AssetOut:
        return cls(
            id=a.id,
            project_id=a.project_id,
            type=a.type,
            width=a.width,
            height=a.height,
            duration_s=a.duration_s,
            fps=a.fps,
            has_audio=a.has_audio,
            source=a.source,
            parent_asset_ids=a.parent_asset_ids_json,
            generation=a.generation_json,
            tags=a.tags_json,
            favorite=a.favorite,
            created_at=a.created_at,
            updated_at=a.updated_at,
            url=f"/api/files/{a.project_id}/{a.id}",
            thumb_url=f"/api/thumbs/{a.project_id}/{a.id}" if a.thumb_path else None,
        )


class AssetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: list[str] | None = None
    favorite: bool | None = None


# --- Jobs --------------------------------------------------------------------------------------
class JobOut(BaseModel):
    id: str
    project_id: str | None
    type: str
    status: JobStatus
    priority: int
    params: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    progress: float
    progress_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    target_ref: dict[str, Any] | None

    @classmethod
    def of(cls, j: Job) -> JobOut:
        return cls(
            id=j.id,
            project_id=j.project_id,
            type=j.type,
            status=j.status,
            priority=j.priority,
            params=j.params_json,
            result=j.result_json,
            error=j.error,
            progress=j.progress,
            progress_message=j.progress_message,
            created_at=j.created_at,
            updated_at=j.updated_at,
            started_at=j.started_at,
            finished_at=j.finished_at,
            target_ref=j.target_ref_json,
        )


class JobReorder(BaseModel):
    job_ids: list[str] = Field(min_length=1)


# --- System ------------------------------------------------------------------------------------
class GpuInfo(BaseModel):
    name: str
    vram_used_mb: int
    vram_total_mb: int
    driver_version: str | None = None


class WorkerInfo(BaseModel):
    state: Literal["stopped", "starting", "idle", "busy", "restarting", "disabled"]
    pid: int | None
    current_job_id: str | None
    loaded_model: dict[str, Any] | None
    restarts: int
    last_exit_code: int | None
    engine: str
    engine_note: str | None = None  # e.g. why `auto` fell back to the FakeEngine
    vram: dict[str, Any] | None = None  # worker's last VRAM snapshot (MB): device_used, allocated, baseline
    error: str | None = None
    # torch / CUDA as the worker sees them: {torch, cuda, cuda_available, device, cudnn}; None until it
    # reports (and for the FakeEngine, which never imports torch).
    runtime: dict[str, Any] | None = None


class SystemStatus(BaseModel):
    version: str
    gpu: GpuInfo | None
    ram_used_mb: int | None
    ram_total_mb: int | None
    disk_free_mb: int | None
    data_dir: str
    worker: WorkerInfo
    ffmpeg_found: bool
    ffprobe_found: bool
    ffmpeg_version: str | None
    python_version: str
    torch_version: str | None
    cuda_version: str | None
    cuda_available: bool | None = Field(default=None, description="From the worker; None = not reported")
    diffusers_version: str | None
    vram_profile: str
    dev_mode: bool


class AdapterOut(BaseModel):
    adapter_key: str
    adapter_version: str
    name: str
    capabilities: list[str]
    supported_base_models: list[str]
    param_schema: dict[str, Any]
    vram_profiles: dict[str, dict[str, Any]]
    supports_lora: bool
    supports_native_audio: bool
    available: bool
    unavailable_reason: str | None
    heavy: bool


class PlaygroundInfo(BaseModel):
    project_id: str
