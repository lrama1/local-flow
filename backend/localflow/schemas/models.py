"""Model registry API schemas (Sections 9 and 16)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AccessState = Literal[
    "ok",
    "needs_token",
    "license_not_accepted",
    "pending_approval",
    "token_no_access",
    "not_found",
    "invalid_token",
    "forbidden",
]
ModelStatus = Literal["available", "downloading", "installed", "error"]
ModelSource = Literal["hf", "civitai", "local"]
ModelKind = Literal["checkpoint", "lora", "vae", "text_encoder", "audio"]


class ModelFileOut(BaseModel):
    path: str
    size: int | None = None


class ModelPartOut(BaseModel):
    role: str
    source: ModelSource
    repo_id: str | None = None
    revision: str | None = None
    allow_patterns: list[str] = Field(default_factory=list)
    files: list[ModelFileOut] = Field(default_factory=list)
    access: AccessState | None = None
    gated: bool = False
    model_url: str | None = None


class CivitaiRef(BaseModel):
    model_id: int
    version_id: int
    file_id: int | None = None
    version_name: str | None = None
    type: str | None = None
    base_model_raw: str | None = None
    permissions: dict[str, Any] = Field(default_factory=dict)


class ModelEntryOut(BaseModel):
    id: str
    catalog_key: str | None
    source: ModelSource
    source_ref: str
    name: str
    description: str
    purpose: str | None = None
    recommended: bool = False
    tier: Literal["final", "draft"] = "final"
    kind: str
    capabilities: list[str]
    adapter_key: str | None
    adapter_name: str | None = None
    adapter_available: bool = False
    base_model: str | None
    license: str | None
    license_url: str | None
    license_acknowledged: bool
    license_acknowledged_at: datetime | None
    nsfw: bool
    parts: list[ModelPartOut]
    local_path: str | None
    size_bytes: int | None
    approx_size_bytes: int | None = None
    status: ModelStatus
    partial: bool = Field(default=False, description="An interrupted download left files to resume from")
    error: str | None
    default_params: dict[str, Any]
    vram_profiles: dict[str, Any]
    gated: bool
    access: AccessState | None
    access_message: str | None
    access_checked_at: datetime | None
    trigger_words: list[str]
    preview_url: str | None
    requires_pickle: bool
    pickle_blocked: bool
    experimental: bool
    notes: str
    model_url: str | None
    job_id: str | None = None
    civitai: CivitaiRef | None = None
    installed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class AdapterOption(BaseModel):
    key: str
    name: str
    capabilities: list[str]
    base_models: list[str]
    supports_lora: bool
    milestone: str
    available: bool


class HubAccount(BaseModel):
    configured: bool
    username: str | None = None
    source: str | None = None


class ModelOptions(BaseModel):
    adapters: list[AdapterOption]
    kinds: list[str]
    allow_pickle_weights: bool
    allow_nsfw_models: bool
    models_dir: str
    free_bytes: int
    hf: HubAccount
    civitai: HubAccount


class DeleteModelOut(BaseModel):
    id: str
    freed_bytes: int
    deleted_entry: bool
    entry: ModelEntryOut | None = None


# --- Hugging Face --------------------------------------------------------------------------------
_REPO_ID = r"^(https?://(www\.)?huggingface\.co/)?[\w.\-]+/[\w.\-]+(/.*)?$"


class HfInspectBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(
        min_length=3, max_length=300, pattern=_REPO_ID, description="org/name or a model URL"
    )
    revision: str | None = Field(default=None, max_length=200)


class HfFileOut(BaseModel):
    path: str
    size: int | None
    file_class: str
    blocked_reason: str | None
    default_selected: bool


class SuggestionOut(BaseModel):
    kind: str
    base_model: str | None
    adapter_key: str | None
    capabilities: list[str]
    supported: bool
    reason: str


class HfInspectOut(BaseModel):
    repo_id: str
    revision: str | None
    sha: str | None
    gated: bool
    gated_mode: str | None
    private: bool
    access: AccessState
    access_message: str | None
    model_url: str
    license: str | None
    license_url: str
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    files: list[HfFileOut]
    total_size: int
    default_selection: list[str]
    selected_size: int
    suggestion: SuggestionOut
    existing_entry_id: str | None
    account: str | None = Field(default=None, description="Signed-in Hugging Face username, if known")


class HfInstallBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(min_length=3, max_length=300, pattern=_REPO_ID)
    revision: str | None = Field(default=None, max_length=200)
    allow_patterns: list[str] | None = Field(
        default=None, max_length=500, description="Default: inspect's selection"
    )
    adapter_key: str | None = Field(
        description="Confirmed adapter; null = no adapter (component/unsupported)"
    )
    name: str | None = Field(default=None, max_length=200)
    kind: ModelKind | None = None
    base_model: str | None = Field(default=None, max_length=100)
    capabilities: list[str] | None = None


class ModelInstallOut(BaseModel):
    entry: ModelEntryOut
    job_id: str | None


# --- Civitai ---------------------------------------------------------------------------------------
class CivitaiFileOut(BaseModel):
    id: int
    name: str
    size_bytes: int | None
    format: str | None
    fp: str | None
    file_type: str | None
    primary: bool
    sha256: str | None
    scan_ok: bool
    blocked_reason: str | None


class CivitaiVersionOut(BaseModel):
    id: int
    name: str
    base_model_raw: str | None
    base_model: str | None
    adapter_key: str | None
    supported: bool
    capabilities: list[str]
    trained_words: list[str]
    published_at: str | None
    early_access: bool
    early_access_ends_at: str | None
    downloadable: bool
    lock_reason: str | None
    files: list[CivitaiFileOut]
    preview_url: str | None
    installed_entry_id: str | None


class CivitaiModelOut(BaseModel):
    id: int
    name: str
    type: str
    kind: str | None
    nsfw: bool
    creator: str | None
    tags: list[str]
    description: str | None = None
    downloads: int | None
    thumbs_up: int | None
    preview_url: str | None
    permissions: dict[str, Any]
    versions: list[CivitaiVersionOut]
    model_url: str


class CivitaiSearchOut(BaseModel):
    items: list[CivitaiModelOut]
    next_cursor: str | None
    hidden_nsfw: int = 0


class CivitaiInstallBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: int = Field(gt=0)
    file_id: int | None = Field(default=None, gt=0)
    adapter_key: str | None = Field(default=None, description="Default: inferred from the base model")


CIVITAI_TYPES = ("Checkpoint", "LORA", "LoCon", "DoRA", "VAE")
CIVITAI_SORTS = ("Highest Rated", "Most Downloaded", "Newest")
