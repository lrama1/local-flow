"""Application configuration.

Precedence: defaults < `<data_dir>/config.toml` < `LOCALFLOW_*` env vars < explicit overrides."""

from __future__ import annotations

import os
import secrets
import shutil
import tomllib
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

VramProfile = Literal["low", "balanced", "quality"]
EngineKind = Literal["auto", "fake", "real"]
LlmProviderName = Literal["openai", "anthropic", "openrouter"]
SecretBackend = Literal["auto", "keyring", "file", "memory"]

DEFAULT_CONFIG_TOML = """\
# LocalFlow configuration. Every key can be overridden with an env var: LOCALFLOW_<KEY>.
# data_dir is read from LOCALFLOW_DATA_DIR (it locates this file, so it can't be set here).

# ffmpeg_path = "ffmpeg"
# ffprobe_path = "ffprobe"
# host = "127.0.0.1"
# port = 8765
# vram_profile = "balanced"       # low | balanced | quality
# engine = "auto"                 # auto (real if a CUDA GPU + torch are found) | real | fake (placeholders)
# allow_nsfw_models = false
# allow_pickle_weights = false    # allow .bin/.pt/.pth/.ckpt weights (needed by Kokoro and MMAudio)
# max_concurrent_downloads = 2
# hf_home = "D:/LocalFlow/models/hf"  # defaults to <data_dir>/models/hf

# API keys are read from environment variables first (OPENAI_API_KEY, ANTHROPIC_API_KEY,
# OPENROUTER_API_KEY, HF_TOKEN, CIVITAI_API_TOKEN), then from keys pasted in Settings (OS keyring).

# LLM defaults. Settings → "Default models" overrides these per purpose.
# default_llm_provider = "openai"  # openai | anthropic | openrouter; unset = first with a working key
# default_max_clip_seconds = 5.0
# auto_approve_keyframes = false   # "Generate all": use the first keyframe and queue the video
# cancel_grace_s = 15.0            # restart the GPU worker if a cancelled job hasn't stopped by then
#
# [default_llm_models]
# openai = "gpt-5.4-mini"
# anthropic = "claude-sonnet-5"
# openrouter = "anthropic/claude-sonnet-5"
#
# [llm_purpose_models]             # refine | bible | breakdown | prompt_rewrite | vision_review | ...
# breakdown = "openai/gpt-5.5"
"""


_toml_values: dict[str, Any] = {}


def default_data_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("USERPROFILE") or Path.home()) / "LocalFlow"
    return Path.home() / "LocalFlow"


class _TomlSource(PydanticBaseSettingsSource):
    """Reads `<data_dir>/config.toml`; placed after env vars so env wins."""

    def __init__(self, settings_cls: type[BaseSettings], values: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._values = values

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._values.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._values.items() if k in self.settings_cls.model_fields}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOCALFLOW_", extra="ignore")

    data_dir: Path = Field(default_factory=default_data_dir)
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    host: str = "127.0.0.1"
    port: int = 8765
    vram_profile: VramProfile = "balanced"
    engine: EngineKind = "auto"
    preview_every_n_steps: int = 4  # image step previews (0 = off)
    hf_home: Path | None = None
    allow_nsfw_models: bool = False
    allow_pickle_weights: bool = False
    max_concurrent_downloads: int = 2
    max_upload_mb: int = 500

    # LLM (Section 8). Model ids live in config, never in code; Settings can override per purpose.
    default_llm_provider: LlmProviderName | None = None  # None → first provider with a working key
    default_llm_models: dict[str, str] = Field(
        default_factory=lambda: {
            "openai": "gpt-5.4-mini",
            "anthropic": "claude-sonnet-5",
            "openrouter": "anthropic/claude-sonnet-5",
        }
    )
    llm_purpose_models: dict[str, str] = Field(default_factory=dict)  # e.g. {"breakdown": "openai/gpt-5.5"}
    llm_timeout_s: float = 120.0
    llm_max_repairs: int = 2
    # Offline demo / smoke tests: every LLM call gets a canned response (services/llm/demo.py); no keys used.
    fake_llm: bool = False
    default_max_clip_seconds: float = 5.0  # until a real video model is selected (M2+)
    # Video (Section 11.6): "Generate all" selects the first keyframe and queues the video when true.
    auto_approve_keyframes: bool = False
    # A cancelled GPU job that hasn't stopped after this many seconds gets its worker restarted.
    cancel_grace_s: float = 15.0

    # Secrets (Section 7.2): env vars first, then keyring (or an encrypted file if no keyring backend).
    secrets_from_env: bool = True
    secret_backend: SecretBackend = "auto"

    # Runtime / dev options
    dev_mode: bool = False
    dev_origin: str = "http://127.0.0.1:5173"
    api_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost", "[::1]"])
    frontend_dist: Path | None = None
    start_worker: bool = True
    batch_top_k: int = 3
    fake_step_delay_s: float = 0.08
    log_level: str = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, _TomlSource(settings_cls, _toml_values)

    # Derived paths -----------------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "localflow.db"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def library_dir(self) -> Path:
        return self.data_dir / "library"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def resolved_hf_home(self) -> Path:
        return self.hf_home or (self.models_dir / "hf")

    @cached_property
    def ffmpeg(self) -> str | None:
        return _which(self.ffmpeg_path)

    @cached_property
    def ffprobe(self) -> str | None:
        return _which(self.ffprobe_path)

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.projects_dir,
            self.models_dir / "hf",
            self.models_dir / "civitai",
            self.models_dir / "loras",
            self.library_dir,
            self.logs_dir,
            self.tmp_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _which(cmd: str) -> str | None:
    p = Path(cmd)
    if p.is_absolute() or len(p.parts) > 1:
        return str(p) if p.exists() else None
    return shutil.which(cmd)


def env_data_dir() -> str | None:
    """`LOCALFLOW_DATA_DIR` from the process, else (Windows) from the user/system environment in the registry,
    so a value set with `setx` works without restarting the terminal or editor."""
    value = os.environ.get("LOCALFLOW_DATA_DIR")
    if value:
        return value
    from localflow.services.secrets import windows_registry_env

    return windows_registry_env("LOCALFLOW_DATA_DIR")


def load_settings(**overrides: Any) -> Settings:
    data_dir = Path(overrides.get("data_dir") or env_data_dir() or default_data_dir())
    toml_path = data_dir / "config.toml"
    values: dict[str, Any] = {}
    if toml_path.exists():
        values = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        values.pop("data_dir", None)
    global _toml_values
    _toml_values = values
    settings = Settings(**{**overrides, "data_dir": data_dir})
    settings.ensure_dirs()
    if not toml_path.exists():
        toml_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return settings
