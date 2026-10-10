from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from localflow.errors import bad_request
from localflow.routers.cancel import run_cancellable
from localflow.schemas.story import LlmChoice
from localflow.services.container import Services, services
from localflow.services.keys import SecretStatus
from localflow.services.llm.service import ProviderStatus, UsageSummary
from localflow.services.llm.types import LLMModelInfo, ProviderName

router = APIRouter(prefix="/api", tags=["settings"])


class AppSettingsOut(BaseModel):
    llm_defaults: dict[str, LlmChoice]
    default_llm_models: dict[str, str]
    purposes: list[str]
    config: dict[str, Any]


class AppSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_defaults: dict[str, LlmChoice | None] = Field(
        default_factory=dict, description="Per purpose ('refine', 'bible', ... or '*' for all); null clears"
    )


class SecretValue(BaseModel):
    value: str = Field(min_length=1, max_length=4096)


_PURPOSES = [
    "*",
    "refine",
    "bible",
    "breakdown",
    "prompt_rewrite",
    "character_expand",
    "character_describe",
]


def _settings_out(svc: Services) -> AppSettingsOut:
    st = svc.settings
    return AppSettingsOut(
        llm_defaults=svc.llm.defaults(),
        default_llm_models=st.default_llm_models,
        purposes=_PURPOSES,
        config={
            "data_dir": str(st.data_dir),
            "vram_profile": st.vram_profile,
            "engine": st.engine,
            "ffmpeg_path": st.ffmpeg,
            "ffprobe_path": st.ffprobe,
            "default_llm_provider": st.default_llm_provider,
            "default_max_clip_seconds": st.default_max_clip_seconds,
            "secret_store": svc.secrets.stored_source,
            "allow_nsfw_models": st.allow_nsfw_models,
        },
    )


@router.get("/settings", response_model=AppSettingsOut)
def get_settings(svc: Services = Depends(services)) -> AppSettingsOut:
    return _settings_out(svc)


@router.patch("/settings", response_model=AppSettingsOut)
def patch_settings(body: AppSettingsPatch, svc: Services = Depends(services)) -> AppSettingsOut:
    unknown = [p for p in body.llm_defaults if p not in _PURPOSES]
    if unknown:
        raise bad_request("unknown_purpose", f"Unknown purpose(s): {', '.join(unknown)}")
    svc.llm.set_defaults(body.llm_defaults)
    return _settings_out(svc)


@router.get("/settings/secrets", response_model=list[SecretStatus])
def list_secrets(svc: Services = Depends(services)) -> list[SecretStatus]:
    return svc.keys.all_statuses()


@router.post("/settings/secrets/hf_token/import-cli", response_model=SecretStatus)
async def import_hf_cli_token(svc: Services = Depends(services)) -> SecretStatus:
    """Copy an existing `hf auth login` token into the keyring, then validate it (Section 7.3)."""
    status = svc.keys.import_hf_cli(svc.settings.resolved_hf_home)
    if status.source != "env":  # an env var would still win; validating would test that one instead
        status = await svc.keys.test("hf_token")
    return status


@router.put("/settings/secrets/{name}", response_model=SecretStatus)
def set_secret(name: str, body: SecretValue, svc: Services = Depends(services)) -> SecretStatus:
    return svc.keys.set(name, body.value)


@router.delete("/settings/secrets/{name}", response_model=SecretStatus)
def delete_secret(name: str, svc: Services = Depends(services)) -> SecretStatus:
    return svc.keys.delete(name)


@router.post("/settings/secrets/{name}/test", response_model=SecretStatus)
async def test_secret(name: str, svc: Services = Depends(services)) -> SecretStatus:
    return await svc.keys.test(name)


@router.get("/llm/providers", response_model=list[ProviderStatus], tags=["llm"])
async def llm_providers(request: Request, svc: Services = Depends(services)) -> list[ProviderStatus]:
    return await run_cancellable(request, svc.llm.statuses(validate=True))


@router.get("/llm/models", response_model=list[LLMModelInfo], tags=["llm"])
async def llm_models(
    request: Request, provider: ProviderName, refresh: bool = False, svc: Services = Depends(services)
) -> list[LLMModelInfo]:
    return await run_cancellable(request, svc.llm.list_models(provider, refresh=refresh))


@router.get("/llm/usage", response_model=UsageSummary, tags=["llm"])
def llm_usage(
    since: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    svc: Services = Depends(services),
) -> UsageSummary:
    return svc.llm.usage(since, limit)
