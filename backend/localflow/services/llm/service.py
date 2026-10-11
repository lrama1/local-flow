"""LLM access: provider/model resolution per purpose, usage tracking, model lists (Section 8.1)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlmodel import col, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError
from localflow.models import LlmUsage, Setting
from localflow.schemas.story import LlmChoice
from localflow.services.keys import KeyService
from localflow.services.llm.litellm_provider import LiteLLMProvider
from localflow.services.llm.provider import LLMProvider, SemanticValidator
from localflow.services.llm.types import (
    PROVIDERS,
    LLMError,
    LLMModelInfo,
    Msg,
    ProviderName,
    StructuredResult,
)
from localflow.services.secrets import LLM_SECRET, SECRET_DEFS

log = logging.getLogger("localflow.llm")

ProviderFactory = Callable[[ProviderName, str], LLMProvider]
FAKE_AS: ProviderName = "openai"  # the provider name the fake LLM answers as
DEFAULTS_KEY = "llm_defaults"
MODELS_TTL_S = 600.0


class ProviderStatus(BaseModel):
    provider: ProviderName
    label: str
    configured: bool
    ok: bool | None
    error: str | None
    source: str | None
    default_model: str


class UsageRow(BaseModel):
    provider: str
    model: str
    purpose: str
    project_id: str | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    attempts: int
    ok: bool
    duration_ms: int | None
    created_at: datetime


class UsageSummary(BaseModel):
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    by_model: dict[str, dict[str, float]]
    recent: list[UsageRow]


class LLMService:
    def __init__(
        self, settings: Settings, db: Database, keys: KeyService, factory: ProviderFactory | None = None
    ) -> None:
        self.settings = settings
        self.db = db
        self.keys = keys
        # Fake mode (config `fake_llm`): one canned-response provider stands in for OpenAI; keys are ignored.
        self.fake = settings.fake_llm
        if self.fake and factory is None:
            from localflow.services.llm.demo import demo_script
            from localflow.services.llm.fake import FakeLLMProvider

            demo = FakeLLMProvider(demo_script(), max_repairs=settings.llm_max_repairs)
            factory = lambda name, key: demo  # noqa: E731
            log.warning("fake_llm is on: LLM features return canned demo responses")
        self._factory: ProviderFactory = factory or (
            lambda name, key: LiteLLMProvider(
                name, key, timeout_s=settings.llm_timeout_s, max_repairs=settings.llm_max_repairs
            )
        )
        self._models_cache: dict[str, tuple[float, list[LLMModelInfo]]] = {}

    # --- providers -------------------------------------------------------------------------------
    def configured(self) -> list[ProviderName]:
        if self.fake:
            return [FAKE_AS]
        return [p for p in PROVIDERS if self.keys.secrets.get(LLM_SECRET[p])]

    def usable(self) -> list[ProviderName]:
        """Configured and not known to be broken (unchecked keys count as usable)."""
        if self.fake:
            return [FAKE_AS]
        return [p for p in self.configured() if self.keys.status(LLM_SECRET[p]).ok is not False]

    def provider(self, name: ProviderName) -> LLMProvider:
        if self.fake:
            return self._factory(name, "fake")
        key = self.keys.secrets.get(LLM_SECRET[name])
        if not key:
            raise LLMError(
                "llm_not_configured",
                f"No {name} API key: set {SECRET_DEFS[LLM_SECRET[name]].env_vars[0]} or add one in Settings.",
                409,
            )
        return self._factory(name, key)

    async def statuses(self, validate: bool = True) -> list[ProviderStatus]:
        if self.fake:
            return [
                ProviderStatus(
                    provider=p,
                    label=SECRET_DEFS[LLM_SECRET[p]].label,
                    configured=p == FAKE_AS,
                    ok=True if p == FAKE_AS else None,
                    error=None,
                    source="fake" if p == FAKE_AS else None,
                    default_model=self.settings.default_llm_models.get(p, ""),
                )
                for p in PROVIDERS
            ]
        if validate:
            await self.keys.ensure_checked([LLM_SECRET[p] for p in PROVIDERS])
        out = []
        for p in PROVIDERS:
            st = self.keys.status(LLM_SECRET[p])
            out.append(
                ProviderStatus(
                    provider=p,
                    label=st.label,
                    configured=st.configured,
                    ok=st.ok,
                    error=st.error,
                    source=st.source,
                    default_model=self.settings.default_llm_models.get(p, ""),
                )
            )
        return out

    # --- defaults / resolution -------------------------------------------------------------------
    def defaults(self) -> dict[str, LlmChoice]:
        with self.db.session() as s:
            row = s.get(Setting, DEFAULTS_KEY)
            raw = row.value_json if row and isinstance(row.value_json, dict) else {}
        out = {}
        for purpose, v in raw.items():
            try:
                out[purpose] = LlmChoice.model_validate(v)
            except ValueError:
                continue
        return out

    def set_defaults(self, values: dict[str, LlmChoice | None]) -> dict[str, LlmChoice]:
        current = {k: v.model_dump() for k, v in self.defaults().items()}
        for purpose, choice in values.items():
            if choice is None:
                current.pop(purpose, None)
            else:
                current[purpose] = choice.model_dump()
        with self.db.session() as s:
            row = s.get(Setting, DEFAULTS_KEY) or Setting(key=DEFAULTS_KEY)
            row.value_json = current
            s.add(row)
            s.commit()
        return self.defaults()

    def resolve(self, purpose: str, choice: LlmChoice | None = None) -> tuple[ProviderName, str]:
        """Request override → Settings default for the purpose → config `llm_purpose_models` →
        Settings "*" default → config `default_llm_provider` → first usable provider."""
        if choice is not None:
            return choice.provider, choice.model
        configured = set(self.configured())
        defaults = self.defaults()
        for c in (defaults.get(purpose),):
            if c is not None and c.provider in configured:
                return c.provider, c.model
        cfg = self.settings.llm_purpose_models.get(purpose)
        if cfg and "/" in cfg:
            prov, model = cfg.split("/", 1)
            if prov in configured:
                return prov, model
        star = defaults.get("*")
        if star is not None and star.provider in configured:
            return star.provider, star.model
        order = list(self.usable()) or list(self.configured())
        if self.settings.default_llm_provider in order:
            order.remove(self.settings.default_llm_provider)
            order.insert(0, self.settings.default_llm_provider)
        if not order:
            raise LLMError(
                "llm_not_configured",
                "No LLM API key is configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY "
                "in your environment, or add a key in Settings.",
                409,
            )
        prov = order[0]
        default_model = self.settings.default_llm_models.get(prov)
        if not default_model:
            raise AppError(409, "llm_model_not_set", f"No default model configured for {prov}.")
        return prov, default_model

    def vision(self, purpose: str) -> tuple[bool, str | None, str | None]:
        """(available, "provider/model", reason if not) for a purpose that sends images."""
        try:
            prov, model = self.resolve(purpose)
        except (LLMError, AppError) as e:
            return False, None, getattr(e, "message", str(e))
        if self.provider(prov).supports_vision(model):
            return True, f"{prov}/{model}", None
        return False, f"{prov}/{model}", f"{model} can't read images; pick a vision model in Settings."

    # --- calls -----------------------------------------------------------------------------------
    async def structured[T: BaseModel](
        self,
        *,
        purpose: str,
        project_id: str | None,
        choice: LlmChoice | None,
        system: str,
        user: str,
        schema: type[T],
        validate: SemanticValidator | None = None,
        temperature: float = 0.7,
        images: list[str] | None = None,
    ) -> StructuredResult[T]:
        prov_name, model = self.resolve(purpose, choice)
        provider = self.provider(prov_name)
        if images and not provider.supports_vision(model):
            raise LLMError(
                "llm_no_vision",
                f"{model} can't read images. Pick a vision-capable model for '{purpose}' in Settings.",
                409,
            )
        started = time.monotonic()
        try:
            result = await provider.complete_structured(
                model=model,
                system=system,
                messages=[Msg("user", user, images=list(images or []))],
                schema=schema,
                temperature=temperature,
                validate=validate,
            )
        except LLMError:
            self._record(prov_name, model, purpose, project_id, None, ok=False, started=started)
            raise
        self._record(prov_name, model, purpose, project_id, result, ok=True, started=started)
        return result

    def _record(
        self,
        provider: str,
        model: str,
        purpose: str,
        project_id: str | None,
        result: StructuredResult[Any] | None,
        *,
        ok: bool,
        started: float,
    ) -> None:
        row = LlmUsage(
            provider=provider,
            model=model,
            purpose=purpose,
            project_id=project_id,
            prompt_tokens=result.usage.prompt_tokens if result else 0,
            completion_tokens=result.usage.completion_tokens if result else 0,
            cost_usd=result.usage.cost_usd if result else None,
            attempts=result.attempts if result else 0,
            ok=ok,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        with self.db.session() as s:
            s.add(row)
            s.commit()

    # --- models & usage --------------------------------------------------------------------------
    async def list_models(self, provider: ProviderName, refresh: bool = False) -> list[LLMModelInfo]:
        hit = self._models_cache.get(provider)
        if hit and not refresh and time.monotonic() - hit[0] < MODELS_TTL_S:
            return hit[1]
        try:
            models = await self.provider(provider).list_models()
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                "llm_models_unavailable", f"Couldn't list {provider} models: {type(e).__name__}"
            ) from e
        self._models_cache[provider] = (time.monotonic(), models)
        return models

    def usage(self, since: datetime | None = None, limit: int = 50) -> UsageSummary:
        with self.db.session() as s:
            q = select(LlmUsage)
            if since is not None:
                q = q.where(LlmUsage.created_at >= since)
            rows = list(s.exec(q.order_by(col(LlmUsage.created_at).desc())).all())
        by_model: dict[str, dict[str, float]] = {}
        for r in rows:
            m = by_model.setdefault(f"{r.provider}/{r.model}", {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            m["calls"] += 1
            m["tokens"] += r.prompt_tokens + r.completion_tokens
            m["cost_usd"] += r.cost_usd or 0.0
        return UsageSummary(
            calls=len(rows),
            prompt_tokens=sum(r.prompt_tokens for r in rows),
            completion_tokens=sum(r.completion_tokens for r in rows),
            cost_usd=round(sum(r.cost_usd or 0.0 for r in rows), 6),
            by_model=by_model,
            recent=[UsageRow(**r.model_dump(exclude={"id"})) for r in rows[:limit]],
        )
