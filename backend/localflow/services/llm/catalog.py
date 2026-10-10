"""Provider model lists and API-key checks over plain HTTP (cheap calls; no completions)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx2

from localflow.services.llm.types import LLMModelInfo, ProviderName

TIMEOUT = httpx2.Timeout(20.0)

# OpenAI's /models lists everything; keep chat models usable through Chat Completions.
_OPENAI_CHAT = re.compile(r"^(gpt-|o\d|chatgpt-)")
_OPENAI_EXCLUDE = re.compile(
    r"(audio|realtime|tts|transcribe|image|search|embedding|codex|instruct|-pro\b|-pro-|live)"
)


_GPT_VERSION = re.compile(r"^gpt-(\d+(?:\.\d+)?)")
_DATED = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{4}$")


def _openai_order(model_id: str) -> tuple[float, int, str]:
    """Newest GPT family first, aliases before dated snapshots, then o-series."""
    m = _GPT_VERSION.match(model_id)
    version = float(m.group(1)) if m else 0.0
    return (-version, 1 if _DATED.search(model_id) else 0, model_id)


@dataclass
class KeyCheck:
    ok: bool
    account: dict[str, Any] | None = None
    token_type: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def _client(client: httpx2.AsyncClient | None) -> httpx2.AsyncClient:
    return client or httpx2.AsyncClient(timeout=TIMEOUT, follow_redirects=True)


def _litellm_cost(model_ids: list[str]) -> tuple[float | None, float | None, int | None, bool | None]:
    try:
        import litellm
    except ImportError:  # pragma: no cover
        return None, None, None, None
    for mid in model_ids:
        info = litellm.model_cost.get(mid)
        if info:
            i, o = info.get("input_cost_per_token"), info.get("output_cost_per_token")
            ctx = info.get("max_input_tokens")
            return (i * 1e6 if i else None), (o * 1e6 if o else None), ctx, info.get("supports_vision")
    return None, None, None, None


def _structured(full_id: str) -> bool | None:
    try:
        import litellm

        return bool(litellm.supports_response_schema(model=full_id))
    except Exception:
        return None


async def list_models(
    provider: ProviderName, api_key: str, client: httpx2.AsyncClient | None = None
) -> list[LLMModelInfo]:
    c = _client(client)
    try:
        if provider == "openai":
            r = await c.get(
                "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"}
            )
            r.raise_for_status()
            ids = sorted(
                (m["id"] for m in r.json().get("data", []) if _OPENAI_CHAT.match(m["id"])), key=_openai_order
            )
            out = []
            for mid in ids:
                if _OPENAI_EXCLUDE.search(mid):
                    continue
                i, o, ctx, vis = _litellm_cost([mid, f"openai/{mid}"])
                out.append(
                    LLMModelInfo(
                        id=mid,
                        provider="openai",
                        name=mid,
                        context_length=ctx,
                        supports_vision=vis,
                        supports_structured_output=_structured(f"openai/{mid}"),
                        input_cost_per_mtok=i,
                        output_cost_per_mtok=o,
                    )
                )
            return out
        if provider == "anthropic":
            r = await c.get(
                "https://api.anthropic.com/v1/models",
                params={"limit": 100},
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            r.raise_for_status()
            out = []
            for m in r.json().get("data", []):
                mid = m["id"]
                i, o, ctx, vis = _litellm_cost([mid, f"anthropic/{mid}"])
                out.append(
                    LLMModelInfo(
                        id=mid,
                        provider="anthropic",
                        name=m.get("display_name") or mid,
                        context_length=ctx,
                        supports_vision=vis if vis is not None else True,
                        supports_structured_output=True,
                        input_cost_per_mtok=i,
                        output_cost_per_mtok=o,
                    )
                )
            return out
        # OpenRouter: public catalog with pricing, context length, and supported parameters.
        r = await c.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            params = set(m.get("supported_parameters") or [])
            arch = m.get("architecture") or {}
            if "text" not in (arch.get("output_modalities") or ["text"]):
                continue
            pricing = m.get("pricing") or {}
            out.append(
                LLMModelInfo(
                    id=m["id"],
                    provider="openrouter",
                    name=m.get("name") or m["id"],
                    context_length=m.get("context_length"),
                    supports_structured_output=bool(params & {"structured_outputs", "response_format"}),
                    supports_vision="image" in (arch.get("input_modalities") or []),
                    input_cost_per_mtok=_per_mtok(pricing.get("prompt")),
                    output_cost_per_mtok=_per_mtok(pricing.get("completion")),
                )
            )
        return sorted(out, key=lambda x: x.id)
    finally:
        if client is None:
            await c.aclose()


def _per_mtok(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v * 1e6 if v >= 0 else None


async def check_key(secret_name: str, api_key: str, client: httpx2.AsyncClient | None = None) -> KeyCheck:
    """Cheap authenticated call per provider (Sections 7.2-7.3)."""
    c = _client(client)
    try:
        if secret_name == "openai_api_key":
            r = await c.get(
                "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"}
            )
            return _status_check(r)
        if secret_name == "anthropic_api_key":
            r = await c.get(
                "https://api.anthropic.com/v1/models",
                params={"limit": 1},
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            return _status_check(r)
        if secret_name == "openrouter_api_key":
            r = await c.get(
                "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"}
            )
            check = _status_check(r)
            if check.ok:
                data = (r.json() or {}).get("data") or {}
                label = str(data.get("label") or "")
                # Unnamed keys are labeled with a masked copy of the key; don't show that as an account.
                check.account = {"username": label} if label and not label.startswith("sk-") else None
                limit, usage = data.get("limit"), data.get("usage")
                if isinstance(limit, (int, float)) and isinstance(usage, (int, float)) and usage >= limit:
                    check.warnings.append("This key has reached its credit limit.")
            return check
        if secret_name == "hf_token":
            r = await c.get(
                "https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {api_key}"}
            )
            check = _status_check(r)
            if check.ok:
                data = r.json() or {}
                token = (data.get("auth") or {}).get("accessToken") or {}
                check.account = {"username": data.get("name"), "avatar_url": data.get("avatarUrl")}
                check.token_type = token.get("role")
                if check.token_type == "write":
                    check.warnings.append(
                        "This token has write access; a read token is enough for LocalFlow."
                    )
                fine = token.get("fineGrained") if isinstance(token.get("fineGrained"), dict) else None
                if check.token_type == "fineGrained" and not (fine or {}).get("canReadGatedRepos"):
                    check.warnings.append(
                        "This fine-grained token can't read public gated repos (e.g. FLUX.1-dev). Enable "
                        '"Read access to contents of all public gated repos you can access" in its settings.'
                    )
            return check
        if secret_name == "civitai_api_key":
            # Verify the endpoint against the current Civitai API (Section 0.2); 404 → can't verify.
            r = await c.get("https://civitai.com/api/v1/me", headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 404:
                return KeyCheck(
                    ok=True, warnings=["Civitai doesn't expose an account check; key not verified."]
                )
            check = _status_check(r)
            if check.ok:
                data = r.json() or {}
                if data.get("username"):
                    check.account = {"username": data.get("username")}
            return check
        return KeyCheck(ok=False, error=f"Unknown secret '{secret_name}'")
    except httpx2.RequestError as e:
        return KeyCheck(ok=False, error=f"Network error: {type(e).__name__}")
    finally:
        if client is None:
            await c.aclose()


def _status_check(r: httpx2.Response) -> KeyCheck:
    if r.status_code in (401, 403):
        return KeyCheck(ok=False, error="The key was rejected (unauthorized).")
    if r.status_code >= 400:
        return KeyCheck(ok=False, error=f"Unexpected response ({r.status_code}).")
    return KeyCheck(ok=True)
