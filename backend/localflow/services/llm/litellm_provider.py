"""LiteLLM-backed provider (Section 8.1).

Model strings: `openai/<model>`, `anthropic/<model>`, `openrouter/<vendor>/<model>`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import litellm
from pydantic import BaseModel

from localflow.services.llm import catalog
from localflow.services.llm.provider import BaseProvider
from localflow.services.llm.types import LLMError, LLMModelInfo, Msg, ProviderName, RawCompletion, Usage

log = logging.getLogger("localflow.llm")

litellm.drop_params = True  # e.g. GPT-5 reasoning models reject `temperature`
litellm.suppress_debug_info = True

_LABEL = {"openai": "OpenAI", "anthropic": "Anthropic", "openrouter": "OpenRouter"}
_RETRYABLE: tuple[type[Exception], ...] = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
    litellm.BadGatewayError,
    litellm.Timeout,
)
MAX_ATTEMPTS = 3


class LiteLLMProvider(BaseProvider):
    def __init__(
        self,
        name: ProviderName,
        api_key: str,
        *,
        timeout_s: float = 120.0,
        max_repairs: int = 2,
        reasoning_effort: str | None = "low",
    ) -> None:
        self.name = name
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.max_repairs = max_repairs
        self.reasoning_effort = reasoning_effort

    def full_model(self, model: str) -> str:
        return f"{self.name}/{model}"

    async def list_models(self) -> list[LLMModelInfo]:
        return await catalog.list_models(self.name, self._api_key)

    def response_format_for(self, model: str, schema: type[BaseModel]) -> dict[str, Any] | None:
        try:
            native = bool(litellm.supports_response_schema(model=self.full_model(model)))
        except Exception:
            native = False
        if native:
            # Non-strict: our schemas have optional fields/defaults; Pydantic validates and we repair anyway.
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": False,
                },
            }
        return {"type": "json_object"} if self.name in ("openai", "openrouter") else None

    def supports_vision(self, model: str) -> bool:
        try:
            return bool(litellm.supports_vision(model=self.full_model(model)))
        except Exception:
            return False

    def _wants_reasoning_effort(self, model: str) -> bool:
        # Only OpenAI-family reasoning models: for Anthropic this would switch on extended thinking.
        if not self.reasoning_effort or not (self.name == "openai" or model.startswith("openai/")):
            return False
        try:
            return bool(litellm.supports_reasoning(model=self.full_model(model)))
        except Exception:
            return False

    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> RawCompletion:
        kwargs: dict[str, Any] = {
            "model": self.full_model(model),
            "messages": messages,
            "api_key": self._api_key,
            "timeout": self.timeout_s,
            "temperature": temperature,
            "drop_params": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        if self._wants_reasoning_effort(model):
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.name == "openrouter":
            kwargs["extra_headers"] = {"HTTP-Referer": "http://127.0.0.1", "X-Title": "LocalFlow"}

        resp = await self._call_with_retries(kwargs, model)
        message = resp.choices[0].message
        text = message.content or ""
        if not text and getattr(message, "tool_calls", None):  # JSON returned via a forced tool call
            text = message.tool_calls[0].function.arguments or ""
        u = getattr(resp, "usage", None)
        usage = Usage(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            cost_usd=_cost(resp),
        )
        return RawCompletion(text, usage)

    async def _call_with_retries(self, kwargs: dict[str, Any], model: str) -> Any:
        label = _LABEL[self.name]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await litellm.acompletion(**kwargs)
            except litellm.AuthenticationError as e:
                raise LLMError("llm_auth_failed", f"{label} rejected the API key.") from e
            except litellm.NotFoundError as e:
                raise LLMError("llm_model_not_found", f"Model '{model}' isn't available from {label}.") from e
            except litellm.ContextWindowExceededError as e:
                raise LLMError("llm_context_exceeded", f"The request is too long for '{model}'.") from e
            except litellm.ContentPolicyViolationError as e:
                raise LLMError("llm_content_policy", f"{label} refused the request (content policy).") from e
            except _RETRYABLE as e:
                if attempt == MAX_ATTEMPTS:
                    if isinstance(e, litellm.Timeout):
                        raise LLMError(
                            "llm_timeout", f"{label} didn't respond within {self.timeout_s:.0f} s.", 504
                        ) from e
                    if isinstance(e, litellm.RateLimitError):
                        raise LLMError(
                            "llm_rate_limited", f"{label} rate limit reached; try again shortly.", 429
                        ) from e
                    raise LLMError("llm_unavailable", f"{label} is unavailable: {_short(e)}") from e
                log.warning(
                    "%s call failed (%s), retrying (%d/%d)", label, type(e).__name__, attempt, MAX_ATTEMPTS
                )
                await asyncio.sleep(2 ** (attempt - 1))
            except litellm.BadRequestError as e:
                raise LLMError("llm_bad_request", f"{label} rejected the request: {_short(e)}") from e
            except litellm.APIError as e:
                raise LLMError("llm_error", f"{label} error: {_short(e)}") from e
        raise AssertionError("unreachable")

    async def stream_text(
        self, *, model: str, system: str, messages: list[Msg], temperature: float = 0.7
    ) -> AsyncIterator[str]:
        stream = await litellm.acompletion(
            model=self.full_model(model),
            messages=[{"role": "system", "content": system}, *(m.to_dict() for m in messages)],
            api_key=self._api_key,
            timeout=self.timeout_s,
            temperature=temperature,
            drop_params=True,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def _cost(resp: Any) -> float | None:
    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception:
        return None
    return float(cost) if cost else None


def _short(e: Exception, limit: int = 300) -> str:
    msg = str(e).replace("\n", " ")
    return msg[:limit]
