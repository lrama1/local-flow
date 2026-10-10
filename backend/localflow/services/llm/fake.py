"""Deterministic provider for tests and offline development (Section 8.1)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from localflow.services.llm.provider import BaseProvider
from localflow.services.llm.types import LLMModelInfo, Msg, ProviderName, RawCompletion, Usage

Responder = Callable[[list[dict[str, Any]]], str | dict[str, Any]]


class FakeLLMProvider(BaseProvider):
    """Answers each call with the next scripted response for the detected purpose.

    `script[purpose]` is a list of responses (dicts are JSON-encoded, strings are returned as-is, callables
    receive the messages); the last one repeats. The purpose is read from a `PURPOSE: <name>` line that every
    LocalFlow system prompt starts with.
    """

    def __init__(
        self,
        script: dict[str, list[str | dict[str, Any] | Responder]],
        name: ProviderName = "openai",
        max_repairs: int = 2,
    ) -> None:
        self.name = name
        self.script = script
        self.max_repairs = max_repairs
        self.calls: list[dict[str, Any]] = []

    async def list_models(self) -> list[LLMModelInfo]:
        return [
            LLMModelInfo(
                id="fake-model", provider=self.name, name="Fake model", supports_structured_output=True
            )
        ]

    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> RawCompletion:
        purpose = _purpose(messages)
        n = sum(1 for c in self.calls if c["purpose"] == purpose)
        self.calls.append({"purpose": purpose, "model": model, "messages": messages})
        responses = self.script.get(purpose)
        if not responses:
            raise AssertionError(f"FakeLLMProvider has no scripted response for purpose '{purpose}'")
        item = responses[min(n, len(responses) - 1)]
        if callable(item):
            item = item(messages)
        text = item if isinstance(item, str) else json.dumps(item)
        prompt_tokens = sum(len(str(m["content"])) for m in messages) // 4
        return RawCompletion(text, Usage(prompt_tokens, len(text) // 4, 0.0))

    def supports_vision(self, model: str) -> bool:
        return True

    async def stream_text(
        self, *, model: str, system: str, messages: list[Msg], temperature: float = 0.7
    ) -> AsyncIterator[str]:
        result = await self.complete_text(model=model, system=system, messages=messages)
        for word in result.text.split(" "):
            yield word + " "


def _purpose(messages: list[dict[str, Any]]) -> str:
    first = messages[0]["content"] if messages else ""
    for line in first.splitlines()[:3]:
        if line.startswith("PURPOSE:"):
            return line.split(":", 1)[1].strip()
    return "unknown"
