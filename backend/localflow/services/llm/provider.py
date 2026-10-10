"""Provider abstraction (Section 8.1) with the shared structured-output + repair loop."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from localflow.services.llm.types import (
    LLMError,
    LLMModelInfo,
    Msg,
    ProviderName,
    RawCompletion,
    StructuredResult,
    TextResult,
    Usage,
)

SemanticValidator = Callable[[Any], list[str]]


class LLMProvider(Protocol):
    name: ProviderName

    async def list_models(self) -> list[LLMModelInfo]: ...

    async def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        messages: list[Msg],
        schema: type[T],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        validate: SemanticValidator | None = None,
    ) -> StructuredResult[T]: ...

    async def complete_text(
        self,
        *,
        model: str,
        system: str,
        messages: list[Msg],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> TextResult: ...

    def stream_text(
        self, *, model: str, system: str, messages: list[Msg], temperature: float = 0.7
    ) -> AsyncIterator[str]: ...

    def supports_vision(self, model: str) -> bool: ...


_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError("No JSON object found in the response")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise ValueError("Unterminated JSON object in the response")


def format_validation_errors(e: ValidationError, limit: int = 15) -> list[str]:
    out = []
    for err in e.errors()[:limit]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
        out.append(f"{loc}: {err.get('msg')}")
    return out


def repair_message(errors: list[str]) -> str:
    bullet = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous response did not pass validation:\n"
        f"{bullet}\n\n"
        "Fix every issue and return ONLY the complete corrected JSON object that matches the schema. "
        "No prose, no code fences."
    )


class BaseProvider:
    """Implements structured output on top of `_chat`; subclasses implement transport."""

    name: ProviderName
    max_repairs: int = 2

    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> RawCompletion:
        raise NotImplementedError

    def response_format_for(self, model: str, schema: type[BaseModel]) -> dict[str, Any] | None:
        return None

    def supports_vision(self, model: str) -> bool:
        return False

    async def complete_structured[T: BaseModel](
        self,
        *,
        model: str,
        system: str,
        messages: list[Msg],
        schema: type[T],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        validate: SemanticValidator | None = None,
    ) -> StructuredResult[T]:
        convo = [{"role": "system", "content": system}, *(m.to_dict() for m in messages)]
        response_format = self.response_format_for(model, schema)
        usage = Usage()
        last_errors: list[str] = []
        best: T | None = None
        best_errors: list[str] = []
        for attempt in range(1, self.max_repairs + 2):
            raw = await self._chat(
                model=model,
                messages=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            usage.add(raw.usage)
            try:
                value = schema.model_validate(extract_json(raw.text))
            except ValidationError as e:
                last_errors = format_validation_errors(e)
            except ValueError as e:  # includes json.JSONDecodeError
                last_errors = [f"Invalid JSON: {e}"]
            else:
                last_errors = validate(value) if validate else []
                if not last_errors:
                    return StructuredResult(value, usage, attempt, self.name, model)
                best, best_errors = value, last_errors
            if attempt <= self.max_repairs:
                convo += [
                    {"role": "assistant", "content": raw.text[:20000]},
                    {"role": "user", "content": repair_message(last_errors)},
                ]
        if best is not None:  # parses, but semantic rules still fail: the caller decides (e.g. auto-fix)
            return StructuredResult(best, usage, self.max_repairs + 1, self.name, model, best_errors)
        raise LLMError(
            "llm_invalid_output",
            f"The model's response didn't match the expected format after {self.max_repairs} repairs.",
            details=last_errors,
        )

    async def complete_text(
        self,
        *,
        model: str,
        system: str,
        messages: list[Msg],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> TextResult:
        convo = [{"role": "system", "content": system}, *(m.to_dict() for m in messages)]
        raw = await self._chat(
            model=model, messages=convo, temperature=temperature, max_tokens=max_tokens, response_format=None
        )
        return TextResult(raw.text, raw.usage)
