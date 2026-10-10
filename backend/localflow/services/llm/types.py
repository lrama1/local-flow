from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

ProviderName = Literal["openai", "anthropic", "openrouter"]
PROVIDERS: tuple[ProviderName, ...] = ("openai", "anthropic", "openrouter")
Purpose = Literal[
    "refine",
    "bible",
    "breakdown",
    "prompt_rewrite",
    "character_expand",
    "character_describe",
    "voice_design",
    "vision_review",
]
PURPOSES: tuple[Purpose, ...] = (
    "refine",
    "bible",
    "breakdown",
    "prompt_rewrite",
    "character_expand",
    "character_describe",
    "voice_design",
)


@dataclass
class Msg:
    role: Literal["system", "user", "assistant"]
    content: str
    images: list[str] = field(default_factory=list)  # data: URLs, sent as OpenAI-style image parts

    def to_dict(self) -> dict[str, Any]:
        if not self.images:
            return {"role": self.role, "content": self.content}
        parts: list[dict[str, Any]] = [{"type": "text", "text": self.content}]
        parts += [{"type": "image_url", "image_url": {"url": url}} for url in self.images]
        return {"role": self.role, "content": parts}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd


@dataclass
class RawCompletion:
    text: str
    usage: Usage


@dataclass
class StructuredResult[T: BaseModel]:
    value: T
    usage: Usage
    attempts: int
    provider: str
    model: str
    remaining_issues: list[str] = field(default_factory=list)  # semantic issues still present after repairs


@dataclass
class TextResult:
    text: str
    usage: Usage


class LLMModelInfo(BaseModel):
    id: str
    provider: ProviderName
    name: str
    context_length: int | None = None
    supports_structured_output: bool | None = None
    supports_vision: bool | None = None
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None


class LLMError(Exception):
    """Errors surfaced to the API as `{error: {code, message}}`."""

    def __init__(self, code: str, message: str, status: int = 502, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details
