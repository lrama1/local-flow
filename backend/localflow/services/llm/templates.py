"""Versioned Jinja2 prompt templates (Section 8.4).

Each `<purpose>.<version>.j2` renders a system prompt and a user message separated by a `---- USER ----` line.
User-supplied text is wrapped in tags with `| data("tag")` and treated as data (prompt-injection hygiene)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
TEMPLATE_VERSIONS: dict[str, str] = {
    "refine": "v1",
    "bible": "v2",  # v2 (M3.5): existing cast carries character_info and voice
    "breakdown": "v1",
    "prompt_rewrite": "v1",
    "character_expand": "v1",
    "character_describe": "v1",
    "voice_design": "v1",
    "vision_review": "v1",
}
_SPLIT = "---- USER ----"


def _data(value: Any, tag: str) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    # Neutralize anything that could close (or fake) our delimiter tags.
    text = re.sub(rf"</?\s*{re.escape(tag)}\s*>", "", text, flags=re.IGNORECASE)
    return f"<{tag}>\n{text.strip()}\n</{tag}>"


def _tojson_pretty(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, ensure_ascii=False)


@lru_cache(maxsize=1)
def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=False,  # plain-text prompts, not HTML
    )
    env.filters["data"] = _data
    env.filters["tojson_pretty"] = _tojson_pretty
    env.filters["num"] = _num
    return env


def _num(value: float) -> str:
    """5.0 → "5", 2.5 → "2.5"."""
    return f"{value:g}"


def render(purpose: str, context: dict[str, Any], version: str | None = None) -> tuple[str, str, str]:
    """Returns (system, user, version)."""
    version = version or TEMPLATE_VERSIONS[purpose]
    text = _env().get_template(f"{purpose}.{version}.j2").render(**context)
    system, sep, user = text.partition(_SPLIT)
    if not sep:
        raise ValueError(f"Template {purpose}.{version} is missing the '{_SPLIT}' separator")
    return _collapse(system), _collapse(user), version


def _collapse(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()
