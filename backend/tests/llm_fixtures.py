"""Deterministic LLM responses for pipeline tests (FakeLLMProvider scripts).

The canned story lives in the package (`localflow.services.llm.demo`): `fake_llm` mode serves the same one."""

from __future__ import annotations

import copy
from typing import Any

from localflow.services.llm.demo import BIBLE, BREAKDOWN, IDEA, REFINED, SHOT_PROMPTS

__all__ = [
    "BIBLE",
    "BREAKDOWN",
    "IDEA",
    "REFINED",
    "SHOT_PROMPTS",
    "breakdown_with_long_shot",
    "default_script",
]


def breakdown_with_long_shot(seconds: float = 9.0) -> dict[str, Any]:
    bd = copy.deepcopy(BREAKDOWN)
    bd["scenes"][0]["shots"][0]["duration_s"] = seconds
    return bd


def default_script() -> dict[str, list[Any]]:
    return {
        "refine": [REFINED],
        "bible": [BIBLE],
        "breakdown": [BREAKDOWN],
        "prompt_rewrite": [SHOT_PROMPTS],
    }
