"""Deterministic LLM responses for pipeline tests (FakeLLMProvider scripts)."""

from __future__ import annotations

import copy
from typing import Any

IDEA = (
    "A lonely lighthouse keeper finds a mechanical seagull washed up on the rocks. "
    "She repairs it over a stormy night. At dawn it flies off and returns with a message from the mainland."
)

REFINED: dict[str, Any] = {
    "title": "The Brass Gull",
    "logline": "A lighthouse keeper repairs a mechanical seagull that carries her back to the world.",
    "synopsis": "Mara, a solitary keeper, finds a broken brass gull. She mends it through a storm; "
    "it delivers a letter at dawn.",
    "story": "Mara walks the rocks at dusk and finds a brass gull.\n\n"
    "She repairs it by lamplight as the storm rages.\n\n"
    "At dawn the gull flies and returns with a letter.",
    "estimated_runtime_s": 20,
    "tone": "wistful, hopeful",
    "visual_style": "Painterly cinematic realism, teal and amber palette.",
}

BIBLE: dict[str, Any] = {
    "style": {
        "look": "painterly cinematic realism",
        "palette": "teal and amber",
        "lighting": "low-key lamplight, stormy blue",
        "lens": "35mm, shallow depth of field",
        "era": "1930s",
        "negative": "text, watermark, deformed hands",
    },
    "characters": [
        {
            "key": "mara",
            "name": "Mara",
            "description": "The keeper",
            "visual_prompt": "woman in her 50s, weathered face, grey braid, navy wool coat, yellow oilskin",
            "voice": {"gender": "female", "age": "50s", "tone": "warm, husky"},
        },
        {
            "key": "brass-gull",
            "name": "The Brass Gull",
            "description": "A clockwork seagull",
            "visual_prompt": "mechanical seagull of polished brass with copper gears and glass eyes",
            "voice": {"gender": "", "age": "", "tone": "metallic chirps"},
        },
    ],
    "locations": [
        {
            "key": "lighthouse",
            "name": "The Lighthouse",
            "visual_prompt": "white stone lighthouse on black rocks",
        }
    ],
    "props": [
        {"key": "letter", "name": "The Letter", "visual_prompt": "folded cream letter with a red wax seal"}
    ],
}


def _shot(
    duration: float, keys: list[str], prompt: str, dialogue: list[dict[str, str]] | None = None
) -> dict:
    return {
        "duration_s": duration,
        "shot_type": "medium",
        "camera_move": "dolly_in",
        "ingredient_keys": keys,
        "generation_mode": "i2v",
        "keyframe_prompt": f"{prompt}, still frame",
        "end_keyframe_prompt": None,
        "video_prompt": f"{prompt}, gentle motion",
        "negative_prompt": "blur",
        "dialogue": dialogue or [],
        "sfx_prompt": "waves",
        "music_cue": None,
        "needs_extend": False,
        "continuity_notes": "",
    }


BREAKDOWN: dict[str, Any] = {
    "scenes": [
        {
            "title": "Dusk on the rocks",
            "summary": "Mara finds the gull.",
            "location_key": "lighthouse",
            "shots": [
                _shot(4, ["mara"], "Mara walks the rocks"),
                _shot(
                    3,
                    ["mara", "brass-gull"],
                    "Mara kneels by the brass gull",
                    [{"character_key": "mara", "line": "What are you?", "emotion": "curious"}],
                ),
            ],
        },
        {
            "title": "Storm night",
            "summary": "She repairs it.",
            "location_key": "lighthouse",
            "shots": [
                _shot(5, ["mara", "brass-gull"], "Mara works on the gull by lamplight"),
                _shot(4, ["brass-gull", "letter"], "The gull returns with the letter"),
                _shot(4, ["mara", "letter"], "Mara reads the letter"),
            ],
        },
    ]
}

SHOT_PROMPTS: dict[str, Any] = {
    "keyframe_prompt": "Mara walks the rocks at night, moonlight",
    "end_keyframe_prompt": None,
    "video_prompt": "Mara walks slowly, coat flapping in the night wind",
    "negative_prompt": "blur",
}


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
