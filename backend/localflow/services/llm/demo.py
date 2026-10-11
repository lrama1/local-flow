"""Canned LLM responses for every purpose: `fake_llm` mode (offline demos, the Playwright smoke test) and the
tests' FakeLLMProvider scripts. The story is the lighthouse-keeper example used throughout the docs."""

from __future__ import annotations

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


EXPANSION: dict[str, Any] = {
    "name": "Mara",
    "visual_prompt": "a woman in her fifties with a weathered face, grey braid, green eyes, navy wool coat",
    "body_outfit_prompt": "navy wool coat over a yellow oilskin, grey trousers, black rubber boots",
    "negative_prompt": "young, smooth skin",
    "character_info": "Patient and dry-humored; hums sea shanties while she works.",
    "voice": {"gender": "female", "age": "50s", "tone": "warm and husky"},
}

DESCRIPTION: dict[str, Any] = {
    "visual_prompt": "a person with short dark hair and a calm expression, plain clothing",
    "body_outfit_prompt": "",
}

VOICE_DESIGN: dict[str, Any] = {
    "base_weight": 1.0,
    "mix": [],
    "speed": 1.0,
    "notes": "Demo mode: the base voice unchanged.",
}

VISION_REVIEW: dict[str, Any] = {"consistent": True, "issues": [], "suggested_prompt_fix": ""}

# One line per shot of BREAKDOWN, in order, within each shot's word budget (the second shot has dialogue).
NARRATION: dict[str, Any] = {
    "lines": [
        {"n": 1, "text": "Mara walks the rocks at dusk."},
        {"n": 2, "text": ""},
        {"n": 3, "text": "All night she works, as the storm rages."},
        {"n": 4, "text": "By morning, a message arrives."},
        {"n": 5, "text": "Some letters change everything."},
    ]
}


def demo_script() -> dict[str, list[Any]]:
    """One response per purpose (repeated for every call)."""
    return {
        "refine": [REFINED],
        "bible": [BIBLE],
        "breakdown": [BREAKDOWN],
        "prompt_rewrite": [SHOT_PROMPTS],
        "character_expand": [EXPANSION],
        "character_describe": [DESCRIPTION],
        "voice_design": [VOICE_DESIGN],
        "vision_review": [VISION_REVIEW],
        "narration": [NARRATION],
    }
