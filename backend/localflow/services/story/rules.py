"""Semantic validation for LLM outputs (Section 8.3).

Hard errors trigger a repair turn; whatever is left after the last repair is auto-fixed, so the board always
respects the render constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from localflow.schemas.story import Breakdown, RenderConstraints, StoryBible

_EPS = 1e-6
WORDS_PER_SECOND = 2.5
LENGTH_TOLERANCE = 0.15
SMALL_OVERRUN = 0.25  # fraction of max_clip_seconds


@dataclass
class BreakdownContext:
    rc: RenderConstraints
    character_keys: set[str]
    location_keys: set[str]
    other_keys: set[str]  # props / style / any other ingredient keys
    target_length_s: float
    allow_extend: bool = True
    known_keys: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.known_keys = self.character_keys | self.location_keys | self.other_keys


def bible_errors(bible: StoryBible, existing_cast_keys: list[str]) -> list[str]:
    errors = []
    seen: dict[str, str] = {}
    for kind, items in (
        ("character", bible.characters),
        ("location", bible.locations),
        ("prop", bible.props),
    ):
        for item in items:
            if item.key in seen:
                errors.append(
                    f"Duplicate key '{item.key}' (a {seen[item.key]} and a {kind}); keys must be unique."
                )
            seen[item.key] = kind
    present = {c.key for c in bible.characters}
    for key in existing_cast_keys:
        if key not in present:
            errors.append(
                f"Existing character '{key}' is missing; include it with the same key and visual_prompt."
            )
    return errors


def breakdown_errors(bd: Breakdown, ctx: BreakdownContext) -> list[str]:
    errors: list[str] = []
    rc = ctx.rc
    for si, scene in enumerate(bd.scenes, 1):
        if scene.location_key and scene.location_key not in ctx.location_keys:
            errors.append(
                f"Scene {si}: location_key '{scene.location_key}' is not a location key in the bible."
            )
        for hi, shot in enumerate(scene.shots, 1):
            where = f"Scene {si} shot {hi}"
            if shot.duration_s > rc.max_clip_seconds + _EPS:
                errors.append(
                    f"{where}: duration_s {shot.duration_s:g} exceeds the {rc.max_clip_seconds:g} s maximum; "
                    "split it into several shots."
                )
            if shot.generation_mode not in rc.supported_modes:
                errors.append(
                    f"{where}: generation_mode '{shot.generation_mode}' is not supported "
                    f"(use one of {', '.join(rc.supported_modes)})."
                )
            if shot.generation_mode == "flf2v" and not shot.end_keyframe_prompt:
                errors.append(f"{where}: flf2v shots need an end_keyframe_prompt.")
            if shot.needs_extend and not ctx.allow_extend:
                errors.append(f"{where}: needs_extend isn't allowed; split the beat into shots instead.")
            unknown = [k for k in shot.ingredient_keys if k not in ctx.known_keys]
            if unknown:
                errors.append(f"{where}: unknown ingredient_keys {unknown}; use only keys from the bible.")
            for d in shot.dialogue:
                if d.character_key not in ctx.character_keys:
                    errors.append(
                        f"{where}: dialogue character_key '{d.character_key}' is not a character key."
                    )
    return errors


def breakdown_warnings(bd: Breakdown, ctx: BreakdownContext) -> list[str]:
    warnings = []
    total = bd.total_duration()
    if ctx.target_length_s and abs(total - ctx.target_length_s) > LENGTH_TOLERANCE * ctx.target_length_s:
        warnings.append(
            f"Total shot duration is {total:.0f} s; the target is {ctx.target_length_s:.0f} s (±15%)."
        )
    for si, scene in enumerate(bd.scenes, 1):
        for hi, shot in enumerate(scene.shots, 1):
            words = sum(len(d.line.split()) for d in shot.dialogue)
            budget = shot.duration_s * WORDS_PER_SECOND * (1 + (shot.extend_segments or 0))
            if words > budget + 1:
                warnings.append(
                    f"Scene {si} shot {hi}: {words} words of dialogue may not fit in {shot.duration_s:g} s."
                )
    return warnings


def autofix_breakdown(bd: Breakdown, ctx: BreakdownContext) -> tuple[Breakdown, list[str]]:
    """Make a breakdown valid without another LLM call. Returns the fixed copy and what was changed."""
    fixed = bd.model_copy(deep=True)
    notes: list[str] = []
    rc = ctx.rc
    fallback_mode = "i2v" if "i2v" in rc.supported_modes else rc.supported_modes[0]
    for si, scene in enumerate(fixed.scenes, 1):
        if scene.location_key and scene.location_key not in ctx.location_keys:
            notes.append(f"Scene {si}: dropped unknown location '{scene.location_key}'.")
            scene.location_key = None
        for hi, shot in enumerate(scene.shots, 1):
            where = f"Scene {si} shot {hi}"
            if shot.duration_s > rc.max_clip_seconds + _EPS:
                overrun = shot.duration_s - rc.max_clip_seconds
                # Small overruns just lose a moment; longer beats keep their length via extensions.
                extend = ctx.allow_extend and overrun > SMALL_OVERRUN * rc.max_clip_seconds
                extra = math.ceil(shot.duration_s / rc.max_clip_seconds) - 1
                notes.append(
                    f"{where}: {shot.duration_s:g} s clamped to {rc.max_clip_seconds:g} s"
                    + (f" and marked for {extra} extension(s)." if extend else ".")
                )
                shot.duration_s = rc.max_clip_seconds
                if extend:
                    shot.needs_extend = True
                    shot.extend_segments = max(shot.extend_segments or 0, extra)
            if shot.needs_extend and not ctx.allow_extend:
                shot.needs_extend, shot.extend_segments = False, None
            if shot.generation_mode not in rc.supported_modes:
                notes.append(f"{where}: mode '{shot.generation_mode}' → '{fallback_mode}'.")
                shot.generation_mode = fallback_mode
            if shot.generation_mode == "flf2v" and not shot.end_keyframe_prompt:
                shot.end_keyframe_prompt = shot.keyframe_prompt
                notes.append(f"{where}: missing end keyframe prompt filled from the keyframe prompt.")
            if shot.generation_mode != "flf2v":
                shot.end_keyframe_prompt = None
            unknown = [k for k in shot.ingredient_keys if k not in ctx.known_keys]
            if unknown:
                shot.ingredient_keys = [k for k in shot.ingredient_keys if k in ctx.known_keys]
                notes.append(f"{where}: removed unknown ingredient keys {unknown}.")
            bad = [d for d in shot.dialogue if d.character_key not in ctx.character_keys]
            if bad:
                shot.dialogue = [d for d in shot.dialogue if d.character_key in ctx.character_keys]
                notes.append(f"{where}: removed {len(bad)} dialogue line(s) with unknown speakers.")
    return fixed, notes
