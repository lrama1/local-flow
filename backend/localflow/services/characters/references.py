"""Keyframe reference selection by shot type (Section 11.7, used by Section 11.2 step 1).

For each character in a shot, the shot type decides which views to use:

| shot_type                                | references                        |
|------------------------------------------|-----------------------------------|
| close, extreme_close, over_shoulder      | primary (+ face_closeup)          |
| medium, pov, insert                      | primary + full_body               |
| wide                                     | full_body + primary               |

Props and locations contribute their primary image. The total is capped (the edit adapter puts all
references on one reference sheet, and a wider sheet loses detail per reference): first every character's
first choice, then props, then the location, then the characters' second choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

MAX_REFERENCES = 3

VIEW_PREFERENCES: dict[str, tuple[str, ...]] = {
    "close": ("primary", "face_closeup"),
    "extreme_close": ("primary", "face_closeup"),
    "over_shoulder": ("primary", "face_closeup"),
    "medium": ("primary", "full_body"),
    "pov": ("primary", "full_body"),
    "insert": ("primary", "full_body"),
    "wide": ("full_body", "primary"),
}
DEFAULT_PREFERENCES = ("primary", "full_body")

VIEW_LABELS = {
    "primary": "portrait",
    "full_body": "full body",
    "three_view": "turnaround",
    "side": "side view",
    "back": "back view",
    "face_closeup": "face close-up",
}
_KIND_ORDER = {"character": 0, "prop": 1, "location": 2}


class Referenceable(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def primary_asset_id(self) -> str | None: ...


@dataclass(frozen=True)
class RefImage:
    asset_id: str
    ingredient_id: str | None
    name: str
    kind: str
    view: str
    label: str  # how the edit prompt names this tile, e.g. "Mara (full body)"


def _options(ing: Referenceable, views: Mapping[str, str], shot_type: str) -> list[tuple[str, str]]:
    available: dict[str, str] = {k: v for k, v in views.items() if v}
    if ing.primary_asset_id:
        available["primary"] = ing.primary_asset_id
    if ing.kind != "character":
        return [("primary", available["primary"])] if "primary" in available else []
    prefs = VIEW_PREFERENCES.get(shot_type, DEFAULT_PREFERENCES)
    out = [(v, available[v]) for v in prefs if v in available]
    if not out and "primary" in available:
        out = [("primary", available["primary"])]
    return out


def select_references(
    shot_type: str,
    ingredients: Sequence[Referenceable],
    views: Mapping[str, Mapping[str, str]],
    max_refs: int = MAX_REFERENCES,
) -> list[RefImage]:
    """`ingredients` in shot order; `views[ingredient_id]` maps extra views to asset ids."""
    order = sorted(
        (i for i in ingredients if i.kind in _KIND_ORDER),
        key=lambda i: _KIND_ORDER[i.kind],  # stable: keeps shot order within a kind
    )
    options = {i.id: _options(i, views.get(i.id, {}), shot_type) for i in order}
    picked: list[tuple[Referenceable, str, str]] = []
    used: set[str] = set()

    def take(ing: Referenceable, view: str, asset: str) -> None:
        if len(picked) < max_refs and asset not in used:
            picked.append((ing, view, asset))
            used.add(asset)

    for kind in ("character", "prop", "location"):
        for ing in order:
            if ing.kind == kind and options[ing.id]:
                take(ing, *options[ing.id][0])
    for ing in order:
        if ing.kind == "character":
            for view, asset in options[ing.id][1:]:
                take(ing, view, asset)

    position = {ing.id: n for n, ing in enumerate(order)}
    picked.sort(key=lambda p: (position[p[0].id], [v for v, _ in options[p[0].id]].index(p[1])))
    counts: dict[str, int] = {}
    for ing, _, _ in picked:
        counts[ing.id] = counts.get(ing.id, 0) + 1
    return [
        RefImage(
            asset_id=asset,
            ingredient_id=ing.id,
            name=ing.name,
            kind=ing.kind,
            view=view,
            label=f"{ing.name} ({VIEW_LABELS.get(view, view)})" if counts[ing.id] > 1 else ing.name,
        )
        for ing, view, asset in picked
    ]
