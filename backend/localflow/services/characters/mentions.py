"""`@key` mentions in prompts (Section 11.7.3).

- `@key` names an ingredient by its key (a lowercase slug; matching is case-insensitive).
- `@@` is a literal `@`. An `@` right after a letter or digit isn't a mention, so e-mail addresses are safe.
- Resolution, "full" mode (keyframe, t2i, t2v, image edit): the first mention of an ingredient becomes
  "{name}, {visual_prompt}" and later ones become "{name}". "name" mode (i2v, flf2v, extend) always uses the
  name, because the appearance already comes from the keyframe.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from localflow.errors import AppError

Mode = Literal["full", "name"]

_MENTION = re.compile(r"@@|(?<![\w@])@([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)")

# Which generation kinds resolve mentions, and how. Others (tts, v2a, music) leave text untouched.
MODE_FOR_KIND: dict[str, Mode] = {
    "keyframe": "full",
    "t2i": "full",
    "image_edit": "full",
    "t2v": "full",
    "ref2v": "full",
    "i2v": "name",
    "flf2v": "name",
    "extend": "name",
}


@dataclass(frozen=True)
class Mention:
    key: str  # normalized (lowercase)
    text: str  # as written, e.g. "@Mara"
    start: int
    end: int


@dataclass(frozen=True)
class Mentionable:
    id: str
    key: str
    name: str
    visual_prompt: str = ""
    kind: str = "character"


@dataclass
class Resolved:
    raw: str
    text: str
    ingredient_ids: list[str] = field(default_factory=list)  # in order of first mention
    keys: list[str] = field(default_factory=list)
    unknown: list[Mention] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.text != self.raw


def parse(text: str) -> list[Mention]:
    return [
        Mention(m.group(1).lower(), m.group(0), m.start(), m.end())
        for m in _MENTION.finditer(text or "")
        if m.group(1)
    ]


def resolve(text: str, lookup: Mapping[str, Mentionable], mode: Mode = "full") -> Resolved:
    out: list[str] = []
    seen: set[str] = set()
    res = Resolved(raw=text, text=text)
    pos = 0
    for m in _MENTION.finditer(text or ""):
        out.append(text[pos : m.start()])
        pos = m.end()
        if not m.group(1):
            out.append("@")  # @@ → @
            continue
        key = m.group(1).lower()
        item = lookup.get(key)
        if item is None:
            res.unknown.append(Mention(key, m.group(0), m.start(), m.end()))
            out.append(m.group(0))
            continue
        visual = item.visual_prompt.strip().rstrip(". ")
        if re.match(r"(A|An|The) ", visual):
            visual = visual[0].lower() + visual[1:]  # "Old Tom, a man in his sixties…" reads as one phrase
        if mode == "full" and key not in seen and visual:
            out.append(f"{item.name}, {visual}")
        else:
            out.append(item.name)
        if key not in seen:
            seen.add(key)
            res.keys.append(key)
            if item.id not in res.ingredient_ids:
                res.ingredient_ids.append(item.id)
    out.append(text[pos:])
    res.text = "".join(out)
    return res


def unknown_error(resolved: Resolved) -> AppError:
    keys = list(dict.fromkeys(m.text for m in resolved.unknown))
    return AppError(
        422,
        "unknown_mention",
        f"Unknown mention{'s' if len(keys) > 1 else ''}: {', '.join(keys)}. Mention an existing ingredient "
        "by its key, or write @@ for a literal @.",
        {
            "mentions": [
                {"key": m.key, "text": m.text, "start": m.start, "end": m.end} for m in resolved.unknown
            ]
        },
    )


def require_known(resolved: Resolved) -> Resolved:
    if resolved.unknown:
        raise unknown_error(resolved)
    return resolved
