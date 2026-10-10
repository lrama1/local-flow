"""M3.5 units: mention parsing/resolution, reference selection by shot type, templates, key collisions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from localflow.errors import AppError
from localflow.services.board import unique_key
from localflow.services.characters import mentions
from localflow.services.characters.mentions import Mentionable
from localflow.services.characters.references import select_references
from localflow.services.characters.templates import THUMBNAILS_DIR, builtin_templates

LOOKUP = {
    "mara": Mentionable("i-mara", "mara", "Mara", "a weathered harbor keeper, silver hair."),
    "old-tom": Mentionable("i-tom", "old-tom", "Old Tom", "a fisherman in a yellow coat"),
    "pier": Mentionable("i-pier", "pier", "The pier", "", "location"),
}


# --- mentions --------------------------------------------------------------------------------------
def test_parse_mentions_escapes_and_emails() -> None:
    found = mentions.parse("@Mara meets @old-tom. Mail me at tom@harbor.com, @@mara is literal, @pier-")
    assert [(m.key, m.text) for m in found] == [("mara", "@Mara"), ("old-tom", "@old-tom"), ("pier", "@pier")]
    assert mentions.parse("no mentions @ all, @-x, @@") == []


def test_resolve_full_mode_expands_first_mention_only() -> None:
    r = mentions.resolve("@mara waves at @old-tom, then @mara smiles at @pier.", LOOKUP, "full")
    assert r.text == (
        "Mara, a weathered harbor keeper, silver hair waves at Old Tom, a fisherman in a yellow coat, then "
        "Mara smiles at The pier."
    )
    assert r.keys == ["mara", "old-tom", "pier"] and r.ingredient_ids == ["i-mara", "i-tom", "i-pier"]
    assert r.changed and not r.unknown


def test_resolve_name_mode_and_literal_at() -> None:
    r = mentions.resolve("@mara raises the lantern; email @@harbor", LOOKUP, "name")
    assert r.text == "Mara raises the lantern; email @harbor"
    assert mentions.resolve("plain prompt", LOOKUP).changed is False


def test_unknown_mentions_are_a_422() -> None:
    r = mentions.resolve("@mara and @Nobody and @nobody", LOOKUP)
    assert [m.text for m in r.unknown] == ["@Nobody", "@nobody"]
    assert "@Nobody and @nobody" in r.text  # left as written
    with pytest.raises(AppError) as e:
        mentions.require_known(r)
    assert e.value.status == 422 and e.value.code == "unknown_mention"
    assert [m["key"] for m in e.value.details["mentions"]] == ["nobody", "nobody"]


# --- reference selection ---------------------------------------------------------------------------
@dataclass
class Ing:
    id: str
    name: str
    kind: str
    primary_asset_id: str | None


MARA = Ing("mara", "Mara", "character", "p-mara")
TOM = Ing("tom", "Tom", "character", "p-tom")
LANTERN = Ing("lantern", "Lantern", "prop", "p-lantern")
PIER = Ing("pier", "Pier", "location", "p-pier")
VIEWS = {"mara": {"full_body": "fb-mara", "face_closeup": "fc-mara"}, "tom": {"full_body": "fb-tom"}}


def ids(refs: list) -> list[str]:  # type: ignore[type-arg]
    return [r.asset_id for r in refs]


def test_reference_views_by_shot_type() -> None:
    assert ids(select_references("wide", [MARA], VIEWS)) == ["fb-mara", "p-mara"]
    assert ids(select_references("medium", [MARA], VIEWS)) == ["p-mara", "fb-mara"]
    assert ids(select_references("pov", [MARA], VIEWS)) == ["p-mara", "fb-mara"]
    assert ids(select_references("close", [MARA], VIEWS)) == ["p-mara", "fc-mara"]
    assert ids(select_references("over_shoulder", [MARA], {})) == ["p-mara"]
    # No full-body view yet: a wide shot falls back to the primary.
    assert ids(select_references("wide", [MARA], {})) == ["p-mara"]


def test_reference_priority_and_cap() -> None:
    # Each character's first choice, then props, then the location; second choices only if room is left.
    refs = select_references("wide", [PIER, LANTERN, MARA, TOM], VIEWS)
    assert ids(refs) == ["fb-mara", "fb-tom", "p-lantern"]
    refs = select_references("wide", [PIER, MARA], VIEWS)
    assert ids(refs) == ["fb-mara", "p-mara", "p-pier"]  # grouped per ingredient on the sheet
    assert [r.label for r in refs] == ["Mara (full body)", "Mara (portrait)", "Pier"]
    assert [r.view for r in refs] == ["full_body", "primary", "primary"]
    no_image = Ing("ghost", "Ghost", "character", None)
    assert ids(select_references("medium", [no_image, PIER], {})) == ["p-pier"]


# --- templates & keys ------------------------------------------------------------------------------
def test_builtin_templates() -> None:
    items = builtin_templates()
    assert len(items) >= 12
    required = {"realistic", "fantastical", "sci-fi", "historical", "anime", "cartoon", "creature", "animal"}
    assert required <= {t.category for t in items}
    for t in items:
        assert t.visual_prompt and t.character_info and t.body_outfit_prompt
        assert t.voice_hints.gender and t.voice_hints.tone
        assert (THUMBNAILS_DIR / f"{t.key}.jpg").is_file(), f"missing thumbnail for {t.key}"


def test_unique_key_suffixes() -> None:
    assert unique_key("mara", set()) == "mara"
    assert unique_key("mara", {"mara"}) == "mara-2"
    assert unique_key("mara", {"mara", "mara-2", "mara-3"}) == "mara-4"


# --- keyframe instruction --------------------------------------------------------------------------
def test_edit_prompt_treats_views_of_one_character_as_one_person() -> None:
    from localflow.services.imagegen import edit_prompt

    names = ["Mara (full body)", "Mara (portrait)", "Lantern"]
    p = edit_prompt("@mara on the pier", names, "", ["Mara", "Mara", "Lantern"])
    assert "Mara (full body) (left), Mara (portrait) (middle), Lantern (right)" in p
    assert "The images of Mara are different views of the same character; show Mara only once." in p
    assert "place them together in one scene" in p  # Mara and the lantern are still two things
    plain = edit_prompt("x", ["Mara", "Lantern"], "")
    assert "different views" not in plain and "place them together" in plain


def test_full_mode_lowercases_a_leading_article() -> None:
    lookup = {"tom": Mentionable("i-tom", "tom", "Old Tom", "A man in his sixties. ")}
    assert mentions.resolve("@tom waves", lookup, "full").text == "Old Tom, a man in his sixties waves"
    lookup = {"ann": Mentionable("i-ann", "ann", "Ann", "Anna-like twin with red hair")}
    assert mentions.resolve("@ann", lookup, "full").text == "Ann, Anna-like twin with red hair"
