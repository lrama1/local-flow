"""M3.5 through the API (FakeEngine worker + FakeLLMProvider): wizard, views, Cast library, mentions,
reference selection, templates."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from localflow.main import create_app
from localflow.services.llm.catalog import KeyCheck
from localflow.services.llm.fake import FakeLLMProvider
from localflow.services.secrets import MemoryKeyStore, SecretStore
from tests.conftest import TOKEN, make_settings, wait_job, wait_worker_idle
from tests.test_m3_imagegen import install, ok, uninstall

EXPANSION = {
    "name": "Mara",
    "visual_prompt": "a woman in her sixties with short silver hair and green eyes, navy wool peacoat",
    "body_outfit_prompt": "navy peacoat with brass buttons, yellow scarf, grey trousers, black boots",
    "negative_prompt": "young",
    "character_info": "Patient and dry-humored; hums sea shanties.",
    "voice": {"gender": "female", "age": "60s", "tone": "warm and low"},
}
DESCRIPTION = {"visual_prompt": "an orange square figure with a flat face", "body_outfit_prompt": ""}


@pytest.fixture(scope="module")
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider({"character_expand": [EXPANSION], "character_describe": [DESCRIPTION]})


@pytest.fixture(scope="module")
def c(tmp_path_factory: pytest.TempPathFactory, fake_llm: FakeLLMProvider) -> Iterator[TestClient]:
    store = MemoryKeyStore()
    store.set("openai_api_key", "sk-test-openai-key-1234")

    async def checker(name: str, value: str) -> KeyCheck:
        return KeyCheck(ok=True)

    app = create_app(
        make_settings(tmp_path_factory.mktemp("m35")),
        secrets=SecretStore(store, env=None),
        llm_factory=lambda name, key: fake_llm,
        key_checker=checker,
    )
    with TestClient(app, headers={"X-LocalFlow-Token": TOKEN}) as client:
        wait_worker_idle(client)
        yield client


def png(color: str = "orange", size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def project(c: TestClient, name: str = "Harbor", style: str | None = "moody watercolor") -> str:
    pid = ok(c, "POST", "/api/projects", 201, json={"name": name})["id"]
    if style:
        body = {"kind": "style", "name": "Style", "key": "style", "visual_prompt": style}
        ok(c, "POST", f"/api/projects/{pid}/ingredients", 201, json=body)
    return str(pid)


def ingredient(c: TestClient, ing_id: str) -> dict[str, Any]:
    return ok(c, "GET", f"/api/ingredients/{ing_id}")  # type: ignore[no-any-return]


def run(c: TestClient, job: dict[str, Any]) -> dict[str, Any]:
    done = wait_job(c, job["id"])
    assert done["status"] == "completed", done
    return done


def character(c: TestClient, pid: str, name: str = "Mara", views: tuple[str, ...] = ()) -> dict[str, Any]:
    """A saved character with a primary image (and optional extra views), built through the wizard API."""
    d = ok(c, "POST", f"/api/projects/{pid}/characters", 201, json={"fields": {"name": name}})
    ok(c, "PATCH", f"/api/ingredients/{d['id']}", json={"visual_prompt": f"{name.lower()} visual"})
    run(c, ok(c, "POST", f"/api/ingredients/{d['id']}/generate", 201, json={"count": 1}))
    if views:
        out = ok(
            c,
            "POST",
            f"/api/ingredients/{d['id']}/generate-views",
            201,
            json={"views": list(views), "count": 1},
        )
        for job in out["jobs"]:
            run(c, job)
    return ok(c, "PATCH", f"/api/ingredients/{d['id']}", json={"draft": False})  # type: ignore[no-any-return]


# --- wizard ------------------------------------------------------------------------------------------
def test_template_wizard_with_views_and_save(c: TestClient) -> None:
    templates = ok(c, "GET", "/api/character-templates")
    fisher = next(t for t in templates if t["key"] == "weathered-fisherman")
    assert (
        fisher["builtin"] and fisher["category"] == "realistic" and fisher["voice_hints"]["gender"] == "male"
    )
    pid = project(c)

    draft = ok(
        c,
        "POST",
        f"/api/projects/{pid}/characters",
        201,
        json={"source": "template", "template_key": "weathered-fisherman"},
    )
    assert (
        draft["draft"] and draft["key"].startswith("draft-") and draft["locked"] and draft["origin"] == "user"
    )
    assert (
        draft["visual_prompt"] == fisher["visual_prompt"] and draft["template_key"] == "weathered-fisherman"
    )
    assert draft["voice_hints"] == fisher["voice_hints"]
    assert all(i["id"] != draft["id"] for i in ok(c, "GET", f"/api/projects/{pid}/ingredients"))
    with_drafts = ok(c, "GET", f"/api/projects/{pid}/ingredients", params={"include_drafts": True})
    assert any(i["id"] == draft["id"] for i in with_drafts)

    # Step 1: candidates from the template + reference-shot suffix + project style; the first is primary.
    job = run(c, ok(c, "POST", f"/api/ingredients/{draft['id']}/generate", 201, json={"count": 2}))
    prompt = job["params"]["request"]["prompt"]
    # The portrait already wears the outfit the views will show; the framing leads so it stays a portrait.
    assert prompt.startswith(f"Close-up head and shoulders portrait. {fisher['visual_prompt'][:40]}")
    assert f"wearing {fisher['body_outfit_prompt']}" in prompt
    assert prompt.rstrip(".").endswith("moody watercolor")
    primary = ingredient(c, draft["id"])["primary_asset_id"]
    assert primary

    # Step 2 without an edit model: text-to-image views with a lower-consistency note.
    out = ok(
        c,
        "POST",
        f"/api/ingredients/{draft['id']}/generate-views",
        201,
        json={
            "views": ["full_body", "three_view"],
            "body_outfit_prompt": "a yellow oilskin coat",
            "count": 1,
        },
    )
    assert out["mode"] == "t2i" and "lower consistency" in out["note"] and len(out["jobs"]) == 2
    assert [j["target_ref"]["view"] for j in out["jobs"]] == ["full_body", "three_view"]
    assert "wearing a yellow oilskin coat" in out["jobs"][0]["params"]["request"]["prompt"]
    for j in out["jobs"]:
        run(c, j)
    views = ingredient(c, draft["id"])["views"]
    assert set(views) == {"primary", "full_body", "three_view"} and views["primary"] == primary
    assert ingredient(c, draft["id"])["body_outfit_prompt"] == "a yellow oilskin coat"

    # With FLUX Kontext installed, views are image edits of the primary.
    kontext = install(c, "flux1-kontext-dev-q4")
    try:
        out = ok(
            c, "POST", f"/api/ingredients/{draft['id']}/generate-views", 201, json={"views": ["full_body"]}
        )
        assert out["mode"] == "edit" and out["note"] is None
        req = out["jobs"][0]["params"]["request"]
        assert req["reference_asset_ids"] == [primary] and req["model_entry_id"] == kontext
        assert "full-body shot" in req["prompt"] and "a yellow oilskin coat" in req["prompt"]
        assert (req["width"], req["height"]) == (832, 1248) and req["num_images"] == 2
        run(c, out["jobs"][0])
    finally:
        uninstall(c, kontext)
    cands = ok(
        c,
        "GET",
        f"/api/projects/{pid}/assets",
        params={"ingredient_id": draft["id"], "view": "full_body"},
    )["items"]
    assert len(cands) == 3
    first_choice = ingredient(c, draft["id"])["views"]["full_body"]
    other = next(a["id"] for a in cands if a["id"] != first_choice)
    assert (
        ok(c, "PUT", f"/api/ingredients/{draft['id']}/views/full_body", json={"asset_id": other})["views"][
            "full_body"
        ]
        == other
    )
    primaries = ok(
        c, "GET", f"/api/projects/{pid}/assets", params={"ingredient_id": draft["id"], "view": "primary"}
    )
    assert primaries["total"] == 2

    # Steps 3-5: identity, then save. The key is editable until saved, then immutable.
    saved = ok(
        c,
        "PATCH",
        f"/api/ingredients/{draft['id']}",
        json={
            "name": "Old Tom",
            "key": "tom",
            "character_info": "Quiet; rubs an old compass.",
            "voice_hints": {"gender": "male", "age": "60s", "tone": "gravelly"},
            "draft": False,
        },
    )
    assert not saved["draft"] and saved["key"] == "tom" and saved["locked"] and saved["origin"] == "user"
    assert saved["voice_hints"]["tone"] == "gravelly"
    r = c.patch(f"/api/ingredients/{draft['id']}", json={"key": "thomas"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "key_immutable"
    assert any(i["key"] == "tom" for i in ok(c, "GET", f"/api/projects/{pid}/ingredients"))


def test_draft_cancel_and_key_rules(c: TestClient) -> None:
    pid = project(c, style=None)
    character(c, pid, "Mara")
    draft = ok(c, "POST", f"/api/projects/{pid}/characters", 201, json={"fields": {"name": "Mara"}})
    r = c.patch(f"/api/ingredients/{draft['id']}", json={"draft": False})  # derived key "mara" is taken
    assert r.status_code == 409 and r.json()["error"]["code"] == "key_taken"
    r = c.patch(f"/api/ingredients/{draft['id']}", json={"key": "draft-x"})
    assert r.status_code == 400
    assert (
        ok(c, "PATCH", f"/api/ingredients/{draft['id']}", json={"key": "mara-b", "draft": False})["key"]
        == "mara-b"
    )

    # Cancelling the wizard deletes the draft and the images generated for it.
    d2 = ok(c, "POST", f"/api/projects/{pid}/characters", 201, json={})
    assert d2["name"] == "New character"
    run(c, ok(c, "POST", f"/api/ingredients/{d2['id']}/generate", 201, json={"count": 1}))
    asset_id = ingredient(c, d2["id"])["primary_asset_id"]
    assert c.delete(f"/api/ingredients/{d2['id']}").status_code == 204
    assert c.get(f"/api/assets/{asset_id}").status_code == 404


def test_upload_describe_expand_and_clean_reference(c: TestClient, fake_llm: FakeLLMProvider) -> None:
    pid = project(c, style=None)
    caps = ok(c, "GET", "/api/characters/capabilities", params={"project_id": pid})
    assert caps["expand"] and caps["describe"] and not caps["edit_model"] and caps["voices"]  # fake TTS (M6)

    draft = ok(c, "POST", f"/api/projects/{pid}/characters", 201, json={"source": "upload"})
    up = ok(
        c,
        "POST",
        f"/api/ingredients/{draft['id']}/assets",
        201,
        files={"file": ("me.png", png(), "image/png")},
    )
    assert ingredient(c, draft["id"])["primary_asset_id"] == up["id"]

    described = ok(c, "POST", "/api/characters/describe", json={"asset_ids": [up["id"]], "name": "Pip"})
    assert described["visual_prompt"] == DESCRIPTION["visual_prompt"]
    call = [x for x in fake_llm.calls if x["purpose"] == "character_describe"][-1]
    content = call["messages"][1]["content"]
    assert isinstance(content, list) and content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert '"Pip"' in call["messages"][0]["content"]

    expanded = ok(
        c,
        "POST",
        "/api/characters/expand",
        json={"description": "an old harbor keeper", "template_key": "weathered-fisherman"},
    )
    assert expanded["visual_prompt"] == EXPANSION["visual_prompt"] and expanded["voice_hints"]["age"] == "60s"
    call = [x for x in fake_llm.calls if x["purpose"] == "character_expand"][-1]
    assert (
        "<description>" in call["messages"][1]["content"] and "<template>" in call["messages"][1]["content"]
    )

    r = c.post(f"/api/ingredients/{draft['id']}/clean-reference", json={"asset_id": up["id"]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_edit_model"
    kontext = install(c, "flux1-kontext-dev-q4")
    try:
        job = ok(
            c, "POST", f"/api/ingredients/{draft['id']}/clean-reference", 201, json={"asset_id": up["id"]}
        )
        assert job["type"] == "image_edit" and job["params"]["request"]["reference_asset_ids"] == [up["id"]]
        assert job["target_ref"] == {"kind": "ingredient", "ingredient_id": draft["id"]}
        run(c, job)
    finally:
        uninstall(c, kontext)
    assert ok(c, "GET", f"/api/projects/{pid}/assets", params={"ingredient_id": draft["id"]})["total"] == 2


# --- Cast library -------------------------------------------------------------------------------------
def file_hash(c: TestClient, pid: str, asset_id: str) -> str:
    return hashlib.sha256(c.get(f"/api/files/{pid}/{asset_id}").content).hexdigest()


def test_library_save_cast_update_push(c: TestClient) -> None:
    p1, p2 = project(c, "One", style=None), project(c, "Two", style=None)
    mara = character(c, p1, "Mara", views=("full_body", "three_view"))
    aria = next(v for v in ok(c, "GET", "/api/voices") if v["native_id"] == "fake_aria")["id"]
    mara = ok(
        c,
        "PATCH",
        f"/api/ingredients/{mara['id']}",
        json={
            "character_info": "Patient; hums shanties.",
            "voice_id": aria,
            "voice_hints": {"gender": "female", "age": "60s", "tone": "warm"},
            "body_outfit_prompt": "navy peacoat",
        },
    )
    lib = ok(c, "POST", f"/api/ingredients/{mara['id']}/save-to-library", 201)
    assert lib["key"] == "mara" and lib["character_info"] == "Patient; hums shanties."
    assert lib["voice_id"] == aria and lib["voice_hints"]["tone"] == "warm"
    assert [(i["view"], i["is_primary"]) for i in lib["images"]] == [
        ("primary", True),
        ("full_body", False),
        ("three_view", False),
    ]
    assert ingredient(c, mara["id"])["library_status"] == "synced"
    r = c.post(f"/api/ingredients/{mara['id']}/save-to-library")
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_in_library"
    # Library images load in <img> tags with the token in the query string.
    plain = TestClient(c.app)
    img = plain.get(f"{lib['images'][0]['thumb_url']}&token={TOKEN}", headers={"host": "127.0.0.1"})
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    # Cast into a second project: fields, voice and all three images come along as imported assets.
    cast = ok(
        c, "POST", f"/api/projects/{p2}/ingredients/import", 201, json={"library_character_id": lib["id"]}
    )
    assert cast["origin"] == "library" and cast["key"] == "mara" and cast["locked"]
    assert cast["character_info"] == "Patient; hums shanties." and cast["voice_id"] == aria
    assert cast["voice_hints"]["tone"] == "warm" and cast["body_outfit_prompt"] == "navy peacoat"
    assert set(cast["views"]) == {"primary", "full_body", "three_view"} and cast["library_status"] == "synced"
    for view in ("primary", "full_body", "three_view"):
        a = ok(c, "GET", f"/api/assets/{cast['views'][view]}")
        assert a["project_id"] == p2 and a["source"] == "imported"
        assert file_hash(c, p2, cast["views"][view]) == file_hash(c, p1, mara["views"][view])
    again = ok(
        c, "POST", f"/api/projects/{p2}/ingredients/import", 201, json={"library_character_id": lib["id"]}
    )
    assert again["key"] == "mara-2"

    # Snapshots: a library edit shows "update available" until the copy pulls it.
    ok(c, "PATCH", f"/api/library/characters/{lib['id']}", json={"character_info": "Now grumpier."})
    assert ingredient(c, cast["id"])["library_status"] == "update_available"
    updated = ok(c, "POST", f"/api/ingredients/{cast['id']}/update-from-library")
    assert updated["character_info"] == "Now grumpier." and updated["library_status"] == "synced"

    # Push overwrites the library entry from a project copy.
    ok(c, "PATCH", f"/api/ingredients/{mara['id']}", json={"visual_prompt": "mara, now with a red scarf"})
    pushed = ok(c, "POST", f"/api/ingredients/{mara['id']}/push-to-library")
    assert pushed["visual_prompt"] == "mara, now with a red scarf" and len(pushed["images"]) == 3
    assert ingredient(c, mara["id"])["library_status"] == "synced"
    assert ingredient(c, cast["id"])["library_status"] == "update_available"
    assert any(
        r["id"] == lib["id"] for r in ok(c, "GET", "/api/library/characters", params={"q": "red scarf"})
    )

    assert c.delete(f"/api/library/characters/{lib['id']}").status_code == 204
    assert ingredient(c, cast["id"])["library_status"] == "missing"


def test_import_character_from_another_project(c: TestClient) -> None:
    p1, p2 = project(c, "A", style=None), project(c, "B", style=None)
    src = character(c, p1, "Pip", views=("full_body",))
    character(c, p2, "Pip")
    copy = ok(c, "POST", f"/api/projects/{p2}/ingredients/import", 201, json={"ingredient_id": src["id"]})
    assert copy["origin"] == "import" and copy["key"] == "pip-2" and copy["name"] == "Pip"
    assert set(copy["views"]) == {"primary", "full_body"}
    assert ok(c, "GET", f"/api/assets/{copy['views']['primary']}")["project_id"] == p2
    r = c.post(f"/api/projects/{p2}/ingredients/import", json={})
    assert r.status_code == 422


def test_library_wizard_saves_without_a_project(c: TestClient) -> None:
    playground = ok(c, "GET", "/api/playground")["project_id"]
    draft = ok(c, "POST", f"/api/projects/{playground}/characters", 201, json={"fields": {"name": "Wren"}})
    run(c, ok(c, "POST", f"/api/ingredients/{draft['id']}/generate", 201, json={"count": 1}))
    lib = ok(c, "POST", f"/api/ingredients/{draft['id']}/save-to-library", 201)
    assert lib["key"] == "wren" and len(lib["images"]) == 1
    assert ingredient(c, draft["id"])["library_character_id"] is None  # drafts aren't linked
    assert c.delete(f"/api/ingredients/{draft['id']}").status_code == 204
    assert c.get(lib["images"][0]["url"]).status_code == 200  # the library keeps its own copy


# --- mentions ----------------------------------------------------------------------------------------
def scene_shot(c: TestClient, pid: str, **shot: Any) -> dict[str, Any]:
    scene = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "Pier"})
    return ok(c, "POST", f"/api/scenes/{scene['id']}/shots", 201, json={"video_prompt": "she walks", **shot})  # type: ignore[no-any-return]


def test_mentions_add_references_and_are_recorded(c: TestClient) -> None:
    pid = project(c)
    mara = character(c, pid, "Mara")
    shot = scene_shot(c, pid, keyframe_prompt="@mara walks the pier; later @mara waves (@@dawn)")
    assert shot["ingredient_ids"] == [mara["id"]]  # the mention added her to the shot

    out = ok(c, "POST", f"/api/shots/{shot['id']}/generate-keyframe", 201, json={"count": 1})
    assert out["mode"] == "t2i"
    assert out["prompt"].startswith("Mara, mara visual walks the pier; later Mara waves (@dawn)")
    done = run(c, out["job"])
    assert done["params"]["prompt_raw"] == "@mara walks the pier; later @mara waves (@@dawn)"
    asset = ok(c, "GET", f"/api/assets/{done['result']['asset_ids'][0]}")
    gen = asset["generation"]
    assert gen["prompt_raw"] == "@mara walks the pier; later @mara waves (@@dawn)" and gen["mentions"] == [
        "mara"
    ]
    assert gen["params"]["prompt"].startswith("Mara, mara visual")

    bad = ok(c, "PATCH", f"/api/shots/{shot['id']}", json={"keyframe_prompt": "@nobody waits"})
    assert bad["keyframe_prompt"] == "@nobody waits"  # saving is lenient; the editor shows the error
    r = c.post(f"/api/shots/{shot['id']}/generate-keyframe", json={"count": 1})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_mention"

    res = ok(
        c, "POST", "/api/prompts/resolve", json={"project_id": pid, "text": "@Mara and @ghost", "mode": "i2v"}
    )
    assert res["resolved_text"] == "Mara and @ghost" and res["ingredient_ids"] == [mara["id"]]
    assert [(m["key"], m["known"]) for m in res["mentions"]] == [("mara", True), ("ghost", False)]
    assert res["errors"] == ["Unknown mention @ghost"]


def test_playground_mentions_resolve_against_the_cast_library(c: TestClient) -> None:
    pid = project(c, "Lib", style=None)
    lib = ok(c, "POST", f"/api/ingredients/{character(c, pid, 'Juno')['id']}/save-to-library", 201)
    job = ok(c, "POST", "/api/generate/t2i", 201, json={"prompt": "@juno on a boat", "adapter_key": "fake"})
    assert job["params"]["request"]["prompt"] == "Juno, juno visual on a boat"
    assert job["params"]["prompt_raw"] == "@juno on a boat" and job["params"]["mentions"] == [lib["key"]]
    r = c.post("/api/generate/t2i", json={"prompt": "@nobody on a boat", "adapter_key": "fake"})
    assert r.status_code == 422
    tts = ok(c, "POST", "/api/generate/tts", 201, json={"text": "email me @juno", "adapter_key": "fake"})
    assert tts["params"]["request"]["text"] == "email me @juno"  # speech text is never resolved
    run(c, job)


# --- reference selection -----------------------------------------------------------------------------
def test_wide_shot_keyframe_uses_the_full_body_view(c: TestClient) -> None:
    pid = project(c, style=None)
    mara = character(c, pid, "Mara", views=("full_body",))
    full_body, primary = mara["views"]["full_body"], mara["views"]["primary"]
    shot = scene_shot(c, pid, keyframe_prompt="@mara on the pier", shot_type="wide")
    kontext = install(c, "flux1-kontext-dev-q4")
    try:
        out = ok(c, "POST", f"/api/shots/{shot['id']}/generate-keyframe", 201, json={"count": 1})
        assert out["mode"] == "edit"
        assert out["reference_asset_ids"] == [full_body, primary]
        assert out["reference_names"] == ["Mara (full body)", "Mara (portrait)"]
        assert "Mara (full body) (left), Mara (portrait) (right)" in out["prompt"]
        # Two views of one character are one person (not twins in the frame).
        assert "different views of the same character; show Mara only once" in out["prompt"]
        assert "place them together" not in out["prompt"]
        assert out["job"]["params"]["request"]["reference_asset_ids"] == [full_body, primary]
        run(c, out["job"])

        refs = ok(c, "GET", f"/api/shots/{shot['id']}/references", params={"shot_type": "close"})
        assert [r["asset_id"] for r in refs["automatic"]] == [primary] and not refs["overridden"]
        assert {i["view"] for i in refs["options"][0]["images"]} == {"primary", "full_body"}

        # The user can override the references per shot; null restores the automatic choice.
        ok(c, "PATCH", f"/api/shots/{shot['id']}", json={"reference_asset_ids": [primary]})
        out = ok(c, "POST", f"/api/shots/{shot['id']}/generate-keyframe", 201, json={"count": 1})
        assert out["reference_asset_ids"] == [primary] and out["reference_names"] == ["Mara"]
        refs = ok(c, "GET", f"/api/shots/{shot['id']}/references")
        assert refs["overridden"] and [r["asset_id"] for r in refs["selected"]] == [primary]
        assert (
            ok(c, "PATCH", f"/api/shots/{shot['id']}", json={"reference_asset_ids": None})[
                "reference_asset_ids"
            ]
            is None
        )
        run(c, out["job"])
    finally:
        uninstall(c, kontext)

    # Deleting a chosen view image clears the view (and any override that used it).
    ok(c, "PATCH", f"/api/shots/{shot['id']}", json={"reference_asset_ids": [full_body]})
    assert c.delete(f"/api/assets/{full_body}").status_code == 204
    assert "full_body" not in ingredient(c, mara["id"])["views"]
    assert ok(c, "GET", f"/api/shots/{shot['id']}")["reference_asset_ids"] is None


# --- templates ---------------------------------------------------------------------------------------
def test_user_templates(c: TestClient) -> None:
    pid = project(c, style=None)
    mara = character(c, pid, "Mara")
    tpl = ok(
        c,
        "POST",
        "/api/character-templates",
        201,
        json={"from_ingredient_id": mara["id"], "name": "Harbor keeper", "category": "realistic"},
    )
    assert not tpl["builtin"] and tpl["key"] == "harbor-keeper" and tpl["visual_prompt"] == "mara visual"
    thumb = c.get(tpl["thumbnail_url"])
    assert thumb.status_code == 200 and Image.open(io.BytesIO(thumb.content)).size == (384, 384)
    assert (
        ok(c, "PATCH", f"/api/character-templates/{tpl['id']}", json={"name": "Keeper"})["name"] == "Keeper"
    )
    r = c.patch("/api/character-templates/weathered-fisherman", json={"name": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "builtin_template"
    draft = ok(
        c,
        "POST",
        f"/api/projects/{pid}/characters",
        201,
        json={"source": "template", "template_key": "harbor-keeper"},
    )
    assert draft["visual_prompt"] == "mara visual" and draft["template_key"] == "harbor-keeper"
    assert c.delete(f"/api/character-templates/{tpl['id']}").status_code == 204
    assert all(t["key"] != "harbor-keeper" for t in ok(c, "GET", "/api/character-templates"))
    r = c.post("/api/character-templates", json={"category": "custom"})
    assert r.status_code == 422
