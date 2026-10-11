"""M7 vision review (Section 8.3 step 5) through the API: FakeEngine worker + FakeLLMProvider."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.services.llm.catalog import KeyCheck
from localflow.services.llm.fake import FakeLLMProvider
from localflow.services.secrets import MemoryKeyStore, SecretStore
from tests.conftest import TOKEN, make_settings, wait_worker_idle
from tests.test_m3_imagegen import install, ok, uninstall
from tests.test_m35_characters import character, project, run, scene_shot

FIX = "@mara on the pier at dawn, long red hair, green wool coat"
ISSUE = "Mara's hair is short and blonde; her reference shows long red hair."


@pytest.fixture(scope="module")
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider(
        {
            "vision_review": [
                # Inconsistent without saying why: the semantic check asks for a repair.
                {"consistent": False, "issues": [], "suggested_prompt_fix": ""},
                {"consistent": False, "issues": [ISSUE, "  "], "suggested_prompt_fix": FIX},
            ]
        }
    )


@pytest.fixture(scope="module")
def c(tmp_path_factory: pytest.TempPathFactory, fake_llm: FakeLLMProvider) -> Iterator[TestClient]:
    store = MemoryKeyStore()
    store.set("openai_api_key", "sk-test-openai-key-1234")

    async def checker(name: str, value: str) -> KeyCheck:
        return KeyCheck(ok=True)

    app = create_app(
        make_settings(tmp_path_factory.mktemp("m7r")),
        secrets=SecretStore(store, env=None),
        llm_factory=lambda name, key: fake_llm,
        key_checker=checker,
    )
    with TestClient(app, headers={"X-LocalFlow-Token": TOKEN}) as client:
        wait_worker_idle(client)
        yield client


def first_user(call: dict[str, Any]) -> list[dict[str, Any]]:
    return next(m["content"] for m in call["messages"] if m["role"] == "user")  # type: ignore[no-any-return]


def image_parts(call: dict[str, Any]) -> int:
    return sum(1 for p in first_user(call) if p.get("type") == "image_url")


def test_review_a_keyframe_then_a_take(c: TestClient, fake_llm: FakeLLMProvider) -> None:
    cap = ok(c, "GET", "/api/llm/vision", params={"purpose": "vision_review"})
    assert cap["available"] and cap["model"].startswith("openai/") and cap["reason"] is None
    assert c.get("/api/llm/vision", params={"purpose": "nope"}).status_code == 400
    assert "vision_review" in ok(c, "GET", "/api/settings")["purposes"]

    pid = project(c, style=None)
    mara = character(c, pid, "Mara")
    shot = scene_shot(c, pid, keyframe_prompt="@mara on the pier at dawn", shot_type="medium")
    r = c.post(f"/api/shots/{shot['id']}/review", json={})
    assert r.status_code == 400 and r.json()["error"]["code"] == "nothing_to_review"

    out = ok(c, "POST", f"/api/shots/{shot['id']}/generate-keyframe", 201, json={"count": 1})
    keyframe = run(c, out["job"])["result"]["asset_ids"][0]
    ok(c, "POST", f"/api/shots/{shot['id']}/select", json={"slot": "keyframe", "asset_id": keyframe})

    review = ok(c, "POST", f"/api/shots/{shot['id']}/review")  # no body: the selected keyframe
    assert review["asset_id"] == keyframe and review["target"] == "keyframe"
    assert review["prompt_field"] == "keyframe_prompt"
    assert review["current_prompt"] == "@mara on the pier at dawn"
    assert not review["consistent"] and review["issues"] == [ISSUE]  # blank issues dropped
    assert review["suggested_prompt_fix"] == FIX
    assert [r["asset_id"] for r in review["references"]] == [mara["primary_asset_id"]]
    assert review["usage"]["attempts"] == 2 and review["frames_s"] == []

    calls = [x for x in fake_llm.calls if x["purpose"] == "vision_review"]
    assert len(calls) == 2 and image_parts(calls[0]) == 2  # the keyframe + Mara's portrait
    system, user = calls[0]["messages"][0]["content"], first_user(calls[0])[0]["text"]
    assert system.startswith("PURPOSE: vision_review") and "Image 1 is the generated keyframe" in system
    assert "<references>" in user and '"image": 2' in user and "mara visual" in user
    assert '"keyframe_prompt": "@mara on the pier at dawn"' in user

    # A take: two frames of the video against the same references; the fix targets the video prompt.
    wan = install(c, "wan22-ti2v-5b")
    try:
        vid = ok(c, "POST", f"/api/shots/{shot['id']}/generate-video", 201, json={})
        take = run(c, vid["jobs"][0])["result"]["asset_ids"][0]
    finally:
        uninstall(c, wan)
    review = ok(c, "POST", f"/api/shots/{shot['id']}/review", json={"asset_id": take})
    assert review["target"] == "video" and review["prompt_field"] == "video_prompt"
    assert review["current_prompt"] == "she walks" and len(review["frames_s"]) == 2
    assert 0 < review["frames_s"][0] < review["frames_s"][1]
    last = [x for x in fake_llm.calls if x["purpose"] == "vision_review"][-1]
    assert (
        image_parts(last) == 3
        and "Images 1-2 are frames of the generated take" in last["messages"][0]["content"]
    )
    assert not list((c.app.state.services.settings.tmp_dir / "review").glob("*/*.png"))  # type: ignore[attr-defined]


def test_review_needs_references_and_a_vision_model(
    c: TestClient, fake_llm: FakeLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = project(c, "Empty", style=None)
    shot = scene_shot(c, pid, keyframe_prompt="an empty pier")
    out = ok(c, "POST", f"/api/shots/{shot['id']}/generate-keyframe", 201, json={"count": 1})
    keyframe = run(c, out["job"])["result"]["asset_ids"][0]
    r = c.post(f"/api/shots/{shot['id']}/review", json={"asset_id": keyframe})
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_references"

    other = project(c, "Other", style=None)
    r = c.post(f"/api/shots/{scene_shot(c, other)['id']}/review", json={"asset_id": keyframe})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_asset"  # another project's

    monkeypatch.setattr(fake_llm, "supports_vision", lambda model: False)
    cap = ok(c, "GET", "/api/llm/vision")
    assert not cap["available"] and "can't read images" in cap["reason"]
    mara = character(c, pid, "Mara")
    ok(c, "PATCH", f"/api/shots/{shot['id']}", json={"ingredient_ids": [mara["id"]]})
    r = c.post(f"/api/shots/{shot['id']}/review", json={"asset_id": keyframe})
    assert r.status_code == 409 and r.json()["error"]["code"] == "llm_no_vision"
