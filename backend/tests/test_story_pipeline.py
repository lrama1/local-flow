"""Story pipeline end to end through the API with FakeLLMProvider (M1 acceptance, offline)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.services.llm.catalog import KeyCheck
from localflow.services.llm.fake import FakeLLMProvider
from localflow.services.secrets import MemoryKeyStore, SecretStore
from tests import llm_fixtures as fx
from tests.conftest import TOKEN, make_settings


class Harness:
    def __init__(self, client: TestClient, fake: FakeLLMProvider, store: MemoryKeyStore) -> None:
        self.client = client
        self.fake = fake
        self.store = store

    def project(self, name: str = "Gull") -> str:
        return str(self.client.post("/api/projects", json={"name": name}).json()["id"])

    def post(self, path: str, body: dict[str, Any] | None = None, status: int = 200) -> dict[str, Any]:
        r = self.client.post(path, json=body or {})
        assert r.status_code == status, r.text
        return r.json()  # type: ignore[no-any-return]

    def run_all(self, pid: str, target: int = 20) -> dict[str, Any]:
        self.post(
            f"/api/projects/{pid}/story/refine",
            {"raw_idea": fx.IDEA, "options": {"target_length_seconds": target}},
        )
        self.post(f"/api/projects/{pid}/story/bible")
        return self.post(f"/api/projects/{pid}/story/breakdown", {"target_length_seconds": target})


@pytest.fixture
def h(tmp_path: Path) -> Iterator[Harness]:
    fake = FakeLLMProvider(fx.default_script())
    store = MemoryKeyStore()
    store.set("openai_api_key", "sk-test-openai-key-1234")

    async def checker(name: str, value: str) -> KeyCheck:
        return KeyCheck(ok=True, account={"username": "tester"})

    def factory(name: Any, key: str) -> FakeLLMProvider:
        fake.name = name  # one shared fake, answering as whichever provider was requested
        return fake

    app = create_app(
        make_settings(tmp_path / "data", start_worker=False),
        secrets=SecretStore(store, env=None),
        llm_factory=factory,
        key_checker=checker,
    )
    with TestClient(app, headers={"X-LocalFlow-Token": TOKEN}) as client:
        yield Harness(client, fake, store)


def test_full_pipeline_idea_to_shots(h: Harness) -> None:
    pid = h.project()
    r = h.post(
        f"/api/projects/{pid}/story/refine", {"raw_idea": fx.IDEA, "options": {"target_length_seconds": 20}}
    )
    assert r["saved"] and r["story"]["revision"] == 1 and r["story"]["status"] == "refined"
    assert r["story"]["refined"]["title"] == "The Brass Gull"
    assert r["usage"]["provider"] == "openai" and r["usage"]["model"] == "gpt-5.4-mini"
    assert r["story"]["template_versions"] == {"refine": "v1"}

    r = h.post(f"/api/projects/{pid}/story/bible")
    assert r["story"]["revision"] == 2
    assert len(r["story"]["bible"]["characters"]) >= 2
    assert r["ingredients"]["created"] == 5  # 2 characters, 1 location, 1 prop, style
    ings = h.client.get(f"/api/projects/{pid}/ingredients").json()
    assert [i["kind"] for i in ings] == ["character", "character", "location", "prop", "style"]
    assert all(i["origin"] == "bible" and not i["locked"] for i in ings)

    r = h.post(f"/api/projects/{pid}/story/breakdown", {"target_length_seconds": 20})
    assert r["story"]["status"] == "broken_down" and r["scenes"] == 2 and r["shots"] == 5
    assert r["warnings"] == []

    board = h.client.get(f"/api/projects/{pid}/scenes").json()
    assert board["shot_count"] == 5 and board["total_duration_s"] == 20
    shots = [sh for sc in board["scenes"] for sh in sc["shots"]]
    assert all(sh["duration_s"] <= board["max_clip_seconds"] for sh in shots)
    by_key = {i["key"]: i["id"] for i in ings}
    assert board["scenes"][0]["location_ingredient_id"] == by_key["lighthouse"]
    assert shots[1]["ingredient_ids"] == [by_key["mara"], by_key["brass-gull"]]
    assert shots[1]["dialogue"][0] == {
        "character_ingredient_id": by_key["mara"],
        "line": "What are you?",
        "emotion": "curious",
    }

    usage = h.client.get("/api/llm/usage").json()
    assert usage["calls"] == 3 and {r["purpose"] for r in usage["recent"]} == {"refine", "bible", "breakdown"}


def test_breakdown_prompt_contains_constraints_and_cast(h: Harness) -> None:
    pid = h.project()
    h.run_all(pid)
    call = next(c for c in h.fake.calls if c["purpose"] == "breakdown")
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert "at most 5 seconds" in system and '"max_clip_seconds": 5.0' in system
    assert "<bible>" in user and "brass-gull" in user and "Target length: 20 seconds" in user


def test_malformed_breakdown_json_is_repaired(h: Harness) -> None:
    h.fake.script["breakdown"] = ['```json\n{"scenes": [{"title": "x", "shots": [', fx.BREAKDOWN]
    pid = h.project()
    r = h.run_all(pid)
    assert r["usage"]["attempts"] == 2 and r["shots"] == 5
    repair = h.fake.calls[-1]["messages"][-1]["content"]
    assert "did not pass validation" in repair and "Invalid JSON" in repair


def test_overlong_shot_is_repaired_by_the_llm(h: Harness) -> None:
    h.fake.script["breakdown"] = [fx.breakdown_with_long_shot(9), fx.BREAKDOWN]
    r = h.run_all(h.project())
    assert r["usage"]["attempts"] == 2
    assert "exceeds the 5 s maximum" in h.fake.calls[-1]["messages"][-1]["content"]


def test_overlong_shot_is_auto_fixed_when_repairs_fail(h: Harness) -> None:
    h.fake.script["breakdown"] = [fx.breakdown_with_long_shot(9)]
    pid = h.project()
    r = h.run_all(pid)
    assert r["usage"]["attempts"] == 3
    assert any(w.startswith("Auto-fixed: Scene 1 shot 1: 9 s clamped to 5 s") for w in r["warnings"])
    shot = h.client.get(f"/api/projects/{pid}/scenes").json()["scenes"][0]["shots"][0]
    assert shot["duration_s"] == 5 and shot["needs_extend"] and shot["extend_segments"] == 1


def test_small_overrun_is_clamped_without_extension(h: Harness) -> None:
    h.fake.script["breakdown"] = [fx.breakdown_with_long_shot(6)]
    pid = h.project()
    r = h.run_all(pid)
    assert any("6 s clamped to 5 s." in w for w in r["warnings"])
    shot = h.client.get(f"/api/projects/{pid}/scenes").json()["scenes"][0]["shots"][0]
    assert shot["duration_s"] == 5 and not shot["needs_extend"] and shot["extend_segments"] is None


def test_length_off_target_is_a_warning_not_a_repair(h: Harness) -> None:
    pid = h.project()
    h.run_all(pid, target=60)  # fixture breakdown totals 20 s
    r = h.post(
        f"/api/projects/{pid}/story/breakdown", {"target_length_seconds": 60, "replace_existing": True}
    )
    assert r["usage"]["attempts"] == 1
    assert r["warnings"] == ["Total shot duration is 20 s; the target is 60 s (±15%)."]
    system = h.fake.calls[-1]["messages"][0]["content"]
    assert "roughly\n  14 shots" in system or "roughly 14 shots" in system


def test_unknown_keys_are_reported_then_dropped(h: Harness) -> None:
    bad = json.loads(json.dumps(fx.BREAKDOWN))
    bad["scenes"][0]["shots"][0]["ingredient_keys"] = ["mara", "ghost"]
    bad["scenes"][0]["shots"][0]["generation_mode"] = "flf2v"
    h.fake.script["breakdown"] = [bad]
    r = h.run_all(h.project())
    text = " ".join(r["warnings"])
    assert "removed unknown ingredient keys ['ghost']" in text and "mode 'flf2v' → 'i2v'" in text


def test_breakdown_requires_confirmation_to_replace(h: Harness) -> None:
    pid = h.project()
    h.run_all(pid)
    r = h.client.post(f"/api/projects/{pid}/story/breakdown", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "shots_exist"
    r = h.post(f"/api/projects/{pid}/story/breakdown", {"replace_existing": True})
    assert r["shots"] == 5
    assert h.client.get(f"/api/projects/{pid}/scenes").json()["shot_count"] == 5


def test_steps_require_previous_steps(h: Harness) -> None:
    pid = h.project()
    assert (
        h.client.post(f"/api/projects/{pid}/story/bible", json={}).json()["error"]["code"]
        == "story_not_refined"
    )
    h.post(f"/api/projects/{pid}/story/refine", {"raw_idea": fx.IDEA})
    assert (
        h.client.post(f"/api/projects/{pid}/story/breakdown", json={}).json()["error"]["code"]
        == "bible_missing"
    )


def test_dry_run_then_apply(h: Harness) -> None:
    pid = h.project()
    h.post(f"/api/projects/{pid}/story/refine", {"raw_idea": fx.IDEA})
    edited = {**fx.REFINED, "title": "My Edited Title"}
    r = h.client.patch(f"/api/projects/{pid}/story", json={"refined": edited})
    assert r.json()["story"]["edited_fields"] == ["refined"] and r.json()["story"]["step"] == "edit"

    r = h.post(f"/api/projects/{pid}/story/refine", {"raw_idea": fx.IDEA, "dry_run": True})
    assert r["saved"] is False and r["story"]["revision"] is None
    assert r["story"]["refined"]["title"] == "The Brass Gull"
    assert h.client.get(f"/api/projects/{pid}/story").json()["refined"]["title"] == "My Edited Title"

    r = h.client.patch(
        f"/api/projects/{pid}/story",
        json={
            "refined": r["story"]["refined"],
            "applied_from": "refine",
            "llm": {"provider": "openai", "model": "gpt-5.4-mini"},
        },
    )
    story = r.json()["story"]
    assert story["revision"] == 3 and story["step"] == "refine" and story["edited_fields"] == []
    assert [x["revision"] for x in h.client.get(f"/api/projects/{pid}/story/revisions").json()] == [3, 2, 1]

    r = h.client.post(f"/api/projects/{pid}/story/restore", json={"revision": 2})
    assert r.json()["revision"] == 4 and r.json()["refined"]["title"] == "My Edited Title"


def test_locked_and_user_ingredients_survive_bible_reruns(h: Harness) -> None:
    pid = h.project()
    h.run_all(pid)
    ings = {i["key"]: i for i in h.client.get(f"/api/projects/{pid}/ingredients").json()}
    h.client.patch(
        f"/api/ingredients/{ings['mara']['id']}", json={"visual_prompt": "my mara", "locked": True}
    )
    h.client.post(
        f"/api/projects/{pid}/ingredients",
        json={
            "kind": "character",
            "name": "Old Tom",
            "visual_prompt": "old fisherman",
            "character_info": "grumpy",
        },
    )

    r = h.post(f"/api/projects/{pid}/story/bible")
    assert r["ingredients"]["locked"] == 1
    after = {i["key"]: i for i in h.client.get(f"/api/projects/{pid}/ingredients").json()}
    assert after["mara"]["visual_prompt"] == "my mara"
    assert after["old-tom"]["origin"] == "user"

    bible_call = [c for c in h.fake.calls if c["purpose"] == "bible"][-1]
    user_msg = bible_call["messages"][1]["content"]
    assert "<existing_cast>" in user_msg and "old-tom" in user_msg and "my mara" in user_msg
    # The fake bible omits old-tom, so the repair loop asked for it; the user character is kept anyway.
    assert any("old-tom" in m["content"] for m in h.fake.calls[-1]["messages"] if m["role"] == "user")

    bd_call_count = len([c for c in h.fake.calls if c["purpose"] == "breakdown"])
    h.post(f"/api/projects/{pid}/story/breakdown", {"replace_existing": True})
    bd_call = [c for c in h.fake.calls if c["purpose"] == "breakdown"][bd_call_count]
    assert '"character_info": "grumpy"' in bd_call["messages"][1]["content"]


def test_wizard_character_survives_bible_rerun_with_info_and_voice(h: Harness) -> None:
    """M3.5 acceptance 5: a created character keeps its key and visual prompt through a bible re-run, and
    the LLM sees its character info and voice. Unsaved wizard drafts are invisible to the pipeline."""
    pid = h.project()
    h.run_all(pid)
    assert h.client.get(f"/api/projects/{pid}/ingredients").json()[0]["voice_hints"] == {
        "gender": "female",
        "age": "50s",
        "tone": "warm, husky",
    }
    draft = h.post(f"/api/projects/{pid}/characters", {"fields": {"name": "Pip"}}, status=201)
    r = h.client.patch(
        f"/api/ingredients/{draft['id']}",
        json={
            "visual_prompt": "a small boy in a red raincoat",
            "character_info": "Curious; collects sea glass.",
            "voice_id": "am_puck",
            "voice_hints": {"gender": "male", "age": "child", "tone": "bright"},
            "draft": False,
        },
    )
    assert r.status_code == 200 and r.json()["key"] == "pip"
    unsaved = h.post(f"/api/projects/{pid}/characters", {"fields": {"name": "Ghost"}}, status=201)
    h.client.patch(f"/api/ingredients/{unsaved['id']}", json={"visual_prompt": "a translucent sailor"})

    r = h.post(f"/api/projects/{pid}/story/bible")
    assert r["story"]["template_versions"]["bible"] == "v2"
    pip = next(i for i in h.client.get(f"/api/projects/{pid}/ingredients").json() if i["id"] == draft["id"])
    assert pip["key"] == "pip" and pip["visual_prompt"] == "a small boy in a red raincoat"
    assert pip["origin"] == "user" and pip["voice_hints"]["tone"] == "bright"

    bible_call = [c for c in h.fake.calls if c["purpose"] == "bible"][-1]
    user_msg = bible_call["messages"][1]["content"]
    assert '"key": "pip"' in user_msg and "Curious; collects sea glass." in user_msg
    assert '"name": "am_puck"' in user_msg and '"description": "male, child, bright"' in user_msg
    assert "Ghost" not in user_msg and "translucent sailor" not in user_msg


def test_stale_bible_ingredients_are_removed(h: Harness) -> None:
    pid = h.project()
    h.run_all(pid)
    smaller = json.loads(json.dumps(fx.BIBLE))
    smaller["props"] = []
    h.fake.script["bible"] = [smaller]
    r = h.post(f"/api/projects/{pid}/story/bible")
    assert r["ingredients"]["removed"] == 1
    keys = [i["key"] for i in h.client.get(f"/api/projects/{pid}/ingredients").json()]
    assert "letter" not in keys
    # Shots that referenced the removed prop no longer resolve it, but the board still loads.
    assert h.client.get(f"/api/projects/{pid}/scenes").status_code == 200


def test_rewrite_prompt_returns_a_proposal(h: Harness) -> None:
    pid = h.project()
    h.run_all(pid)
    shot = h.client.get(f"/api/projects/{pid}/scenes").json()["scenes"][0]["shots"][0]
    r = h.post(f"/api/shots/{shot['id']}/rewrite-prompt", {"instruction": "make it night"})
    assert r["proposal"]["video_prompt"].startswith("Mara walks slowly")
    assert h.client.get(f"/api/shots/{shot['id']}").json()["video_prompt"] == shot["video_prompt"]
    call = h.fake.calls[-1]["messages"][1]["content"]
    assert "<instruction>\nmake it night\n</instruction>" in call and "teal and amber" in call


def test_no_llm_key_gives_a_clear_error(h: Harness) -> None:
    h.store.delete("openai_api_key")
    pid = h.project()
    r = h.client.post(f"/api/projects/{pid}/story/refine", json={"raw_idea": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "llm_not_configured"
    providers = h.client.get("/api/llm/providers").json()
    assert all(not p["configured"] for p in providers)


def test_settings_defaults_choose_provider_and_model(h: Harness) -> None:
    h.store.set("openrouter_api_key", "sk-or-test-key-0000")
    r = h.client.patch(
        "/api/settings", json={"llm_defaults": {"refine": {"provider": "openrouter", "model": "x/y"}}}
    )
    assert r.json()["llm_defaults"]["refine"] == {"provider": "openrouter", "model": "x/y"}
    res = h.post(f"/api/projects/{h.project()}/story/refine", {"raw_idea": fx.IDEA})
    assert (res["usage"]["provider"], res["usage"]["model"]) == ("openrouter", "x/y")
    res = h.post(
        f"/api/projects/{h.project()}/story/refine",
        {"raw_idea": fx.IDEA, "llm": {"provider": "openai", "model": "gpt-x"}},
    )
    assert res["usage"]["model"] == "gpt-x"
    assert h.client.patch("/api/settings", json={"llm_defaults": {"bogus": None}}).status_code == 400


def test_secret_endpoints_never_return_values(h: Harness) -> None:
    r = h.client.put("/api/settings/secrets/anthropic_api_key", json={"value": "sk-ant-very-secret-4242"})
    assert r.status_code == 200
    body = h.client.get("/api/settings/secrets").text
    assert "very-secret" not in body and "sk-test-openai-key-1234" not in body
    statuses = {s["name"]: s for s in json.loads(body)}
    assert statuses["anthropic_api_key"]["last4"] == "4242" and statuses["anthropic_api_key"]["stored"]
    tested = h.client.post("/api/settings/secrets/openai_api_key/test").json()
    assert tested["ok"] is True and tested["account"] == {"username": "tester"}
    assert h.client.delete("/api/settings/secrets/anthropic_api_key").json()["configured"] is False
    assert h.client.post("/api/settings/secrets/hf_token/test").status_code == 409
