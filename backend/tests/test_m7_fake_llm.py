"""M7 `fake_llm` mode: the whole LLM surface answers with canned demo responses and no keys (for offline demos
and the Playwright smoke test)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localflow.config import Settings
from localflow.main import create_app
from localflow.services.llm.demo import BIBLE, REFINED, demo_script
from localflow.services.llm.types import PURPOSES
from tests.conftest import TOKEN, make_settings
from tests.test_m3_imagegen import ok


@pytest.fixture
def c(tmp_path: Path) -> Iterator[TestClient]:
    # make_settings already keeps the developer's env keys and keyring out: no LLM key exists here.
    settings = make_settings(tmp_path / "data", start_worker=False, fake_llm=True)
    with TestClient(create_app(settings), headers={"X-LocalFlow-Token": TOKEN}) as client:
        yield client


def test_demo_script_covers_every_purpose() -> None:
    assert set(demo_script()) == set(PURPOSES)


def test_fake_llm_runs_the_story_pipeline_without_keys(c: TestClient) -> None:
    providers = {p["provider"]: p for p in ok(c, "GET", "/api/llm/providers")}
    assert providers["openai"]["configured"] and providers["openai"]["source"] == "fake"
    assert not providers["anthropic"]["configured"]
    assert ok(c, "GET", "/api/llm/vision")["available"]

    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Demo"})["id"]
    refined = ok(c, "POST", f"/api/projects/{pid}/story/refine", json={"raw_idea": "a lighthouse keeper"})
    assert refined["story"]["refined"]["title"] == REFINED["title"]
    ok(c, "POST", f"/api/projects/{pid}/story/bible")
    keys = {i["key"] for i in ok(c, "GET", f"/api/projects/{pid}/ingredients")}
    assert {ch["key"] for ch in BIBLE["characters"]} <= keys
    ok(c, "POST", f"/api/projects/{pid}/story/breakdown", json={})
    board = ok(c, "GET", f"/api/projects/{pid}/scenes")
    assert board["shot_count"] == 5 and all(
        sh["duration_s"] <= board["max_clip_seconds"] for sc in board["scenes"] for sh in sc["shots"]
    )
    expanded = ok(c, "POST", "/api/characters/expand", json={"description": "an old keeper"})
    assert expanded["name"] == "Mara"
    usage = ok(c, "GET", "/api/llm/usage")
    assert usage["calls"] == 4 and usage["cost_usd"] == 0

    diag = {x["key"]: x for x in ok(c, "GET", "/api/system/diagnostics")["checks"]}
    assert diag["llm"]["status"] == "warn" and "Fake LLM mode" in diag["llm"]["detail"]


def test_fake_llm_is_off_by_default() -> None:
    assert Settings.model_fields["fake_llm"].default is False
