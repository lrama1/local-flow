"""Draft models and the project's Draft mode: catalog entries, default-model ordering, the project API and
the FLUX adapter's bundled Turbo-Alpha LoRA."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.services.registry.catalog import load_catalog
from localflow.worker.adapters.flux import FluxAdapter, FluxKontextAdapter
from localflow.worker.messages import WorkerConfig
from localflow.worker.spec import LoraSpec, ModelSpec, PartSpec
from tests.conftest import TOKEN, make_settings, wait_worker_idle
from tests.test_m3_imagegen import install, ok, uninstall
from tests.test_m4_video import make_project

DRAFTS = {"flux1-dev-q4-turbo", "flux1-kontext-dev-q4-turbo", "wan22-ti2v-5b-turbo"}


@pytest.fixture(scope="module")
def c(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = make_settings(tmp_path_factory.mktemp("draft"))
    with TestClient(create_app(settings), headers={"X-LocalFlow-Token": TOKEN}) as client:
        wait_worker_idle(client)
        yield client


# --- catalog -----------------------------------------------------------------------------------------
def test_catalog_has_the_draft_models() -> None:
    entries = {e.key: e for e in load_catalog()}
    assert {k for k, e in entries.items() if e.tier == "draft"} == DRAFTS
    assert all(e.tier == "final" for k, e in entries.items() if k not in DRAFTS)

    for key, final in (
        ("flux1-dev-q4-turbo", "flux1-dev-q4"),
        ("flux1-kontext-dev-q4-turbo", "flux1-kontext-dev-q4"),
    ):
        draft, base = entries[key], entries[final]
        assert (draft.adapter_key, draft.base_model, draft.capabilities) == (
            base.adapter_key,
            base.base_model,
            base.capabilities,
        )
        assert draft.default_params["steps"] == 8
        lora = [p for p in draft.parts if p.role == "lora"]
        assert [(p.repo_id, p.allow_patterns) for p in lora] == [
            ("alimama-creative/FLUX.1-Turbo-Alpha", ["diffusion_pytorch_model.safetensors"])
        ]
        # The other parts are the final model's: the downloads are shared.
        assert [(p.role, p.repo_id) for p in draft.parts if p.role != "lora"] == [
            (p.role, p.repo_id) for p in base.parts
        ]
        assert draft.license == base.license  # Turbo-Alpha inherits FLUX [dev]'s non-commercial license

    wan = entries["wan22-ti2v-5b-turbo"]
    assert wan.adapter_key == "wan" and wan.base_model == entries["wan22-ti2v-5b"].base_model
    assert (wan.default_params["steps"], wan.default_params["guidance_scale"]) == (4, 1.0)
    assert wan.default_params["sizes"] == entries["wan22-ti2v-5b"].default_params["sizes"]
    assert wan.render == entries["wan22-ti2v-5b"].render
    assert next(p for p in wan.parts if p.role == "base").repo_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def test_models_api_exposes_the_tier(c: TestClient) -> None:
    models = {m["catalog_key"]: m for m in ok(c, "GET", "/api/models")}
    assert models["wan22-ti2v-5b-turbo"]["tier"] == "draft"
    assert models["wan22-ti2v-5b"]["tier"] == "final"


# --- default model ordering ------------------------------------------------------------------------
def test_draft_models_are_only_the_default_in_draft_mode_or_when_alone(c: TestClient) -> None:
    reg = c.app.state.services.registry  # type: ignore[attr-defined]

    def pick(draft: bool, **kw: Any) -> str | None:
        e = reg.default_entry("i2v", adapter_key=None, usable=None, preferred=[], draft=draft, **kw)
        return e.catalog_key if e else None

    assert pick(False) is None and pick(True) is None
    turbo = install(c, "wan22-ti2v-5b-turbo")
    assert pick(False) == pick(True) == "wan22-ti2v-5b-turbo"  # the only one installed
    final = install(c, "wan22-ti2v-5b")
    try:
        assert pick(False) == "wan22-ti2v-5b"
        assert pick(True) == "wan22-ti2v-5b-turbo"
        uninstall(c, turbo)
        assert pick(True) == "wan22-ti2v-5b"  # no draft model installed: falls back
        # An explicit project default still wins over catalog order outside Draft mode.
        a14b = install(c, "wan22-i2v-a14b-q4")
        try:
            e = reg.default_entry("i2v", adapter_key=None, usable=None, preferred=[a14b], draft=False)
            assert e.catalog_key == "wan22-i2v-a14b-q4"
            install(c, "wan22-ti2v-5b-turbo")
            e = reg.default_entry("i2v", adapter_key=None, usable=None, preferred=[a14b], draft=True)
            assert e.catalog_key == "wan22-ti2v-5b-turbo"  # Draft mode puts draft models first
        finally:
            uninstall(c, a14b)
    finally:
        uninstall(c, final)
        uninstall(c, turbo)


# --- project API -------------------------------------------------------------------------------------
def test_project_draft_mode_picks_the_draft_video_model(c: TestClient) -> None:
    final = install(c, "wan22-ti2v-5b")
    turbo = install(c, "wan22-ti2v-5b-turbo")
    try:
        p = make_project(c)
        project = ok(c, "GET", f"/api/projects/{p['pid']}")
        assert project["draft_mode"] is False
        plan = ok(c, "POST", f"/api/shots/{p['shot']}/video-plan", json={})
        assert plan["model_entry_id"] == final

        project = ok(c, "PATCH", f"/api/projects/{p['pid']}", json={"draft_mode": True})
        assert project["draft_mode"] is True
        plan = ok(c, "POST", f"/api/shots/{p['shot']}/video-plan", json={})
        assert plan["model_entry_id"] == turbo
        # Naming a model still beats Draft mode.
        plan = ok(c, "POST", f"/api/shots/{p['shot']}/video-plan", json={"model_entry_id": final})
        assert plan["model_entry_id"] == final

        ok(c, "PATCH", f"/api/projects/{p['pid']}", json={"draft_mode": False})
        plan = ok(c, "POST", f"/api/shots/{p['shot']}/video-plan", json={})
        assert plan["model_entry_id"] == final
    finally:
        uninstall(c, final)
        uninstall(c, turbo)


def test_projects_can_be_created_in_draft_mode(c: TestClient) -> None:
    created = ok(c, "POST", "/api/projects", 201, json={"name": "Quick look", "draft_mode": True})
    assert created["draft_mode"] is True


# --- adapter ---------------------------------------------------------------------------------------
class _Pipe:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, str]] = []
        self.active: tuple[list[str], list[float]] | None = None

    def load_lora_weights(self, path: str, adapter_name: str) -> None:
        self.loaded.append((Path(path).name, adapter_name))

    def enable_lora(self) -> None: ...

    def disable_lora(self) -> None: ...

    def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
        self.active = (names, adapter_weights)


def _adapter(cls: type, parts: tuple[PartSpec, ...]) -> tuple[Any, _Pipe, Any]:
    a = cls(
        WorkerConfig(
            data_dir=".",
            ffmpeg=None,
            ffprobe=None,
            engine="fake",
            vram_profile="balanced",
            fake_step_delay_s=0,
        )
    )
    a.pipe = _Pipe()
    a.spec = ModelSpec("m1", "Draft", "flux", "checkpoint", "flux1", "hf", parts=parts)
    ctx = SimpleNamespace(job_id="j", report_progress=lambda *a_, **k: None)
    return a, a.pipe, ctx


@pytest.mark.parametrize("cls", [FluxAdapter, FluxKontextAdapter])
def test_bundled_lora_is_always_applied(cls: type, tmp_path: Path) -> None:
    lora_dir = tmp_path / "alimama-creative--FLUX.1-Turbo-Alpha"
    parts = (PartSpec("lora", str(lora_dir), ("diffusion_pytorch_model.safetensors",)),)
    a, pipe, ctx = _adapter(cls, parts)

    a.apply_loras([], ctx)  # a job with no LoRAs of its own
    assert pipe.loaded == [("diffusion_pytorch_model.safetensors", "lora_0")]
    assert pipe.active == (["lora_0"], [1.0])

    style = LoraSpec("style", "Style", str(tmp_path / "style.safetensors"), 0.6)
    a.apply_loras([style], ctx)
    assert pipe.active == (["lora_0", "lora_1"], [1.0, 0.6])  # bundled first, then the job's
    assert [lo.name for lo in a._lora_specs][1:] == ["Style"]

    # Reloading the transformer re-applies the recorded LoRAs, bundled included: no duplicates.
    again = list(a._lora_specs)
    a._active = ()
    a.apply_loras(again, ctx)
    assert pipe.active == (["lora_0", "lora_1"], [1.0, 0.6])
    assert len(a._lora_specs) == 2


def test_models_without_a_lora_part_are_unchanged(tmp_path: Path) -> None:
    a, pipe, ctx = _adapter(FluxAdapter, (PartSpec("transformer", str(tmp_path), ("m.gguf",)),))
    a.apply_loras([], ctx)
    assert pipe.loaded == [] and pipe.active is None
