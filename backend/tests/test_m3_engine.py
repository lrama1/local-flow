"""Engine behavior without a GPU: ModelCache keys, OOM retry with a lower VRAM profile, helpers."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from localflow.worker.adapters.base import Adapter, AdapterInfo, ImageResult, T2IRequest
from localflow.worker.context import JobContext
from localflow.worker.engine import Engine, EngineError, is_oom, resolve_engine
from localflow.worker.images import reference_sheet, save_png, snap
from localflow.worker.messages import JobRequest, WorkerConfig, WorkerEvent
from localflow.worker.spec import LoraSpec, ModelSpec


class OutOfMemoryError(RuntimeError):  # same class name as torch.cuda.OutOfMemoryError
    pass


class ScriptedAdapter(Adapter):
    requires_model = True
    info = AdapterInfo(
        adapter_key="scripted", adapter_version="1", name="Scripted", capabilities=["t2i"], supports_lora=True
    )

    def __init__(self, oom_profiles: set[str]) -> None:
        self.oom_profiles = oom_profiles
        self.loads: list[tuple[str, str]] = []
        self.unloads = 0
        self.loras: list[list[str]] = []
        self.profile = ""

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        assert model is not None
        self.loads.append((model.entry_id, profile))
        self.profile = profile
        return {"mode": "resident"}

    def apply_loras(self, loras: list[LoraSpec], ctx: JobContext) -> None:
        self.loras.append([lo.entry_id for lo in loras])

    def unload(self) -> None:
        self.unloads += 1

    def generate_t2i(self, req: T2IRequest, ctx: JobContext) -> list[ImageResult]:
        if self.profile in self.oom_profiles:
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        path = ctx.output_dir / "a.png"
        Image.new("RGB", (8, 8)).save(path)
        return [ImageResult(str(path), req.width, req.height, ctx.seed)]


def _engine(tmp_path: Path, adapter: Adapter) -> tuple[Engine, queue.Queue[Any]]:
    events: queue.Queue[Any] = queue.Queue()
    cfg = WorkerConfig(
        data_dir=str(tmp_path),
        ffmpeg=None,
        ffprobe=None,
        engine="fake",
        vram_profile="balanced",
        fake_step_delay_s=0,
    )
    eng = Engine(cfg, events)
    eng.adapters = {"scripted": adapter}
    return eng, events


def _spec(entry_id: str = "m1") -> dict[str, Any]:
    return ModelSpec(
        entry_id=entry_id,
        name="Model One",
        adapter_key="scripted",
        kind="checkpoint",
        base_model="x",
        source="hf",
    ).to_dict()


def _run(
    eng: Engine,
    events: queue.Queue[Any],
    tmp_path: Path,
    *,
    profile: str = "balanced",
    entry: str = "m1",
    loras: list[dict[str, Any]] | None = None,
) -> tuple[Any, list[Any]]:
    ctx = JobContext(
        job_id="j",
        seed=1,
        output_dir=tmp_path / "out",
        preview_dir=tmp_path / "prev",
        vram_profile=profile,
        cancel_event=threading.Event(),
        event_queue=events,
    )
    req = JobRequest(
        job_id="j",
        job_type="t2i",
        adapter_key="scripted",
        params={"prompt": "x", "width": 512, "height": 512, "num_images": 2},
        output_dir=str(tmp_path / "out"),
        preview_dir=str(tmp_path / "prev"),
        seed=1,
        vram_profile=profile,
        model_entry_id=entry,
        model=_spec(entry),
        loras=loras or [],
    )
    result = eng.run(req, ctx)
    drained = []
    while not events.empty():
        drained.append(events.get())
    return result, drained


def test_cache_reuses_model_and_swaps_loras(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(set())
    eng, events = _engine(tmp_path, adapter)
    _run(eng, events, tmp_path)
    lora = LoraSpec(entry_id="l1", name="L", path="x.safetensors", weight=0.8).to_dict()
    (_outs, meta), _ = _run(eng, events, tmp_path, loras=[lora])
    assert adapter.loads == [("m1", "balanced")]  # same model + profile: no reload
    assert adapter.loras == [[], ["l1"]]
    assert meta["loras"] == [{"model_entry_id": "l1", "weight": 0.8}]
    _, evs = _run(eng, events, tmp_path, entry="m2")
    assert adapter.loads[-1] == ("m2", "balanced") and adapter.unloads == 1
    kinds = [e.kind for e in evs if isinstance(e, WorkerEvent)]
    assert kinds == ["model_unloaded", "model_loaded"]
    loaded = next(e for e in evs if isinstance(e, WorkerEvent) and e.kind == "model_loaded")
    assert loaded.data["model_name"] == "Model One" and loaded.data["vram_profile"] == "balanced"


def test_oom_retries_once_with_lower_profile(tmp_path: Path) -> None:
    adapter = ScriptedAdapter({"balanced"})
    eng, events = _engine(tmp_path, adapter)
    (outs, meta), evs = _run(eng, events, tmp_path)
    assert meta["vram_profile"] == "low" and len(outs) == 1
    assert adapter.loads == [("m1", "balanced"), ("m1", "low")]
    messages = [getattr(e, "message", "") or "" for e in evs]
    assert any("retrying with the 'low' VRAM profile" in m for m in messages)


def test_second_oom_is_actionable(tmp_path: Path) -> None:
    adapter = ScriptedAdapter({"balanced", "low"})
    eng, events = _engine(tmp_path, adapter)
    with pytest.raises(EngineError) as exc:
        _run(eng, events, tmp_path)
    msg = str(exc.value)
    assert "Out of GPU memory generating 2 images at 512x512 with Model One (low profile)" in msg
    assert "fewer images" in msg
    assert eng.cache.adapter is None  # memory was released


def test_low_profile_oom_fails_without_retry(tmp_path: Path) -> None:
    adapter = ScriptedAdapter({"low"})
    eng, events = _engine(tmp_path, adapter)
    with pytest.raises(EngineError):
        _run(eng, events, tmp_path, profile="low")
    assert adapter.loads == [("m1", "low")]


def test_real_adapter_requires_model(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(set())
    eng, events = _engine(tmp_path, adapter)
    ctx = JobContext(
        job_id="j",
        seed=1,
        output_dir=tmp_path / "o",
        preview_dir=tmp_path / "p",
        vram_profile="balanced",
        cancel_event=threading.Event(),
        event_queue=events,
    )
    req = JobRequest(
        job_id="j",
        job_type="t2i",
        adapter_key="scripted",
        params={"prompt": "x"},
        output_dir=str(tmp_path / "o"),
        preview_dir=str(tmp_path / "p"),
        seed=1,
        vram_profile="balanced",
    )
    with pytest.raises(EngineError, match="needs an installed model"):
        eng.run(req, ctx)


def test_is_oom_and_resolve_engine() -> None:
    assert is_oom(OutOfMemoryError("x"))
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not is_oom(ValueError("nope"))
    assert resolve_engine("fake") == ("fake", None)


def test_helpers(tmp_path: Path) -> None:
    assert snap(1010, 16) == 1008 and snap(720, 16) == 720 and snap(100, 16) == 256 and snap(9999, 8) == 2048
    a, b = Image.new("RGB", (512, 512), "red"), Image.new("RGB", (400, 800), "blue")
    sheet = reference_sheet([a, b], height=512, gutter=10)
    assert sheet.size == (512 + 10 + 256, 512)
    assert sheet.getpixel((5, 5)) == (255, 0, 0) and sheet.getpixel((520, 5)) == (255, 255, 255)
    assert reference_sheet([a]).size == (512, 512)
    wide = reference_sheet([a] * 6, height=1024, max_width=2560)
    assert wide.width == 2560
    p = save_png(a, tmp_path / "x.png", {"prompt": "hi", "seed": 3})
    with Image.open(p) as im:
        assert '"seed": 3' in im.text["localflow"]  # type: ignore[attr-defined]


def test_spec_roundtrip() -> None:
    spec = ModelSpec(
        entry_id="e",
        name="N",
        adapter_key="flux",
        kind="checkpoint",
        base_model="flux1",
        source="hf",
        parts=(),
        single_file=None,
        default_params={"steps": 20},
        aux={"vae_dir": "v"},
    )
    again = ModelSpec.from_dict(spec.to_dict())
    assert again == spec and again.default("steps", 1) == 20 and again.default("x", 5) == 5
