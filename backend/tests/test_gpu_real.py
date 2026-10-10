"""Real-GPU regression checks for the M3 image adapters (Section 18). Opt-in: `uv run pytest -m gpu`.

Uses the models installed in the configured data dir (LOCALFLOW_DATA_DIR) and skips whatever is missing.
Short step counts keep it to a few minutes; full-quality timings are measured through the app instead.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from localflow.config import load_settings
from localflow.services.container import build_services
from localflow.services.registry.service import RegistryService
from localflow.worker.context import JobContext
from localflow.worker.engine import Engine, vram_snapshot
from localflow.worker.messages import JobRequest, WorkerConfig, WorkerEvent

pytestmark = pytest.mark.gpu

BASELINE_TOLERANCE_MB = 500  # SPEC M3 acceptance: unloading returns VRAM to baseline ± 500 MB


@pytest.fixture(scope="module")
def registry() -> RegistryService:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    settings = load_settings()
    if not settings.db_path.exists():
        pytest.skip(f"no LocalFlow database in {settings.data_dir}")
    return build_services(settings).registry


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Engine, queue.Queue[Any]]]:
    events: queue.Queue[Any] = queue.Queue()
    cfg = WorkerConfig(
        data_dir=str(tmp_path_factory.mktemp("gpu")),
        ffmpeg=None,
        ffprobe=None,
        engine="real",
        vram_profile="balanced",
        fake_step_delay_s=0,
    )
    eng = Engine(cfg, events)
    assert eng.engine == "real", eng.reason
    yield eng, events
    eng.shutdown()


def _spec(registry: RegistryService, catalog_key: str) -> dict[str, Any]:
    entry = registry._installed_catalog(catalog_key)
    if entry is None:
        pytest.skip(f"{catalog_key} is not installed")
    return registry.model_spec(entry.id)


def _run(
    eng: Engine,
    events: queue.Queue[Any],
    out: Path,
    job_type: str,
    adapter: str,
    spec: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    ctx = JobContext(
        job_id=f"gpu-{adapter}",
        seed=7,
        output_dir=out,
        preview_dir=out / "previews",
        vram_profile="balanced",
        cancel_event=threading.Event(),
        event_queue=events,
        settings={"preview_every_n_steps": 4},
    )
    req = JobRequest(
        job_id=ctx.job_id,
        job_type=job_type,
        adapter_key=adapter,
        params=params,
        output_dir=str(out),
        preview_dir=str(out / "previews"),
        seed=7,
        vram_profile="balanced",
        model_entry_id=spec["entry_id"],
        model=spec,
    )
    outputs, _meta = eng.run(req, ctx)
    drained = []
    while not events.empty():
        drained.append(events.get())
    return outputs, drained


def test_flux_t2i_then_kontext_edit_then_sdxl_and_unload(
    registry: RegistryService, engine: tuple[Engine, queue.Queue[Any]], tmp_path: Path
) -> None:
    eng, events = engine
    flux = _spec(registry, "flux1-dev-q4")
    outs, evs = _run(
        eng,
        events,
        tmp_path / "flux",
        "t2i",
        "flux",
        flux,
        {
            "prompt": "a red apple on a wooden table",
            "width": 1024,
            "height": 1024,
            "num_images": 2,
            "steps": 8,
        },
    )
    assert len(outs) == 2
    with Image.open(outs[0].path) as im:
        assert im.size == (1024, 1024)
        low, high = im.convert("L").getextrema()
        assert isinstance(low, int) and isinstance(high, int) and high - low > 32  # not a flat/black image
    assert any(isinstance(e, WorkerEvent) and e.kind == "model_loaded" for e in evs)

    kontext = _spec(registry, "flux1-kontext-dev-q4")
    refs = [o.path for o in outs]
    edits, _ = _run(
        eng,
        events,
        tmp_path / "kontext",
        "image_edit",
        "flux_kontext",
        kontext,
        {
            "prompt": "Put both apples side by side on a marble counter",
            "reference_image_paths": refs,
            "width": 1280,
            "height": 720,
            "num_images": 1,
            "steps": 8,
        },
    )
    assert len(edits) == 1
    with Image.open(edits[0].path) as im:
        assert im.size == (1280, 720)

    sdxl = _spec(registry, "sdxl-base")
    images, evs = _run(
        eng,
        events,
        tmp_path / "sdxl",
        "t2i",
        "sdxl",
        sdxl,
        {"prompt": "a lighthouse at dusk", "width": 1024, "height": 1024, "num_images": 1, "steps": 10},
    )
    assert len(images) == 1
    unloaded = [e for e in evs if isinstance(e, WorkerEvent) and e.kind == "model_unloaded"]
    assert unloaded, "switching models must unload the previous one"

    baseline = eng.cache.baseline
    assert baseline is not None
    eng.cache.unload()
    after = vram_snapshot()
    assert after is not None
    assert abs(after["device_used_mb"] - baseline["device_used_mb"]) <= BASELINE_TOLERANCE_MB, (
        after,
        baseline,
    )
