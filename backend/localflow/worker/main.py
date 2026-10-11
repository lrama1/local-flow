"""GPU worker process entry point (Section 4.1). Started by the API with the `spawn` context."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import threading
import traceback
from pathlib import Path
from typing import Any

from localflow.logging_setup import setup_logging
from localflow.services.media.ffmpeg import FFmpegCancelled
from localflow.worker.context import JobCancelled, JobContext
from localflow.worker.engine import Engine, torch_runtime
from localflow.worker.messages import (
    CancelRequest,
    JobEvent,
    JobRequest,
    ShutdownRequest,
    WorkerConfig,
    WorkerEvent,
)

log = logging.getLogger("localflow.worker")


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_job_id: str | None = None
        self.cancel_event = threading.Event()
        self.pending_cancels: set[str] = set()


def _control_loop(control_q: Any, state: _State) -> None:
    while True:
        msg = control_q.get()
        if msg is None:
            return
        if isinstance(msg, CancelRequest):
            with state.lock:
                if state.current_job_id == msg.job_id:
                    state.cancel_event.set()
                else:
                    state.pending_cancels.add(msg.job_id)


def _watch_parent() -> None:
    """Exit if the API process dies without shutting us down (e.g. killed by a reloader)."""
    parent = mp.parent_process()
    if parent is None:
        return

    def _wait() -> None:
        parent.join()
        os._exit(0)

    threading.Thread(target=_wait, daemon=True, name="parent-watch").start()


def _run_job(engine: Engine, req: JobRequest, state: _State, events: Any, config: WorkerConfig) -> None:
    with state.lock:
        state.current_job_id = req.job_id
        state.cancel_event = threading.Event()
        if req.job_id in state.pending_cancels:
            state.pending_cancels.discard(req.job_id)
            state.cancel_event.set()
        cancel_event = state.cancel_event

    ctx = JobContext(
        job_id=req.job_id,
        seed=req.seed,
        output_dir=Path(req.output_dir),
        preview_dir=Path(req.preview_dir),
        vram_profile=req.vram_profile,
        cancel_event=cancel_event,
        event_queue=events,
        settings={"preview_every_n_steps": config.preview_every_n_steps},
    )
    events.put(JobEvent(req.job_id, "started"))
    try:
        ctx.check_cancelled()
        outputs, meta = engine.run(req, ctx)
        events.put(
            JobEvent(
                req.job_id, "completed", fraction=1.0, outputs=outputs, result={**meta, "seed": req.seed}
            )
        )
    except (JobCancelled, FFmpegCancelled):
        events.put(JobEvent(req.job_id, "cancelled"))
    except Exception as e:
        log.error("job failed: %s\n%s", e, traceback.format_exc(), extra={"job_id": req.job_id})
        events.put(JobEvent(req.job_id, "failed", error=f"{type(e).__name__}: {e}"))
    finally:
        with state.lock:
            state.current_job_id = None


def _prepare_environment(config: WorkerConfig) -> None:
    """Library env for the worker. Must run before torch/diffusers/huggingface_hub are imported."""
    if config.hf_home:
        os.environ["HF_HOME"] = config.hf_home  # configs fetched by from_single_file stay in the data dir
    # Tokens are only ever passed explicitly (Section 7.3); the worker never needs one.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def worker_main(config: WorkerConfig, job_q: Any, control_q: Any, event_q: Any) -> None:
    # The supervisor owns our lifecycle (ShutdownRequest / terminate). Console Ctrl+C, e.g. from
    # uvicorn's reloader on Windows, must not kill a job mid-step.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _prepare_environment(config)
    setup_logging(Path(config.data_dir) / "logs", config.log_level, filename="worker.log")
    _watch_parent()
    state = _State()
    threading.Thread(target=_control_loop, args=(control_q, state), daemon=True, name="control").start()
    try:
        engine = Engine(config, event_q)
    except Exception as e:
        log.exception("engine init failed")
        event_q.put(WorkerEvent("ready", {"pid": os.getpid(), "adapters": [], "error": str(e)}))
        return
    event_q.put(
        WorkerEvent(
            "ready",
            {
                "pid": os.getpid(),
                "adapters": engine.describe(),
                "engine": engine.engine,
                "engine_note": engine.reason,
                "vram_baseline": engine.cache.baseline,
                "runtime": torch_runtime(),
            },
        )
    )
    log.info(
        "worker ready (pid %s, engine=%s%s; VRAM baseline %s)",
        os.getpid(),
        engine.engine,
        f": {engine.reason}" if engine.reason else "",
        engine.cache.baseline,
    )
    while True:
        msg = job_q.get()
        if msg is None or isinstance(msg, ShutdownRequest):
            break
        if isinstance(msg, JobRequest):
            _run_job(engine, msg, state, event_q, config)
    engine.shutdown()
    control_q.put(None)
    log.info("worker stopped")
