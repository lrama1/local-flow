"""Spawns/supervises the GPU worker, dispatches queued GPU jobs, handles worker events (Section 4.1)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import queue
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from localflow.config import Settings
from localflow.errors import AppError
from localflow.models import Asset, Job
from localflow.schemas.core import AssetOut, WorkerInfo
from localflow.services.assets import AssetService
from localflow.services.events import EventBus
from localflow.services.jobs.generation import GenerationService
from localflow.services.jobs.service import TERMINAL, JobService, model_key
from localflow.worker.main import worker_main
from localflow.worker.messages import (
    CancelRequest,
    JobEvent,
    ShutdownRequest,
    WorkerConfig,
    WorkerEvent,
)

log = logging.getLogger("localflow.supervisor")

_RESTART_BACKOFF_S = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


def exit_code_text(code: int | None) -> str:
    """Windows exit codes are unsigned 32-bit: a killed worker reports 4294967295 (-1) and a native crash
    3221225477 (0xC0000005), so they're shown signed with their hex form, which is what people search for.
    Elsewhere a negative code is the signal that killed the process."""
    if code is None:
        return "unknown"
    if sys.platform == "win32":
        if code > 0x7FFFFFFF:
            code -= 1 << 32
        return f"{code} (0x{code & 0xFFFFFFFF:08X})" if code < 0 else str(code)
    return f"{code} (signal {-code})" if code < 0 else str(code)


class WorkerSupervisor:
    def __init__(
        self,
        settings: Settings,
        jobs: JobService,
        generation: GenerationService,
        assets: AssetService,
        bus: EventBus,
    ) -> None:
        self.settings = settings
        self.jobs = jobs
        self.generation = generation
        self.assets = assets
        self.bus = bus

        self.state: str = "stopped" if settings.start_worker else "disabled"
        self.pid: int | None = None
        self.adapters: list[dict[str, Any]] | None = None
        self.loaded_model: dict[str, Any] | None = None
        self.current_job_id: str | None = None
        self.restarts = 0
        self.last_exit_code: int | None = None
        self.error: str | None = None
        self.previews: dict[str, str] = {}
        self.engine: str = settings.engine
        self.engine_note: str | None = None
        self.vram: dict[str, Any] | None = None
        # Called after a job's outputs are ingested (links candidates to ingredients/shots).
        self.on_assets: Callable[[Job, list[Asset]], None] | None = None

        self._ctx = mp.get_context("spawn")
        self._proc: Any = None
        self._job_q: Any = None
        self._control_q: Any = None
        self._event_q: Any = None
        self._generation = 0
        self._consecutive_crashes = 0
        self._cancel_requested: tuple[str, float] | None = None
        self._stopping = False
        self._wakeup = asyncio.Event()
        self._inbox: asyncio.Queue[JobEvent | WorkerEvent] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

        jobs.add_listener(on_change=self.wake, on_cancel=self._cancel_running)

    # --- lifecycle -------------------------------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._spawn()
        self._tasks = [
            asyncio.create_task(self._dispatch_loop(), name="dispatch"),
            asyncio.create_task(self._event_loop(), name="worker-events"),
            asyncio.create_task(self._watchdog_loop(), name="watchdog"),
        ]

    async def stop(self) -> None:
        self._stopping = True
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        if self.current_job_id:
            with contextlib.suppress(AppError):
                self.jobs.finish(self.current_job_id, "failed", error="Interrupted (LocalFlow was shut down)")
            self.current_job_id = None
        await asyncio.to_thread(self._terminate)
        self._set_state("stopped")

    def _spawn(self) -> None:
        self._generation += 1
        gen = self._generation
        self._job_q = self._ctx.Queue()
        self._control_q = self._ctx.Queue()
        self._event_q = self._ctx.Queue()
        cfg = WorkerConfig(
            data_dir=str(self.settings.data_dir),
            ffmpeg=self.settings.ffmpeg,
            ffprobe=self.settings.ffprobe,
            engine=self.settings.engine,
            vram_profile=self.settings.vram_profile,
            fake_step_delay_s=self.settings.fake_step_delay_s,
            log_level=self.settings.log_level,
            hf_home=str(self.settings.resolved_hf_home),
            preview_every_n_steps=self.settings.preview_every_n_steps,
        )
        self._proc = self._ctx.Process(
            target=worker_main,
            args=(cfg, self._job_q, self._control_q, self._event_q),
            name="localflow-worker",
            daemon=True,
        )
        self._set_state("starting")
        self.adapters = None
        self.loaded_model = None
        self._proc.start()
        self.pid = self._proc.pid
        threading.Thread(
            target=self._reader, args=(gen, self._event_q), daemon=True, name=f"worker-reader-{gen}"
        ).start()
        log.info("spawned worker pid=%s", self.pid)

    def _terminate(self, graceful: bool = True) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.is_alive() and graceful:
            with contextlib.suppress(Exception):
                self._job_q.put(ShutdownRequest())
            proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        self._generation += 1  # stops the reader thread
        for q in (self._job_q, self._control_q, self._event_q):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self._proc = None

    def _reader(self, gen: int, q: Any) -> None:
        """Blocking reads from the worker's queue, handed to the event loop in order."""
        while gen == self._generation and not self._stopping:
            try:
                ev = q.get(timeout=0.25)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            if self._loop is not None:
                with contextlib.suppress(RuntimeError):
                    self._loop.call_soon_threadsafe(self._inbox.put_nowait, ev)

    # --- public API ------------------------------------------------------------------------------
    def wake(self) -> None:
        if self._loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._wakeup.set()
        else:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._wakeup.set)

    def info(self) -> WorkerInfo:
        return WorkerInfo(
            state=self.state,
            pid=self.pid if self._proc is not None else None,
            current_job_id=self.current_job_id,
            loaded_model=self.loaded_model,
            restarts=self.restarts,
            last_exit_code=self.last_exit_code,
            engine=self.engine,
            engine_note=self.engine_note,
            vram=self.vram,
            error=self.error,
        )

    def preview_path(self, job_id: str) -> Path | None:
        p = self.previews.get(job_id)
        return Path(p) if p else None

    def _cancel_running(self, job: Job) -> None:
        if self._control_q is not None and job.id == self.current_job_id:
            self._control_q.put(CancelRequest(job.id))
            self._cancel_requested = (job.id, time.monotonic())

    async def _hard_cancel(self, job_id: str) -> None:
        """The job didn't stop cooperatively (e.g. stuck in a long CUDA kernel or a model load): restart the
        worker. The model has to load again, but the queue keeps moving."""
        log.warning(
            "job %s ignored cancel for %.0f s; restarting the worker",
            job_id,
            self.settings.cancel_grace_s,
            extra={"job_id": job_id},
        )
        self._cancel_requested = None
        self._set_state("restarting")
        await asyncio.to_thread(self._terminate, graceful=False)
        with contextlib.suppress(AppError):
            if self.jobs.get(job_id).status not in TERMINAL:
                self.jobs.finish(job_id, "cancelled")
        if self.current_job_id == job_id:
            self.current_job_id = None
            shutil.rmtree(self.settings.tmp_dir / "jobs" / job_id, ignore_errors=True)
        if self._stopping:
            return
        self.restarts += 1
        self._spawn()

    # --- loops -----------------------------------------------------------------------------------
    async def _dispatch_loop(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=1.0)
            self._wakeup.clear()
            if self.state != "idle" or self.current_job_id is not None:
                continue
            loaded = None
            if self.loaded_model:
                loaded = (self.loaded_model.get("adapter_key"), self.loaded_model.get("model_entry_id"))
            job = self.jobs.next_gpu_job(loaded, self.settings.batch_top_k)
            if job is None:
                continue
            try:
                req = self.generation.build_request(job)
            except AppError as e:
                self.jobs.finish(job.id, "failed", error=e.message)
                self._wakeup.set()
                continue
            self.jobs.mark_running(job.id)
            self.current_job_id = job.id
            self._set_state("busy")
            self._job_q.put(req)
            log.info(
                "dispatched %s job %s (model %s)", job.type, job.id, model_key(job), extra={"job_id": job.id}
            )

    async def _event_loop(self) -> None:
        while True:
            ev = await self._inbox.get()
            try:
                if isinstance(ev, WorkerEvent):
                    self._on_worker_event(ev)
                else:
                    await self._on_job_event(ev)
            except Exception:
                log.exception("error handling worker event %r", ev)

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            pending = self._cancel_requested
            if pending is not None:
                job_id, at = pending
                if job_id != self.current_job_id:
                    self._cancel_requested = None
                elif time.monotonic() - at > self.settings.cancel_grace_s and not self._stopping:
                    await self._hard_cancel(job_id)
                    continue
            proc = self._proc
            if proc is None or proc.is_alive() or self._stopping:
                continue
            code = proc.exitcode
            self.last_exit_code = code
            log.error("worker exited unexpectedly (exit code %s)", exit_code_text(code))
            self._set_state("restarting")
            await asyncio.to_thread(self._terminate)
            if self.current_job_id:
                with contextlib.suppress(AppError):
                    self.jobs.finish(
                        self.current_job_id,
                        "failed",
                        error=f"The GPU worker crashed (exit code {exit_code_text(code)}). "
                        "It was restarted; retry the job.",
                    )
                self.current_job_id = None
            delay = _RESTART_BACKOFF_S[min(self._consecutive_crashes, len(_RESTART_BACKOFF_S) - 1)]
            self._consecutive_crashes += 1
            await asyncio.sleep(delay)
            if self._stopping:
                return
            self.restarts += 1
            self._spawn()

    # --- event handlers --------------------------------------------------------------------------
    def _on_worker_event(self, ev: WorkerEvent) -> None:
        if ev.kind == "ready":
            self.adapters = list(ev.data.get("adapters", []))
            self.pid = ev.data.get("pid", self.pid)
            self.error = ev.data.get("error")
            self.engine = ev.data.get("engine") or self.settings.engine
            self.engine_note = ev.data.get("engine_note")
            baseline = ev.data.get("vram_baseline")
            self.vram = {"baseline": baseline, **(baseline or {})} if baseline else None
            self._set_state("idle")
            self.wake()
        elif ev.kind == "model_loaded":
            self.loaded_model = {k: v for k, v in ev.data.items() if k != "vram"}
            self._update_vram(ev.data.get("vram"))
            self._publish_status()
        elif ev.kind == "model_unloaded":
            self.loaded_model = None
            self._update_vram(ev.data.get("vram"))
            self._publish_status()

    def _update_vram(self, snapshot: dict[str, Any] | None) -> None:
        if snapshot:
            baseline = (self.vram or {}).get("baseline")
            self.vram = {**snapshot, "baseline": baseline}

    async def _on_job_event(self, ev: JobEvent) -> None:
        if ev.job_id != self.current_job_id:
            log.warning("ignoring %s event for non-current job %s", ev.kind, ev.job_id)
            return
        if ev.kind == "started":
            self._consecutive_crashes = 0
            self.jobs.progress(ev.job_id, 0.0, "Running")
        elif ev.kind == "progress":
            self.jobs.progress(ev.job_id, ev.fraction or 0.0, ev.message)
        elif ev.kind == "preview" and ev.preview_path:
            self.previews[ev.job_id] = ev.preview_path
            n = len(self.previews)
            self.bus.publish("job.preview", job_id=ev.job_id, url=f"/api/jobs/{ev.job_id}/preview?n={n}")
        elif ev.kind in ("completed", "failed", "cancelled"):
            await self._finish(ev)

    async def _finish(self, ev: JobEvent) -> None:
        job_id = ev.job_id
        try:
            job = self.jobs.get(job_id)
            if job.status in TERMINAL:
                return
            if ev.kind == "completed":
                try:
                    assets = await asyncio.to_thread(
                        self.assets.ingest_job_outputs, job, ev.outputs, ev.result
                    )
                except Exception as e:
                    log.exception("ingesting outputs failed", extra={"job_id": job_id})
                    self.jobs.finish(job_id, "failed", error=f"Saving outputs failed: {e}")
                else:
                    result = {**ev.result, "asset_ids": [a.id for a in assets]}
                    if self.on_assets is not None:
                        try:
                            await asyncio.to_thread(self.on_assets, job, assets)
                        except Exception:
                            log.exception("linking outputs failed", extra={"job_id": job_id})
                    self.jobs.finish(job_id, "completed", result=result)
                    for a in assets:
                        self.bus.publish("asset.created", asset=AssetOut.of(a).model_dump(mode="json"))
            elif ev.kind == "failed":
                self.jobs.finish(job_id, "failed", error=ev.error)
            else:
                self.jobs.finish(job_id, "cancelled")
        finally:
            self.current_job_id = None
            shutil.rmtree(self.settings.tmp_dir / "jobs" / job_id, ignore_errors=True)
            self._cleanup_previews(job_id)
            if self.state == "busy":
                self._set_state("idle")
            self.wake()

    def _cleanup_previews(self, job_id: str) -> None:
        # Keep the last preview briefly for the UI; delete older numbered files.
        last = self.previews.get(job_id)
        for p in (self.settings.tmp_dir / "previews").glob(f"{job_id}.*.jpg"):
            if str(p) != last:
                with contextlib.suppress(OSError):
                    p.unlink()

    def _set_state(self, state: str) -> None:
        self.state = state
        self._publish_status()

    def _publish_status(self) -> None:
        self.bus.publish("worker.status", worker=self.info().model_dump(mode="json"))
