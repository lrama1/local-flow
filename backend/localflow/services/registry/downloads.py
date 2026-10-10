"""Runs `download_model` jobs in the API process with progress, cancel and resume (Sections 4.1, 9.6, 14)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque

from localflow.errors import AppError
from localflow.logging_setup import redact
from localflow.models import Job
from localflow.services.events import EventBus
from localflow.services.jobs.service import JobService
from localflow.services.registry.policy import format_bytes
from localflow.services.registry.service import RegistryService

log = logging.getLogger("localflow.downloads")


class _Reporter:
    """Throttles progress to ~2 updates/s; speed is averaged over a sliding window because some
    transports (hf_xet) write in bursts."""

    WINDOW_S = 6.0

    def __init__(self, jobs: JobService, bus: EventBus, job_id: str, entry_id: str) -> None:
        self.jobs, self.bus, self.job_id, self.entry_id = jobs, bus, job_id, entry_id
        self._last_t = 0.0
        self._samples: deque[tuple[float, int]] = deque()

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last_t < 0.5 and done < total:
            return
        self._last_t = now
        self._samples.append((now, done))
        while len(self._samples) > 2 and now - self._samples[0][0] > self.WINDOW_S:
            self._samples.popleft()
        t0, d0 = self._samples[0]
        speed = max(0.0, (done - d0) / (now - t0)) if now > t0 else 0.0
        frac = min(1.0, done / total) if total else 0.0
        msg = f"{format_bytes(done)} of {format_bytes(total)}" if total else format_bytes(done)
        if speed:
            msg += f" · {format_bytes(speed)}/s"
        self.jobs.progress(self.job_id, frac, msg)
        self.bus.publish(
            "download.progress",
            job_id=self.job_id,
            model_entry_id=self.entry_id,
            downloaded_bytes=done,
            total_bytes=total,
            bytes_per_s=round(speed),
        )


class DownloadRunner:
    def __init__(
        self, jobs: JobService, registry: RegistryService, bus: EventBus, max_concurrent: int = 2
    ) -> None:
        self.jobs = jobs
        self.registry = registry
        self.bus = bus
        self.max_concurrent = max(1, max_concurrent)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._wakeup = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main: asyncio.Task[None] | None = None
        self._shutting_down = False
        jobs.add_listener(on_change=self.wake, on_cancel=self.cancel)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._main = asyncio.create_task(self._run_loop(), name="downloads")
        self.wake()

    async def stop(self) -> None:
        self._shutting_down = True
        if self._main:
            self._main.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t

    def wake(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._wakeup.set()
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._wakeup.set)

    def cancel(self, job: Job) -> None:
        task = self._tasks.get(job.id)
        if task is None or self._loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            task.cancel()
        else:
            self._loop.call_soon_threadsafe(task.cancel)

    @property
    def active_job_ids(self) -> list[str]:
        return list(self._tasks)

    async def _run_loop(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=1.0)
            self._wakeup.clear()
            free = self.max_concurrent - len(self._tasks)
            if free <= 0:
                continue
            for job in self.jobs.queued_jobs(("download_model",), limit=free + len(self._tasks)):
                if job.id in self._tasks:
                    continue
                if len(self._tasks) >= self.max_concurrent:
                    break
                self._tasks[job.id] = asyncio.create_task(self._run(job), name=f"download-{job.id}")

    async def _run(self, job: Job) -> None:
        entry_id = str(job.params_json.get("model_entry_id") or "")
        try:
            self.jobs.mark_running(job.id)
            result = await self.registry.run_download(
                entry_id, job.id, _Reporter(self.jobs, self.bus, job.id, entry_id)
            )
            self.jobs.finish(job.id, "completed", result=result)
            log.info("download %s finished (%s)", job.id, entry_id, extra={"job_id": job.id})
        except asyncio.CancelledError:
            if self._shutting_down:
                self.registry.download_stopped(entry_id, interrupted=True)
                with contextlib.suppress(AppError):
                    self.jobs.finish(job.id, "failed", error="Interrupted (LocalFlow was shut down)")
                raise
            self.registry.download_stopped(entry_id)
            self.jobs.finish(job.id, "cancelled")
        except AppError as e:
            self.registry.download_failed(entry_id, e)
            self.jobs.finish(job.id, "failed", error=e.message, result={"code": e.code, "details": e.details})
        except Exception as e:
            log.exception("download %s crashed", job.id, extra={"job_id": job.id})
            err = AppError(500, "download_failed", redact(f"Unexpected error: {type(e).__name__}: {e}"))
            self.registry.download_failed(entry_id, err)
            self.jobs.finish(job.id, "failed", error=err.message, result={"code": err.code})
        finally:
            self._tasks.pop(job.id, None)
            self.wake()
