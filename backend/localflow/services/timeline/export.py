"""Runs `export` jobs in the API process: one ffmpeg at a time, with progress and cancel (Sections 14.1,
15.3). The result is a video asset with `source="export"` under `projects/<id>/exports/`."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from pathlib import Path

from localflow.config import Settings
from localflow.errors import AppError, bad_request
from localflow.logging_setup import redact
from localflow.models import Asset, Job
from localflow.schemas.core import AssetOut
from localflow.schemas.timeline import ExportBody, TimelineData
from localflow.services.assets import AssetService
from localflow.services.events import EventBus
from localflow.services.jobs.service import JobService
from localflow.services.media.ffmpeg import FFmpegCancelled, FFmpegError, probe, run_ffmpeg
from localflow.services.timeline.render import Source, build_export
from localflow.services.timeline.service import TimelineService

log = logging.getLogger("localflow.export")

_ENCODE_END = 0.97  # the rest is probing, thumbnailing and the asset row


class ExportRunner:
    def __init__(
        self,
        settings: Settings,
        jobs: JobService,
        assets: AssetService,
        timeline: TimelineService,
        bus: EventBus,
    ) -> None:
        self.settings = settings
        self.jobs = jobs
        self.assets = assets
        self.timeline = timeline
        self.bus = bus
        self._task: tuple[str, asyncio.Task[None], threading.Event] | None = None
        self._wakeup = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main: asyncio.Task[None] | None = None
        self._shutting_down = False
        jobs.add_listener(on_change=self.wake, on_cancel=self.cancel)

    # --- submitting ----------------------------------------------------------------------------------
    def submit(self, project_id: str, body: ExportBody) -> Job:
        """Snapshots the timeline into the job, so later edits don't change a queued export."""
        tl = self.timeline.load(project_id)
        if not tl.video.clips:
            raise bad_request(
                "empty_timeline", "The timeline is empty. Auto-assemble it from the shots first."
            )
        if not self.settings.ffmpeg:
            raise AppError(409, "ffmpeg_missing", "ffmpeg wasn't found. Install it or set ffmpeg_path.")
        return self.jobs.create(
            type_="export",
            params={**body.model_dump(), "timeline": tl.model_dump(mode="json")},
            project_id=project_id,
            target_ref={"kind": "timeline", "project_id": project_id},
        )

    # --- runner loop (like DownloadRunner) --------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._main = asyncio.create_task(self._run_loop(), name="exports")
        self.wake()

    async def stop(self) -> None:
        self._shutting_down = True
        if self._main:
            self._main.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main
        if self._task is not None:
            _, task, event = self._task
            event.set()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def wake(self) -> None:
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._wakeup.set)

    def cancel(self, job: Job) -> None:
        if self._task is not None and self._task[0] == job.id:
            self._task[2].set()  # ffmpeg is killed; the job ends as cancelled

    async def _run_loop(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=1.0)
            self._wakeup.clear()
            if self._task is not None:
                continue
            queued = self.jobs.queued_jobs(("export",), limit=1)
            if queued:
                job, event = queued[0], threading.Event()
                self._task = (
                    job.id,
                    asyncio.create_task(self._run(job, event), name=f"export-{job.id}"),
                    event,
                )

    async def _run(self, job: Job, event: threading.Event) -> None:
        try:
            self.jobs.mark_running(job.id)
            asset = await asyncio.to_thread(self._export, job, event)
            self.jobs.finish(
                job.id,
                "completed",
                result={
                    "asset_ids": [asset.id],
                    "duration_s": asset.duration_s,
                    "width": asset.width,
                    "height": asset.height,
                },
            )
            self.bus.publish("asset.created", asset=AssetOut.of(asset).model_dump(mode="json"))
            log.info("export %s finished (%s)", job.id, asset.id, extra={"job_id": job.id})
        except FFmpegCancelled:
            if self._shutting_down:
                self.jobs.finish(job.id, "failed", error="Interrupted (LocalFlow was shut down)")
            else:
                self.jobs.finish(job.id, "cancelled")
        except AppError as e:
            self.jobs.finish(job.id, "failed", error=e.message, result={"code": e.code})
        except FFmpegError as e:
            log.error("export %s failed: %s", job.id, e, extra={"job_id": job.id})
            self.jobs.finish(job.id, "failed", error=redact(str(e))[-1500:])
        except Exception as e:
            log.exception("export %s crashed", job.id, extra={"job_id": job.id})
            self.jobs.finish(job.id, "failed", error=redact(f"Unexpected error: {type(e).__name__}: {e}"))
        finally:
            self._task = None
            self.wake()

    # --- the export itself (worker thread) --------------------------------------------------------
    def _export(self, job: Job, event: threading.Event) -> Asset:
        assert job.project_id is not None
        body = ExportBody.model_validate({k: job.params_json[k] for k in ("preset", "loudnorm")})
        tl = TimelineData.model_validate(job.params_json["timeline"])
        sources: dict[str, Source] = {}
        for c in (c for t in tl.tracks for c in t.clips):
            if c.asset_id in sources:
                continue
            try:
                a = self.assets.get(c.asset_id)
            except AppError as e:
                raise AppError(
                    409, "asset_missing", "A clip's file was deleted; open the timeline again."
                ) from e
            path = self.assets.file_path(a)
            if not path.is_file():
                raise AppError(409, "asset_missing", f"A clip's file is missing ({a.path}).")
            sources[a.id] = Source(path, a.duration_s, a.has_audio, a.width, a.height)

        tmp = Path(self.settings.tmp_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        dst = tmp / f"export-{job.id}.mp4"
        plan = build_export(tl, sources, body.preset, dst, loudnorm=body.loudnorm)
        self.jobs.progress(job.id, 0.0, f"Encoding {plan.width}x{plan.height}…")

        def progress(f: float) -> None:
            self.jobs.progress(job.id, _ENCODE_END * f, f"Encoding {plan.width}x{plan.height} · {f:.0%}")

        try:
            run_ffmpeg(
                self.settings.ffmpeg,
                plan.args,
                total_s=plan.duration_s,
                on_progress=progress,
                cancel_event=event,
                timeout_s=6 * 3600,
            )
            info = probe(self.settings.ffprobe, dst)
            if not info.get("has_video"):
                raise FFmpegError("The export has no video stream.")
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
        self.jobs.progress(job.id, 0.99, "Saving…")
        generation = {
            "job_id": job.id,
            "job_type": "export",
            "preset": body.preset,
            "loudnorm": body.loudnorm,
            "timeline_duration_s": plan.duration_s,
            "clips": len(tl.video.clips),
            "timeline": job.params_json["timeline"],
        }
        parents = sorted({c.asset_id for c in tl.video.clips})
        # Thumbnail 1 s in: exports usually open with a fade from black.
        return self.assets.ingest_file(
            job.project_id,
            dst,
            "video",
            source="export",
            generation=generation,
            parents=parents,
            subdir="exports",
            thumb_at_s=1.0,
        )
