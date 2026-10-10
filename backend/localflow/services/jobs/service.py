"""SQLite-backed job queue (Section 14)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from sqlmodel import col, func, select

from localflow.db import Database
from localflow.errors import AppError, conflict, not_found
from localflow.models import Job, utcnow
from localflow.schemas.core import JobOut
from localflow.services.events import EventBus
from localflow.worker.adapters.base import GPU_JOB_TYPES

log = logging.getLogger("localflow.jobs")

TERMINAL = ("completed", "failed", "cancelled")
API_JOB_TYPES = ("export", "download_model")
ALL_JOB_TYPES = (*GPU_JOB_TYPES, *API_JOB_TYPES)


def model_key(job: Job) -> tuple[str | None, str | None]:
    p = job.params_json
    return p.get("adapter_key"), (p.get("request") or {}).get("model_entry_id")


class JobService:
    def __init__(self, db: Database, bus: EventBus) -> None:
        self.db = db
        self.bus = bus
        self._change_listeners: list[Callable[[], None]] = []
        self._cancel_handlers: list[Callable[[Job], None]] = []
        self._last_persist: dict[str, float] = {}

    def add_listener(
        self, *, on_change: Callable[[], None] | None = None, on_cancel: Callable[[Job], None] | None = None
    ) -> None:
        """Runners subscribe here: `on_change` when the queue changes, `on_cancel` for running jobs."""
        if on_change:
            self._change_listeners.append(on_change)
        if on_cancel:
            self._cancel_handlers.append(on_cancel)

    def on_change(self) -> None:
        for fn in self._change_listeners:
            fn()

    def cancel_running(self, job: Job) -> None:
        for fn in self._cancel_handlers:
            fn(job)

    # --- queries ---------------------------------------------------------------------------------
    def get(self, job_id: str) -> Job:
        with self.db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise not_found("Job", job_id)
            return job

    def list_jobs(
        self,
        *,
        statuses: list[str] | None = None,
        project_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Job], int]:
        with self.db.session() as s:
            q = select(Job)
            cq = select(func.count()).select_from(Job)
            if statuses:
                q = q.where(col(Job.status).in_(statuses))
                cq = cq.where(col(Job.status).in_(statuses))
            if project_id:
                q = q.where(Job.project_id == project_id)
                cq = cq.where(Job.project_id == project_id)
            q = q.order_by(col(Job.created_at).desc(), col(Job.id).desc()).offset(offset).limit(limit)
            return list(s.exec(q).all()), int(s.exec(cq).one())

    def queued_gpu_jobs(self, limit: int) -> list[Job]:
        with self.db.session() as s:
            q = (
                select(Job)
                .where(Job.status == "queued", col(Job.type).in_(GPU_JOB_TYPES))
                .order_by(col(Job.priority).desc(), col(Job.id).asc())  # ULIDs sort by creation time
                .limit(limit)
            )
            return list(s.exec(q).all())

    def queued_jobs(self, types: tuple[str, ...], limit: int) -> list[Job]:
        with self.db.session() as s:
            q = (
                select(Job)
                .where(Job.status == "queued", col(Job.type).in_(types))
                .order_by(col(Job.priority).desc(), col(Job.id).asc())
                .limit(limit)
            )
            return list(s.exec(q).all())

    def next_gpu_job(self, loaded: tuple[str | None, str | None] | None, top_k: int) -> Job | None:
        """Priority desc + FIFO; prefer a job using the loaded model within the top K (Section 14.2)."""
        candidates = self.queued_gpu_jobs(max(1, top_k))
        if not candidates:
            return None
        if loaded is not None:
            for job in candidates:
                if model_key(job) == loaded:
                    return job
        return candidates[0]

    # --- mutations -------------------------------------------------------------------------------
    def create(
        self,
        *,
        type_: str,
        params: dict[str, Any],
        project_id: str | None,
        priority: int = 0,
        target_ref: dict[str, Any] | None = None,
    ) -> Job:
        if type_ not in ALL_JOB_TYPES:
            raise AppError(400, "invalid_job_type", f"Unknown job type '{type_}'")
        job = Job(
            type=type_,
            params_json=params,
            project_id=project_id,
            priority=priority,
            target_ref_json=target_ref,
        )
        with self.db.session() as s:
            s.add(job)
            s.commit()
            s.refresh(job)
        self.bus.publish("job.created", job=self._out(job))
        self.on_change()
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status in TERMINAL:
            raise conflict("job_not_cancellable", f"Job is already {job.status}")
        if job.status == "queued":
            return self._update(
                job_id, status="cancelled", finished_at=utcnow(), progress_message="Cancelled"
            )
        self.cancel_running(job)
        return self._update(job_id, progress_message="Cancelling…")

    def retry(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status not in ("failed", "cancelled"):
            raise conflict("job_not_retryable", "Only failed or cancelled jobs can be retried")
        return self.create(
            type_=job.type,
            params=job.params_json,
            project_id=job.project_id,
            priority=job.priority,
            target_ref=job.target_ref_json,
        )

    def reorder(self, job_ids: list[str]) -> list[Job]:
        """Listed queued jobs run in the given order, ahead of unlisted queued jobs."""
        with self.db.session() as s:
            jobs = {j.id: j for j in s.exec(select(Job).where(col(Job.id).in_(job_ids))).all()}
            missing = [i for i in job_ids if i not in jobs]
            if missing:
                raise not_found("Job", missing[0])
            not_queued = [i for i in job_ids if jobs[i].status != "queued"]
            if not_queued:
                raise conflict("job_not_queued", f"Job '{not_queued[0]}' is not queued")
            top = s.exec(select(func.max(Job.priority)).where(Job.status == "queued")).one() or 0
            n = len(job_ids)
            for i, jid in enumerate(job_ids):
                jobs[jid].priority = int(top) + n - i
                s.add(jobs[jid])
            s.commit()
            ordered = [jobs[i] for i in job_ids]
        for j in ordered:
            self.bus.publish("job.updated", job=self._out(j))
        self.on_change()
        return ordered

    def mark_running(self, job_id: str) -> Job:
        return self._update(
            job_id, status="running", started_at=utcnow(), progress=0.0, progress_message="Starting"
        )

    def progress(self, job_id: str, fraction: float, message: str | None) -> None:
        now = time.monotonic()
        if now - self._last_persist.get(job_id, 0.0) >= 1.0:
            self._last_persist[job_id] = now
            with self.db.session() as s:
                job = s.get(Job, job_id)
                if job is not None and job.status == "running":
                    job.progress, job.progress_message = fraction, message
                    s.add(job)
                    s.commit()
        self.bus.publish("job.progress", job_id=job_id, progress=fraction, message=message)

    def finish(
        self, job_id: str, status: str, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> Job:
        self._last_persist.pop(job_id, None)
        fields: dict[str, Any] = {"status": status, "finished_at": utcnow()}
        if status == "completed":
            fields |= {"progress": 1.0, "progress_message": "Done", "result_json": result or {}}
        elif status == "failed":
            fields |= {"error": error or "Unknown error", "progress_message": "Failed"}
            if result is not None:
                fields["result_json"] = result
        elif status == "cancelled":
            fields |= {"progress_message": "Cancelled"}
        job = self._update(job_id, publish=False, **fields)
        event = {"completed": "job.completed", "failed": "job.failed"}.get(status, "job.updated")
        self.bus.publish(event, job=self._out(job))
        return job

    def fail_interrupted(self) -> int:
        """On startup, jobs left `running` are marked failed ("interrupted")."""
        with self.db.session() as s:
            jobs = s.exec(select(Job).where(Job.status == "running")).all()
            for job in jobs:
                job.status, job.error, job.finished_at = (
                    "failed",
                    "Interrupted (LocalFlow was restarted)",
                    utcnow(),
                )
                job.progress_message = "Interrupted"
                s.add(job)
            s.commit()
            return len(jobs)

    # --- helpers ---------------------------------------------------------------------------------
    def _update(self, job_id: str, *, publish: bool = True, **fields: Any) -> Job:
        with self.db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise not_found("Job", job_id)
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = utcnow()
            s.add(job)
            s.commit()
            s.refresh(job)
        if publish:
            self.bus.publish("job.updated", job=self._out(job))
        return job

    @staticmethod
    def _out(job: Job) -> dict[str, Any]:
        return JobOut.of(job).model_dump(mode="json")
