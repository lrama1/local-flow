from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from localflow.worker.messages import JobEvent

if TYPE_CHECKING:
    from PIL import Image


class JobCancelled(Exception):
    pass


class JobContext:
    """Passed to adapters (Section 10.1). Progress is throttled to keep queue traffic low."""

    def __init__(
        self,
        *,
        job_id: str,
        seed: int,
        output_dir: Path,
        preview_dir: Path,
        vram_profile: str,
        cancel_event: threading.Event,
        event_queue: Any,
        settings: dict[str, Any] | None = None,
        progress_interval_s: float = 0.2,
    ) -> None:
        self.job_id = job_id
        self.seed = seed
        self.output_dir = output_dir
        self.preview_dir = preview_dir
        self.vram_profile = vram_profile
        self.cancel_event = cancel_event
        self.settings = settings or {}
        self.logger = logging.LoggerAdapter(logging.getLogger("localflow.worker.job"), {"job_id": job_id})
        self._events = event_queue
        self._interval = progress_interval_s
        self._last_progress = 0.0
        self._preview_n = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

    def report_progress(self, fraction: float, message: str | None = None, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and fraction < 1.0 and now - self._last_progress < self._interval:
            return
        self._last_progress = now
        self._events.put(
            JobEvent(self.job_id, "progress", fraction=max(0.0, min(1.0, fraction)), message=message)
        )

    def report_preview(self, image: Image.Image) -> None:
        # Numbered files instead of overwrite-in-place: on Windows a file being served can't be replaced.
        self._preview_n += 1
        path = self.preview_dir / f"{self.job_id}.{self._preview_n}.jpg"
        img = image.convert("RGB")
        img.thumbnail((512, 512))
        img.save(path, "JPEG", quality=80)
        self._events.put(JobEvent(self.job_id, "preview", preview_path=str(path)))
        if self._preview_n > 3:
            with contextlib.suppress(OSError):
                (self.preview_dir / f"{self.job_id}.{self._preview_n - 3}.jpg").unlink(missing_ok=True)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()
