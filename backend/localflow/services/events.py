"""In-process pub/sub fanned out to WebSocket clients (Section 14.3).

`publish` is thread-safe: each subscriber queue is fed on the event loop that created it."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(eq=False)
class _Subscription:
    queue: asyncio.Queue[dict[str, Any]]
    loop: asyncio.AbstractEventLoop


class EventBus:
    def __init__(self, maxsize: int = 1000) -> None:
        self._subs: dict[int, _Subscription] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self._maxsize)
        with self._lock:
            self._subs[id(q)] = _Subscription(q, asyncio.get_running_loop())
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subs.pop(id(q), None)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def publish(self, type_: str, **data: Any) -> None:
        event = {"type": type_, "ts": datetime.now(UTC).isoformat(), **data}
        try:
            current: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        with self._lock:
            subs = list(self._subs.values())
        for sub in subs:
            if sub.loop is current:
                _put(sub.queue, event)
            else:
                with contextlib.suppress(RuntimeError):  # subscriber's loop already closed
                    sub.loop.call_soon_threadsafe(_put, sub.queue, event)


def _put(q: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    if q.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            q.get_nowait()  # drop the oldest event for slow clients
    q.put_nowait(event)
