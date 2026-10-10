"""Cancel long LLM calls when the client goes away (UI cancel = AbortController, Section 16)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from typing import Any

from fastapi import Request

from localflow.errors import AppError


async def run_cancellable[T](request: Request, coro: Coroutine[Any, Any, T], poll_s: float = 0.5) -> T:
    task: asyncio.Task[T] = asyncio.ensure_future(coro)
    while True:
        done, _ = await asyncio.wait({task}, timeout=poll_s)
        if done:
            return task.result()
        if await request.is_disconnected():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise AppError(499, "client_cancelled", "The request was cancelled.")
