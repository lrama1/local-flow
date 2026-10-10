"""WebSocket `/ws/events` (Section 14.3): JSON events, heartbeat every 15 s."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from localflow.services.container import Services

router = APIRouter()

HEARTBEAT_S = 15.0


@router.websocket("/ws/events")
async def events(ws: WebSocket) -> None:
    svc: Services = ws.app.state.services
    await ws.accept()
    q = svc.bus.subscribe()
    try:
        await ws.send_json(
            {
                "type": "worker.status",
                "ts": datetime.now(UTC).isoformat(),
                "worker": svc.supervisor.info().model_dump(mode="json"),
            }
        )

        async def _drain_client() -> None:
            while True:
                await ws.receive_text()  # clients don't send anything meaningful; detects disconnects

        reader = asyncio.create_task(_drain_client())
        try:
            while True:
                getter = asyncio.ensure_future(q.get())
                done, _ = await asyncio.wait(
                    {getter, reader}, timeout=HEARTBEAT_S, return_when=asyncio.FIRST_COMPLETED
                )
                if reader in done:
                    getter.cancel()
                    break
                if getter in done:
                    event = getter.result()
                else:
                    getter.cancel()
                    event = {"type": "heartbeat", "ts": datetime.now(UTC).isoformat()}
                await ws.send_json(event)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await reader
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        svc.bus.unsubscribe(q)
