"""Host-header check + per-launch token on all API/WebSocket calls (Section 18).

Browsers can't set headers on <video>/<img> or WebSocket requests, so those may pass `?token=`."""

from __future__ import annotations

import secrets
from urllib.parse import parse_qs

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from localflow.config import Settings
from localflow.errors import error_body

TOKEN_HEADER = "x-localflow-token"
_PUBLIC_API = {"/api/health"}
_QUERY_TOKEN_GET_PREFIXES = ("/api/files/", "/api/thumbs/")


def _host_only(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):
        return host[: host.find("]") + 1]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _query_token_allowed(scope: Scope) -> bool:
    if scope["type"] == "websocket":
        return True
    path: str = scope["path"]
    if scope.get("method") != "GET":
        return False
    return (
        path.startswith(_QUERY_TOKEN_GET_PREFIXES)
        or (path.startswith("/api/jobs/") and path.endswith("/preview"))
        # <img> sources: Cast library images and character template thumbnails (Section 11.7)
        or (path.startswith("/api/library/characters/") and "/images/" in path)
        or (path.startswith("/api/character-templates/") and path.endswith("/thumbnail"))
    )


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.token = settings.api_token
        self.allowed_hosts = {h.lower() for h in settings.allowed_hosts}
        self.allowed_origins = {f"http://{h}:{settings.port}" for h in ("127.0.0.1", "localhost")}
        if settings.dev_mode:
            self.allowed_origins.add(settings.dev_origin.rstrip("/"))
            self.allowed_origins.add(settings.dev_origin.rstrip("/").replace("127.0.0.1", "localhost"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if _host_only(headers.get("host", "")) not in self.allowed_hosts:
            await self._reject(scope, receive, send, 400, "invalid_host", "Host header not allowed")
            return

        path: str = scope["path"]
        needs_token = (path.startswith("/api/") and path not in _PUBLIC_API) or path.startswith("/ws")
        if needs_token:
            token = headers.get(TOKEN_HEADER)
            if token is None and _query_token_allowed(scope):
                qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
                token = (qs.get("token") or [None])[0]
            if not token or not secrets.compare_digest(token, self.token):
                await self._reject(
                    scope, receive, send, 401, "unauthorized", "Missing or invalid LocalFlow token"
                )
                return

        if scope["type"] == "websocket":
            origin = headers.get("origin")
            if origin and not self._origin_ok(origin):
                await self._reject(scope, receive, send, 403, "forbidden_origin", "Origin not allowed")
                return

        await self.app(scope, receive, send)

    def _origin_ok(self, origin: str) -> bool:
        return origin.rstrip("/").lower() in self.allowed_origins

    @staticmethod
    async def _reject(
        scope: Scope, receive: Receive, send: Send, status: int, code: str, message: str
    ) -> None:
        if scope["type"] == "websocket":
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 4000 + status})
            return
        await JSONResponse(error_body(code, message), status_code=status)(scope, receive, send)
