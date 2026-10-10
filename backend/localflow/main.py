"""FastAPI app factory. The lifespan runs migrations and starts the GPU worker supervisor."""

from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from localflow import __version__
from localflow.config import Settings, load_settings
from localflow.db import BACKEND_DIR, run_migrations
from localflow.errors import error_body, install_error_handlers, not_found
from localflow.logging_setup import setup_logging
from localflow.routers import (
    assets,
    board,
    characters,
    events,
    generate,
    imagegen,
    jobs,
    projects,
    story,
    system,
    timeline,
    video,
    voices,
)
from localflow.routers import models as models_router
from localflow.routers import settings as settings_router
from localflow.security import SecurityMiddleware
from localflow.services.container import build_services
from localflow.services.llm.types import LLMError

log = logging.getLogger("localflow")


def create_app(settings: Settings | None = None, **service_hooks: Any) -> FastAPI:
    """`service_hooks` are passed to `build_services` (tests inject fakes, e.g. `llm_factory`)."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.logs_dir, settings.log_level)
        await asyncio.to_thread(run_migrations, settings.db_path)
        svc = build_services(settings, **service_hooks)
        interrupted = svc.jobs.fail_interrupted()
        if interrupted:
            log.warning("marked %d interrupted job(s) as failed", interrupted)
        await asyncio.to_thread(svc.registry.sync_catalog)
        svc.registry.reset_interrupted()
        await asyncio.to_thread(svc.characters.purge_stale_drafts)
        app.state.services = svc
        if settings.ffmpeg is None:
            log.warning("ffmpeg not found; video/audio features will fail until it is installed")
        await svc.downloads.start()
        await svc.exports.start()
        if settings.start_worker:
            await svc.supervisor.start()
        log.info(
            "LocalFlow %s ready on http://%s:%s (data dir %s)",
            __version__,
            settings.host,
            settings.port,
            settings.data_dir,
        )
        try:
            yield
        finally:
            await svc.downloads.stop()
            await svc.exports.stop()
            if settings.start_worker:
                await svc.supervisor.stop()
            svc.db.dispose()

    app = FastAPI(title="LocalFlow", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    install_error_handlers(app)

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message, exc.details), status_code=exc.status)

    for r in (
        system.router,
        projects.router,
        assets.router,
        jobs.router,
        generate.router,
        settings_router.router,
        story.router,
        board.router,
        models_router.router,  # after `system`, which owns /models/adapters
        imagegen.router,
        video.router,
        timeline.router,
        voices.router,
        characters.router,
        events.router,
    ):
        app.include_router(r)

    if settings.dev_mode:
        origins = {settings.dev_origin, settings.dev_origin.replace("127.0.0.1", "localhost")}
        app.add_middleware(
            CORSMiddleware,
            allow_origins=sorted(origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(SecurityMiddleware, settings=settings)

    dist = settings.frontend_dist or (BACKEND_DIR.parent / "frontend" / "dist")
    if (dist / "index.html").is_file():
        _mount_frontend(app, dist.resolve(), settings.api_token)
    return app


def _mount_frontend(app: FastAPI, dist: Path, token: str) -> None:
    """Production mode: serve the built SPA from one origin, injecting the per-launch token."""
    index = (dist / "index.html").read_text(encoding="utf-8")
    meta = f'<meta name="localflow-token" content="{html.escape(token)}">'
    index_html = index.replace("</head>", f"  {meta}\n  </head>", 1)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> Response:
        if full_path.startswith(("api/", "ws/")):
            raise not_found("Route", "/" + full_path)
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        return HTMLResponse(index_html, headers={"Cache-Control": "no-store"})
