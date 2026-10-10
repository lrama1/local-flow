"""`localflow` command: `serve` (default) and `openapi` (prints the OpenAPI schema for typegen)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    # Settings are read from env inside the app factory (also in reloader subprocesses).
    if args.dev:
        os.environ["LOCALFLOW_DEV_MODE"] = "true"
    if args.port:
        os.environ["LOCALFLOW_PORT"] = str(args.port)
    if args.no_worker:
        os.environ["LOCALFLOW_START_WORKER"] = "false"

    from localflow.config import load_settings

    settings = load_settings()
    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Refusing to bind to non-loopback host {settings.host!r} (Section 18).", file=sys.stderr)
        sys.exit(2)
    pkg_dir = Path(__file__).resolve().parent
    uvicorn.run(
        "localflow.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        reload_dirs=[str(pkg_dir)] if args.reload else None,
        log_level="info",
    )


def _openapi(args: argparse.Namespace) -> None:
    from localflow.config import Settings
    from localflow.main import create_app

    app = create_app(Settings(start_worker=False))  # no lifespan runs, so nothing touches the data dir
    schema = json.dumps(app.openapi(), indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(schema + "\n", encoding="utf-8")
    else:
        sys.stdout.write(schema + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="localflow")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Run the API server (default)")
    serve.add_argument("--dev", action="store_true", help="Dev mode (CORS for the Vite dev server)")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on backend code changes")
    serve.add_argument("--port", type=int)
    serve.add_argument("--no-worker", action="store_true", help="Don't start the GPU worker")
    serve.set_defaults(func=_serve)

    oa = sub.add_parser("openapi", help="Write the OpenAPI schema")
    oa.add_argument("-o", "--output")
    oa.set_defaults(func=_openapi)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["serve", *(argv or [])])
    args.func(args)


if __name__ == "__main__":
    main()
