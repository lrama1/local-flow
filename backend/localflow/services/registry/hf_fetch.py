"""Child process that downloads Hugging Face files (run as `python -m localflow.services.registry.hf_fetch`).

Reads one JSON line from stdin: {repo_id, revision, files, local_dir, token}. The token travels over stdin
only (never argv or env). Exits when the parent closes stdin, so a dead API process never leaves an
orphaned download. Cancelling = killing this process; huggingface_hub resumes from `.incomplete` files.
Writes one JSON result line to stdout."""

from __future__ import annotations

import json
import os
import sys
import threading


def _watch_parent() -> None:
    sys.stdin.read()  # returns at EOF, i.e. when the parent exits or closes the pipe
    os._exit(3)


def main() -> int:
    req = json.loads(sys.stdin.readline())
    threading.Thread(target=_watch_parent, daemon=True).start()
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    result: dict[str, object]
    try:
        snapshot_download(
            repo_id=req["repo_id"],
            revision=req.get("revision") or None,
            local_dir=req["local_dir"],
            allow_patterns=req["files"],
            token=req.get("token") or False,
            max_workers=int(req.get("max_workers") or 4),
        )
        result = {"ok": True}
    except GatedRepoError as e:
        status = e.response.status_code if e.response is not None else None
        result = {"ok": False, "kind": "gated", "status": status, "message": _last_line(e)}
    except RepositoryNotFoundError as e:
        result = {"ok": False, "kind": "not_found", "message": _last_line(e)}
    except HfHubHTTPError as e:
        status = e.response.status_code if e.response is not None else None
        result = {"ok": False, "kind": "http", "status": status, "message": _last_line(e)}
    except OSError as e:
        result = {"ok": False, "kind": "os", "message": f"{type(e).__name__}: {e}"}
    except Exception as e:  # network errors surface as various httpx exception types
        result = {"ok": False, "kind": "error", "message": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0 if result["ok"] else 1


def _last_line(e: Exception) -> str:
    lines = [ln for ln in str(e).splitlines() if ln.strip()]
    return lines[-1] if lines else type(e).__name__


if __name__ == "__main__":
    sys.exit(main())
