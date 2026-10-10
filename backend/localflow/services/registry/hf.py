"""Hugging Face Hub access: repo metadata, gated-access states (Section 9.3) and downloads.

The token is always passed explicitly (`token=` or `token=False`), never discovered implicitly
(Section 7.3)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from localflow.errors import AppError
from localflow.logging_setup import redact

log = logging.getLogger("localflow.registry.hf")

AccessState = Literal[
    "ok",
    "needs_token",
    "license_not_accepted",
    "pending_approval",
    "token_no_access",
    "not_found",
    "invalid_token",
    "forbidden",
]
# Worst first: the entry's access is the worst state across its parts.
ACCESS_SEVERITY: tuple[str, ...] = (
    "not_found",
    "needs_token",
    "invalid_token",
    "token_no_access",
    "forbidden",
    "license_not_accepted",
    "pending_approval",
    "ok",
)


def worst_access(states: list[str | None]) -> str | None:
    known = [s for s in states if s]
    if not known:
        return None
    return min(known, key=lambda s: ACCESS_SEVERITY.index(s) if s in ACCESS_SEVERITY else 0)


def model_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


class HubAccessError(AppError):
    """An access problem (Section 9.3/9.4) with a stable `access` state in `details`."""

    def __init__(self, access: str, message: str, details: dict[str, Any] | None = None) -> None:
        status = 404 if access == "not_found" else 403 if access in ("forbidden", "token_no_access") else 409
        super().__init__(status, access, message, {"access": access, **(details or {})})
        self.access = access


@dataclass
class RepoFile:
    path: str
    size: int | None = None
    sha256: str | None = None


@dataclass
class RepoInfo:
    repo_id: str
    sha: str | None
    gated: str | None  # None (public) | "auto" | "manual"
    private: bool
    files: list[RepoFile]
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    license: str | None = None
    license_name: str | None = None
    license_link: str | None = None
    base_models: list[str] = field(default_factory=list)

    @property
    def license_url(self) -> str:
        link = self.license_link
        if link and link.startswith(("http://", "https://")):
            return link
        if link:
            return f"{model_url(self.repo_id)}/blob/main/{link.lstrip('./')}"
        return model_url(self.repo_id)


@dataclass
class Access:
    state: str
    message: str | None = None


@dataclass
class FetchRequest:
    repo_id: str
    revision: str | None
    files: list[RepoFile]
    local_dir: Path
    token: str | None
    hf_home: Path


ProgressFn = Callable[[int, int], None]


class HfBackend(Protocol):
    def repo_info(self, repo_id: str, revision: str | None, token: str | None) -> RepoInfo: ...
    def access(self, info: RepoInfo, token: str | None, token_type: str | None) -> Access: ...
    async def fetch(self, req: FetchRequest, on_progress: ProgressFn) -> None: ...


def _card(card: Any, key: str) -> Any:
    if card is None:
        return None
    try:
        return card.get(key)
    except AttributeError:
        return getattr(card, key, None)


def _server_message(e: Exception) -> str:
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    return lines[-1] if lines else type(e).__name__


def classify_gated_error(
    repo_id: str, status: int | None, message: str, has_token: bool, token_type: str | None
) -> Access:
    """Maps a GatedRepoError to an access state (Section 9.3). Messages verified against the Hub 2026-10."""
    msg = message.lower()
    if not has_token:
        return Access("needs_token", f"{repo_id} is gated. Add a Hugging Face token in Settings.")
    if status == 401:
        return Access("needs_token", "Your Hugging Face token was rejected. Replace it in Settings.")
    if "fine-grained" in msg or "fine grained" in msg:
        return Access(
            "token_no_access",
            "Your fine-grained token can't read gated repos. Enable read access to public gated repos "
            "in the token's settings, then click Check again.",
        )
    if "awaiting" in msg or "pending" in msg:
        return Access(
            "pending_approval", f"Your access request for {repo_id} is awaiting approval by the repo owner."
        )
    if token_type == "fineGrained" and "authorized list" not in msg:
        return Access(
            "token_no_access",
            "Your fine-grained token may lack access to this gated repo. Check its permissions.",
        )
    return Access(
        "license_not_accepted",
        f"{repo_id} is gated and you haven't been granted access. Open the model page, accept the "
        "license (or request access), then click Check again.",
    )


class HubClient:
    """The real backend, built on huggingface_hub."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def _api(self) -> Any:
        from huggingface_hub import HfApi

        return HfApi()

    def repo_info(self, repo_id: str, revision: str | None, token: str | None) -> RepoInfo:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        try:
            mi = self._api().model_info(repo_id, revision=revision, files_metadata=True, token=token or False)
        except RevisionNotFoundError as e:
            raise AppError(404, "revision_not_found", f"Revision '{revision}' not found in {repo_id}") from e
        except GatedRepoError:
            # Metadata of gated repos is public; this only happens for odd setups. Report as gated/unknown.
            return RepoInfo(repo_id=repo_id, sha=None, gated="auto", private=False, files=[])
        except RepositoryNotFoundError as e:
            hint = "" if token else " If it's private, add a Hugging Face token in Settings."
            raise HubAccessError(
                "not_found", f"Repository '{repo_id}' was not found.{hint}", {"repo_id": repo_id}
            ) from e
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401 and token:
                raise HubAccessError(
                    "needs_token", "Your Hugging Face token was rejected. Replace it in Settings."
                ) from e
            raise AppError(502, "hub_error", f"Hugging Face error: {_server_message(e)}") from e
        except Exception as e:
            raise AppError(
                502, "hub_unreachable", f"Couldn't reach Hugging Face ({type(e).__name__})."
            ) from e
        card = mi.card_data
        base = _card(card, "base_model")
        files = [
            RepoFile(
                path=s.rfilename,
                size=s.size if s.size is not None else (s.lfs.size if s.lfs else None),
                sha256=s.lfs.sha256 if s.lfs else None,
            )
            for s in (mi.siblings or [])
        ]
        return RepoInfo(
            repo_id=mi.id or repo_id,
            sha=mi.sha,
            gated=mi.gated if isinstance(mi.gated, str) else None,
            private=bool(mi.private),
            files=files,
            tags=list(mi.tags or []),
            pipeline_tag=mi.pipeline_tag,
            library_name=mi.library_name,
            license=_card(card, "license"),
            license_name=_card(card, "license_name"),
            license_link=_card(card, "license_link"),
            base_models=[base] if isinstance(base, str) else list(base or []),
        )

    def access(self, info: RepoInfo, token: str | None, token_type: str | None) -> Access:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

        if not info.gated and not info.private:
            return Access("ok")
        if not token:
            what = "gated" if info.gated else "private"
            return Access("needs_token", f"{info.repo_id} is {what}. Add a Hugging Face token in Settings.")
        try:
            self._api().auth_check(info.repo_id, token=token)
            return Access("ok")
        except GatedRepoError as e:
            status = e.response.status_code if e.response is not None else None
            return classify_gated_error(info.repo_id, status, _server_message(e), True, token_type)
        except RepositoryNotFoundError:
            return Access("token_no_access", f"Your token can't see {info.repo_id}. Check its permissions.")
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401:
                return Access("needs_token", "Your Hugging Face token was rejected. Replace it in Settings.")
            if status == 403:
                return Access("token_no_access", f"Your token isn't allowed to read {info.repo_id}.")
            raise AppError(502, "hub_error", f"Hugging Face error: {_server_message(e)}") from e
        except Exception as e:
            raise AppError(
                502, "hub_unreachable", f"Couldn't reach Hugging Face ({type(e).__name__})."
            ) from e

    async def fetch(self, req: FetchRequest, on_progress: ProgressFn) -> None:
        req.local_dir.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        env |= {
            "HF_HOME": str(req.hf_home),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        log_path = req.local_dir / ".cache" / "localflow-fetch.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        payload = {
            "repo_id": req.repo_id,
            "revision": req.revision,
            "files": [f.path for f in req.files],
            "local_dir": str(req.local_dir),
            "token": req.token,
            "max_workers": self.max_workers,
        }
        total = sum(f.size or 0 for f in req.files)
        with log_path.open("wb") as errlog:
            proc = _spawn_fetcher(env, errlog, flags if sys.platform == "win32" else 0)
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
                proc.stdin.flush()
                while proc.poll() is None:
                    await asyncio.sleep(0.5)
                    on_progress(measure_progress(req.local_dir, req.files), total)
                out = proc.stdout.read().decode("utf-8", "replace").strip().splitlines()
            except BaseException:
                with contextlib.suppress(OSError):
                    proc.kill()
                await asyncio.to_thread(proc.wait)
                raise
            finally:
                with contextlib.suppress(OSError):
                    proc.stdin.close()
        result: dict[str, Any] = {}
        with contextlib.suppress(ValueError, IndexError):
            result = json.loads(out[-1])
        if not result.get("ok"):
            tail = redact(log_path.read_text(encoding="utf-8", errors="replace")[-2000:])
            if tail.strip():
                log.warning("hf fetch of %s failed; stderr tail:\n%s", req.repo_id, tail)
            raise _fetch_error(req.repo_id, result, proc.returncode)
        on_progress(measure_progress(req.local_dir, req.files), total)
        missing = [f.path for f in req.files if not (req.local_dir / f.path).is_file()]
        if missing:
            raise AppError(502, "download_incomplete", f"Download finished but {missing[0]} is missing.")
        wrong = [
            f.path
            for f in req.files
            if f.size is not None and (req.local_dir / f.path).stat().st_size != f.size
        ]
        if wrong:
            raise AppError(502, "download_corrupt", f"{wrong[0]} has the wrong size; delete and reinstall.")


def _spawn_fetcher(env: dict[str, str], errlog: Any, creationflags: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "localflow.services.registry.hf_fetch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=errlog,
        env=env,
        creationflags=creationflags,
    )


def _fetch_error(repo_id: str, result: dict[str, Any], code: int | None) -> AppError:
    kind, status, message = result.get("kind"), result.get("status"), str(result.get("message") or "")
    if kind == "gated":
        a = classify_gated_error(repo_id, status, message, True, None)
        return HubAccessError(
            a.state, a.message or message, {"repo_id": repo_id, "model_url": model_url(repo_id)}
        )
    if kind == "not_found":
        return HubAccessError("not_found", f"Repository '{repo_id}' was not found.", {"repo_id": repo_id})
    if kind == "http" and status == 401:
        return HubAccessError("needs_token", "Hugging Face rejected the request; check your token.")
    if kind == "os":
        return AppError(507 if "space" in message.lower() else 500, "download_failed", redact(message))
    if not result:
        return AppError(500, "download_failed", f"The download process exited unexpectedly (code {code}).")
    return AppError(502, "download_failed", f"Download failed: {redact(message)}")


def measure_progress(local_dir: Path, files: list[RepoFile]) -> int:
    """Bytes on disk for the requested files: finished files plus huggingface_hub's `.incomplete` parts."""
    done = 0
    for f in files:
        p = local_dir / f.path
        with contextlib.suppress(OSError):
            if p.is_file():
                size = p.stat().st_size
                done += min(size, f.size) if f.size else size
    dl = local_dir / ".cache" / "huggingface" / "download"
    if dl.is_dir():
        for p in dl.rglob("*.incomplete"):
            with contextlib.suppress(OSError):
                done += p.stat().st_size
    total = sum(f.size or 0 for f in files)
    return min(done, total) if total else done
