"""Civitai REST client (Section 9.4): search, metadata, and resumable downloads.

Behavior verified against the live API (2026-10): downloads 307-redirect to a signed CDN URL that supports
Range; early-access versions return 403 JSON; a login-required version returns 401. The API key is sent as a
Bearer header to civitai.com only, never to the CDN, and no URL with a signature or token is persisted."""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx2

from localflow.errors import AppError
from localflow.services.registry.hf import HubAccessError

log = logging.getLogger("localflow.registry.civitai")

API_BASE = "https://civitai.com/api/v1"
_CHUNK = 1 << 20
_MAX_REDIRECTS = 5

ProgressFn = Callable[[int, int], None]


def _is_civitai(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return host == "civitai.com" or host.endswith(".civitai.com")


def _auth(url: str, token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token and _is_civitai(url) else {}


def _json_message(r: httpx2.Response) -> str | None:
    with contextlib.suppress(Exception):
        data = r.json()
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error")
            return str(msg) if msg else None
    return None


def access_error(status: int, has_token: bool, message: str | None) -> HubAccessError:
    """Maps a Civitai HTTP failure to an access state (Section 9.4)."""
    if status == 401:
        if not has_token:
            return HubAccessError(
                "needs_token",
                "This Civitai model requires you to be logged in to download it. Add a Civitai API key "
                "in Settings.",
            )
        return HubAccessError("invalid_token", "Civitai rejected your API key. Replace it in Settings.")
    if status == 403:
        detail = f" ({message})" if message else ""
        return HubAccessError(
            "forbidden",
            f"Civitai refused the download{detail}. Early-access or paid versions need to be unlocked on "
            "civitai.com with your account.",
        )
    if status == 404:
        return HubAccessError("not_found", "Civitai couldn't find this file.")
    raise AppError(502, "hub_error", f"Civitai returned an unexpected response ({status}).")


class CivitaiClient:
    def __init__(self, transport: httpx2.AsyncBaseTransport | None = None, base_url: str = API_BASE) -> None:
        self._transport = transport
        self.base_url = base_url.rstrip("/")

    def _client(self, timeout: float = 30.0) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=self._transport,
            timeout=httpx2.Timeout(timeout, read=60.0),
            follow_redirects=False,
            headers={"User-Agent": "LocalFlow"},
        )

    async def _get_json(self, path: str, params: Any, token: str | None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            async with self._client() as c:
                r = await c.get(url, params=params, headers=_auth(url, token))
        except httpx2.RequestError as e:
            raise AppError(502, "hub_unreachable", f"Couldn't reach Civitai ({type(e).__name__}).") from e
        if r.status_code == 404:
            raise AppError(404, "not_found", "Not found on Civitai.")
        if r.status_code in (401, 403):
            raise access_error(r.status_code, bool(token), _json_message(r))
        if r.status_code >= 400:
            raise AppError(
                502, "hub_error", f"Civitai error {r.status_code}: {_json_message(r) or ''}".strip()
            )
        return r.json()

    async def search(self, params: list[tuple[str, str]], token: str | None) -> dict[str, Any]:
        data = await self._get_json("/models", params, token)
        return data if isinstance(data, dict) else {"items": []}

    async def model(self, model_id: int, token: str | None) -> dict[str, Any]:
        data = await self._get_json(f"/models/{model_id}", None, token)
        return data if isinstance(data, dict) else {}

    async def version(self, version_id: int, token: str | None) -> dict[str, Any]:
        data = await self._get_json(f"/model-versions/{version_id}", None, token)
        return data if isinstance(data, dict) else {}

    async def probe(self, url: str, token: str | None) -> str:
        """Cheap access check for a download URL: 'ok' or raises HubAccessError. Doesn't download anything."""
        try:
            async with self._client() as c:
                r = await c.get(url, headers=_auth(url, token))
                status = r.status_code
                msg = _json_message(r) if status >= 400 else None
        except httpx2.RequestError as e:
            raise AppError(502, "hub_unreachable", f"Couldn't reach Civitai ({type(e).__name__}).") from e
        if status in (301, 302, 303, 307, 308, 200, 206):
            return "ok"
        raise access_error(status, bool(token), msg)

    async def download(
        self,
        url: str,
        dest: Path,
        *,
        token: str | None,
        expected_size: int | None,
        sha256: str | None,
        on_progress: ProgressFn,
    ) -> None:
        """Streams to `<dest>.part` with HTTP Range resume, verifies size/SHA-256, then renames into place."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        total = expected_size or 0
        try:
            async with self._client(timeout=60.0) as c:
                await self._download(c, url, part, token, total, on_progress)
        except httpx2.RequestError as e:
            raise AppError(
                502, "download_failed", f"Network error while downloading ({type(e).__name__})."
            ) from e
        size = part.stat().st_size
        if expected_size and abs(size - expected_size) > max(1024, expected_size // 1000):
            # Civitai reports sizeKB as a float; allow rounding, but not a truncated file.
            part.unlink(missing_ok=True)
            raise AppError(
                502, "download_corrupt", f"Downloaded {size} bytes, expected about {expected_size}."
            )
        if sha256:
            digest = await _sha256_file(part)
            if digest.lower() != sha256.lower():
                part.unlink(missing_ok=True)
                raise AppError(
                    502, "sha256_mismatch", "The downloaded file's SHA-256 doesn't match Civitai's."
                )
        part.replace(dest)

    async def _download(
        self,
        c: httpx2.AsyncClient,
        url: str,
        part: Path,
        token: str | None,
        total: int,
        on_progress: ProgressFn,
    ) -> None:
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            have = _file_size(part)
            if total and have >= total:
                return
            headers = _auth(current, token)
            if have:
                headers["Range"] = f"bytes={have}-"
            async with c.stream("GET", current, headers=headers) as r:
                if r.status_code in (301, 302, 303, 307, 308):
                    location = r.headers.get("location")
                    if not location:
                        raise AppError(502, "download_failed", "Civitai redirected without a location.")
                    current = urljoin(current, location)
                    continue
                if r.status_code == 416 and have:
                    return  # nothing left to fetch; size/SHA checks decide whether the file is good
                if r.status_code >= 400:
                    await r.aread()
                    raise access_error(r.status_code, bool(token), _json_message(r))
                ctype = r.headers.get("content-type", "").lower()
                if "text/html" in ctype:
                    # A login page instead of the file: never save it as a model (Section 9.4).
                    raise access_error(401, bool(token), None)
                if r.status_code == 200 and have:
                    have = 0  # server ignored Range; start over
                length = int(r.headers.get("content-length") or 0)
                if not total and length:
                    total = have + length
                mode = "ab" if have else "wb"
                done = have
                first = True
                with part.open(mode) as fh:
                    async for chunk in r.aiter_bytes(_CHUNK):
                        if first and not have and _looks_like_html(chunk):
                            fh.close()
                            _remove(part)
                            raise access_error(401, bool(token), None)
                        first = False
                        fh.write(chunk)
                        done += len(chunk)
                        on_progress(done, total or done)
                return
        raise AppError(502, "download_failed", "Too many redirects from Civitai.")


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)


def _looks_like_html(chunk: bytes) -> bool:
    head = chunk[:512].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html", b"<head"))


async def _sha256_file(path: Path) -> str:
    import asyncio

    def run() -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(_CHUNK * 4), b""):
                h.update(block)
        return h.hexdigest()

    return await asyncio.to_thread(run)
