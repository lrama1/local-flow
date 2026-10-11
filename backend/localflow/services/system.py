"""`GET /api/system/status` data. The API process never imports torch, so GPU info comes from nvidia-smi."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from importlib import metadata

import psutil

from localflow import __version__
from localflow.config import Settings
from localflow.schemas.core import GpuInfo, SystemStatus, WorkerInfo

_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: float, fn):  # type: ignore[no-untyped-def]
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


def _gpu() -> GpuInfo | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = (
            subprocess.run(
                [
                    exe,
                    "--query-gpu=name,memory.used,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, used, total, driver = (x.strip() for x in out[0].split(","))
    return GpuInfo(
        name=name, vram_used_mb=int(float(used)), vram_total_mb=int(float(total)), driver_version=driver
    )


def _ffmpeg_version(ffmpeg: str | None) -> str | None:
    if not ffmpeg:
        return None
    try:
        first = subprocess.run(
            [ffmpeg, "-version"], capture_output=True, text=True, timeout=5
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    parts = first[0].split() if first else []
    return parts[2] if len(parts) > 2 else None


def gpu_info() -> GpuInfo | None:
    return _cached("gpu", 2.0, _gpu)


def ffmpeg_version(ffmpeg: str | None) -> str | None:
    return _cached("ffmpeg", 300.0, lambda: _ffmpeg_version(ffmpeg))


def dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def system_status(settings: Settings, worker: WorkerInfo) -> SystemStatus:
    mem = psutil.virtual_memory()
    runtime = worker.runtime or {}
    try:
        disk_free = shutil.disk_usage(settings.data_dir).free // (1024 * 1024)
    except OSError:
        disk_free = None
    return SystemStatus(
        version=__version__,
        gpu=gpu_info(),
        ram_used_mb=int((mem.total - mem.available) / (1024 * 1024)),
        ram_total_mb=int(mem.total / (1024 * 1024)),
        disk_free_mb=disk_free,
        data_dir=str(settings.data_dir),
        worker=worker,
        ffmpeg_found=settings.ffmpeg is not None,
        ffprobe_found=settings.ffprobe is not None,
        ffmpeg_version=ffmpeg_version(settings.ffmpeg),
        python_version=platform.python_version(),
        torch_version=runtime.get("torch") or dist_version("torch"),
        cuda_version=runtime.get("cuda"),
        cuda_available=runtime.get("cuda_available"),
        diffusers_version=dist_version("diffusers"),
        vram_profile=settings.vram_profile,
        dev_mode=settings.dev_mode,
    )
