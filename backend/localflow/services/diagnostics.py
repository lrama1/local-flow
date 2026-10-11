"""Settings → Diagnostics (Section 19): health checks, storage use, log tails and a redacted support bundle.

Nothing here reads secrets: the bundle has versions, checks, config.toml and log tails, all passed through
`redact()` (logs are already redacted when written)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import shutil
import time
import zipfile
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from localflow import __version__
from localflow.errors import not_found
from localflow.logging_setup import redact
from localflow.models import new_id
from localflow.schemas.diagnostics import DiagCheck, DiagnosticsOut, FolderUsage, LogFile, LogTail
from localflow.services.system import dist_version, ffmpeg_version, gpu_info, system_status

if TYPE_CHECKING:
    from localflow.services.container import Services

LOW_DISK_MB = 20 * 1024
BUNDLE_LOG_LINES = 5000
_sizes: dict[str, tuple[float, list[FolderUsage]]] = {}
SIZES_TTL_S = 60.0


# --- checks ------------------------------------------------------------------------------------------
def write_test(data_dir: Path) -> DiagCheck:
    probe = data_dir / "tmp" / f"diag-{new_id()}.txt"
    payload = f"localflow write test {time.time()}"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(payload, encoding="utf-8")
        ok = probe.read_text(encoding="utf-8") == payload
    except OSError as e:
        return DiagCheck(
            key="data_dir", label="Data folder", status="error", detail=f"Can't write to {data_dir}: {e}"
        )
    finally:
        # Not just missing_ok: under a non-directory, Linux raises NotADirectoryError.
        with contextlib.suppress(OSError):
            probe.unlink()
    if not ok:
        return DiagCheck(
            key="data_dir", label="Data folder", status="error", detail="A test file read back wrong."
        )
    return DiagCheck(key="data_dir", label="Data folder", status="ok", detail=f"{data_dir} is writable")


def run_checks(svc: Services) -> list[DiagCheck]:
    st = svc.settings
    worker = svc.supervisor.info()
    checks = [write_test(st.data_dir)]

    try:
        free = shutil.disk_usage(st.data_dir).free // (1024 * 1024)
        low = free < LOW_DISK_MB
        checks.append(
            DiagCheck(
                key="disk",
                label="Disk space",
                status="warn" if low else "ok",
                detail=f"{free / 1024:.1f} GB free"
                + (" (models and videos need room; free some space or move the data folder)" if low else ""),
            )
        )
    except OSError as e:
        checks.append(DiagCheck(key="disk", label="Disk space", status="error", detail=str(e)))

    version = ffmpeg_version(st.ffmpeg)
    checks.append(
        DiagCheck(key="ffmpeg", label="ffmpeg", status="ok", detail=f"{version or 'found'} at {st.ffmpeg}")
        if st.ffmpeg
        else DiagCheck(
            key="ffmpeg",
            label="ffmpeg",
            status="error",
            detail="Not found. Install it (Windows: winget install Gyan.FFmpeg), then restart LocalFlow.",
        )
    )
    checks.append(
        DiagCheck(key="ffprobe", label="ffprobe", status="ok", detail=str(st.ffprobe))
        if st.ffprobe
        else DiagCheck(
            key="ffprobe", label="ffprobe", status="error", detail="Not found (it ships with ffmpeg)."
        )
    )

    gpu = gpu_info()
    checks.append(
        DiagCheck(
            key="gpu",
            label="NVIDIA GPU",
            status="ok",
            detail=f"{gpu.name} · {gpu.vram_total_mb / 1024:.0f} GB · driver {gpu.driver_version or '?'}",
        )
        if gpu
        else DiagCheck(
            key="gpu",
            label="NVIDIA GPU",
            status="warn",
            detail="nvidia-smi not found: no NVIDIA GPU or driver. Real generation needs one.",
        )
    )
    checks.append(_cuda_check(worker.engine, worker.runtime, worker.engine_note, worker.state))

    if worker.error or worker.state == "disabled":
        status, detail = "error", worker.error or "The worker is disabled after repeated crashes."
    elif worker.state in ("starting", "restarting", "stopped"):
        status, detail = "warn", f"The worker is {worker.state}."
    else:
        adapters = len(svc.supervisor.adapters or [])
        status, detail = "ok", f"{worker.state} · {worker.engine} engine · {adapters} adapters"
    checks.append(DiagCheck(key="worker", label="GPU worker", status=status, detail=detail))

    source = svc.secrets.stored_source
    checks.append(
        DiagCheck(
            key="keyring",
            label="Secret storage",
            status="ok" if source == "keyring" else "warn",
            detail=svc.secrets.backend_name(),
        )
    )
    providers = svc.llm.configured()
    if svc.llm.fake:
        llm_detail = "Fake LLM mode (fake_llm): canned demo responses, no API calls"
    else:
        llm_detail = ", ".join(providers) if providers else "No API key: the story tools are unavailable."
    checks.append(
        DiagCheck(
            key="llm",
            label="LLM providers",
            status="warn" if svc.llm.fake or not providers else "ok",
            detail=llm_detail,
        )
    )
    return checks


def _cuda_check(engine: str, runtime: dict[str, Any] | None, note: str | None, state: str) -> DiagCheck:
    label = "PyTorch + CUDA"
    if engine == "fake":
        detail = "The worker runs the FakeEngine (placeholder outputs)" + (f": {note}" if note else ".")
        return DiagCheck(key="cuda", label=label, status="warn" if note else "skip", detail=detail)
    if not runtime:
        return DiagCheck(key="cuda", label=label, status="warn", detail=f"Not reported yet (worker {state}).")
    if runtime.get("error"):
        return DiagCheck(key="cuda", label=label, status="error", detail=str(runtime["error"]))
    if runtime.get("cuda_available"):
        detail = f"torch {runtime.get('torch')} · CUDA {runtime.get('cuda')} · {runtime.get('device')}"
        return DiagCheck(key="cuda", label=label, status="ok", detail=detail)
    return DiagCheck(
        key="cuda",
        label=label,
        status="error",
        detail=f"torch {runtime.get('torch')} can't see a CUDA GPU (CUDA build "
        f"{runtime.get('cuda') or 'none'}). Check the NVIDIA driver, then reinstall the CUDA build of "
        "PyTorch (`uv sync` in backend/).",
    )


# --- storage -----------------------------------------------------------------------------------------
def _folder(path: Path) -> tuple[int, int]:
    """(bytes, files) under `path`, not following symlinks (the HF cache links snapshots to blobs)."""
    total = files = 0
    stack = [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    total += e.stat(follow_symlinks=False).st_size
                    files += 1
            except OSError:
                continue
    return total, files


def storage(svc: Services, refresh: bool = False) -> list[FolderUsage]:
    st = svc.settings
    key = str(st.data_dir)
    hit = _sizes.get(key)
    if hit and not refresh and time.monotonic() - hit[0] < SIZES_TTL_S:
        return hit[1]
    out = []
    for name, path in (
        ("models", st.models_dir),
        ("projects", st.projects_dir),
        ("library", st.library_dir),
        ("tmp", st.tmp_dir),
        ("logs", st.logs_dir),
    ):
        size, files = _folder(path)
        out.append(FolderUsage(key=name, path=str(path), size_mb=round(size / 2**20, 1), files=files))
    db = [p for p in (st.db_path, st.db_path.with_name(st.db_path.name + "-wal")) if p.is_file()]
    out.append(
        FolderUsage(
            key="database",
            path=str(st.db_path),
            size_mb=round(sum(p.stat().st_size for p in db) / 2**20, 1),
            files=len(db),
        )
    )
    _sizes[key] = (time.monotonic(), out)
    return out


# --- logs ---------------------------------------------------------------------------------------------
def log_files(logs_dir: Path) -> list[LogFile]:
    files = [p for p in logs_dir.glob("*.log*") if p.is_file()]
    files.sort(key=lambda p: (p.name.split(".log")[0], p.name))
    return [
        LogFile(
            name=p.name,
            size_kb=round(p.stat().st_size / 1024, 1),
            modified=datetime.fromtimestamp(p.stat().st_mtime, UTC),
        )
        for p in files
    ]


def _format(line: str) -> str:
    """JSON log records → `time level logger: message` (other lines pass through)."""
    try:
        r = json.loads(line)
    except ValueError:
        return redact(line.rstrip("\n"))
    if not isinstance(r, dict):
        return redact(line.rstrip("\n"))
    ctx = " ".join(f"{k}={r[k]}" for k in ("job_id", "project_id") if r.get(k))
    text = f"{str(r.get('ts', ''))[:19].replace('T', ' ')} {r.get('level', ''):<7} {r.get('logger', '')}: "
    text += str(r.get("msg", "")) + (f" [{ctx}]" if ctx else "")
    if r.get("exc"):
        text += "\n" + str(r["exc"])
    return redact(text)


def log_tail(logs_dir: Path, name: str, lines: int) -> LogTail:
    if name not in {f.name for f in log_files(logs_dir)}:  # only listed files: no path traversal
        raise not_found("Log file", name)
    tail: deque[str] = deque(maxlen=lines)
    total = 0
    with (logs_dir / name).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            tail.append(line)
    return LogTail(name=name, lines=[_format(x) for x in tail], total_lines=total)


# --- everything ---------------------------------------------------------------------------------------
def versions(svc: Services) -> dict[str, str | None]:
    runtime = svc.supervisor.info().runtime or {}
    return {
        "localflow": __version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": runtime.get("torch") or dist_version("torch"),
        "cuda": runtime.get("cuda"),
        "cudnn": str(runtime["cudnn"]) if runtime.get("cudnn") else None,
        "diffusers": dist_version("diffusers"),
        "transformers": dist_version("transformers"),
        "litellm": dist_version("litellm"),
        "ffmpeg": ffmpeg_version(svc.settings.ffmpeg),
        "engine": svc.supervisor.info().engine,
        "vram_profile": svc.settings.vram_profile,
    }


def diagnostics(svc: Services, refresh: bool = False) -> DiagnosticsOut:
    free: int | None
    total: int | None
    try:
        usage = shutil.disk_usage(svc.settings.data_dir)
        free, total = usage.free // 2**20, usage.total // 2**20
    except OSError:
        free = total = None
    return DiagnosticsOut(
        generated_at=datetime.now(UTC),
        checks=run_checks(svc),
        versions=versions(svc),
        storage=storage(svc, refresh),
        disk_free_mb=free,
        disk_total_mb=total,
        logs=log_files(svc.settings.logs_dir),
    )


def bundle(svc: Services) -> bytes:
    """A zip to attach to a bug report: diagnostics, system status, config.toml and recent logs."""
    st = svc.settings
    diag = diagnostics(svc)
    status = system_status(st, svc.supervisor.info())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("diagnostics.json", redact(diag.model_dump_json(indent=2)))
        z.writestr("system-status.json", redact(status.model_dump_json(indent=2)))
        config = st.data_dir / "config.toml"
        if config.is_file():
            z.writestr("config.toml", redact(config.read_text(encoding="utf-8", errors="replace")))
        for log in diag.logs:
            if log.name.endswith(".log"):  # current files; rotated ones are older
                tail = log_tail(st.logs_dir, log.name, BUNDLE_LOG_LINES)
                z.writestr(f"logs/{log.name}.txt", "\n".join(tail.lines) + "\n")
        z.writestr(
            "README.txt",
            "LocalFlow diagnostics bundle. API keys and tokens are redacted; no project media or database "
            "is included.\n",
        )
    return buf.getvalue()
