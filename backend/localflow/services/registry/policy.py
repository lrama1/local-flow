"""File-type policy, URL hygiene and disk checks for model downloads (Sections 9.6 and 18)."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from localflow.errors import AppError

FileClass = Literal["weight", "pickle", "other_weight", "config", "code", "media", "doc", "other"]

SAFE_WEIGHT_EXT = {".safetensors", ".gguf"}
PICKLE_EXT = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
OTHER_WEIGHT_EXT = {".onnx", ".msgpack", ".h5", ".tflite", ".mlmodel", ".npz", ".npy", ".ot", ".zip", ".tar"}
CONFIG_EXT = {".json", ".txt", ".model", ".yaml", ".yml", ".tiktoken", ".vocab", ".spm", ".jinja", ".csv"}
CODE_EXT = {".py", ".pyc", ".sh", ".bat", ".ps1", ".exe", ".dll", ".so", ".dylib", ".js", ".ipynb"}
MEDIA_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".webm",
    ".mov",
    ".wav",
    ".mp3",
    ".flac",
    ".svg",
}
DOC_EXT = {".md", ".pdf", ".rst", ".html"}
_IGNORED_NAMES = {".gitattributes", ".gitignore"}
# Folders holding alternative runtimes that diffusers/transformers don't load.
_ALT_RUNTIME_DIRS = ("onnx/", "openvino/", "flax/", "coreml/", "tf/", "mlx/")
_FP16_VARIANT = re.compile(r"\.fp16\.[^./]+$")
_DISK_MARGIN_BYTES = 1 << 30  # keep at least 1 GiB free after a download


def classify(path: str) -> FileClass:
    pp = PurePosixPath(path)
    ext = pp.suffix.lower()
    if ext in SAFE_WEIGHT_EXT:
        return "weight"
    if ext in PICKLE_EXT:
        return "pickle"
    if ext in OTHER_WEIGHT_EXT:
        return "other_weight"
    if ext in CODE_EXT:
        return "code"
    if ext in MEDIA_EXT:
        return "media"
    if ext in DOC_EXT or pp.name.upper().startswith(("LICENSE", "NOTICE")):
        return "doc"
    if ext in CONFIG_EXT:
        return "config"
    return "other"


def blocked_reason(path: str, allow_pickle: bool) -> str | None:
    """Why a file may never be downloaded under the current settings, or None if it may."""
    cls = classify(path)
    if cls == "code":
        return "Code files are never downloaded."
    if cls == "pickle" and not allow_pickle:
        return "Pickle weights are blocked; set allow_pickle_weights = true in config.toml to allow them."
    if PurePosixPath(path).name in _IGNORED_NAMES:
        return "Repository metadata."
    return None


def default_selection(paths: Iterable[str], allow_pickle: bool) -> list[str]:
    """Default files: safe weights + configs/tokenizers, minus alternative runtimes and duplicates."""
    present = set(paths)
    has_index = "model_index.json" in present
    picked: list[str] = []
    for p in sorted(present):
        if blocked_reason(p, allow_pickle):
            continue
        cls = classify(p)
        if cls not in ("weight", "config", "pickle"):
            continue
        if p.startswith(_ALT_RUNTIME_DIRS):
            continue
        # A diffusers folder layout makes root-level single-file checkpoints redundant duplicates.
        if has_index and cls in ("weight", "pickle") and "/" not in p:
            continue
        # Prefer `x.fp16.safetensors` over `x.safetensors` when both exist.
        if cls == "weight" and not _FP16_VARIANT.search(p):
            stem, ext = p.rsplit(".", 1)
            if f"{stem}.fp16.{ext}" in present:
                continue
        # Skip pickle duplicates of safetensors weights (e.g. pytorch_model.bin next to model.safetensors).
        if cls == "pickle" and _has_safe_sibling(p, present):
            continue
        picked.append(p)
    return picked


def _has_safe_sibling(path: str, present: set[str]) -> bool:
    parent = str(PurePosixPath(path).parent)
    prefix = "" if parent == "." else parent + "/"
    return any(
        q.startswith(prefix) and "/" not in q[len(prefix) :] and classify(q) == "weight" for q in present
    )


def match_patterns(paths: Iterable[str], patterns: Iterable[str]) -> list[str]:
    """huggingface_hub `allow_patterns` semantics: fnmatch; a trailing '/' means the whole folder."""
    pats = [p + "*" if p.endswith("/") else p for p in patterns]
    return sorted({p for p in paths if any(fnmatch(p, pat) for pat in pats)})


def validate_selection(paths: Iterable[str], allow_pickle: bool) -> None:
    bad = [p for p in paths if classify(p) in ("code", "pickle") and blocked_reason(p, allow_pickle)]
    if bad:
        raise AppError(
            400,
            "file_blocked",
            f"'{bad[0]}' can't be downloaded: {blocked_reason(bad[0], allow_pickle)}",
            {"files": bad},
        )


# --- URLs --------------------------------------------------------------------------------------
_KEEP_QUERY = {"type", "format", "size", "fp"}


def safe_url(url: str) -> str:
    """Drops userinfo and every query parameter except Civitai's file selectors (no tokens/signatures)."""
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() in _KEEP_QUERY])
    netloc = parts.hostname or ""
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


# --- disk ----------------------------------------------------------------------------------------
def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def check_disk(path: Path, needed_bytes: int) -> None:
    free = free_bytes(path)
    if needed_bytes + _DISK_MARGIN_BYTES > free:
        raise AppError(
            507,
            "insufficient_disk",
            f"Not enough disk space: the download needs {format_bytes(needed_bytes)} "
            f"but only {format_bytes(free)} is free on the models drive.",
            {"needed_bytes": needed_bytes, "free_bytes": free},
        )


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total
