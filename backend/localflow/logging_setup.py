"""Structured JSON logging to a rotating file plus human-readable console output, with secret redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)([?&]token=)[^&\s\"']+"),
    # Signed CDN URLs (e.g. Civitai → R2/S3 redirects): drop the whole query string.
    re.compile(
        r"(?i)(https?://[^\s\"'?]+\?)[^\s\"']*(?:signature|x-amz-|sig=|expires=|key-pair-id)[^\s\"']*"
    ),
    re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])"),  # Civitai API keys (32 hex chars)
    re.compile(r"(?i)(x-localflow-token[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{8,}"),
]


def redact(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


class _JsonFormatter(logging.Formatter):
    _context_keys = ("job_id", "project_id")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "process": record.processName,
            "msg": redact(record.getMessage()),
        }
        for key in self._context_keys:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


class RedactFilter(logging.Filter):
    """Redacts secrets in records from loggers with their own handlers (e.g. uvicorn's access log)."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(logs_dir: Path, level: str = "INFO", filename: str = "localflow.log") -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        if getattr(h, "_localflow", False):
            root.removeHandler(h)
            h.close()

    file_handler = RotatingFileHandler(
        logs_dir / filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(_JsonFormatter())
    file_handler._localflow = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(
        _ConsoleFormatter("%(asctime)s %(levelname)-7s %(processName)s %(name)s: %(message)s")
    )
    console._localflow = True  # type: ignore[attr-defined]
    root.addHandler(console)
    # Per-request client logs are noisy and URLs may carry tokens.
    for noisy in ("httpx", "httpx2", "httpcore", "LiteLLM", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for name in ("uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, RedactFilter) for f in lg.filters):
            lg.addFilter(RedactFilter())
