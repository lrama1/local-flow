"""Settings → Diagnostics (Section 19)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DiagCheck(BaseModel):
    key: str
    label: str
    status: Literal["ok", "warn", "error", "skip"]
    detail: str


class FolderUsage(BaseModel):
    key: str = Field(description="models | projects | library | tmp | logs | database")
    path: str
    size_mb: float
    files: int


class LogFile(BaseModel):
    name: str
    size_kb: float
    modified: datetime


class DiagnosticsOut(BaseModel):
    generated_at: datetime
    checks: list[DiagCheck]
    versions: dict[str, str | None]
    storage: list[FolderUsage]
    disk_free_mb: int | None
    disk_total_mb: int | None
    logs: list[LogFile]


class LogTail(BaseModel):
    name: str
    lines: list[str] = Field(description="Oldest first, formatted as `time level logger: message`")
    total_lines: int
