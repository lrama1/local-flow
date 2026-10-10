"""Secret status + validation results (Sections 7.2-7.3).

Values never leave SecretStore except to the checkers."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from localflow.db import Database
from localflow.errors import AppError, not_found
from localflow.models import Setting
from localflow.services.llm.catalog import KeyCheck, check_key
from localflow.services.secrets import SECRET_DEFS, SecretStore, fingerprint

Checker = Callable[[str, str], Awaitable[KeyCheck]]
RECHECK_AFTER = timedelta(hours=24)


class SecretStatus(BaseModel):
    name: str
    label: str
    kind: Literal["llm", "hub"]
    configured: bool
    source: Literal["env", "keyring", "file", "memory"] | None
    env_var: str | None
    env_vars: list[str]
    stored: bool  # a key pasted in Settings exists
    overridden: bool  # stored key exists but an environment variable takes precedence
    last4: str | None
    validated_at: datetime | None
    ok: bool | None  # None = not checked yet for the current key
    account: dict[str, Any] | None
    token_type: str | None
    warnings: list[str]
    error: str | None
    token_url: str


class KeyService:
    def __init__(self, db: Database, secrets: SecretStore, checker: Checker | None = None) -> None:
        self.db = db
        self.secrets = secrets
        self._checker: Checker = checker or (lambda name, value: check_key(name, value))
        self._locks: dict[str, asyncio.Lock] = {}

    def _check_record(self, name: str) -> dict[str, Any] | None:
        with self.db.session() as s:
            row = s.get(Setting, f"secret_check.{name}")
            return row.value_json if row and isinstance(row.value_json, dict) else None

    def status(self, name: str) -> SecretStatus:
        if name not in SECRET_DEFS:
            raise not_found("Secret", name)
        d = SECRET_DEFS[name]
        resolved = self.secrets.resolve(name)
        stored = self.secrets.has_stored(name)
        rec = self._check_record(name)
        current = bool(resolved and rec and rec.get("fingerprint") == fingerprint(resolved.value))
        checked_at = datetime.fromisoformat(rec["checked_at"]) if current and rec else None
        return SecretStatus(
            name=name,
            label=d.label,
            kind=d.kind,
            configured=resolved is not None,
            source=resolved.source if resolved else None,
            env_var=resolved.env_var if resolved else None,
            env_vars=list(d.env_vars),
            stored=stored,
            overridden=bool(stored and resolved and resolved.source == "env"),
            last4=resolved.value[-4:] if resolved and len(resolved.value) >= 8 else None,
            validated_at=checked_at,
            ok=rec.get("ok") if current and rec else None,
            account=rec.get("account") if current and rec else None,
            token_type=rec.get("token_type") if current and rec else None,
            warnings=list(rec.get("warnings") or []) if current and rec else [],
            error=rec.get("error") if current and rec else None,
            token_url=d.token_url,
        )

    def all_statuses(self) -> list[SecretStatus]:
        return [self.status(n) for n in SECRET_DEFS]

    async def test(self, name: str) -> SecretStatus:
        if name not in SECRET_DEFS:
            raise not_found("Secret", name)
        resolved = self.secrets.resolve(name)
        if resolved is None:
            raise AppError(409, "secret_not_configured", f"No {SECRET_DEFS[name].label} key is configured.")
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            result = await self._checker(name, resolved.value)
            record = {
                "fingerprint": fingerprint(resolved.value),
                "checked_at": datetime.now(UTC).isoformat(),
                "ok": result.ok,
                "account": result.account,
                "token_type": result.token_type,
                "warnings": result.warnings,
                "error": result.error,
            }
            with self.db.session() as s:
                row = s.get(Setting, f"secret_check.{name}") or Setting(key=f"secret_check.{name}")
                row.value_json = record
                s.add(row)
                s.commit()
        return self.status(name)

    async def ensure_checked(self, names: list[str]) -> None:
        """Validate configured keys that haven't been checked (for the current value) in the last 24 h."""
        todo = []
        now = datetime.now(UTC)
        for n in names:
            st = self.status(n)
            if st.configured and (st.validated_at is None or now - st.validated_at > RECHECK_AFTER):
                todo.append(n)
        if todo:
            await asyncio.gather(*(self.test(n) for n in todo), return_exceptions=True)

    def set(self, name: str, value: str) -> SecretStatus:
        if name not in SECRET_DEFS:
            raise not_found("Secret", name)
        if not value.strip():
            raise AppError(400, "empty_secret", "The key is empty.")
        self.secrets.set(name, value)
        return self.status(name)

    def delete(self, name: str) -> SecretStatus:
        if name not in SECRET_DEFS:
            raise not_found("Secret", name)
        self.secrets.delete(name)
        return self.status(name)

    def import_hf_cli(self, hf_home: Path | None = None) -> SecretStatus:
        """Copies a token saved by `hf auth login` into the keyring (Section 7.3). Never writes to HF_HOME."""
        for path in hf_cli_token_paths(hf_home):
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                self.secrets.set("hf_token", value)
                return self.status("hf_token")
        tried = ", ".join(str(p) for p in hf_cli_token_paths(hf_home))
        raise AppError(
            404,
            "hf_cli_token_not_found",
            f"No Hugging Face CLI login found (looked in {tried}). Run `hf auth login`, or paste a token.",
        )


def hf_cli_token_paths(own_hf_home: Path | None = None) -> list[Path]:
    """`$HF_TOKEN_PATH`, then `$HF_HOME/token` (unless it's LocalFlow's own), then the default location."""
    out: list[Path] = []
    if os.environ.get("HF_TOKEN_PATH"):
        out.append(Path(os.environ["HF_TOKEN_PATH"]).expanduser())
    env_home = os.environ.get("HF_HOME")
    if env_home and (own_hf_home is None or Path(env_home).resolve() != own_hf_home.resolve()):
        out.append(Path(env_home).expanduser() / "token")
    default = Path.home() / ".cache" / "huggingface" / "token"
    if default not in out:
        out.append(default)
    return out
