"""Secrets (Section 7.2): environment variables first, then keys pasted in Settings.

Pasted keys live in the OS keyring (or an encrypted file if there's none). Values never leave this module
except to the code that calls a provider."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

log = logging.getLogger("localflow.secrets")

SERVICE_NAME = "LocalFlow"
SecretSource = Literal["env", "keyring", "file", "memory"]


@dataclass(frozen=True)
class SecretDef:
    name: str
    label: str
    kind: Literal["llm", "hub"]
    env_vars: tuple[str, ...]
    token_url: str


SECRET_DEFS: dict[str, SecretDef] = {
    d.name: d
    for d in (
        SecretDef(
            "openai_api_key", "OpenAI", "llm", ("OPENAI_API_KEY",), "https://platform.openai.com/api-keys"
        ),
        SecretDef(
            "anthropic_api_key",
            "Anthropic",
            "llm",
            ("ANTHROPIC_API_KEY",),
            "https://console.anthropic.com/settings/keys",
        ),
        SecretDef(
            "openrouter_api_key", "OpenRouter", "llm", ("OPENROUTER_API_KEY",), "https://openrouter.ai/keys"
        ),
        SecretDef(
            "hf_token",
            "Hugging Face",
            "hub",
            ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
            "https://huggingface.co/settings/tokens",
        ),
        SecretDef(
            "civitai_api_key",
            "Civitai",
            "hub",
            ("CIVITAI_API_TOKEN", "CIVITAI_API_KEY"),
            "https://civitai.com/user/account",
        ),
    )
}
LLM_SECRET = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "openrouter": "openrouter_api_key",
}


def fingerprint(value: str) -> str:
    """Short, non-reversible id used to notice that a key changed (e.g. to reset its validation status)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


# --- environment -------------------------------------------------------------------------------
def windows_registry_env(name: str) -> str | None:
    """User then system environment from the registry: picks up variables set after this process started."""
    if sys.platform != "win32":
        return None
    import winreg

    locations = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for hive, path in locations:
        try:
            with winreg.OpenKey(hive, path) as key:
                value, kind = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if isinstance(value, str) and value.strip():
            return winreg.ExpandEnvironmentStrings(value) if kind == winreg.REG_EXPAND_SZ else value
    return None


class EnvReader:
    def __init__(
        self, environ: Mapping[str, str] | None = None, registry: Callable[[str], str | None] | None = None
    ) -> None:
        self._environ = environ if environ is not None else os.environ
        self._registry = registry

    def lookup(self, names: tuple[str, ...]) -> tuple[str, str] | None:
        for n in names:
            v = self._environ.get(n)
            if v and v.strip():
                return n, v.strip()
        if self._registry is not None:
            for n in names:
                v = self._registry(n)
                if v and v.strip():
                    return n, v.strip()
        return None


# --- stored keys -------------------------------------------------------------------------------
class KeyStore(Protocol):
    source: SecretSource

    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class MemoryKeyStore:
    source: SecretSource = "memory"

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._data.get(name)

    def set(self, name: str, value: str) -> None:
        self._data[name] = value

    def delete(self, name: str) -> None:
        self._data.pop(name, None)


class KeyringStore:
    source: SecretSource = "keyring"

    def get(self, name: str) -> str | None:
        import keyring

        return keyring.get_password(SERVICE_NAME, name)

    def set(self, name: str, value: str) -> None:
        import keyring

        keyring.set_password(SERVICE_NAME, name, value)

    def delete(self, name: str) -> None:
        import keyring
        from keyring.errors import PasswordDeleteError

        with contextlib.suppress(PasswordDeleteError):
            keyring.delete_password(SERVICE_NAME, name)


class EncryptedFileStore:
    """Fallback when no keyring backend exists: Fernet-encrypted JSON keyed by a machine-local key file."""

    source: SecretSource = "file"

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "secrets.enc"
        self._key_path = data_dir / "secrets.key"
        self._lock = threading.Lock()

    def _fernet(self):  # type: ignore[no-untyped-def]
        from cryptography.fernet import Fernet

        if not self._key_path.exists():
            self._key_path.write_bytes(Fernet.generate_key())
            with contextlib.suppress(OSError):
                os.chmod(self._key_path, 0o600)
        return Fernet(self._key_path.read_bytes())

    def _load(self) -> dict[str, str]:
        import json

        if not self._path.exists():
            return {}
        return dict(json.loads(self._fernet().decrypt(self._path.read_bytes())))

    def _save(self, data: dict[str, str]) -> None:
        import json

        self._path.write_bytes(self._fernet().encrypt(json.dumps(data).encode("utf-8")))

    def get(self, name: str) -> str | None:
        with self._lock:
            return self._load().get(name)

    def set(self, name: str, value: str) -> None:
        with self._lock:
            data = self._load()
            data[name] = value
            self._save(data)

    def delete(self, name: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(name, None)
            self._save(data)


def make_key_store(backend: str, data_dir: Path) -> KeyStore:
    if backend == "memory":
        return MemoryKeyStore()
    if backend == "file":
        return EncryptedFileStore(data_dir)
    try:
        import keyring
        from keyring.backends import fail

        if isinstance(keyring.get_keyring(), fail.Keyring):
            raise RuntimeError("no keyring backend")
        return KeyringStore()
    except Exception as e:
        if backend == "keyring":
            raise
        log.warning(
            "No OS keyring available (%s); storing pasted keys in an encrypted file in the data dir", e
        )
        return EncryptedFileStore(data_dir)


# --- facade ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedSecret:
    value: str
    source: SecretSource
    env_var: str | None


class SecretStore:
    def __init__(self, store: KeyStore, env: EnvReader | None) -> None:
        self._store = store
        self._env = env

    @property
    def stored_source(self) -> SecretSource:
        return self._store.source

    def resolve(self, name: str) -> ResolvedSecret | None:
        d = SECRET_DEFS[name]
        if self._env is not None:
            hit = self._env.lookup(d.env_vars)
            if hit:
                return ResolvedSecret(hit[1], "env", hit[0])
        stored = self._stored(name)
        return ResolvedSecret(stored, self._store.source, None) if stored else None

    def get(self, name: str) -> str | None:
        r = self.resolve(name)
        return r.value if r else None

    def has_stored(self, name: str) -> bool:
        return bool(self._stored(name))

    def set(self, name: str, value: str) -> None:
        self._store.set(name, value.strip())

    def delete(self, name: str) -> None:
        self._store.delete(name)

    def _stored(self, name: str) -> str | None:
        try:
            return self._store.get(name)
        except Exception as e:  # a broken keyring must not take the app down
            log.warning("reading stored secret %s failed: %s", name, e)
            return None


def build_secret_store(backend: str, data_dir: Path, from_env: bool) -> SecretStore:
    env = EnvReader(registry=windows_registry_env) if from_env else None
    return SecretStore(make_key_store(backend, data_dir), env)
