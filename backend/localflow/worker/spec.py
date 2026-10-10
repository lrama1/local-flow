"""Resolved model/LoRA descriptions sent from the API to the worker with each job (torch-free).

The API resolves registry entries to absolute local paths, so the worker never touches the database."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PartSpec:
    role: str
    local_dir: str
    files: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return Path(self.local_dir)

    def first(self, suffix: str) -> Path | None:
        for f in self.files:
            if f.lower().endswith(suffix):
                return self.path / f
        return None


@dataclass(frozen=True)
class ModelSpec:
    entry_id: str
    name: str
    adapter_key: str | None
    kind: str
    base_model: str | None
    source: str
    parts: tuple[PartSpec, ...] = ()
    single_file: str | None = None  # a single checkpoint file (Civitai, or a single-file HF repo)
    default_params: dict[str, Any] = field(default_factory=dict)
    aux: dict[str, str] = field(default_factory=dict)  # e.g. {"sdxl_config_dir": ..., "vae_dir": ...}

    def part(self, *roles: str) -> PartSpec | None:
        for role in roles:
            for p in self.parts:
                if p.role == role:
                    return p
        return None

    def default(self, key: str, fallback: Any) -> Any:
        value = self.default_params.get(key)
        return fallback if value is None else value

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "name": self.name,
            "adapter_key": self.adapter_key,
            "kind": self.kind,
            "base_model": self.base_model,
            "source": self.source,
            "parts": [{"role": p.role, "local_dir": p.local_dir, "files": list(p.files)} for p in self.parts],
            "single_file": self.single_file,
            "default_params": dict(self.default_params),
            "aux": dict(self.aux),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(
            entry_id=str(d["entry_id"]),
            name=str(d.get("name") or d["entry_id"]),
            adapter_key=d.get("adapter_key"),
            kind=str(d.get("kind") or "checkpoint"),
            base_model=d.get("base_model"),
            source=str(d.get("source") or "hf"),
            parts=tuple(
                PartSpec(
                    role=str(p["role"]), local_dir=str(p["local_dir"]), files=tuple(p.get("files") or ())
                )
                for p in d.get("parts") or []
            ),
            single_file=d.get("single_file"),
            default_params=dict(d.get("default_params") or {}),
            aux={str(k): str(v) for k, v in (d.get("aux") or {}).items()},
        )


@dataclass(frozen=True)
class LoraSpec:
    entry_id: str
    name: str
    path: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoraSpec:
        return cls(
            entry_id=str(d["entry_id"]),
            name=str(d.get("name") or d["entry_id"]),
            path=str(d["path"]),
            weight=float(d.get("weight", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"entry_id": self.entry_id, "name": self.name, "path": self.path, "weight": self.weight}
