"""The curated catalog (`catalog.yaml`) and its sync into `model_entries` (Section 9.1)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import col, select

from localflow.db import Database
from localflow.models import ModelEntry, utcnow

log = logging.getLogger("localflow.registry")

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")


class CatalogPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    repo_id: str
    revision: str = "main"
    allow_patterns: list[str] = Field(min_length=1)


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    purpose: str = ""
    description: str = ""
    kind: str
    capabilities: list[str]
    adapter_key: str
    base_model: str
    source_ref: str
    license: str
    license_url: str
    approx_size_gb: float
    recommended: bool = False
    experimental: bool = False
    requires_pickle: bool = False
    notes: str = ""
    default_params: dict[str, Any] = Field(default_factory=dict)
    vram_profiles: dict[str, Any] = Field(default_factory=dict)
    render: dict[str, Any] | None = None
    parts: list[CatalogPart] = Field(min_length=1)

    def parts_json(self) -> list[dict[str, Any]]:
        return [{"source": "hf", **p.model_dump(), "files": [], "access": None} for p in self.parts]


@lru_cache(maxsize=1)
def load_catalog(path: Path = CATALOG_PATH) -> tuple[CatalogEntry, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    entries = tuple(CatalogEntry.model_validate(e) for e in raw)
    keys = [e.key for e in entries]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate catalog keys: {sorted(dupes)}")
    return entries


def _apply(row: ModelEntry, e: CatalogEntry) -> None:
    row.source, row.source_ref, row.name = "hf", e.source_ref, e.name
    row.description, row.kind = e.description, e.kind
    row.capabilities_json, row.adapter_key, row.base_model = list(e.capabilities), e.adapter_key, e.base_model
    row.license, row.license_url = e.license, e.license_url
    row.default_params_json, row.vram_profiles_json = dict(e.default_params), dict(e.vram_profiles)
    row.requires_pickle, row.experimental, row.notes = e.requires_pickle, e.experimental, e.notes
    row.meta_json = {
        **(row.meta_json or {}),
        "purpose": e.purpose,
        "recommended": e.recommended,
        "approx_size_bytes": int(e.approx_size_gb * 1e9),
        "render": e.render,
    }
    if row.status not in ("installed", "downloading"):
        # Not installed: the catalog defines what will be downloaded. Keep resolved file lists if unchanged.
        old = {(p.get("repo_id"), tuple(p.get("allow_patterns") or [])): p for p in row.files_json or []}
        parts: list[dict[str, Any]] = []
        for p in e.parts_json():
            prev = old.get((p["repo_id"], tuple(p["allow_patterns"])))
            kept: dict[str, Any] = {k: prev[k] for k in ("files", "access", "gated") if prev and k in prev}
            parts.append({**p, **kept})
        row.files_json = parts
        row.size_bytes = sum(int(f.get("size") or 0) for p in parts for f in p["files"]) or int(
            e.approx_size_gb * 1e9
        )
    row.updated_at = utcnow()


def sync_catalog(db: Database, entries: tuple[CatalogEntry, ...] | None = None) -> int:
    """Upserts catalog entries by `catalog_key`, keeping install state; drops removed, uninstalled entries."""
    entries = entries if entries is not None else load_catalog()
    keys = {e.key for e in entries}
    with db.session() as s:
        existing = {
            r.catalog_key: r
            for r in s.exec(select(ModelEntry).where(col(ModelEntry.catalog_key).is_not(None)))
        }
        for e in entries:
            row = existing.get(e.key) or ModelEntry(
                catalog_key=e.key, source="hf", source_ref=e.source_ref, name=e.name
            )
            _apply(row, e)
            s.add(row)
        for key, row in existing.items():
            if key not in keys:
                if row.status in ("installed", "downloading"):
                    row.meta_json = {**(row.meta_json or {}), "catalog_removed": True}
                    s.add(row)
                else:
                    s.delete(row)
        s.commit()
    return len(entries)
