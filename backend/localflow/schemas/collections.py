"""Asset library: collections, facets, bulk edits and frame extraction (Sections 13, 16)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CollectionOut(BaseModel):
    id: str
    project_id: str
    name: str
    description: str
    asset_count: int
    cover_thumb_url: str | None
    created_at: datetime
    updated_at: datetime


class CollectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    asset_ids: list[str] = Field(default_factory=list, max_length=500, description="Assets to start with")


class CollectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)


class CollectionAssetsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: list[str] = Field(min_length=1, max_length=500)


class TagCount(BaseModel):
    tag: str
    count: int


class ModelCount(BaseModel):
    model: str = Field(description="Model entry id (or name for older assets): the `model` filter value")
    name: str
    count: int


class AssetFacets(BaseModel):
    total: int
    favorites: int
    types: dict[str, int]
    sources: dict[str, int]
    tags: list[TagCount]
    models: list[ModelCount]


class BulkAssetsBody(BaseModel):
    """Tag, favorite or delete several assets at once."""

    model_config = ConfigDict(extra="forbid")

    asset_ids: list[str] = Field(min_length=1, max_length=500)
    add_tags: list[str] = Field(default_factory=list, max_length=20)
    remove_tags: list[str] = Field(default_factory=list, max_length=20)
    favorite: bool | None = None
    delete: bool = False


class BulkAssetsOut(BaseModel):
    updated: int
    deleted: int


class ExtractFrameBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_s: float = Field(ge=0, le=36_000)
