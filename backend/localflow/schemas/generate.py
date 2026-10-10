"""Bodies for `POST /api/generate/{kind}`. Inputs reference assets by id; the dispatcher resolves paths."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from localflow.worker.adapters.base import (
    ExtendParams,
    ImageParams,
    MusicRequest,
    T2IRequest,
    T2VRequest,
    TTSRequest,
    V2AParams,
    VideoParams,
)


class GenerateCommon(BaseModel):
    project_id: str | None = Field(default=None, description="Defaults to the Playground project")
    adapter_key: str | None = Field(default=None, description="Defaults to the first installed adapter")
    priority: int = Field(default=0, ge=-100, le=100)
    target_ref: dict[str, Any] | None = None
    prompt_raw: str | None = Field(
        default=None,
        description="Internal: the prompt as written, when the caller already resolved its @mentions",
    )
    mentions: list[str] | None = Field(default=None, description="Internal: keys resolved in the prompt")


class T2IBody(T2IRequest, GenerateCommon):
    pass


class ImageEditBody(ImageParams, GenerateCommon):
    reference_asset_ids: list[str] = Field(min_length=1, max_length=8)


class T2VBody(T2VRequest, GenerateCommon):
    pass


class I2VBody(VideoParams, GenerateCommon):
    image_asset_id: str


class FLF2VBody(VideoParams, GenerateCommon):
    first_asset_id: str
    last_asset_id: str


class Ref2VBody(VideoParams, GenerateCommon):
    reference_asset_ids: list[str] = Field(min_length=1, max_length=8)


class ExtendBody(ExtendParams, GenerateCommon):
    video_asset_id: str


class TTSBody(TTSRequest, GenerateCommon):
    pass


class V2ABody(V2AParams, GenerateCommon):
    video_asset_id: str


class MusicBody(MusicRequest, GenerateCommon):
    pass


GenerateBody = (
    T2IBody
    | ImageEditBody
    | T2VBody
    | I2VBody
    | FLF2VBody
    | Ref2VBody
    | ExtendBody
    | TTSBody
    | V2ABody
    | MusicBody
)

COMMON_FIELDS = set(GenerateCommon.model_fields)

# job type -> {asset-id field in the body: (path field in the worker request, expected asset type)}
ASSET_FIELDS: dict[str, dict[str, tuple[str, str]]] = {
    "image_edit": {"reference_asset_ids": ("reference_image_paths", "image")},
    "i2v": {"image_asset_id": ("image_path", "image")},
    "flf2v": {"first_asset_id": ("first_image_path", "image"), "last_asset_id": ("last_image_path", "image")},
    "ref2v": {"reference_asset_ids": ("reference_image_paths", "image")},
    "extend": {"video_asset_id": ("video_path", "video")},
    "v2a": {"video_asset_id": ("video_path", "video")},
}
