"""Vision review (Section 8.3 step 5): does a keyframe or take match its ingredients' reference images?

The image under review (or two frames of a take) and the shot's reference images (the same ones keyframes
use, Section 11.7) go to a vision-capable LLM, which answers `{consistent, issues, suggested_prompt_fix}`."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from localflow.config import Settings
from localflow.errors import AppError, bad_request
from localflow.models import Asset, Shot, new_id
from localflow.schemas.characters import RefOut
from localflow.schemas.review import (
    PromptField,
    ReviewBody,
    ReviewTarget,
    ShotReviewOut,
    VisionCapability,
    VisionReview,
)
from localflow.schemas.story import UsageOut
from localflow.services.assets import AssetService
from localflow.services.characters.references import VIEW_LABELS
from localflow.services.imagegen import ImageGenService
from localflow.services.llm.service import LLMService
from localflow.services.llm.templates import render
from localflow.services.media.ffmpeg import FFmpegError, extract_frame
from localflow.services.media.images import jpeg_data_url

MAX_ISSUES = 6
FIELDS: dict[ReviewTarget, PromptField] = {
    "keyframe": "keyframe_prompt",
    "end_keyframe": "end_keyframe_prompt",
    "video": "video_prompt",
}
TAKE_FRAMES = (0.3, 0.7)  # where in a take to look (fractions of its length): early drift and late drift


def review_errors(r: VisionReview) -> list[str]:
    """Semantic checks the provider repairs against (Section 8.4)."""
    errors = []
    if not r.consistent and not any(i.strip() for i in r.issues):
        errors.append("`consistent` is false but `issues` is empty: list what doesn't match.")
    if not r.consistent and not r.suggested_prompt_fix.strip():
        errors.append("`consistent` is false: put the complete revised prompt in `suggested_prompt_fix`.")
    return errors


def review_target(asset: Asset, shot: Shot) -> ReviewTarget:
    if asset.type == "video":
        return "video"
    slot = (asset.generation_json.get("target_ref") or {}).get("slot")
    end = slot == "end_keyframe" or asset.id == shot.selected_end_keyframe_asset_id
    return "end_keyframe" if end and shot.end_keyframe_prompt is not None else "keyframe"


class ReviewService:
    def __init__(self, settings: Settings, assets: AssetService, imagegen: ImageGenService, llm: LLMService):
        self.settings = settings
        self.assets = assets
        self.imagegen = imagegen
        self.llm = llm

    def capability(self, purpose: str = "vision_review") -> VisionCapability:
        ok, model, reason = self.llm.vision(purpose)
        return VisionCapability(purpose=purpose, available=ok, model=model, reason=reason)

    async def review_shot(self, shot_id: str, body: ReviewBody) -> ShotReviewOut:
        ctx = self.imagegen.shot_context(shot_id)
        shot = ctx.shot
        asset_id = body.asset_id or shot.selected_keyframe_asset_id or shot.selected_video_asset_id
        if not asset_id:
            raise bad_request("nothing_to_review", "Generate or pick a keyframe or a take first.")
        asset = self.assets.get(asset_id)
        if asset.project_id != ctx.project.id or asset.type == "audio":
            raise bad_request("invalid_asset", "Pick an image or a video from this project.")
        target = review_target(asset, shot)
        field = FIELDS[target]
        current = getattr(shot, field) or ""

        refs = [
            (r, p) for r in self.imagegen.references_for(shot_id).selected if (p := self._path(r.asset_id))
        ]
        if not refs:
            raise bad_request(
                "no_references",
                "This shot's characters, location and props have no reference images to compare against yet.",
            )
        frames_s, shot_images = await asyncio.to_thread(self._shot_images, asset)
        ref_images = await asyncio.to_thread(lambda: [jpeg_data_url(p) for _, p in refs])

        location = ctx.rows.get(ctx.scene.location_ingredient_id or "")
        visible = [ctx.rows[i].name for i in shot.ingredient_ids_json if i in ctx.rows]
        shot_ctx: dict[str, Any] = {
            "shot_type": shot.shot_type,
            "camera_move": shot.camera_move,
            "scene": ctx.scene.title,
            "location": location.name if location else None,
            "in_shot": visible,
            field: current,
            "style": self.imagegen.style(ctx.project.id) or None,
        }
        references = [
            {
                "image": len(shot_images) + n,
                "name": r.name,
                "kind": r.kind,
                "view": VIEW_LABELS.get(r.view, r.view),
                "description": self._describe(ctx.rows.get(r.ingredient_id or "")),
            }
            for n, (r, _) in enumerate(refs, 1)
        ]
        frame_labels = (
            [f"{t:g} s" for t in frames_s]
            if frames_s
            else ["end frame" if target == "end_keyframe" else "keyframe"]
        )
        system, user, _ = render(
            "vision_review",
            {
                "target": target,
                "prompt_field": field,
                "frame_labels": frame_labels,
                "shot": shot_ctx,
                "references": references,
                "schema": VisionReview.model_json_schema(),
            },
        )
        res = await self.llm.structured(
            purpose="vision_review",
            project_id=ctx.project.id,
            choice=body.llm,
            system=system,
            user=user,
            schema=VisionReview,
            validate=review_errors,
            temperature=0.2,
            images=shot_images + ref_images,
        )
        v = res.value
        return ShotReviewOut(
            shot_id=shot.id,
            asset_id=asset.id,
            target=target,
            prompt_field=field,
            current_prompt=current,
            consistent=v.consistent,
            issues=[i.strip() for i in v.issues if i.strip()][:MAX_ISSUES],
            suggested_prompt_fix="" if v.consistent else v.suggested_prompt_fix.strip(),
            references=[RefOut.model_validate(r.model_dump()) for r, _ in refs],
            frames_s=frames_s,
            usage=UsageOut(
                provider=res.provider,
                model=res.model,
                prompt_tokens=res.usage.prompt_tokens,
                completion_tokens=res.usage.completion_tokens,
                cost_usd=res.usage.cost_usd,
                attempts=res.attempts,
            ),
        )

    def _path(self, asset_id: str) -> Path | None:
        try:
            path = self.assets.file_path(self.assets.get(asset_id))
        except AppError:
            return None
        return path if path.is_file() else None

    @staticmethod
    def _describe(ing: Any) -> str:
        if ing is None:
            return ""
        parts = [ing.visual_prompt, ing.body_outfit_prompt or ""]
        return " ".join(p.strip() for p in parts if p and p.strip())[:600]

    def _shot_images(self, asset: Asset) -> tuple[list[float], list[str]]:
        """([], [the image]) or ([frame times], [frames]) for a take."""
        path = self.assets.file_path(asset)
        if not path.is_file():
            raise AppError(410, "asset_file_missing", "The file for this asset is missing on disk.")
        if asset.type == "image":
            return [], [jpeg_data_url(path)]
        fps = asset.fps or 24.0
        duration = asset.duration_s or 0.0
        times = sorted({round(duration * f * fps) / fps for f in TAKE_FRAMES})
        tmp = self.settings.tmp_dir / "review" / new_id()
        try:
            urls = []
            for n, t in enumerate(times):
                out = tmp / f"{n}.png"
                extract_frame(self.settings.ffmpeg, path, out, at_s=max(0.0, t - 0.5 / fps))  # see assets
                urls.append(jpeg_data_url(out))
        except (FFmpegError, OSError) as e:
            raise AppError(500, "extract_failed", f"Couldn't read frames from the take: {e}") from e
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return [round(t, 2) for t in times], urls
