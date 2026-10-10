"""Asset files on disk + `assets` rows (Sections 4.2, 6.2, 13)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from sqlmodel import col, func, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, not_found
from localflow.models import Asset, Ingredient, IngredientAsset, Job, Project, Shot, new_id, utcnow
from localflow.services.board import derived_status
from localflow.services.media.ffmpeg import FFmpegError, image_thumbnail, probe, video_thumbnail
from localflow.services.timeline.service import drop_asset_clips
from localflow.worker.messages import OutputFile

log = logging.getLogger("localflow.assets")

_SUBDIR = {"image": "images", "video": "videos", "audio": "audio"}
_DEFAULT_EXT = {"image": ".png", "video": ".mp4", "audio": ".wav"}


class AssetService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db

    # --- paths -----------------------------------------------------------------------------------
    def project_dir(self, project_id: str) -> Path:
        root = self.settings.projects_dir.resolve()
        d = (root / project_id).resolve()
        if not d.is_relative_to(root) or d == root:
            raise AppError(400, "invalid_path", "Invalid project id")
        return d

    def resolve_in_project(self, project_id: str, rel: str) -> Path:
        """Resolve a DB-stored relative path; it must stay inside the project dir (Section 18)."""
        base = self.project_dir(project_id)
        p = (base / rel).resolve()
        if not p.is_relative_to(base):
            raise AppError(403, "forbidden_path", "Path escapes the project directory")
        return p

    def file_path(self, asset: Asset) -> Path:
        return self.resolve_in_project(asset.project_id, asset.path)

    def thumb_file(self, asset: Asset) -> Path | None:
        return self.resolve_in_project(asset.project_id, asset.thumb_path) if asset.thumb_path else None

    # --- queries ---------------------------------------------------------------------------------
    def get(self, asset_id: str) -> Asset:
        with self.db.session() as s:
            a = s.get(Asset, asset_id)
            if a is None:
                raise not_found("Asset", asset_id)
            return a

    def list_assets(
        self,
        project_id: str,
        *,
        type_: str | None = None,
        favorite: bool | None = None,
        tag: str | None = None,
        q: str | None = None,
        target: dict[str, str] | None = None,
        source: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Asset], int]:
        """`target` filters on the generating job's target_ref, e.g. {"ingredient_id": ...} (candidates)."""
        with self.db.session() as s:
            stmt = select(Asset).where(Asset.project_id == project_id)
            if type_:
                stmt = stmt.where(Asset.type == type_)
            if source:
                stmt = stmt.where(Asset.source == source)
            if favorite is not None:
                stmt = stmt.where(Asset.favorite == favorite)
            rows = list(s.exec(stmt.order_by(col(Asset.id).desc())).all())
        # tag/q filters operate on JSON columns; filtering in Python keeps SQL portable.
        if tag:
            rows = [a for a in rows if tag in a.tags_json]
        if target:

            def hit(a: Asset) -> bool:
                ref = a.generation_json.get("target_ref") or {}
                # Ingredient candidates without a view are primary candidates (M3 jobs).
                actual = {**ref, "view": ref.get("view") or "primary"} if "ingredient_id" in ref else ref
                return all(actual.get(k) == v for k, v in target.items())

            rows = [a for a in rows if hit(a)]
        if q:
            ql = q.lower()
            rows = [
                a for a in rows if ql in str(a.generation_json.get("params", {}).get("prompt", "")).lower()
            ]
        return rows[offset : offset + limit], len(rows)

    def latest_thumb(self, project_id: str) -> Asset | None:
        with self.db.session() as s:
            return s.exec(
                select(Asset)
                .where(Asset.project_id == project_id, col(Asset.thumb_path).is_not(None))
                .order_by(col(Asset.id).desc())
                .limit(1)
            ).first()

    def count(self, project_id: str) -> int:
        with self.db.session() as s:
            return int(
                s.exec(select(func.count()).select_from(Asset).where(Asset.project_id == project_id)).one()
            )

    # --- mutations -------------------------------------------------------------------------------
    def ingest_file(
        self,
        project_id: str,
        src: Path,
        type_: str,
        *,
        source: str,
        generation: dict[str, Any] | None = None,
        parents: list[str] | None = None,
        meta: OutputFile | None = None,
        subdir: str | None = None,
        thumb_at_s: float = 0.0,
    ) -> Asset:
        """Move `src` into the project's asset dir (or `subdir`, e.g. "exports"), probe it, thumbnail it
        (videos at `thumb_at_s`, at most halfway), and insert the row."""
        with self.db.session() as s:
            if s.get(Project, project_id) is None:
                raise not_found("Project", project_id)
        asset_id = new_id()
        ext = src.suffix.lower() or _DEFAULT_EXT[type_]
        rel = f"{subdir or 'assets/' + _SUBDIR[type_]}/{asset_id}{ext}"
        dst = self.resolve_in_project(project_id, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)

        asset = Asset(
            id=asset_id,
            project_id=project_id,
            type=type_,
            path=rel,
            source=source,
            generation_json=generation or {},
            parent_asset_ids_json=parents or [],
        )
        self._fill_metadata(asset, dst, meta)
        thumb_rel = f"thumbs/{asset_id}.jpg"
        thumb = self.resolve_in_project(project_id, thumb_rel)
        try:
            if type_ == "image":
                image_thumbnail(dst, thumb)
                asset.thumb_path = thumb_rel
            elif type_ == "video":
                at = min(thumb_at_s, (asset.duration_s or 0) / 2)
                video_thumbnail(self.settings.ffmpeg, dst, thumb, at_s=at)
                asset.thumb_path = thumb_rel
        except (OSError, FFmpegError) as e:
            log.warning("thumbnail failed for %s: %s", asset_id, e)

        with self.db.session() as s:
            s.add(asset)
            s.commit()
            s.refresh(asset)
        return asset

    def _fill_metadata(self, asset: Asset, path: Path, meta: OutputFile | None) -> None:
        if asset.type == "image":
            with Image.open(path) as im:
                asset.width, asset.height = im.size
            return
        info: dict[str, Any] = {}
        try:
            info = probe(self.settings.ffprobe, path)
        except FFmpegError as e:
            log.warning("ffprobe failed for %s: %s", path.name, e)
        asset.duration_s = info.get("duration_s") or (meta.duration_s if meta else None)
        if asset.type == "video":
            asset.width = info.get("width") or (meta.width if meta else None)
            asset.height = info.get("height") or (meta.height if meta else None)
            asset.fps = info.get("fps") or (meta.fps if meta else None)
            asset.has_audio = bool(info.get("has_audio", meta.has_audio if meta else False))
        else:
            asset.has_audio = True

    def ingest_upload(
        self,
        project_id: str,
        src: Path,
        filename: str,
        *,
        generation: dict[str, Any] | None = None,
    ) -> Asset:
        """Validates an uploaded file by content (Pillow / ffprobe, not the extension), then ingests it."""
        type_, ext = self._sniff(src, Path(filename).suffix.lower())
        named = src.with_name(f"{src.stem}{ext}")
        src.replace(named)
        gen = {"uploaded_name": Path(filename).name[:200], **(generation or {})}
        return self.ingest_file(project_id, named, type_, source="uploaded", generation=gen)

    _VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".m4v")
    _AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

    def _sniff(self, path: Path, ext: str) -> tuple[str, str]:
        try:
            with Image.open(path) as im:
                im.verify()
                fmt = (im.format or "").upper()
        except Exception:
            fmt = ""
        if fmt:
            exts = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}
            if fmt not in exts:
                raise AppError(415, "unsupported_media", f"Unsupported image format '{fmt}'.")
            return "image", exts[fmt]
        if ext not in self._VIDEO_EXT + self._AUDIO_EXT:
            raise AppError(415, "unsupported_media", "The file isn't a supported image, video or audio file.")
        try:
            info = probe(self.settings.ffprobe, path)
        except FFmpegError as e:
            raise AppError(415, "unsupported_media", "The file couldn't be read as video or audio.") from e
        if info.get("width") and ext in self._VIDEO_EXT:
            return "video", ext
        if ext in self._AUDIO_EXT:
            return "audio", ext
        raise AppError(415, "unsupported_media", "The file's content doesn't match its extension.")

    def ingest_job_outputs(self, job: Job, outputs: list[OutputFile], result: dict[str, Any]) -> list[Asset]:
        if job.project_id is None:
            raise AppError(400, "no_project", "Generation jobs must belong to a project")
        params = job.params_json
        request = dict(params.get("request") or {})
        parents = _input_asset_ids(request)
        assets = []
        for out in outputs:
            generation = {
                "job_id": job.id,
                "job_type": job.type,
                "adapter_key": result.get("adapter_key") or params.get("adapter_key"),
                "adapter_version": result.get("adapter_version"),
                "model_entry_id": request.get("model_entry_id"),
                "model_name": result.get("model_name"),
                "vram_profile": result.get("vram_profile"),
                "seed": out.seed if out.seed is not None else result.get("seed"),
                "params": request,
            }
            if job.target_ref_json:
                generation["target_ref"] = job.target_ref_json
            if params.get("prompt_raw"):
                # Section 6.2: prompts with @mentions keep both the raw and the resolved text.
                generation["prompt_raw"] = params["prompt_raw"]
                generation["mentions"] = params.get("mentions") or []
            assets.append(
                self.ingest_file(
                    job.project_id,
                    Path(out.path),
                    out.type,
                    source="generated",
                    generation=generation,
                    parents=parents,
                    meta=out,
                )
            )
        return assets

    def update(self, asset_id: str, *, tags: list[str] | None, favorite: bool | None) -> Asset:
        with self.db.session() as s:
            a = s.get(Asset, asset_id)
            if a is None:
                raise not_found("Asset", asset_id)
            if tags is not None:
                a.tags_json = sorted({t.strip() for t in tags if t.strip()})
            if favorite is not None:
                a.favorite = favorite
            s.add(a)
            s.commit()
            s.refresh(a)
            return a

    def delete(self, asset_id: str) -> tuple[str, set[str]]:
        """Deletes the asset and clears primaries/selected keyframes pointing at it. Returns the project id
        and which views changed ("ingredients", "board")."""
        a = self.get(asset_id)
        for p in (self.file_path(a), self.thumb_file(a)):
            if p is not None:
                p.unlink(missing_ok=True)
        changed: set[str] = set()
        with self.db.session() as s:
            for ing in s.exec(select(Ingredient).where(Ingredient.primary_asset_id == asset_id)).all():
                ing.primary_asset_id = None
                ing.updated_at = utcnow()
                s.add(ing)
                changed.add("ingredients")
            for view in s.exec(select(IngredientAsset).where(IngredientAsset.asset_id == asset_id)).all():
                s.delete(view)
                changed.add("ingredients")
            for shot in s.exec(select(Shot).where(col(Shot.reference_asset_ids_json).is_not(None))).all():
                refs = shot.reference_asset_ids_json or []
                if asset_id in refs:
                    kept = [r for r in refs if r != asset_id]
                    shot.reference_asset_ids_json = kept or None  # nothing left → automatic again
                    s.add(shot)
                    changed.add("board")
            shots = s.exec(
                select(Shot).where(
                    (Shot.selected_keyframe_asset_id == asset_id)
                    | (Shot.selected_end_keyframe_asset_id == asset_id)
                    | (Shot.selected_video_asset_id == asset_id)
                )
            ).all()
            for shot in shots:
                if shot.selected_keyframe_asset_id == asset_id:
                    shot.selected_keyframe_asset_id = None
                if shot.selected_end_keyframe_asset_id == asset_id:
                    shot.selected_end_keyframe_asset_id = None
                if shot.selected_video_asset_id == asset_id:
                    shot.selected_video_asset_id = None
                shot.status = derived_status(shot)
                shot.updated_at = utcnow()
                s.add(shot)
                changed.add("board")
            if drop_asset_clips(s, a.project_id, asset_id):
                changed.add("timeline")
            row = s.get(Asset, asset_id)
            if row is not None:
                s.delete(row)
            s.commit()
        return a.project_id, changed

    def delete_project_files(self, project_id: str) -> None:
        d = self.project_dir(project_id)
        if d.exists():
            shutil.rmtree(d)


def _input_asset_ids(request: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for k, v in request.items():
        if k.endswith("_asset_id") and isinstance(v, str):
            ids.append(v)
        elif k.endswith("_asset_ids") and isinstance(v, list):
            ids.extend(x for x in v if isinstance(x, str))
    return ids
