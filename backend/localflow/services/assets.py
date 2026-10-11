"""Asset files on disk + `assets` rows (Sections 4.2, 6.2, 13)."""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from sqlmodel import col, func, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, not_found
from localflow.models import (
    Asset,
    CollectionAsset,
    Ingredient,
    IngredientAsset,
    Job,
    Project,
    Scene,
    Shot,
    new_id,
    utcnow,
)
from localflow.services.board import derived_status
from localflow.services.media.ffmpeg import (
    FFmpegError,
    extract_frame,
    image_thumbnail,
    probe,
    run_ffmpeg,
    video_thumbnail,
)
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
        collection_id: str | None = None,
        model: str | None = None,
        scene_id: str | None = None,
        featuring: str | None = None,
        oldest_first: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Asset], int]:
        """`target` filters on the generating job's target_ref, e.g. {"ingredient_id": ...} (candidates).
        Library filters (Section 13): `collection_id`; `model` (a model entry id or name); `scene_id` (made
        for a shot of that scene); `featuring` (an ingredient id: its candidates, prompts that @mention it,
        and assets made for shots it's in); `q` searches prompts, dialogue, file names and tags."""
        with self.db.session() as s:
            stmt = select(Asset).where(Asset.project_id == project_id)
            if type_:
                stmt = stmt.where(Asset.type == type_)
            if source:
                stmt = stmt.where(Asset.source == source)
            if favorite is not None:
                stmt = stmt.where(Asset.favorite == favorite)
            if collection_id:
                members = select(CollectionAsset.asset_id).where(
                    CollectionAsset.collection_id == collection_id
                )
                stmt = stmt.where(col(Asset.id).in_(members))
            order = col(Asset.id).asc() if oldest_first else col(Asset.id).desc()
            rows = list(s.exec(stmt.order_by(order)).all())
            scene_shots: set[str] | None = None
            if scene_id:
                scene_shots = {sh.id for sh in s.exec(select(Shot).where(Shot.scene_id == scene_id))}
            feature: tuple[str, str, set[str]] | None = None
            if featuring:
                ing = s.get(Ingredient, featuring)
                if ing is None or ing.project_id != project_id:
                    return [], 0
                shots = s.exec(
                    select(Shot)
                    .join(Scene, col(Scene.id) == col(Shot.scene_id))
                    .where(Scene.project_id == project_id)
                ).all()
                feature = (
                    ing.id,
                    ing.key,
                    {sh.id for sh in shots if ing.id in (sh.ingredient_ids_json or [])},
                )
        # JSON-column filters run in Python: portable, and fine at a project's scale.
        if tag:
            rows = [a for a in rows if tag in a.tags_json]
        if target:

            def hit(a: Asset) -> bool:
                ref = a.generation_json.get("target_ref") or {}
                # Ingredient candidates without a view are primary candidates (M3 jobs).
                actual = {**ref, "view": ref.get("view") or "primary"} if "ingredient_id" in ref else ref
                return all(actual.get(k) == v for k, v in target.items())

            rows = [a for a in rows if hit(a)]
        if model:
            rows = [
                a
                for a in rows
                if model in (a.generation_json.get("model_entry_id"), a.generation_json.get("model_name"))
            ]
        if scene_shots is not None:
            rows = [
                a for a in rows if (a.generation_json.get("target_ref") or {}).get("shot_id") in scene_shots
            ]
        if feature is not None:
            ing_id, key, shots_with = feature

            def features(a: Asset) -> bool:
                gen = a.generation_json
                ref = gen.get("target_ref") or {}
                return (
                    ref.get("ingredient_id") == ing_id
                    or key in (gen.get("mentions") or [])
                    or ref.get("shot_id") in shots_with
                )

            rows = [a for a in rows if features(a)]
        if q:
            ql = q.lower()
            rows = [a for a in rows if ql in search_text(a)]
        return rows[offset : offset + limit], len(rows)

    def facets(self, project_id: str) -> dict[str, Any]:
        """Values to filter the library by, with counts: types, sources, tags, models, favorites."""
        with self.db.session() as s:
            rows = list(s.exec(select(Asset).where(Asset.project_id == project_id)).all())
        types: dict[str, int] = {}
        sources: dict[str, int] = {}
        tags: dict[str, int] = {}
        models: dict[str, dict[str, Any]] = {}
        for a in rows:
            types[a.type] = types.get(a.type, 0) + 1
            sources[a.source] = sources.get(a.source, 0) + 1
            for t in a.tags_json:
                tags[t] = tags.get(t, 0) + 1
            gen = a.generation_json
            key = gen.get("model_entry_id") or gen.get("model_name")
            if key:
                m = models.setdefault(
                    str(key), {"model": str(key), "name": gen.get("model_name") or key, "count": 0}
                )
                m["count"] += 1
        return {
            "total": len(rows),
            "favorites": sum(1 for a in rows if a.favorite),
            "types": types,
            "sources": sources,
            "tags": [{"tag": t, "count": n} for t, n in sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))],
            "models": sorted(models.values(), key=lambda m: -int(m["count"])),
        }

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
        """Validates an uploaded file by content (Pillow / ffprobe, not the extension), normalizes it for
        playback and editing (Section 13), then ingests it. Blocking: run it in a thread."""
        type_, ext = self._sniff(src, Path(filename).suffix.lower())
        named = src.with_name(f"{src.stem}{ext}")
        src.replace(named)
        gen = {"uploaded_name": Path(filename).name[:200], **(generation or {})}
        try:
            try:
                normalized, note = self._normalize(named, type_)
            except FFmpegError as e:
                raise AppError(
                    422, "convert_failed", f"Couldn't convert the {type_} for playback: {str(e)[:300]}"
                ) from e
            if note:
                gen["normalized"] = note
            return self.ingest_file(project_id, normalized, type_, source="uploaded", generation=gen)
        finally:
            # The upload and any (partial) converted copy; the ingested file is a copy in the project.
            for p in (named, *(named.with_name(f"{named.stem}.norm{x}") for x in (".mp4", ".wav"))):
                with contextlib.suppress(OSError):
                    p.unlink(missing_ok=True)

    _BROWSER_AUDIO = {"pcm_s16le", "pcm_s24le", "pcm_f32le", "aac", "mp3", "opus", "vorbis", "flac"}

    def _normalize(self, path: Path, type_: str) -> tuple[Path, str | None]:
        """Video → H.264/yuv420p MP4 (AAC audio), remuxing when the streams already fit; mp3/flac/ogg audio
        → WAV. Returns the file to ingest and a note on what changed (None: kept as uploaded)."""
        if type_ == "image":
            return path, None
        info = probe(self.settings.ffprobe, path)
        if type_ == "audio":
            if path.suffix in (".wav", ".m4a", ".aac"):
                return path, None
            out = path.with_name(f"{path.stem}.norm.wav")
            run_ffmpeg(self.settings.ffmpeg, ["-i", str(path), "-vn", "-c:a", "pcm_s16le", str(out)])
            return out, f"{path.suffix[1:]} → wav"
        h264 = info.get("video_codec") == "h264" and info.get("pix_fmt") in ("yuv420p", "yuvj420p")
        audio_ok = not info.get("has_audio") or info.get("audio_codec") == "aac"
        if h264 and audio_ok and path.suffix == ".mp4":
            return path, None
        out = path.with_name(f"{path.stem}.norm.mp4")
        # yuv420p H.264 needs even dimensions: an odd width or height loses its last pixel row/column.
        even = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        video = (
            ["-c:v", "copy"]
            if h264
            else [*even, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
        )
        audio = ["-c:a", "copy"] if audio_ok else ["-c:a", "aac", "-b:a", "192k"]
        run_ffmpeg(
            self.settings.ffmpeg,
            [
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                *video,
                *audio,
                "-movflags",
                "+faststart",
                str(out),
            ],
            total_s=info.get("duration_s"),
        )
        src = f"{path.suffix[1:]}/{info.get('video_codec')}"
        return out, f"{src} → mp4/h264" if not h264 else f"{src} remuxed to mp4"

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
            if result.get("output"):
                generation["output"] = result["output"]  # adapter-reported details, e.g. dialogue placement
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

    def extract_frame(self, asset_id: str, time_s: float) -> Asset:
        """A still from a video at `time_s` (clamped to the last frame) as a new image asset."""
        video = self.get(asset_id)
        if video.type != "video":
            raise AppError(400, "not_a_video", "Frames can only be extracted from videos.")
        fps = video.fps or 24.0
        last = max(0.0, (video.duration_s or 0.0) - 1.0 / fps)
        at = round(min(max(0.0, time_s), last) * fps) / fps  # on the frame grid
        tmp = self.settings.tmp_dir / "frames" / f"{new_id()}.png"
        try:
            # Seek half a frame early: ffmpeg drops frames before the (millisecond-rounded) seek time, so
            # seeking to exactly k/fps rounded up (1/24 → "0.042") would return frame k + 1.
            extract_frame(self.settings.ffmpeg, self.file_path(video), tmp, at_s=max(0.0, at - 0.5 / fps))
        except FFmpegError as e:
            raise AppError(500, "extract_failed", f"Couldn't extract the frame: {e}") from e
        if not tmp.is_file():
            raise AppError(400, "extract_failed", "No frame at that time.")
        generation = {"extracted_from": video.id, "time_s": round(at, 4), "frame": round(at * fps)}
        return self.ingest_file(
            video.project_id, tmp, "image", source="extracted", generation=generation, parents=[video.id]
        )

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
            if a.type == "audio":
                for shot in s.exec(select(Shot)).all():
                    if asset_id in (shot.selected_audio_asset_ids_json or []):
                        shot.selected_audio_asset_ids_json = [
                            x for x in shot.selected_audio_asset_ids_json if x != asset_id
                        ]
                        shot.updated_at = utcnow()
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


def search_text(a: Asset) -> str:
    """What the library's search box matches: prompts (raw and resolved), spoken text, the uploaded file's
    name, the model, and tags."""
    gen = a.generation_json or {}
    params = gen.get("params") or {}
    parts = [
        params.get("prompt"),
        params.get("text"),
        gen.get("prompt_raw"),
        gen.get("uploaded_name"),
        gen.get("model_name"),
        *(ln.get("text") for ln in params.get("lines") or [] if isinstance(ln, dict)),
        *a.tags_json,
    ]
    return " ".join(str(p) for p in parts if p).lower()


def _input_asset_ids(request: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for k, v in request.items():
        if k.endswith("_asset_id") and isinstance(v, str):
            ids.append(v)
        elif k.endswith("_asset_ids") and isinstance(v, list):
            ids.extend(x for x in v if isinstance(x, str))
    return ids
