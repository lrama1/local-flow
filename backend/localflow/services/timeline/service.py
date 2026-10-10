"""The project timeline: load/save with normalization, auto-assemble from the board (Section 15.1)."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from sqlmodel import Session, col, select

from localflow.db import Database
from localflow.errors import bad_request, not_found
from localflow.models import Asset, Project, Scene, Shot, Timeline, utcnow
from localflow.schemas.timeline import (
    AutoAssembleBody,
    AutoAssembleOut,
    Clip,
    SkippedShot,
    TimelineData,
    TimelineOut,
    Transition,
)
from localflow.services.events import EventBus
from localflow.services.timeline.model import default_timeline, duration_s, normalize

# Which audio track a shot's selected audio lands on, by the job that made it (M6 adds more).
_AUDIO_TRACK_BY_JOB = {"tts": "a_dialogue", "v2a": "a_sfx", "music": "a_music"}


class TimelineService:
    def __init__(self, db: Database, bus: EventBus) -> None:
        self.db = db
        self.bus = bus

    # --- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _project(s: Session, project_id: str) -> Project:
        p = s.get(Project, project_id)
        if p is None:
            raise not_found("Project", project_id)
        return p

    @staticmethod
    def _row(s: Session, project_id: str) -> Timeline | None:
        return s.exec(select(Timeline).where(Timeline.project_id == project_id)).first()

    @staticmethod
    def _parse(raw: dict[str, Any], project: Project) -> TimelineData:
        if not raw:
            return default_timeline(project.aspect_ratio, project.fps)
        try:
            return TimelineData.model_validate(raw)
        except ValidationError:
            # A stored timeline from an older shape: start over rather than fail the tab.
            return default_timeline(project.aspect_ratio, project.fps)

    @staticmethod
    def _assets(s: Session, project_id: str, timeline: TimelineData) -> dict[str, Asset]:
        ids = {c.asset_id for t in timeline.tracks for c in t.clips}
        if not ids:
            return {}
        rows = s.exec(select(Asset).where(col(Asset.id).in_(ids), Asset.project_id == project_id)).all()
        return {a.id: a for a in rows}

    def _out(self, project_id: str, tl: TimelineData, row: Timeline | None, notes: list[str]) -> TimelineOut:
        return TimelineOut(
            project_id=project_id,
            timeline=tl,
            duration_s=round(duration_s(tl), 6),
            updated_at=row.updated_at if row else None,
            notes=notes,
        )

    def _save(self, s: Session, project_id: str, tl: TimelineData) -> Timeline:
        row = self._row(s, project_id)
        if row is None:
            row = Timeline(project_id=project_id)
        row.tracks_json = tl.model_dump(mode="json")
        row.updated_at = utcnow()
        s.add(row)
        s.commit()
        s.refresh(row)
        return row

    # --- API -------------------------------------------------------------------------------------
    def load(self, project_id: str) -> TimelineData:
        """The normalized timeline, without clips whose asset is gone."""
        with self.db.session() as s:
            project = self._project(s, project_id)
            row = self._row(s, project_id)
            tl = self._parse(row.tracks_json if row else {}, project)
            assets = self._assets(s, project_id, tl)
        for t in tl.tracks:
            t.clips = [c for c in t.clips if c.asset_id in assets]
        tl, _ = normalize(tl, {a.id: a.duration_s for a in assets.values()})
        return tl

    def get(self, project_id: str) -> TimelineOut:
        with self.db.session() as s:
            row = self._row(s, project_id)
        return self._out(project_id, self.load(project_id), row, [])

    def put(self, project_id: str, timeline: TimelineData) -> TimelineOut:
        with self.db.session() as s:
            self._project(s, project_id)
            assets = self._assets(s, project_id, timeline)
            for track in timeline.tracks:
                for c in track.clips:
                    a = assets.get(c.asset_id)
                    if a is None:
                        raise bad_request("invalid_asset", f"Asset '{c.asset_id}' isn't in this project.")
                    if track.kind == "video" and a.type != "video":
                        raise bad_request("invalid_asset", "The video track only takes videos.")
                    if track.kind == "audio" and a.type != "audio":
                        raise bad_request("invalid_asset", "Audio tracks only take audio.")
            tl, notes = normalize(timeline, {a.id: a.duration_s for a in assets.values()})
            row = self._save(s, project_id, tl)
        self.bus.publish("timeline.updated", project_id=project_id)
        return self._out(project_id, tl, row, notes)

    def auto_assemble(self, project_id: str, body: AutoAssembleBody) -> AutoAssembleOut:
        """Rebuilds the timeline from the board: every shot's selected take in scene/shot order, cut within
        a scene and `scene_transition` between scenes, plus the shots' selected audio at their clip's
        start. The music track is kept (`keep_music`); trims and other edits are replaced."""
        with self.db.session() as s:
            project = self._project(s, project_id)
            scenes = s.exec(
                select(Scene).where(Scene.project_id == project_id).order_by(col(Scene.index))
            ).all()
            order = {sc.id: sc.index for sc in scenes}
            shots = sorted(
                s.exec(select(Shot).where(col(Shot.scene_id).in_(list(order)))).all(),
                key=lambda sh: (order[sh.scene_id], sh.index),
            )
            wanted = {sh.selected_video_asset_id for sh in shots if sh.selected_video_asset_id}
            wanted |= {a for sh in shots for a in sh.selected_audio_asset_ids_json}
            assets = {
                a.id: a
                for a in s.exec(
                    select(Asset).where(col(Asset.id).in_(wanted), Asset.project_id == project_id)
                ).all()
            }
            row = self._row(s, project_id)
            previous = self._parse(row.tracks_json if row else {}, project)

        tl = default_timeline(project.aspect_ratio, project.fps)
        if body.keep_music:
            for t in tl.audio:
                kept = next((p for p in previous.audio if p.id == t.id and t.id == "a_music"), None)
                if kept is not None:
                    t.clips, t.gain_db, t.muted = kept.clips, kept.gain_db, kept.muted
        skipped: list[SkippedShot] = []
        placed: list[tuple[Shot, Clip]] = []
        last_scene: str | None = None
        for shot in shots:
            take = assets.get(shot.selected_video_asset_id or "")
            if take is None or not take.duration_s:
                reason = "No take selected." if not shot.selected_video_asset_id else "The take is missing."
                skipped.append(SkippedShot(shot_id=shot.id, reason=reason))
                continue
            if not placed:
                transition = (
                    Transition(type="fade_from_black", duration_s=body.transition_s)
                    if body.fade_in
                    else Transition()
                )
            elif shot.scene_id != last_scene and body.scene_transition == "crossfade":
                transition = Transition(type="crossfade", duration_s=body.transition_s)
            else:
                transition = Transition()
            clip = Clip(
                asset_id=take.id,
                in_s=0.0,
                out_s=take.duration_s,
                transition_in=transition,
                shot_id=shot.id,
            )
            tl.video.clips.append(clip)
            placed.append((shot, clip))
            last_scene = shot.scene_id

        durations: dict[str, float | None] = {a.id: a.duration_s for a in assets.values()}
        for t in tl.audio:
            durations.update({c.asset_id: None for c in t.clips if c.asset_id not in durations})
        tl, notes = normalize(tl, durations)
        starts = {c.id: c.start_s for c in tl.video.clips}
        tracks = {t.id: t for t in tl.audio}
        for shot, clip in placed:
            for aid in shot.selected_audio_asset_ids_json:
                audio = assets.get(aid)
                if audio is None or audio.type != "audio" or not audio.duration_s:
                    continue
                track_id = _AUDIO_TRACK_BY_JOB.get(str(audio.generation_json.get("job_type")), "a_sfx")
                if track_id == "a_music" and body.keep_music and tracks["a_music"].clips:
                    continue  # the kept music wins
                tracks[track_id].clips.append(
                    Clip(asset_id=aid, start_s=starts[clip.id], out_s=audio.duration_s, shot_id=shot.id)
                )
        tl, more = normalize(tl, durations)
        with self.db.session() as s:
            row = self._save(s, project_id, tl)
        self.bus.publish("timeline.updated", project_id=project_id)
        out = self._out(project_id, tl, row, notes + more)
        return AutoAssembleOut(**out.model_dump(), skipped=skipped)

    # --- placing generated audio (M6) ------------------------------------------------------------------
    def _with_timeline(self, project_id: str, edit: Any) -> bool:
        """Loads the normalized timeline, applies `edit(tl) -> bool` and saves it if it changed."""
        tl = self.load(project_id)
        if not edit(tl):
            return False
        with self.db.session() as s:
            durations = {a.id: a.duration_s for a in self._assets(s, project_id, tl).values()}
            tl, _ = normalize(tl, durations)
            self._save(s, project_id, tl)
        self.bus.publish("timeline.updated", project_id=project_id)
        return True

    def sync_shot_audio(self, project_id: str, shot_id: str, asset: Asset, track_id: str) -> bool:
        """If the timeline has the shot's take, swaps the shot's clip on `track_id` for `asset`, aligned with
        the take (and trimmed like it). Returns whether the timeline changed."""

        def edit(tl: TimelineData) -> bool:
            video = next((c for c in tl.video.clips if c.shot_id == shot_id), None)
            track = next((t for t in tl.audio if t.id == track_id), None)
            if video is None or track is None or not asset.duration_s:
                return False
            track.clips = [c for c in track.clips if c.shot_id != shot_id]
            in_s = min(video.in_s, max(0.0, asset.duration_s - 0.05))
            track.clips.append(
                Clip(
                    asset_id=asset.id,
                    start_s=video.start_s,
                    in_s=in_s,
                    out_s=asset.duration_s,
                    shot_id=shot_id,
                )
            )
            return True

        return self._with_timeline(project_id, edit)

    def scene_start(self, project_id: str, scene_id: str) -> float | None:
        """Where the scene's first clip starts on the timeline, or None if none of its shots are placed."""
        with self.db.session() as s:
            shot_ids = {sh.id for sh in s.exec(select(Shot).where(Shot.scene_id == scene_id))}
        starts = [c.start_s for c in self.load(project_id).video.clips if c.shot_id in shot_ids]
        return min(starts) if starts else None

    def place_audio(self, project_id: str, asset: Asset, track_id: str, start_s: float) -> bool:
        """Adds `asset` to an audio track at `start_s` (e.g. generated music)."""

        def edit(tl: TimelineData) -> bool:
            track = next((t for t in tl.audio if t.id == track_id), None)
            if track is None or not asset.duration_s:
                return False
            track.clips.append(Clip(asset_id=asset.id, start_s=max(0.0, start_s), out_s=asset.duration_s))
            return True

        return self._with_timeline(project_id, edit)


def drop_asset_clips(s: Session, project_id: str, asset_id: str) -> bool:
    """Removes clips that use a deleted asset; the caller commits. Video clips after it move up when the
    timeline is next loaded (normalization)."""
    row = s.exec(select(Timeline).where(Timeline.project_id == project_id)).first()
    if row is None:
        return False
    raw = dict(row.tracks_json or {})
    tracks = [dict(t) for t in raw.get("tracks") or []]
    hit = False
    for t in tracks:
        clips = t.get("clips") or []
        kept = [c for c in clips if c.get("asset_id") != asset_id]
        if len(kept) != len(clips):
            t["clips"], hit = kept, True
    if hit:
        row.tracks_json = {**raw, "tracks": tracks}
        row.updated_at = utcnow()
        s.add(row)
    return hit
