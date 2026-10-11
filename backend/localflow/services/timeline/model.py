"""Pure timeline rules (Section 15.1): sizes, the default layout, and normalization.

Video clips sit on a frame grid. A clip lasts `round((out_s - in_s) * fps)` frames, and the video track
is sequential: each clip starts where the previous one ends, minus its crossfade. So `start_s` of video
clips is derived and never trusted from the client. Audio clips keep their own `start_s` (millisecond
precision). Normalization also clamps trims to the source asset and transitions to what fits."""

from __future__ import annotations

import math
from collections.abc import Mapping

from localflow.schemas.timeline import Clip, TimelineData, Track, Transition

PRESET_SHORT_SIDE = {"720p": 720, "1080p": 1080}
DEFAULT_TRACKS = (
    ("v1", "video", "Video"),
    ("a_narration", "audio", "Narration"),
    ("a_dialogue", "audio", "Dialogue"),
    ("a_sfx", "audio", "Sound effects"),
    ("a_music", "audio", "Music"),
)
MIN_CLIP_FRAMES = 1
NARRATION_TRACK = "a_narration"
NARRATION_LEAD_S = 0.25  # a line starts this long after its shot's clip does (Section 12.3)
FOLLOW_TOLERANCE_S = 0.05


def _even(x: float) -> int:
    return max(2, 2 * round(x / 2))


def size_for(width: int, height: int, short_side: int) -> tuple[int, int]:
    """The (even) size with the given short side and the aspect ratio of width x height."""
    if width >= height:
        return _even(short_side * width / height), _even(short_side)
    return _even(short_side), _even(short_side * height / width)


def base_size(aspect_ratio: str) -> tuple[int, int]:
    """A project's timeline size: 720p in its aspect ratio (16:9 → 1280x720, 9:16 → 720x1280)."""
    try:
        w, h = (float(x) for x in aspect_ratio.split(":"))
    except ValueError:
        w, h = 16.0, 9.0
    return size_for(round(w * 1000), round(h * 1000), 720)


def preset_size(timeline: TimelineData, preset: str) -> tuple[int, int]:
    return size_for(timeline.width, timeline.height, PRESET_SHORT_SIDE[preset])


def default_timeline(aspect_ratio: str, fps: int) -> TimelineData:
    width, height = base_size(aspect_ratio)
    return TimelineData(
        fps=fps,
        width=width,
        height=height,
        tracks=[Track(id=i, kind=k, name=n) for i, k, n in DEFAULT_TRACKS],
    )


def frames(seconds: float, fps: int) -> int:
    return round(seconds * fps + 1e-9)


def with_narration_track(timeline: TimelineData) -> TimelineData:
    """Timelines saved before M8 have no Narration track: add it, first among the audio tracks."""
    if any(t.id == NARRATION_TRACK for t in timeline.tracks):
        return timeline
    tl = timeline.model_copy(deep=True)
    at = next(i for i, t in enumerate(tl.tracks) if t.kind == "video") + 1
    tl.tracks.insert(at, Track(id=NARRATION_TRACK, kind="audio", name="Narration"))
    return tl


def follow_video_moves(before: TimelineData, after: TimelineData) -> int:
    """Moves narration clips along with their shots' video clips (Section 12.3). A narration clip that was
    still at its anchor (its shot's old clip start + the lead) is re-anchored to the clip's new start; one
    the user dragged elsewhere is left alone. Mutates `after`; returns how many clips moved."""
    old = {c.shot_id: c.start_s for c in before.video.clips if c.shot_id}
    new = {c.shot_id: c.start_s for c in after.video.clips if c.shot_id}
    track = next((t for t in after.tracks if t.id == NARRATION_TRACK), None)
    moved = 0
    for c in track.clips if track else []:
        if c.shot_id not in old or c.shot_id not in new:
            continue
        if abs(c.start_s - (old[c.shot_id] + NARRATION_LEAD_S)) <= FOLLOW_TOLERANCE_S:
            target = round(new[c.shot_id] + NARRATION_LEAD_S, 3)
            if abs(target - c.start_s) > 1e-6:
                c.start_s, moved = target, moved + 1
    if moved and track:
        track.clips.sort(key=lambda x: (x.start_s, x.id))
    return moved


def clip_frames(clip: Clip, fps: int) -> int:
    return max(MIN_CLIP_FRAMES, frames(clip.out_s, fps) - frames(clip.in_s, fps))


def duration_frames(timeline: TimelineData) -> int:
    """Frames of the video track (the export's length); 0 when it's empty."""
    clips = timeline.video.clips
    if not clips:
        return 0
    last = clips[-1]
    return frames(last.start_s, timeline.fps) + clip_frames(last, timeline.fps)


def duration_s(timeline: TimelineData) -> float:
    return duration_frames(timeline) / timeline.fps


def normalize(
    timeline: TimelineData, durations: Mapping[str, float | None]
) -> tuple[TimelineData, list[str]]:
    """Returns a normalized copy and notes on what changed. `durations` maps asset id → its length (None:
    unknown, so trims aren't clamped)."""
    tl = timeline.model_copy(deep=True)
    fps, notes = tl.fps, []
    step = 1 / fps
    cursor = 0  # frames
    prev_frames: int | None = None
    for i, c in enumerate(tl.video.clips):
        src = durations.get(c.asset_id)
        last = math.floor(src * fps + 1e-6) if src else None
        a = frames(c.in_s, fps)
        b = frames(c.out_s, fps)
        if last is not None and b > last:
            b = last
            notes.append(f"Clip {i + 1}: trimmed to the take's length.")
        if last is not None and a > last - MIN_CLIP_FRAMES:
            a = max(0, last - MIN_CLIP_FRAMES)
        b = max(b, a + MIN_CLIP_FRAMES)
        c.in_s, c.out_s = round(a * step, 6), round(b * step, 6)
        n = b - a

        t = c.transition_in
        k = frames(t.duration_s, fps)
        if t.type == "crossfade" and prev_frames is None:
            t = Transition(type="fade_from_black", duration_s=t.duration_s)
            notes.append("The first clip can't crossfade; it fades in from black instead.")
        if t.type == "cut" or k == 0:
            t, k = Transition(type="cut"), 0
        elif t.type == "crossfade":
            assert prev_frames is not None
            fit = min(prev_frames, n) // 2  # leaves room for the neighbours' own crossfades
            if k > fit:
                k = fit
                notes.append(f"Clip {i + 1}: crossfade shortened to fit the clips.")
            if k == 0:
                t = Transition(type="cut")
        else:  # fade_from_black happens inside the clip
            k = min(k, n)
        c.transition_in = Transition(type=t.type, duration_s=round(k * step, 6))

        start = cursor - (k if c.transition_in.type == "crossfade" else 0)
        c.start_s = round(start * step, 6)
        cursor = start + n
        prev_frames = n
        c.gain_db = round(c.gain_db, 2)

    for track in tl.audio:
        for c in track.clips:
            src = durations.get(c.asset_id)
            if src and c.out_s > src + 1e-3:
                c.out_s = src
                notes.append(f"{track.name or track.id}: a clip was trimmed to its audio's length.")
            c.in_s = min(round(c.in_s, 3), max(0.0, round(c.out_s, 3) - 0.01))
            c.out_s = max(round(c.out_s, 3), c.in_s + 0.01)
            c.start_s = round(c.start_s, 3)
            c.transition_in = Transition()
        track.clips.sort(key=lambda x: (x.start_s, x.id))
    return tl, notes
