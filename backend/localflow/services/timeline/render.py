"""Timeline → one ffmpeg command (Section 15.3). Pure: no I/O, so the command is unit-tested as is.

Video, frame-exact on the timeline's grid:
- Each clip goes to the timeline fps first, then is trimmed by frame number. It's fitted to the output
  size: scale-and-crop when the aspect ratio is within 3% (Wan's 1280x704 in a 1280x720 frame loses
  8 px rows rather than gaining black bars), else scale-and-pad.
- Clips are joined pairwise: `concat` for cuts, `xfade` at the exact frame offset for crossfades. A fade
  from black is a `fade` inside the clip.

Audio:
- Every audio clip and, unless the video track is muted, each video clip's own sound is trimmed,
  converted to 48 kHz stereo, set to its clip + track gain, and delayed to its start.
- These are mixed over silence of exactly the video's length, so the output always has an AAC track that
  ends with the picture. `loudnorm` (EBU R128, -16 LUFS) is optional.
- Ducking (Section 12.3): when the Narration track has clips and `duck` is on, the music, the sound
  effects and the video clips' own sound are mixed into a bus that a sidechain compressor, keyed by the
  narration, pulls down while it speaks. The narration and dialogue aren't ducked. Without narration, or
  with `duck` off, the mix is the plain one."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from localflow.schemas.timeline import TimelineData
from localflow.services.timeline.model import NARRATION_TRACK, clip_frames, frames, preset_size

SAMPLE_RATE = 48_000
_ASPECT_TOLERANCE = 0.03
_BACKING_TRACKS = ("a_sfx", "a_music")  # ducked under the narration, with the video clips' own sound
# About 8 dB of reduction on typical speech; it recovers over 0.4 s so music doesn't pump.
DUCK_FILTER = "sidechaincompress=threshold=0.03:ratio=5:attack=20:release=400:makeup=1"


@dataclass(frozen=True)
class Source:
    path: Path
    duration_s: float | None = None
    has_audio: bool = False
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ExportPlan:
    args: list[str]
    width: int
    height: int
    fps: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps


class EmptyTimeline(ValueError):
    pass


def fit_filter(w: int | None, h: int | None, width: int, height: int) -> str:
    if w and h and abs((w / h) / (width / height) - 1) <= _ASPECT_TOLERANCE:
        fit = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    else:
        fit = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    return f"{fit},setsar=1"


def _db(gain: float) -> str:
    return f"{gain:.2f}dB"


def _audio_chain(index: int, in_s: float, out_s: float, start_s: float, gain_db: float) -> str:
    delay = round(start_s * 1000)
    return (
        f"[{index}:a]atrim=start={in_s:.6f}:end={out_s:.6f},asetpts=PTS-STARTPTS,"
        f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo,"
        f"volume={_db(gain_db)},adelay=delays={delay}:all=1"
    )


def build_export(
    timeline: TimelineData,
    sources: Mapping[str, Source],
    preset: str,
    dst: Path,
    *,
    loudnorm: bool = False,
    duck: bool = False,
) -> ExportPlan:
    """`timeline` must be normalized; `sources` maps asset id → its file."""
    fps = timeline.fps
    width, height = preset_size(timeline, preset)
    video = timeline.video
    if not video.clips:
        raise EmptyTimeline("The timeline has no video clips.")

    args: list[str] = []
    graph: list[str] = []
    inputs: dict[str, int] = {}

    def add_input(clip_id: str, path: Path) -> int:
        inputs[clip_id] = len(inputs)
        args.extend(["-i", str(path)])
        return inputs[clip_id]

    # --- video ------------------------------------------------------------------------------------
    total = 0
    acc = ""
    for i, c in enumerate(video.clips):
        src = sources[c.asset_id]
        k_in = add_input(c.id, src.path)
        a, n = frames(c.in_s, fps), clip_frames(c, fps)
        chain = (
            f"[{k_in}:v]fps={fps},trim=start_frame={a}:end_frame={a + n},setpts=PTS-STARTPTS,"
            f"{fit_filter(src.width, src.height, width, height)},format=yuv420p"
        )
        k = frames(c.transition_in.duration_s, fps)
        if c.transition_in.type == "fade_from_black" and k:
            chain += f",fade=t=in:s=0:n={k}"
        graph.append(f"{chain},settb=AVTB[v{i}]")
        if i == 0:
            acc, total = "[v0]", n
            continue
        out = f"[x{i}]"
        if c.transition_in.type == "crossfade" and k:
            offset = (total - k) / fps
            graph.append(f"{acc}[v{i}]xfade=transition=fade:duration={k / fps:.6f}:offset={offset:.6f}{out}")
            total += n - k
        else:
            graph.append(f"{acc}[v{i}]concat=n=2:v=1:a=0{out}")
            total += n
        acc = out
    graph.append(f"{acc}fps={fps},trim=end_frame={total},setpts=PTS-STARTPTS[vout]")
    seconds = total / fps

    # --- audio ------------------------------------------------------------------------------------
    parts: list[str] = []  # every labelled stream, in mix order
    narration: list[str] = []
    backing: list[str] = []  # what ducks under the narration
    if not video.muted:
        for c in video.clips:
            src = sources[c.asset_id]
            if src.has_audio:
                label = f"[ea{len(parts)}]"
                chain = _audio_chain(inputs[c.id], c.in_s, c.out_s, c.start_s, c.gain_db + video.gain_db)
                graph.append(chain + label)
                parts.append(label)
                backing.append(label)
    for track in timeline.audio:
        if track.muted:
            continue
        for c in track.clips:
            if c.start_s >= seconds:
                continue  # starts after the picture ends
            k_in = add_input(c.id, sources[c.asset_id].path)
            label = f"[ta{len(parts)}]"
            graph.append(_audio_chain(k_in, c.in_s, c.out_s, c.start_s, c.gain_db + track.gain_db) + label)
            parts.append(label)
            if track.id == NARRATION_TRACK:
                narration.append(label)
            elif track.id in _BACKING_TRACKS:
                backing.append(label)
    silence = (
        f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=duration={seconds:.6f},"
        f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"
    )
    graph.append(f"{silence}[abase]")
    mix = parts
    if duck and narration and backing:
        # The narration bus runs the whole length (over silence) so the sidechain never ends early.
        graph.append(f"{silence}[nbase]")
        graph.append(
            f"[nbase]{''.join(narration)}amix=inputs={len(narration) + 1}:duration=first:"
            "dropout_transition=0:normalize=0,asplit=2[nmix][nkey]"
        )
        if len(backing) == 1:
            graph.append(f"{backing[0]}anull[bbus]")
        else:
            graph.append(
                f"{''.join(backing)}amix=inputs={len(backing)}:dropout_transition=0:normalize=0[bbus]"
            )
        graph.append(f"[bbus][nkey]{DUCK_FILTER}[bduck]")
        taken = set(narration) | set(backing)
        mix = [p for p in parts if p not in taken] + ["[nmix]", "[bduck]"]
    mixed = "[abase]"
    if mix:
        graph.append(
            f"[abase]{''.join(mix)}amix=inputs={len(mix) + 1}:duration=first:dropout_transition=0:"
            "normalize=0[amix]"
        )
        mixed = "[amix]"
    tail = f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={SAMPLE_RATE}," if loudnorm else ""
    graph.append(f"{mixed}{tail}atrim=duration={seconds:.6f}[aout]")

    args += [
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(SAMPLE_RATE),
        "-movflags",
        "+faststart",
        str(dst),
    ]
    return ExportPlan(args=args, width=width, height=height, fps=fps, frames=total)
