"""ffmpeg/ffprobe helpers shared by the API process and the worker (no torch)."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

THUMB_MAX = (480, 480)


class FFmpegError(RuntimeError):
    pass


class FFmpegCancelled(RuntimeError):
    pass


def run_ffmpeg(
    ffmpeg: str | None,
    args: Sequence[str],
    *,
    total_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
    timeout_s: float = 3600,
) -> None:
    """Run ffmpeg with `-progress pipe:1`, reporting fractional progress and honoring cancellation."""
    if not ffmpeg:
        raise FFmpegError("ffmpeg not found. Install ffmpeg or set ffmpeg_path in config.toml.")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        *args,
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
    )
    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            del stderr_lines[:-50]

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    cancelled = False
    watcher: threading.Thread | None = None
    if cancel_event is not None:

        def _watch() -> None:
            nonlocal cancelled
            while proc.poll() is None:
                if cancel_event.wait(0.1):
                    cancelled = True
                    proc.kill()
                    return

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key in ("out_time_us", "out_time_ms") and on_progress and total_s:
            try:
                seconds = int(value) / 1_000_000
            except ValueError:
                continue
            on_progress(max(0.0, min(1.0, seconds / total_s)))
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        raise FFmpegError("ffmpeg timed out") from e
    t.join(timeout=2)
    if cancelled:
        raise FFmpegCancelled()
    if rc != 0:
        raise FFmpegError(f"ffmpeg failed (exit {rc}): {''.join(stderr_lines).strip()[-2000:]}")


def encode_frames(
    ffmpeg: str | None,
    frames: Iterable[Any],
    width: int,
    height: int,
    fps: float,
    dst: Path,
    *,
    crf: int = 18,
    cancel_event: threading.Event | None = None,
) -> None:
    """RGB24 frames (anything with `tobytes()`, e.g. uint8 numpy arrays of shape HxWx3) → H.264 MP4 (Section
    10.4: libx264, yuv420p, crf 18)."""
    if not ffmpeg:
        raise FFmpegError("ffmpeg not found. Install ffmpeg or set ffmpeg_path in config.toml.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "pipe:0",
        *_H264,
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for frame in frames:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                proc.wait(timeout=10)
                raise FFmpegCancelled()
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass  # ffmpeg exited early; its exit code and stderr say why
    err = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait(timeout=600)
    if rc != 0:
        raise FFmpegError(f"ffmpeg failed to encode frames (exit {rc}): {err.strip()[-1500:]}")


_H264 = [
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
]


def concat_args(
    clips: Sequence[Path],
    durations: Sequence[float],
    dst: Path,
    *,
    width: int,
    height: int,
    fps: float,
    crossfade_frames: int = 0,
) -> list[str]:
    """ffmpeg arguments joining clips (video only) at one size and frame rate. With `crossfade_frames` > 0
    each seam is blended over that many frames (`xfade`), so the result is shorter by that much per seam."""
    norm = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,"
        f"fps={fps:g},format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS"
    )
    args: list[str] = []
    for c in clips:
        args += ["-i", str(c)]
    graph = ";".join(f"[{i}:v]{norm}[v{i}]" for i in range(len(clips)))
    if crossfade_frames and len(clips) > 1:
        fade = crossfade_frames / fps
        label, elapsed = "[v0]", 0.0
        for i in range(1, len(clips)):
            elapsed += durations[i - 1]
            offset = max(0.0, elapsed - i * fade)
            graph += f";{label}[v{i}]xfade=transition=fade:duration={fade:.4f}:offset={offset:.4f}[x{i}]"
            label = f"[x{i}]"
        final = label
    else:
        graph += ";" + "".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[out]"
        final = "[out]"
    return [*args, "-filter_complex", graph, "-map", final, *_H264, "-an", str(dst)]


def probe(ffprobe: str | None, path: Path) -> dict[str, Any]:
    """Return {width, height, duration_s, fps, has_audio, has_video, num_frames, video_codec, pix_fmt,
    audio_codec} for a media file."""
    if not ffprobe:
        raise FFmpegError("ffprobe not found. Install ffmpeg or set ffprobe_path in config.toml.")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {out.stderr.strip()[-1000:]}")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    result: dict[str, Any] = {
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": None,
        "height": None,
        "fps": None,
        "num_frames": None,
        "duration_s": None,
        "video_codec": video.get("codec_name") if video else None,
        "pix_fmt": video.get("pix_fmt") if video else None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }
    duration = data.get("format", {}).get("duration")
    if duration is not None:
        result["duration_s"] = float(duration)
    if video is not None:
        result["width"] = video.get("width")
        result["height"] = video.get("height")
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate")
        if rate and rate != "0/0":
            result["fps"] = float(Fraction(rate))
        if video.get("nb_frames", "").isdigit():
            result["num_frames"] = int(video["nb_frames"])
    return result


def video_thumbnail(ffmpeg: str | None, src: Path, dst: Path, at_s: float = 0.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ffmpeg,
        [
            "-ss",
            f"{at_s:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({THUMB_MAX[0]},iw)':-2",
            "-q:v",
            "4",
            str(dst),
        ],
    )


def image_thumbnail(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        img = im.convert("RGB")
        img.thumbnail(THUMB_MAX)
        img.save(dst, "JPEG", quality=85)


def extract_frame(
    ffmpeg: str | None, src: Path, dst: Path, at_s: float | None = None, last: bool = False
) -> None:
    """Extract one frame as PNG. `last=True` grabs the final frame (used by extend)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if last:
        # Decode the last ~0.3 s and keep overwriting the output; the final write is the last frame.
        args = ["-sseof", "-0.3", "-i", str(src), "-update", "1", str(dst)]
    else:
        args = ["-ss", f"{at_s or 0:.3f}", "-i", str(src), "-frames:v", "1", str(dst)]
    run_ffmpeg(ffmpeg, args)
