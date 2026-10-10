"""Shared text-to-speech logic (Section 12): voice params, dialogue layout and fitting lines into a clip.

TTS adapters implement `speak()` (one utterance -> mono float samples); `SpeechAdapter.synthesize` turns a
`TTSRequest` (a single text, or dialogue `lines`) into one WAV. Importable without numpy/torch, because the
FakeEngine uses it too."""

from __future__ import annotations

import math
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from localflow.worker.adapters.base import Adapter, AudioResult, SpeechLine, TTSRequest

if TYPE_CHECKING:
    from localflow.worker.context import JobContext

MIN_LEAD_S = 0.1
MIN_GAP_S = 0.15
TAIL_S = 0.25  # silence after the last line when the output isn't clip-length


class VoiceMix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice: str = Field(description="Native id of a built-in voice of the same adapter and language")
    weight: float = Field(ge=0.05, le=1.0)


class VoiceParams(BaseModel):
    """Custom-voice params for blend-capable TTS (Kokoro): the base voice, optionally blended with up to two
    other voices, at a speaking rate. Weights are relative: (base_weight, *mix weights) are normalized."""

    model_config = ConfigDict(extra="forbid")

    mix: list[VoiceMix] = Field(
        default_factory=list, max_length=2, description="Other voices blended into the base voice"
    )
    base_weight: float = Field(default=1.0, ge=0.2, le=1.0)
    speed: float = Field(default=1.0, ge=0.7, le=1.4, description="Speaking rate (1 = natural)")

    def weights(self, base: str) -> list[tuple[str, float]]:
        parts = [(base, self.base_weight), *((m.voice, m.weight) for m in self.mix if m.voice != base)]
        total = sum(w for _, w in parts)
        return [(v, w / total) for v, w in parts]


@dataclass(frozen=True)
class BuiltinVoice:
    native_id: str
    name: str
    language: str  # e.g. "en-US"
    gender: str  # female | male
    quality: str = ""  # the model author's grade, if any (Kokoro: A..F)
    traits: str = ""  # short description used for suggestions and the voice designer

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rendered:
    line: SpeechLine
    samples: array  # float32 mono
    speed: float

    def seconds(self, sr: int) -> float:
        return len(self.samples) / sr


class SpeechAdapter(Adapter):
    """Base for TTS adapters. Subclasses set `sample_rate` and `builtin_voices` and implement `speak`."""

    sample_rate: ClassVar[int] = 24000
    builtin_voices: ClassVar[tuple[BuiltinVoice, ...]] = ()

    def speak(self, text: str, voice: str, params: VoiceParams, speed: float, ctx: JobContext) -> array:
        """One utterance -> mono float samples in [-1, 1] at `sample_rate`."""
        raise NotImplementedError

    def default_voice(self) -> str:
        return self.builtin_voices[0].native_id

    # --- synthesis ---------------------------------------------------------------------------------
    def synthesize(self, req: TTSRequest, ctx: JobContext) -> AudioResult:
        lines = req.lines or [
            SpeechLine(text=req.text, voice_id=req.voice_id, voice_params=req.voice_params, speed=req.speed)
        ]
        sr = self.sample_rate
        rendered = self._render_all(lines, req, ctx, factor=1.0, offset=0.0, weight=0.5 if req.lines else 0.9)
        factor = 1.0
        if req.duration_s and req.lines and req.max_speed > 1.0:
            factor = fit_factor(
                [r.seconds(sr) for r in rendered], req.duration_s, req.lead_s, req.gap_s, req.max_speed
            )
            if factor > 1.0:
                ctx.report_progress(0.5, f"Speeding up {factor:.2f}x to fit the clip", force=True)
                rendered = self._render_all(lines, req, ctx, factor=factor, offset=0.5, weight=0.4)
        placed = layout(
            [r.seconds(sr) for r in rendered],
            [r.line.start_s for r in rendered],
            req.duration_s if req.lines else None,
            req.lead_s if req.lines else 0.0,
            req.gap_s,
        )
        speech_end = max((end for _, end in placed), default=0.0)
        total = max(req.duration_s or 0.0, speech_end + (TAIL_S if req.lines and not req.duration_s else 0))
        if not req.lines:
            total = max(total, speech_end)
        track = array("f", bytes(4 * max(1, round(total * sr))))
        for r, (start, _end) in zip(rendered, placed, strict=True):
            i = round(start * sr)
            seg = r.samples[: max(0, len(track) - i)]
            track[i : i + len(seg)] = seg
        out = ctx.output_dir / ("dialogue.wav" if req.lines else "speech.wav")
        write_wav(out, track, sr)
        overflow = max(0.0, speech_end - req.duration_s) if req.duration_s else 0.0
        meta = {
            "speed_factor": round(factor, 3),
            "speech_end_s": round(speech_end, 3),
            "overflow_s": round(overflow, 3),
            "lines": [
                {"key": r.line.key, "start_s": round(s, 3), "end_s": round(e, 3), "speed": round(r.speed, 3)}
                for r, (s, e) in zip(rendered, placed, strict=True)
            ],
        }
        ctx.report_progress(1.0, "Done", force=True)
        return AudioResult(str(out), len(track) / sr, sr, ctx.seed, meta=meta)

    def _render_all(
        self,
        lines: list[SpeechLine],
        req: TTSRequest,
        ctx: JobContext,
        *,
        factor: float,
        offset: float,
        weight: float,
    ) -> list[Rendered]:
        out = []
        for n, line in enumerate(lines):
            ctx.check_cancelled()
            params = VoiceParams.model_validate(line.voice_params or {})
            voice = line.voice_id or self.default_voice()
            speed = max(0.5, min(2.0, line.speed * params.speed * factor))
            label = f"Speaking line {n + 1}/{len(lines)}" if len(lines) > 1 else "Speaking"
            ctx.report_progress(offset + weight * n / len(lines), label)
            samples = normalize_peak(
                trim_silence(self.speak(line.text, voice, params, speed, ctx), self.sample_rate)
            )
            out.append(Rendered(line, samples, speed))
        return out


def fit_factor(
    lengths: list[float], duration_s: float, lead_s: float, gap_s: float, max_speed: float
) -> float:
    """Speed-up (>= 1) that makes sequential lines end within the clip, capped at `max_speed`. Lead-in and
    gaps shrink to their minimums before any speed-up."""
    if not lengths:
        return 1.0
    speech = sum(lengths)
    room = duration_s - min(lead_s, MIN_LEAD_S) - min(gap_s, MIN_GAP_S) * (len(lengths) - 1)
    if speech <= room:
        return 1.0
    if room <= 0:
        return max_speed
    return min(max_speed, speech / room)


def layout(
    lengths: list[float], starts: list[float | None], duration_s: float | None, lead_s: float, gap_s: float
) -> list[tuple[float, float]]:
    """(start, end) per line. Explicit starts win; others follow the previous line after `gap_s`. With a clip
    duration, lead-in and gaps shrink (down to MIN_LEAD_S / MIN_GAP_S) before lines run past the end."""
    n = len(lengths)
    if duration_s and n:
        speech = sum(lengths)
        lead, gap = lead_s, gap_s
        if lead + speech + gap * (n - 1) > duration_s:
            spare = max(0.0, duration_s - speech)
            want = lead_s + gap_s * (n - 1)
            scale = spare / want if want else 0.0
            lead = max(min(lead_s, MIN_LEAD_S), lead_s * scale)
            gap = max(min(gap_s, MIN_GAP_S), gap_s * scale)
        lead_s, gap_s = lead, gap
    out: list[tuple[float, float]] = []
    t = lead_s
    for length, start in zip(lengths, starts, strict=True):
        s = start if start is not None else t
        out.append((s, s + length))
        t = s + length + gap_s
    return out


def trim_silence(
    samples: array, sample_rate: int, threshold: float = 0.01, lead_s: float = 0.03, tail_s: float = 0.08
) -> array:
    """Cuts the model's own padding (Kokoro: ~0.3 s before, ~0.5 s after each utterance; DECISIONS M6), so
    the layout's lead-in and gaps set the timing. Keeps a short pad so consonants aren't clipped."""
    n = len(samples)
    first = next((i for i in range(n) if abs(samples[i]) > threshold), None)
    if first is None:
        return samples
    last = next(i for i in range(n - 1, -1, -1) if abs(samples[i]) > threshold)
    a = max(0, first - round(lead_s * sample_rate))
    b = min(n, last + 1 + round(tail_s * sample_rate))
    return samples[a:b]


def normalize_peak(samples: array, peak: float = 0.89, max_gain: float = 4.0) -> array:
    """Raises quiet speech to `peak` (-1 dBFS), at most `max_gain`x; never turns it down."""
    top = max((abs(x) for x in samples), default=0.0)
    if top <= 0 or top >= peak:
        return samples
    gain = min(max_gain, peak / top)
    return array("f", (x * gain for x in samples))


def write_wav(path: Path, samples: array, sample_rate: int) -> None:
    pcm = array("h", (int(max(-1.0, min(1.0, x)) * 32767) for x in samples))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def tone(seconds: float, freq: float, sample_rate: int, volume: float = 0.4) -> array:
    """A syllable-like tone (FakeEngine speech): a sine with a 6 Hz tremolo and short fades."""
    n = max(1, round(seconds * sample_rate))
    fade = min(n // 4, round(0.02 * sample_rate))
    two_pi = 2 * math.pi
    out = array("f", bytes(4 * n))
    for i in range(n):
        t = i / sample_rate
        env = 1.0 if fade == 0 else min(1.0, i / fade, (n - 1 - i) / fade)
        trem = 0.6 + 0.4 * math.sin(two_pi * 6 * t)
        out[i] = volume * env * trem * math.sin(two_pi * freq * t)
    return out
