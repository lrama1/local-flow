"""MMAudio video-to-audio (Section 12.1): synchronized sound effects for a take, at 44.1 kHz.

Catalog parts: "base" (hkchengrex/MMAudio: the large_44k_v2 network, the Synchformer, the 44k VAE), "clip"
(apple/DFN5B-CLIP-ViT-H-14-384, open_clip format) and "vocoder" (nvidia/bigvgan_v2_44khz_128band_512x).
MMAudio fetches the last two from the Hub by name; here they're loaded from the catalog files instead.

We use MMAudio's model classes but not its `eval_utils` (it imports training-only packages): `generate`
below follows it, and frames come from our ffmpeg rather than torchaudio/av. MMAudio is trained on 8 s
clips; longer takes are made in overlapping 8 s windows joined with crossfades (DECISIONS M6)."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import subprocess
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from localflow.services.media.ffmpeg import probe
from localflow.worker.adapters.base import Adapter, AdapterInfo, AudioResult, V2ARequest
from localflow.worker.context import JobContext
from localflow.worker.messages import WorkerConfig
from localflow.worker.spec import ModelSpec

log = logging.getLogger("localflow.worker.mmaudio")

SAMPLE_RATE = 44100
CLIP_FPS, CLIP_SIZE = 8, 384
SYNC_FPS, SYNC_SIZE = 25, 224
WINDOW_S, OVERLAP_S, SINGLE_MAX_S = 8.0, 1.0, 10.0
_DTYPE = torch.bfloat16


class MMAudioExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cfg_strength: float = Field(default=4.5, ge=1.0, le=15.0, description="Prompt guidance")


@contextlib.contextmanager
def local_helpers(clip_dir: Path, vocoder_dir: Path) -> Iterator[None]:
    """Points MMAudio's hard-coded Hub ids (CLIP, BigVGAN v2) at local folders while its models load."""
    import open_clip
    from mmaudio.ext.autoencoder import autoencoder
    from mmaudio.model.utils import features_utils

    real_vocoder = autoencoder.BigVGANv2

    class LocalBigVGAN:
        @staticmethod
        def from_pretrained(_repo_id: str, **_kw: Any) -> Any:
            return real_vocoder._from_pretrained(
                model_id=str(vocoder_dir),
                revision=None,
                cache_dir=None,
                force_download=False,
                local_files_only=True,
                token=None,
                use_cuda_kernel=False,
            )

    def local_clip(_name: str, **kw: Any) -> Any:
        return open_clip.create_model_from_pretrained(f"local-dir:{clip_dir}", **kw)

    real_clip = features_utils.create_model_from_pretrained
    autoencoder.BigVGANv2 = LocalBigVGAN
    features_utils.create_model_from_pretrained = local_clip
    try:
        yield
    finally:
        autoencoder.BigVGANv2 = real_vocoder
        features_utils.create_model_from_pretrained = real_clip


def windows(duration: float) -> list[tuple[float, float]]:
    """(start, length) windows covering `duration`: one window up to SINGLE_MAX_S, else ~8 s windows that
    overlap by at least OVERLAP_S. Starts are whole seconds, where the 8 fps CLIP and 25 fps sync frame
    grids meet, so each window's audio lines up with the frames it's conditioned on."""
    if duration <= SINGLE_MAX_S:
        return [(0.0, duration)]
    # Rounding moves a start by <= 0.5 s, so plan with a step 1 s shorter to keep the overlap.
    n = math.ceil((duration - WINDOW_S) / (WINDOW_S - OVERLAP_S - 1)) + 1
    step = (duration - WINDOW_S) / (n - 1)
    starts = [float(round(i * step)) for i in range(n - 1)] + [float(math.ceil(duration - WINDOW_S))]
    return [(s, min(WINDOW_S, duration - s)) for s in starts]


def join(pieces: list[tuple[float, np.ndarray]], total: int, sr: int) -> np.ndarray:
    """Joins overlapping windows: each one is used alone except for an OVERLAP_S equal-power crossfade at
    the middle of each overlap (independent generations would sound phasey if averaged)."""
    t = np.arange(total, dtype=np.float64) / sr
    ends = [start + len(audio) / sr for start, audio in pieces]
    mids = [(pieces[k + 1][0] + ends[k]) / 2 for k in range(len(pieces) - 1)]
    half = OVERLAP_S / 2
    out = np.zeros(total, dtype=np.float32)
    for k, (start, audio) in enumerate(pieces):
        w = np.ones(total, dtype=np.float64)
        if k > 0:
            w *= np.sin(np.pi / 2 * np.clip((t - (mids[k - 1] - half)) / OVERLAP_S, 0, 1))
        if k < len(pieces) - 1:
            w *= np.cos(np.pi / 2 * np.clip((t - (mids[k] - half)) / OVERLAP_S, 0, 1))
        a = round(start * sr)
        seg = audio[: max(0, total - a)].astype(np.float32)
        out[a : a + len(seg)] += seg * w[a : a + len(seg)].astype(np.float32)
    return out


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


class MMAudioAdapter(Adapter):
    requires_model = True
    info = AdapterInfo(
        adapter_key="mmaudio",
        adapter_version="1.0.0",
        name="MMAudio (video-to-audio)",
        capabilities=["v2a"],
        supported_base_models=["mmaudio"],
        param_schema=MMAudioExtra.model_json_schema(),
        vram_profiles={"low": {}, "balanced": {}, "quality": {}},
    )

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.net: Any = None
        self.features: Any = None
        self.device = "cuda"

    def load(self, model: ModelSpec | None, profile: str, ctx: JobContext) -> dict[str, Any]:
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        assert model is not None
        base, clip, vocoder = model.part("base", "main"), model.part("clip"), model.part("vocoder")
        if base is None or clip is None or vocoder is None:
            raise RuntimeError(
                f"{model.name}: missing files (it needs the MMAudio weights, CLIP and BigVGAN). "
                "Reinstall it from the Models page."
            )
        weights = base.path / "weights" / "mmaudio_large_44k_v2.pth"
        vae = base.path / "ext_weights" / "v1-44.pth"
        sync = base.path / "ext_weights" / "synchformer_state_dict.pth"
        for f in (
            weights,
            vae,
            sync,
            clip.path / "open_clip_config.json",
            vocoder.path / "bigvgan_generator.pt",
        ):
            if not f.is_file():
                raise RuntimeError(f"{model.name}: {f.name} is missing (reinstall it from the Models page).")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ctx.report_progress(0.0, "Loading MMAudio…", force=True)
        net = get_my_mmaudio("large_44k_v2").to(self.device, _DTYPE).eval()
        net.load_weights(torch.load(weights, map_location=self.device, weights_only=True))
        ctx.report_progress(0.0, "Loading CLIP, Synchformer and the vocoder…", force=True)
        with local_helpers(clip.path, vocoder.path):
            features = FeaturesUtils(
                tod_vae_ckpt=str(vae),
                synchformer_ckpt=str(sync),
                enable_conditions=True,
                mode="44k",
                bigvgan_vocoder_ckpt=None,
                need_vae_encoder=False,
            )
        self.net, self.features = net, features.to(self.device, _DTYPE).eval()
        return {"device": self.device, "dtype": "bf16"}

    def unload(self) -> None:
        self.net = self.features = None

    # --- frames ------------------------------------------------------------------------------------
    def _frames(self, video: str, fps: int, vf: str, size: int) -> torch.Tensor:
        """(T, 3, size, size) uint8 frames at `fps`."""
        assert self.config.ffmpeg
        cmd = [
            self.config.ffmpeg,
            "-v",
            "error",
            "-i",
            video,
            "-vf",
            f"fps={fps},{vf}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size, 3)
        return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)

    def _video(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """CLIP frames in [0, 1] (squashed to 384x384, like MMAudio) and Synchformer frames in [-1, 1]
        (short side 224, center crop)."""
        clip = self._frames(path, CLIP_FPS, f"scale={CLIP_SIZE}:{CLIP_SIZE}:flags=bicubic", CLIP_SIZE)
        short = (
            f"scale='if(gt(iw,ih),-2,{SYNC_SIZE})':'if(gt(iw,ih),{SYNC_SIZE},-2)':flags=bicubic,"
            f"crop={SYNC_SIZE}:{SYNC_SIZE}"
        )
        sync = self._frames(path, SYNC_FPS, short, SYNC_SIZE)
        return clip.float() / 255.0, sync.float() / 127.5 - 1.0

    # --- generation --------------------------------------------------------------------------------
    @torch.inference_mode()
    def _generate(
        self,
        clip: torch.Tensor,
        sync: torch.Tensor,
        duration: float,
        req: V2ARequest,
        extra: MMAudioExtra,
        rng: torch.Generator,
        ctx: JobContext,
        progress: tuple[float, float],
    ) -> np.ndarray:
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.sequence_config import CONFIG_44K

        net, fu = self.net, self.features
        seq = dataclasses.replace(CONFIG_44K, duration=duration)
        net.update_seq_lengths(seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len)
        clip = clip[: seq.clip_seq_len].unsqueeze(0).to(self.device, _DTYPE)
        sync = sync[: int(SYNC_FPS * duration)].unsqueeze(0).to(self.device, _DTYPE)
        clip_f = fu.encode_video_with_clip(clip, batch_size=40)
        sync_f = fu.encode_video_with_sync(sync, batch_size=40)
        text_f = fu.encode_text([req.prompt])
        neg = [req.negative_prompt] if req.negative_prompt else None
        conditions = net.preprocess_conditions(clip_f, sync_f, text_f)
        empty = net.get_empty_conditions(1, negative_text_features=fu.encode_text(neg) if neg else None)
        steps = req.steps or 25
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
        x0 = torch.randn(
            1, net.latent_seq_len, net.latent_dim, device=self.device, dtype=_DTYPE, generator=rng
        )
        lo, hi = progress
        done = [0]

        def ode(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            ctx.check_cancelled()
            done[0] += 1
            ctx.report_progress(lo + (hi - lo) * done[0] / steps, f"Step {done[0]}/{steps}")
            return net.ode_wrapper(t, x, conditions, empty, extra.cfg_strength)

        x1 = net.unnormalize(fm.to_data(ode, x0))
        audio = fu.vocode(fu.decode(x1))
        return audio.float().cpu().numpy().reshape(-1)

    def generate_v2a(self, req: V2ARequest, ctx: JobContext) -> AudioResult:
        extra = MMAudioExtra.model_validate(req.extra)
        info = probe(self.config.ffprobe, Path(req.video_path))
        duration = float(info.get("duration_s") or 0)
        if duration < 0.7:
            raise RuntimeError("The take is too short for sound effects (MMAudio needs at least 0.7 s).")
        ctx.report_progress(0.02, "Reading the take…", force=True)
        clip, sync = self._video(req.video_path)
        full = duration  # the output always matches the take's length
        # Whole 8 fps CLIP frames bound what the model sees (5.04 s -> 40 frames = 5.0 s).
        duration = min(duration, len(clip) / CLIP_FPS, len(sync) / SYNC_FPS)
        rng = torch.Generator(device=self.device).manual_seed(ctx.seed % (2**32))
        spans = windows(duration)
        pieces = []
        for i, (start, length) in enumerate(spans):
            lo, hi = 0.05 + 0.9 * i / len(spans), 0.05 + 0.9 * (i + 1) / len(spans)
            c0, s0 = round(start * CLIP_FPS), round(start * SYNC_FPS)
            length = min(length, (len(clip) - c0) / CLIP_FPS, (len(sync) - s0) / SYNC_FPS)
            audio = self._generate(clip[c0:], sync[s0:], length, req, extra, rng, ctx, (lo, hi))
            pieces.append((start, audio))
        made = round(duration * SAMPLE_RATE)
        out = join(pieces, made, SAMPLE_RATE)
        total = round(full * SAMPLE_RATE)
        if total > made:  # pad the last few frames' worth with a short fade instead of a hard stop
            fade = min(made, round(0.02 * SAMPLE_RATE))
            out[made - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)
            out = np.pad(out, (0, total - made))
        path = ctx.output_dir / "sfx.wav"
        write_wav(path, out, SAMPLE_RATE)
        ctx.report_progress(1.0, "Done", force=True)
        return AudioResult(
            str(path), total / SAMPLE_RATE, SAMPLE_RATE, ctx.seed, meta={"windows": len(spans)}
        )
