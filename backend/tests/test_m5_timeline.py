"""M5: timeline model, auto-assemble, the ffmpeg export command, and export jobs (Section 15)."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.schemas.timeline import Clip, TimelineData, Transition
from localflow.services.timeline.model import base_size, default_timeline, duration_s, normalize, size_for
from localflow.services.timeline.render import EmptyTimeline, Source, build_export, fit_filter
from tests.conftest import TOKEN, make_settings, wait_for, wait_job, wait_worker_idle
from tests.test_m3_imagegen import ok

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg not installed")


def timeline(*video: Clip, audio: list[Clip] | None = None) -> TimelineData:
    tl = default_timeline("16:9", 24)
    tl.video.clips = list(video)
    if audio:
        tl.audio[0].clips = audio
    return tl


def streams(path: Path) -> dict[str, Any]:
    assert FFPROBE
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    return {
        "width": v["width"],
        "height": v["height"],
        "frames": int(v["nb_read_frames"]),
        "fps": v["r_frame_rate"],
        "audio": a and {"codec": a["codec_name"], "rate": int(a["sample_rate"]), "channels": a["channels"]},
        "duration": float(data["format"]["duration"]),
    }


# --- model ---------------------------------------------------------------------------------------------
def test_sizes() -> None:
    assert base_size("16:9") == (1280, 720)
    assert base_size("9:16") == (720, 1280)
    assert base_size("1:1") == (720, 720)
    assert size_for(1280, 720, 1080) == (1920, 1080)
    assert size_for(720, 1280, 1080) == (1080, 1920)
    tl = default_timeline("16:9", 24)
    assert [(t.id, t.kind) for t in tl.tracks] == [
        ("v1", "video"),
        ("a_dialogue", "audio"),
        ("a_sfx", "audio"),
        ("a_music", "audio"),
    ]


def test_normalize_sequences_video_on_the_frame_grid() -> None:
    tl = timeline(
        Clip(
            id="a",
            asset_id="A",
            in_s=0,
            out_s=2.0,
            start_s=99,
            transition_in=Transition(type="crossfade", duration_s=1),
        ),
        Clip(
            id="b",
            asset_id="B",
            in_s=0.51,
            out_s=3.0,
            transition_in=Transition(type="crossfade", duration_s=0.25),
        ),
        Clip(
            id="c", asset_id="C", in_s=0, out_s=9.0, transition_in=Transition(type="crossfade", duration_s=3)
        ),
        Clip(id="d", asset_id="D", in_s=0, out_s=1.0, transition_in=Transition(type="cut", duration_s=2)),
    )
    out, notes = normalize(tl, {"A": 5.0, "B": 2.5, "C": 4.0, "D": None})
    a, b, c, d = out.video.clips
    # The first clip can't crossfade: it fades in from black instead.
    assert a.transition_in.type == "fade_from_black" and a.start_s == 0 and a.transition_in.duration_s == 1
    # B: in snapped to frame 12, out clamped to the take (2.5 s = 60 frames), crossfade 6 frames.
    assert (b.in_s, b.out_s) == (0.5, 2.5) and b.start_s == pytest.approx(42 / 24)
    # C: trimmed to 4 s; its 3 s crossfade shrinks to half the shorter neighbour (48 frames // 2).
    assert c.out_s == 4.0 and c.transition_in.duration_s == 1.0
    assert c.start_s == pytest.approx((42 + 48 - 24) / 24)
    # A cut has no duration; D starts where C ends.
    assert d.transition_in.duration_s == 0 and d.start_s == pytest.approx((66 + 96) / 24)
    assert duration_s(out) == pytest.approx((162 + 24) / 24)
    assert any("fades in from black" in n for n in notes)
    assert any("trimmed" in n for n in notes) and any("shortened" in n for n in notes)
    again, more = normalize(out, {"A": 5.0, "B": 2.5, "C": 4.0, "D": None})
    assert again == out and more == []  # idempotent


def test_normalize_keeps_audio_where_it_was_placed() -> None:
    tl = timeline(
        Clip(asset_id="V", out_s=2),
        audio=[
            Clip(id="late", asset_id="X", start_s=3.25, out_s=9),
            Clip(id="early", asset_id="Y", start_s=0.5, out_s=1),
        ],
    )
    out, notes = normalize(tl, {"V": 2.0, "X": 1.5, "Y": 4.0})
    assert [c.id for c in out.audio[0].clips] == ["early", "late"]  # sorted by start
    late = out.audio[0].clips[1]
    assert (late.start_s, late.out_s) == (3.25, 1.5) and notes


def test_timeline_validation() -> None:
    with pytest.raises(ValueError, match="exactly one video track"):
        TimelineData(tracks=[])
    tl = default_timeline("16:9", 24).model_dump()
    tl["tracks"].append({"id": "a4", "kind": "audio"})
    with pytest.raises(ValueError, match="at most 3 audio tracks"):
        TimelineData.model_validate(tl)
    with pytest.raises(ValueError, match="out_s must be after in_s"):
        Clip(asset_id="A", in_s=2, out_s=1)


# --- export command ------------------------------------------------------------------------------------
def test_fit_filter_crops_near_matches_and_pads_others() -> None:
    assert fit_filter(1280, 704, 1280, 720).startswith(
        "scale=1280:720:force_original_aspect_ratio=increase,crop="
    )
    assert "pad=1280:720" in fit_filter(720, 1280, 1280, 720)
    assert "pad=" in fit_filter(None, None, 1280, 720)


def test_export_command() -> None:
    tl = timeline(
        Clip(
            id="c1", asset_id="A", out_s=2.0, transition_in=Transition(type="fade_from_black", duration_s=0.5)
        ),
        Clip(
            id="c2",
            asset_id="B",
            in_s=0.5,
            out_s=2.0,
            transition_in=Transition(type="crossfade", duration_s=0.25),
        ),
        audio=[Clip(id="c3", asset_id="C", start_s=1.0, out_s=1.0, gain_db=-6)],
    )
    tl, _ = normalize(tl, {"A": 4.0, "B": 4.0, "C": 3.0})
    sources = {
        "A": Source(Path("a.mp4"), 4.0, False, 1280, 704),
        "B": Source(Path("b.mp4"), 4.0, True, 720, 1280),
        "C": Source(Path("c.wav"), 3.0, True),
    }
    plan = build_export(tl, sources, "720p", Path("out.mp4"))
    assert (plan.width, plan.height, plan.fps, plan.frames) == (1280, 720, 24, 78)
    assert plan.duration_s == 3.25
    aformat = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    graph = [
        "[0:v]fps=24,trim=start_frame=0:end_frame=48,setpts=PTS-STARTPTS,"
        "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,setsar=1,format=yuv420p,"
        "fade=t=in:s=0:n=12,settb=AVTB[v0]",
        "[1:v]fps=24,trim=start_frame=12:end_frame=48,setpts=PTS-STARTPTS,"
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1,format=yuv420p,settb=AVTB[v1]",
        "[v0][v1]xfade=transition=fade:duration=0.250000:offset=1.750000[x1]",
        "[x1]fps=24,trim=end_frame=78,setpts=PTS-STARTPTS[vout]",
        f"[1:a]atrim=start=0.500000:end=2.000000,asetpts=PTS-STARTPTS,{aformat},volume=0.00dB,"
        "adelay=delays=1750:all=1[ea0]",
        f"[2:a]atrim=start=0.000000:end=1.000000,asetpts=PTS-STARTPTS,{aformat},volume=-6.00dB,"
        "adelay=delays=1000:all=1[ta1]",
        f"anullsrc=r=48000:cl=stereo,atrim=duration=3.250000,{aformat}[abase]",
        "[abase][ea0][ta1]amix=inputs=3:duration=first:dropout_transition=0:normalize=0[amix]",
        "[amix]atrim=duration=3.250000[aout]",
    ]
    assert plan.args == [
        "-i", "a.mp4", "-i", "b.mp4", "-i", "c.wav",
        "-filter_complex", ";".join(graph),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "out.mp4",
    ]  # fmt: skip

    # Muting the video track drops the clips' own sound; a muted audio track drops its clips; 1080p.
    tl.video.muted = True
    tl.audio[0].muted = True
    plan = build_export(tl, sources, "1080p", Path("out.mp4"), loudnorm=True)
    graph_1080 = plan.args[plan.args.index("-filter_complex") + 1]
    assert "[ea" not in graph_1080 and "[ta" not in graph_1080 and "amix" not in graph_1080
    assert "scale=1920:1080" in graph_1080 and "[abase]loudnorm=I=-16:TP=-1.5:LRA=11" in graph_1080
    with pytest.raises(EmptyTimeline):
        build_export(default_timeline("16:9", 24), {}, "720p", Path("o.mp4"))


@needs_ffmpeg
def test_export_command_renders_frame_exact(tmp_path: Path) -> None:
    """Real ffmpeg: mixed sizes and frame rates, a crossfade, a fade-in, embedded and separate audio."""
    assert FFMPEG

    def make(name: str, lavfi: str, *extra: str) -> Path:
        p = tmp_path / name
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", lavfi, *extra, str(p)], check=True)
        return p

    a = make("a.mp4", "testsrc2=size=640x352:rate=24:duration=2", "-pix_fmt", "yuv420p")
    b = make("b.mp4", "testsrc2=size=360x640:rate=16:duration=2", "-pix_fmt", "yuv420p")
    c = make("c.wav", "sine=frequency=440:sample_rate=24000:duration=1.5")
    tl = timeline(
        Clip(
            id="c1", asset_id="A", out_s=2.0, transition_in=Transition(type="fade_from_black", duration_s=0.5)
        ),
        Clip(id="c2", asset_id="B", out_s=2.0, transition_in=Transition(type="crossfade", duration_s=0.5)),
        Clip(id="c3", asset_id="A", in_s=1.0, out_s=2.0),
        audio=[Clip(id="s", asset_id="C", start_s=0.5, out_s=1.5)],
    )
    tl, _ = normalize(tl, {"A": 2.0, "B": 2.0, "C": 1.5})
    sources = {
        "A": Source(a, 2.0, False, 640, 352),
        "B": Source(b, 2.0, False, 360, 640),
        "C": Source(c, 1.5, True),
    }
    dst = tmp_path / "out.mp4"
    plan = build_export(tl, sources, "720p", dst)
    assert plan.frames == 48 + 48 - 12 + 24
    proc = subprocess.run([FFMPEG, "-v", "error", "-y", *plan.args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    info = streams(dst)
    assert (info["width"], info["height"], info["fps"]) == (1280, 720, "24/1")
    assert info["frames"] == plan.frames
    assert info["audio"] == {"codec": "aac", "rate": 48000, "channels": 2}
    assert abs(info["duration"] - plan.duration_s) < 0.05


# --- API -------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def c(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    with TestClient(
        create_app(make_settings(tmp_path_factory.mktemp("m5"))), headers={"X-LocalFlow-Token": TOKEN}
    ) as client:
        wait_worker_idle(client)
        yield client


def fake_video(c: TestClient, pid: str, seconds: float, **kw: Any) -> dict[str, Any]:
    body = {
        "prompt": "x",
        "project_id": pid,
        "duration_s": seconds,
        "width": 320,
        "height": 176,
        "fps": 24,
        **kw,
    }
    job = ok(c, "POST", "/api/generate/t2v", 201, json=body)
    done = wait_job(c, job["id"])
    assert done["status"] == "completed", done
    return ok(c, "GET", f"/api/assets/{done['result']['asset_ids'][0]}")  # type: ignore[no-any-return]


def six_shot_project(c: TestClient) -> dict[str, Any]:
    """Two scenes, six shots, mixed durations/sizes/fps; one take has its own audio."""
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Six shots", "aspect_ratio": "16:9"})["id"]
    specs = [
        [(2.0, {}), (1.5, {"with_audio": True}), (2.5, {"width": 176, "height": 320})],
        [(2.0, {"fps": 16}), (3.0, {}), (1.0, {})],
    ]
    shots: list[str] = []
    takes: list[dict[str, Any]] = []
    for n, scene_specs in enumerate(specs):
        scene = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": f"Scene {n + 1}"})
        for seconds, kw in scene_specs:
            sh = ok(c, "POST", f"/api/scenes/{scene['id']}/shots", 201, json={"duration_s": seconds})
            take = fake_video(c, pid, seconds, **kw)
            ok(c, "POST", f"/api/shots/{sh['id']}/select", json={"slot": "video", "asset_id": take["id"]})
            shots.append(sh["id"])
            takes.append(take)
    return {"pid": pid, "shots": shots, "takes": takes}


def test_empty_timeline_and_validation(c: TestClient) -> None:
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Empty", "aspect_ratio": "9:16", "fps": 30})["id"]
    out = ok(c, "GET", f"/api/projects/{pid}/timeline")
    assert out["duration_s"] == 0 and out["updated_at"] is None
    assert (out["timeline"]["width"], out["timeline"]["height"], out["timeline"]["fps"]) == (720, 1280, 30)
    tl = out["timeline"]
    tl["tracks"][0]["clips"] = [{"asset_id": "nope", "out_s": 1}]
    r = c.put(f"/api/projects/{pid}/timeline", json=tl)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_asset"
    r = c.post(f"/api/projects/{pid}/export", json={"preset": "720p"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_timeline"


def test_six_shot_project_assembles_and_exports(c: TestClient) -> None:
    """M5 acceptance: auto-assemble 6 shots and export a valid MP4 of the right length."""
    p = six_shot_project(c)
    pid = p["pid"]
    unfinished = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "Later"})
    later = ok(c, "POST", f"/api/scenes/{unfinished['id']}/shots", 201, json={"duration_s": 2})

    out = ok(c, "POST", f"/api/projects/{pid}/timeline/auto-assemble", json={})
    clips = out["timeline"]["tracks"][0]["clips"]
    assert [x["shot_id"] for x in clips] == p["shots"]
    assert [x["asset_id"] for x in clips] == [t["id"] for t in p["takes"]]
    assert [x["transition_in"]["type"] for x in clips] == [
        "fade_from_black",
        "cut",
        "cut",
        "crossfade",
        "cut",
        "cut",
    ]
    assert out["skipped"] == [{"shot_id": later["id"], "reason": "No take selected."}]
    frames = [48, 36, 60, 48, 72, 24]
    assert out["duration_s"] == pytest.approx((sum(frames) - 12) / 24)
    assert clips[3]["start_s"] == pytest.approx((48 + 36 + 60 - 12) / 24)
    assert ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"] == out["timeline"]  # persisted

    job = ok(c, "POST", f"/api/projects/{pid}/export", 201, json={"preset": "720p"})
    assert job["type"] == "export" and job["target_ref"] == {"kind": "timeline", "project_id": pid}
    done = wait_job(c, job["id"], timeout=120)
    assert done["status"] == "completed", done
    asset = ok(c, "GET", f"/api/assets/{done['result']['asset_ids'][0]}")
    assert asset["source"] == "export" and asset["type"] == "video" and asset["thumb_url"]
    assert asset["generation"]["preset"] == "720p" and asset["generation"]["clips"] == 6
    exports = ok(c, "GET", f"/api/projects/{pid}/assets", params={"source": "export"})
    assert [a["id"] for a in exports["items"]] == [asset["id"]]

    if FFPROBE:
        path = c.app.state.services.assets.file_path(c.app.state.services.assets.get(asset["id"]))  # type: ignore[attr-defined]
        assert path.parent.name == "exports"
        info = streams(path)
        transitions = 2  # the fade-in and the scene crossfade
        assert (info["width"], info["height"]) == (1280, 720)
        assert abs(info["frames"] - out["duration_s"] * 24) <= transitions
        assert abs(info["duration"] - out["duration_s"]) <= transitions / 24 + 0.05
        assert info["audio"] and info["audio"]["codec"] == "aac"


def test_audio_clips_and_asset_deletion(c: TestClient) -> None:
    p = six_shot_project(c)
    pid = p["pid"]
    tl = ok(
        c,
        "POST",
        f"/api/projects/{pid}/timeline/auto-assemble",
        json={"fade_in": False, "scene_transition": "cut"},
    )
    assert tl["duration_s"] == pytest.approx(288 / 24)
    speech = wait_job(
        c, ok(c, "POST", "/api/generate/tts", 201, json={"text": "Hello there", "project_id": pid})["id"]
    )
    voice = speech["result"]["asset_ids"][0]
    timeline = tl["timeline"]
    timeline["tracks"][1]["clips"] = [{"asset_id": voice, "start_s": 1.0, "out_s": 60, "gain_db": -3}]
    saved = ok(c, "PUT", f"/api/projects/{pid}/timeline", json=timeline)
    dialogue = saved["timeline"]["tracks"][1]["clips"][0]
    assert dialogue["out_s"] < 60 and any("trimmed" in n for n in saved["notes"])
    # A video on an audio track is refused.
    bad = json.loads(json.dumps(timeline))
    bad["tracks"][2]["clips"] = [{"asset_id": p["takes"][0]["id"], "out_s": 1}]
    assert c.put(f"/api/projects/{pid}/timeline", json=bad).json()["error"]["code"] == "invalid_asset"

    # Deleting the second take removes its clip; the clips after it move up.
    assert c.delete(f"/api/assets/{p['takes'][1]['id']}").status_code == 204
    after = ok(c, "GET", f"/api/projects/{pid}/timeline")
    clips = after["timeline"]["tracks"][0]["clips"]
    assert len(clips) == 5 and clips[1]["asset_id"] == p["takes"][2]["id"]
    assert clips[1]["start_s"] == pytest.approx(2.0)
    assert after["duration_s"] == pytest.approx((288 - 36) / 24)
    assert after["timeline"]["tracks"][1]["clips"][0]["asset_id"] == voice


def test_export_cancel(c: TestClient) -> None:
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Long"})["id"]
    long = fake_video(c, pid, 20.0)
    tl = ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"]
    tl["tracks"][0]["clips"] = [{"asset_id": long["id"], "out_s": 20} for _ in range(12)]
    ok(c, "PUT", f"/api/projects/{pid}/timeline", json=tl)
    job = ok(c, "POST", f"/api/projects/{pid}/export", 201, json={"preset": "1080p"})
    wait_for(
        lambda: (j := c.get(f"/api/jobs/{job['id']}").json())["status"] == "running" and j["progress"] > 0
    )
    t0 = time.monotonic()
    ok(c, "POST", f"/api/jobs/{job['id']}/cancel")
    done = wait_job(c, job["id"], timeout=30)
    assert done["status"] == "cancelled" and time.monotonic() - t0 < 5
    assert ok(c, "GET", f"/api/projects/{pid}/assets", params={"source": "export"})["total"] == 0
    services = c.app.state.services  # type: ignore[attr-defined]
    assert not list(Path(services.settings.tmp_dir).glob(f"export-{job['id']}*"))
