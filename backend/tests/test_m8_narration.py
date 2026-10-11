"""M8 narration: the LLM step, voicing and placement on the Narration track, following the picture, stale
takes, ducking in the export command and (with ffmpeg) measured on the real mix."""

from __future__ import annotations

import json
import re
import subprocess
import wave
from array import array
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.models import Timeline
from localflow.schemas.timeline import Clip, TimelineData
from localflow.services.llm.catalog import KeyCheck
from localflow.services.llm.fake import FakeLLMProvider
from localflow.services.narration import budget_words, clean_line
from localflow.services.secrets import MemoryKeyStore, SecretStore
from localflow.services.timeline.model import (
    NARRATION_LEAD_S,
    default_timeline,
    follow_video_moves,
    normalize,
    with_narration_track,
)
from localflow.services.timeline.render import DUCK_FILTER, Source, build_export
from tests.conftest import TOKEN, make_settings, wait_job, wait_worker_idle
from tests.test_m3_imagegen import ok
from tests.test_m5_timeline import FFMPEG, fake_video, needs_ffmpeg, track

# One line per shot of `three_shots` (4 s, 3 s, 5 s); the second shot stays silent.
LINES = {
    "lines": [
        {"n": 1, "text": "\u201cThe tide was turning.\u201d"},
        {"n": 2, "text": ""},
        {"n": 3, "text": "By morning, nothing would be the same again."},
    ]
}


# --- units ---------------------------------------------------------------------------------------------
def test_budget_words_and_line_cleaning() -> None:
    assert [budget_words(s) for s in (0.5, 0.6, 3.0, 4.0, 5.0, 10.0)] == [0, 0, 5, 7, 10, 21]
    assert clean_line("\u201cHello   there.\u201d ") == "Hello there."
    assert clean_line("'quoted'") == "quoted"
    assert clean_line("  ") == ""


def _tl(order: list[str], lengths: dict[str, float]) -> TimelineData:
    tl = default_timeline("16:9", 24)
    tl.video.clips = [Clip(asset_id=f"V{s}", out_s=lengths[s], shot_id=s) for s in order]
    out, _ = normalize(tl, {f"V{s}": lengths[s] for s in order})
    return out


def test_narration_follows_its_clip_unless_it_was_moved() -> None:
    lengths = {"A": 2.0, "B": 3.0, "C": 1.0}
    before = _tl(["A", "B", "C"], lengths)
    starts = {c.shot_id: c.start_s for c in before.video.clips}
    assert starts == {"A": 0.0, "B": 2.0, "C": 5.0}
    after = _tl(["B", "C", "A"], lengths)  # A moves to the end
    narr = next(t for t in after.audio if t.id == "a_narration")
    narr.clips = [
        Clip(asset_id="N1", start_s=starts["A"] + NARRATION_LEAD_S, out_s=1.0, shot_id="A"),
        Clip(asset_id="N2", start_s=starts["B"] + NARRATION_LEAD_S, out_s=1.0, shot_id="B"),
        Clip(asset_id="N3", start_s=7.5, out_s=1.0, shot_id="C"),  # dragged away by hand
    ]
    assert follow_video_moves(before, after) == 2
    got = {c.shot_id: c.start_s for c in narr.clips}
    assert got == {"B": 0.25, "A": 4.25, "C": 7.5}
    assert follow_video_moves(before, after) == 0  # nothing left to move
    assert [c.shot_id for c in narr.clips] == ["B", "A", "C"]  # kept in time order


def test_with_narration_track_adds_it_first_among_the_audio_tracks() -> None:
    tl = default_timeline("16:9", 24)
    tl.tracks = [t for t in tl.tracks if t.id != "a_narration"]
    out = with_narration_track(tl)
    assert [t.id for t in out.tracks] == ["v1", "a_narration", "a_dialogue", "a_sfx", "a_music"]
    assert with_narration_track(out) is out


# --- the export command --------------------------------------------------------------------------------
def _graph(args: list[str]) -> str:
    return args[args.index("-filter_complex") + 1]


def _mix_timeline(narration: bool = True, dialogue: bool = True) -> tuple[TimelineData, dict[str, Source]]:
    tl = default_timeline("16:9", 24)
    tl.video.clips = [Clip(asset_id="V", out_s=6.0, shot_id="s1")]
    clips = {
        "a_narration": Clip(asset_id="N", start_s=1.0, out_s=2.0, shot_id="s1") if narration else None,
        "a_dialogue": Clip(asset_id="D", start_s=0.5, out_s=1.0) if dialogue else None,
        "a_sfx": Clip(asset_id="S", start_s=0.0, out_s=3.0),
        "a_music": Clip(asset_id="M", start_s=0.0, out_s=6.0),
    }
    for t in tl.audio:
        clip = clips.get(t.id)
        t.clips = [clip] if clip else []
    tl, _ = normalize(tl, {"V": 6.0})
    sources = {k: Source(Path(f"{k}.wav"), 6.0) for k in "NDSM"}
    sources["V"] = Source(Path("v.mp4"), 6.0, False, 64, 36)
    return tl, sources


def test_ducking_graph_pulls_music_and_effects_under_the_narration() -> None:
    tl, sources = _mix_timeline()
    g = _graph(build_export(tl, sources, "720p", Path("o.mp4"), duck=True).args)
    assert DUCK_FILTER in g and "asplit=2[nmix][nkey]" in g and "[bbus][nkey]" in g
    # Narration and dialogue aren't ducked: they go straight into the final mix beside the ducked bus.
    final = next(p for p in g.split(";") if p.startswith("[abase]") and p.endswith("[amix]"))
    assert "[nmix]" in final and "[bduck]" in final and final.count("[ta") == 1  # only the dialogue clip
    bus = next(p for p in g.split(";") if p.endswith("[bbus]"))
    assert bus.count("[ta") == 2  # the effects and the music


def test_no_ducking_when_off_or_without_narration_or_backing() -> None:
    tl, sources = _mix_timeline()
    off = _graph(build_export(tl, sources, "720p", Path("o.mp4"), duck=False).args)
    assert "sidechaincompress" not in off and "[nmix]" not in off
    plain, src = _mix_timeline(narration=False)
    assert "sidechaincompress" not in _graph(build_export(plain, src, "720p", Path("o.mp4"), duck=True).args)
    # Narration over nothing to duck: just a normal mix.
    solo = default_timeline("16:9", 24)
    solo.video.clips = [Clip(asset_id="V", out_s=6.0)]
    next(t for t in solo.audio if t.id == "a_narration").clips = [Clip(asset_id="N", start_s=1.0, out_s=2.0)]
    solo, _ = normalize(solo, {"V": 6.0})
    assert "sidechaincompress" not in _graph(build_export(solo, src, "720p", Path("o.mp4"), duck=True).args)
    # A muted narration track doesn't duck anything either.
    muted, src2 = _mix_timeline()
    next(t for t in muted.audio if t.id == "a_narration").muted = True
    assert "sidechaincompress" not in _graph(build_export(muted, src2, "720p", Path("o.mp4"), duck=True).args)


def _tone(path: Path, hz: float, seconds: float, amp: float, sr: int = 48_000) -> None:
    import math

    n = int(sr * seconds)
    data = array("h", (int(amp * 32767 * math.sin(2 * math.pi * hz * i / sr)) for i in range(n)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def _level(video: Path, start: float, hz: float = 440.0) -> float:
    """Mean volume (dB) of one second of the mix around `hz`."""
    assert FFMPEG
    out = subprocess.run(
        [
            FFMPEG,
            "-v",
            "info",
            "-ss",
            str(start),
            "-t",
            "1",
            "-i",
            str(video),
            "-vn",
            "-af",
            f"bandpass=f={hz}:width_type=h:w=100,volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    return float(re.search(r"mean_volume: (-?[\d.]+) dB", out).group(1))  # type: ignore[union-attr]


@needs_ffmpeg
def test_music_is_measurably_quieter_under_the_narration(tmp_path: Path) -> None:
    """Acceptance: with ducking the music dips >= 6 dB while the narration speaks and recovers after it."""
    assert FFMPEG
    for name, hz, secs, amp in (("N", 1000, 2.0, 0.15), ("M", 440, 6.0, 0.25)):
        _tone(tmp_path / f"{name}.wav", hz, secs, amp)
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x36:r=24:d=6",
            "-pix_fmt",
            "yuv420p",
            str(tmp_path / "v.mp4"),
        ],
        check=True,
    )
    tl = default_timeline("16:9", 24)
    tl.video.clips = [Clip(asset_id="V", out_s=6.0, shot_id="s1")]
    next(t for t in tl.audio if t.id == "a_narration").clips = [Clip(asset_id="N", start_s=1.0, out_s=2.0)]
    next(t for t in tl.audio if t.id == "a_music").clips = [Clip(asset_id="M", out_s=6.0)]
    tl, _ = normalize(tl, {"V": 6.0})
    sources = {
        "V": Source(tmp_path / "v.mp4", 6.0, False, 64, 36),
        "N": Source(tmp_path / "N.wav", 2.0),
        "M": Source(tmp_path / "M.wav", 6.0),
    }
    outs: dict[bool, Path] = {}
    for duck in (False, True):
        dst = tmp_path / f"out-{duck}.mp4"
        plan = build_export(tl, sources, "720p", dst, duck=duck)
        subprocess.run([FFMPEG, "-y", "-v", "error", *plan.args], check=True)
        outs[duck] = dst
    during = _level(outs[False], 1.5) - _level(outs[True], 1.5)
    after = _level(outs[False], 4.5) - _level(outs[True], 4.5)
    assert during >= 6.0, f"music only dipped {during:.1f} dB under the narration"
    assert abs(after) <= 1.5, f"music hadn't recovered after the narration ({after:.1f} dB down)"
    assert _level(outs[True], 1.5, hz=1000) == pytest.approx(_level(outs[False], 1.5, hz=1000), abs=1.0)


# --- API -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider({"narration": [LINES]})


@pytest.fixture(scope="module")
def c(tmp_path_factory: pytest.TempPathFactory, fake_llm: FakeLLMProvider) -> Iterator[TestClient]:
    store = MemoryKeyStore()
    store.set("openai_api_key", "sk-test-openai-key-1234")

    async def checker(name: str, value: str) -> KeyCheck:
        return KeyCheck(ok=True)

    app = create_app(
        make_settings(tmp_path_factory.mktemp("m8")),
        secrets=SecretStore(store, env=None),
        llm_factory=lambda name, key: fake_llm,
        key_checker=checker,
    )
    with TestClient(app, headers={"X-LocalFlow-Token": TOKEN}) as client:
        wait_worker_idle(client)
        yield client


def done(c: TestClient, job: dict[str, Any]) -> dict[str, Any]:
    out = wait_job(c, job["id"])
    assert out["status"] == "completed", out
    return out


def three_shots(c: TestClient, name: str = "Narrated") -> dict[str, Any]:
    """Two scenes, shots of 4 s / 3 s / 5 s with takes, auto-assembled (cuts, no fade)."""
    pid = ok(c, "POST", "/api/projects", 201, json={"name": name})["id"]
    shots: list[str] = []
    for title, durations in (("Dusk", [4.0, 3.0]), ("Night", [5.0])):
        scene = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": title})
        for d in durations:
            sh = ok(
                c,
                "POST",
                f"/api/scenes/{scene['id']}/shots",
                201,
                json={"duration_s": d, "video_prompt": f"a {d:g} second shot of the sea"},
            )
            take = fake_video(c, pid, d)
            ok(c, "POST", f"/api/shots/{sh['id']}/select", json={"slot": "video", "asset_id": take["id"]})
            shots.append(sh["id"])
    ok(
        c,
        "POST",
        f"/api/projects/{pid}/timeline/auto-assemble",
        json={"fade_in": False, "scene_transition": "cut"},
    )
    return {"pid": pid, "shots": shots}


def plan(c: TestClient, pid: str, **params: Any) -> dict[str, Any]:
    return ok(c, "GET", f"/api/projects/{pid}/narration", params=params)  # type: ignore[no-any-return]


def by_shot(p: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["shot_id"]: s for s in p["shots"]}


def set_text(c: TestClient, shot_id: str, text: str) -> None:
    ok(c, "PATCH", f"/api/shots/{shot_id}", json={"narration": text})


def test_the_llm_writes_a_line_per_shot_within_its_budget(c: TestClient, fake_llm: FakeLLMProvider) -> None:
    p = three_shots(c)
    pid, shots = p["pid"], p["shots"]
    assert plan(c, pid)["total_words"] == 0
    assert all(s["status"] == "empty" for s in plan(c, pid)["shots"])

    out = ok(c, "POST", f"/api/projects/{pid}/narration/write", json={"style": "a hushed storybook"})
    assert [u["shot_id"] for u in out["updated"]] == [shots[0], shots[2]]  # the silent shot stays silent
    assert out["updated"][0]["text"] == "The tide was turning."  # quotes stripped
    assert out["skipped"] == 1 and out["warnings"] == [] and out["usage"]["attempts"] == 1
    rows = by_shot(plan(c, pid))
    assert rows[shots[0]]["text"] == "The tide was turning." and rows[shots[0]]["status"] == "text"
    assert rows[shots[1]]["status"] == "empty" and rows[shots[2]]["words"] == 8
    assert (rows[shots[0]]["budget_words"], rows[shots[1]]["budget_words"]) == (7, 5)
    assert ok(c, "GET", f"/api/shots/{shots[0]}")["narration"] == "The tide was turning."

    call = [x for x in fake_llm.calls if x["purpose"] == "narration"][-1]
    system, user = (m["content"] for m in call["messages"])
    assert system.startswith("PURPOSE: narration")
    assert "<style>\na hushed storybook\n</style>" in user
    ctx = json.loads(re.search(r"<shots>\n(.*)\n</shots>", user, re.S).group(1))  # type: ignore[union-attr]
    assert [(s["n"], s["write"], s["budget_words"]) for s in ctx] == [
        (1, True, 7),
        (2, True, 5),
        (3, True, 10),
    ]

    # A shot the LLM left silent is still empty, so another write offers it again (and only it)...
    again = ok(c, "POST", f"/api/projects/{pid}/narration/write", json={})
    assert again["updated"] == []
    last = [x for x in fake_llm.calls if x["purpose"] == "narration"][-1]["messages"][1]["content"]
    ctx = json.loads(re.search(r"<shots>\n(.*)\n</shots>", last, re.S).group(1))  # type: ignore[union-attr]
    assert [s["write"] for s in ctx] == [False, True, False]
    assert ctx[0]["narration"] == "The tide was turning."  # written lines are context for the rest

    # ...but once every shot has narration, a rewrite has to be asked for.
    set_text(c, shots[1], "Quiet now.")
    err = c.post(f"/api/projects/{pid}/narration/write", json={})
    assert err.status_code == 400 and err.json()["error"]["code"] == "nothing_to_write"
    ok(c, "POST", f"/api/projects/{pid}/narration/write", json={"overwrite": True, "shot_ids": [shots[2]]})
    last = [x for x in fake_llm.calls if x["purpose"] == "narration"][-1]["messages"][1]["content"]
    ctx = json.loads(re.search(r"<shots>\n(.*)\n</shots>", last, re.S).group(1))  # type: ignore[union-attr]
    assert [s["write"] for s in ctx] == [False, False, True]


def test_over_budget_lines_get_a_warning(c: TestClient, fake_llm: FakeLLMProvider) -> None:
    p = three_shots(c, "Wordy")
    long = "Words " * 30
    fake_llm.script["narration"] = [{"lines": [{"n": 1, "text": long}]}]  # type: ignore[attr-defined]
    try:
        out = ok(c, "POST", f"/api/projects/{p['pid']}/narration/write", json={})
    finally:
        fake_llm.script["narration"] = [LINES]  # type: ignore[attr-defined]
    assert len(out["warnings"]) == 1 and "30 words for a 4 s shot (budget 7)" in out["warnings"][0]


def test_the_narrator_voice_belongs_to_the_project(c: TestClient) -> None:
    p = three_shots(c, "Voices")
    pid = p["pid"]
    narrator = plan(c, pid)["narrator"]
    assert narrator["source"] == "suggested" and narrator["voice_id"]
    bad = c.patch(f"/api/projects/{pid}", json={"narrator_voice_id": "nope"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_voice"
    voices = {v["native_id"]: v for v in ok(c, "GET", "/api/voices")}
    ok(c, "PATCH", f"/api/projects/{pid}", json={"narrator_voice_id": voices["fake_finn"]["id"]})
    narrator = plan(c, pid)["narrator"]
    assert (narrator["source"], narrator["voice_name"]) == ("project", voices["fake_finn"]["name"])
    assert ok(c, "GET", f"/api/projects/{pid}")["narrator_voice_id"] == voices["fake_finn"]["id"]


def test_voicing_places_each_line_on_the_narration_track(c: TestClient) -> None:
    p = three_shots(c, "Voiced")
    pid, shots = p["pid"], p["shots"]
    voices = {v["native_id"]: v for v in ok(c, "GET", "/api/voices")}
    ok(c, "PATCH", f"/api/projects/{pid}", json={"narrator_voice_id": voices["fake_finn"]["id"]})
    err = c.post(f"/api/projects/{pid}/narration/generate", json={})
    assert err.status_code == 400 and err.json()["error"]["code"] == "no_narration"

    set_text(c, shots[0], "The tide was turning.")
    set_text(c, shots[2], "By morning, nothing would be the same again.")
    out = ok(c, "POST", f"/api/projects/{pid}/narration/generate", 201, json={"speed": 1.1})
    assert [j["target_ref"] for j in out["jobs"]] == [
        {"kind": "shot", "shot_id": shots[0], "slot": "narration"},
        {"kind": "shot", "shot_id": shots[2], "slot": "narration"},
    ]
    assert out["skipped"] == [{"shot_id": shots[1], "reason": "No narration text."}]
    req = out["jobs"][0]["params"]["request"]
    assert (req["text"], req["voice_id"], req["speed"]) == ("The tide was turning.", "fake_finn", 1.1)
    assert out["plan"]["narrator"]["source"] == "project" and out["plan"]["speed"] == 1.1
    takes = [done(c, j)["result"]["asset_ids"][0] for j in out["jobs"]]

    tl = ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"]
    video = {x["shot_id"]: x for x in track(tl, "v1")}
    clips = {x["shot_id"]: x for x in track(tl, "a_narration")}
    assert set(clips) == {shots[0], shots[2]} and track(tl, "a_dialogue") == []
    assert clips[shots[0]]["start_s"] == pytest.approx(video[shots[0]]["start_s"] + 0.25)
    assert clips[shots[2]]["start_s"] == pytest.approx(video[shots[2]]["start_s"] + 0.25)
    assert clips[shots[0]]["asset_id"] == takes[0] and clips[shots[0]]["in_s"] == 0
    asset = ok(c, "GET", f"/api/assets/{takes[0]}")
    assert asset["generation"]["target_ref"]["slot"] == "narration"
    assert ok(c, "GET", f"/api/shots/{shots[0]}")["selected_audio_asset_ids"] == [takes[0]]

    rows = by_shot(plan(c, pid))
    assert rows[shots[0]]["status"] == "voiced" and rows[shots[0]]["selected_asset_id"] == takes[0]
    assert rows[shots[0]]["start_s"] == pytest.approx(0.25)
    assert rows[shots[1]]["status"] == "empty"

    # Nothing is stale, so voicing again does nothing; changing a line marks only that one stale.
    again = ok(c, "POST", f"/api/projects/{pid}/narration/generate", 201, json={})
    assert again["jobs"] == [] and [s["reason"] for s in again["skipped"]] == [
        "Already voiced.",
        "No narration text.",
        "Already voiced.",
    ]
    set_text(c, shots[2], "By morning, everything had changed.")
    assert by_shot(plan(c, pid))[shots[2]]["status"] == "stale"
    redo = ok(c, "POST", f"/api/projects/{pid}/narration/generate", 201, json={})
    assert [j["target_ref"]["shot_id"] for j in redo["jobs"]] == [shots[2]]
    new_take = done(c, redo["jobs"][0])["result"]["asset_ids"][0]
    assert by_shot(plan(c, pid))[shots[2]]["status"] == "voiced"
    tl = ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"]
    assert [x["asset_id"] for x in track(tl, "a_narration") if x["shot_id"] == shots[2]] == [new_take]

    # `only_missing: false` re-voices a line that is up to date.
    forced = ok(
        c,
        "POST",
        f"/api/projects/{pid}/narration/generate",
        201,
        json={"only_missing": False, "shot_ids": [shots[0]]},
    )
    assert len(forced["jobs"]) == 1
    done(c, forced["jobs"][0])


def test_overflow_into_the_next_line_is_reported(c: TestClient) -> None:
    p = three_shots(c, "Overflow")
    pid, shots = p["pid"], p["shots"]
    set_text(c, shots[0], " ".join(["word"] * 24))  # ~9 s of speech over a 4 s shot
    set_text(c, shots[1], "Short.")
    rows = by_shot(plan(c, pid))
    assert rows[shots[0]]["overflow_s"] > 3 and rows[shots[1]]["overflow_s"] == 0
    assert any("run into the next narrated line" in n for n in plan(c, pid)["notes"])
    set_text(c, shots[0], "Short too.")
    assert by_shot(plan(c, pid))[shots[0]]["overflow_s"] == 0


def _put(c: TestClient, pid: str, tl: dict[str, Any]) -> dict[str, Any]:
    return ok(c, "PUT", f"/api/projects/{pid}/timeline", json=tl)["timeline"]  # type: ignore[no-any-return]


def test_narration_follows_reordered_clips_and_resyncs(c: TestClient) -> None:
    p = three_shots(c, "Follow")
    pid, shots = p["pid"], p["shots"]
    set_text(c, shots[0], "First line.")
    set_text(c, shots[2], "Last line.")
    for j in ok(c, "POST", f"/api/projects/{pid}/narration/generate", 201, json={})["jobs"]:
        done(c, j)

    tl = ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"]
    first_at = {x["shot_id"]: x["start_s"] for x in track(tl, "a_narration")}
    assert first_at == {shots[0]: pytest.approx(0.25), shots[2]: pytest.approx(7.25)}
    # Move the last shot to the front: its line goes with it, the other one shifts back.
    video = track(tl, "v1")
    video.insert(0, video.pop())
    tl = _put(c, pid, tl)
    moved = {x["shot_id"]: x["start_s"] for x in track(tl, "a_narration")}
    assert moved == {shots[2]: pytest.approx(0.25), shots[0]: pytest.approx(5.25)}

    # A line the user dragged somewhere stays there when the picture moves again...
    narr = next(t for t in tl["tracks"] if t["id"] == "a_narration")
    next(x for x in narr["clips"] if x["shot_id"] == shots[0])["start_s"] = 9.0
    tl = _put(c, pid, tl)
    video = track(tl, "v1")
    video.insert(0, video.pop())
    tl = _put(c, pid, tl)
    moved = {x["shot_id"]: x["start_s"] for x in track(tl, "a_narration")}
    assert moved[shots[0]] == pytest.approx(9.0)

    # ...until it's re-synced to its shot.
    out = ok(c, "POST", f"/api/projects/{pid}/narration/resync")["timeline"]
    starts = {x["shot_id"]: x["start_s"] for x in track(out, "v1")}
    for x in track(out, "a_narration"):
        assert x["start_s"] == pytest.approx(starts[x["shot_id"]] + 0.25)


def test_auto_assemble_puts_narration_on_its_own_track(c: TestClient) -> None:
    p = three_shots(c, "Assemble")
    pid, shots = p["pid"], p["shots"]
    set_text(c, shots[1], "Middle line.")
    for j in ok(c, "POST", f"/api/projects/{pid}/narration/generate", 201, json={})["jobs"]:
        done(c, j)
    tl = ok(c, "POST", f"/api/projects/{pid}/timeline/auto-assemble", json={"fade_in": False})["timeline"]
    video = {x["shot_id"]: x for x in track(tl, "v1")}
    [clip] = track(tl, "a_narration")
    assert clip["shot_id"] == shots[1] and track(tl, "a_dialogue") == []
    assert clip["start_s"] == pytest.approx(video[shots[1]]["start_s"] + 0.25)


def test_a_timeline_saved_before_narration_gains_the_track(c: TestClient) -> None:
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Old"})["id"]
    old = default_timeline("16:9", 24).model_dump(mode="json")
    old["tracks"] = [t for t in old["tracks"] if t["id"] != "a_narration"]
    svc = c.app.state.services  # type: ignore[attr-defined]
    with svc.db.session() as s:
        s.add(Timeline(project_id=pid, tracks_json=old))
        s.commit()
    tl = ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"]
    assert [t["id"] for t in tl["tracks"]] == ["v1", "a_narration", "a_dialogue", "a_sfx", "a_music"]
    # A client that doesn't know the track yet can still save.
    legacy = json.loads(json.dumps(tl))
    legacy["tracks"] = [t for t in legacy["tracks"] if t["id"] != "a_narration"]
    saved = _put(c, pid, legacy)
    assert [t["id"] for t in saved["tracks"]][1] == "a_narration"


@needs_ffmpeg
def test_export_takes_a_duck_option_that_defaults_on(c: TestClient) -> None:
    p = three_shots(c, "Export")
    pid = p["pid"]
    on = ok(c, "POST", f"/api/projects/{pid}/export", 201, json={"preset": "720p"})
    assert on["params"]["duck"] is True
    done(c, on)
    off = ok(c, "POST", f"/api/projects/{pid}/export", 201, json={"preset": "720p", "duck": False})
    assert off["params"]["duck"] is False
    asset = ok(c, "GET", f"/api/assets/{done(c, off)['result']['asset_ids'][0]}")
    assert asset["generation"]["duck"] is False


def test_settings_list_the_narration_purpose(c: TestClient) -> None:
    assert "narration" in ok(c, "GET", "/api/settings")["purposes"]


def test_clip_model_still_validates_narration_clips() -> None:
    tl = default_timeline("16:9", 24)
    assert TimelineData.model_validate(tl.model_dump()).audio[0].id == "a_narration"
    assert len(tl.audio) == 4
