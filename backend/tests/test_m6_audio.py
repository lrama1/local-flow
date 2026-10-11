"""M6: voices, dialogue layout, per-shot dialogue + SFX, music, and timeline placement (Section 12)."""

from __future__ import annotations

import wave
from array import array
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.services.audiogen import emotion_speed, estimate_line_s, sfx_negative
from localflow.services.llm.catalog import KeyCheck
from localflow.services.llm.fake import FakeLLMProvider
from localflow.services.secrets import MemoryKeyStore, SecretStore
from localflow.services.voices import hint_gender, sample_text
from localflow.worker.adapters.base import TTSRequest
from localflow.worker.adapters.speech import (
    VoiceParams,
    fit_factor,
    layout,
    normalize_peak,
    trim_silence,
    write_wav,
)
from tests.conftest import TOKEN, make_settings, wait_for, wait_job, wait_worker_idle
from tests.test_m3_imagegen import ok
from tests.test_m5_timeline import FFPROBE, fake_video, streams, track

DESIGN = {
    "base_weight": 0.7,
    "mix": [{"voice": "fake_finn", "weight": 0.3}, {"voice": "nope", "weight": 0.2}],
    "speed": 0.9,
    "notes": "Older and gruffer.",
}


# --- units ---------------------------------------------------------------------------------------------
def test_fit_factor_and_layout() -> None:
    # Fits as is: no speed-up; default lead-in and gaps.
    assert fit_factor([1.0, 1.0], 4.0, 0.3, 0.35, 1.15) == 1.0
    assert layout([1.0, 1.0], [None, None], 4.0, 0.3, 0.35) == [(0.3, 1.3), (1.65, 2.65)]
    # Fits once the lead-in and gap shrink: still no speed-up.
    assert fit_factor([1.8, 1.8], 3.9, 0.3, 0.35, 1.15) == 1.0
    (s1, e1), (s2, e2) = layout([1.8, 1.8], [None, None], 3.9, 0.3, 0.35)
    assert s1 >= 0.1 and s2 - e1 >= 0.15 and e2 <= 3.9 + 1e-9
    # Needs 1.1x: within the cap.
    f = fit_factor([2.0, 2.0], 3.89, 0.3, 0.35, 1.15)
    assert f == pytest.approx(4.0 / (3.89 - 0.1 - 0.15))
    # Too long even at the cap.
    assert fit_factor([5.0], 3.0, 0.3, 0.35, 1.15) == 1.15
    # Explicit starts win; no clip length = plain sequence.
    assert layout([1.0, 1.0], [2.0, None], None, 0.3, 0.5) == [(2.0, 3.0), (3.5, 4.5)]


def test_voice_params_and_helpers() -> None:
    p = VoiceParams.model_validate({"mix": [{"voice": "b", "weight": 0.5}], "base_weight": 1.0})
    assert p.weights("a") == [("a", pytest.approx(2 / 3)), ("b", pytest.approx(1 / 3))]
    with pytest.raises(ValueError):
        VoiceParams.model_validate({"speed": 3})
    with pytest.raises(ValueError):
        TTSRequest.model_validate({"text": "  "})
    assert TTSRequest.model_validate({"lines": [{"text": "Hi"}]}).lines[0].text == "Hi"
    assert hint_gender("an old woman") == "female" and hint_gender("a gruff man") == "male"
    assert hint_gender("") == "" and hint_gender("a boy and a girl") == ""
    assert emotion_speed("Excited, loud") == 1.08 and emotion_speed("neutral") == 1.0
    assert estimate_line_s("one two three four five six seven eight nine ten", 1.0) == pytest.approx(10 / 2.6)
    assert sfx_negative("waves") == "speech, talking, voice, music"
    assert sfx_negative("a music box tinkles") == "speech, talking, voice"
    assert sample_text("es-ES").startswith("¡Hola") and sample_text("xx") == sample_text("en")


def test_trim_silence_and_normalize() -> None:
    sr = 100
    x = array("f", [0.0] * 30 + [0.2, -0.3, 0.1] + [0.0] * 50)
    t = trim_silence(x, sr, lead_s=0.05, tail_s=0.1)
    assert len(t) == 5 + 3 + 10 and t[5] == pytest.approx(0.2)
    assert trim_silence(array("f", [0.0] * 10), sr) == array("f", [0.0] * 10)  # all silence: unchanged
    n = normalize_peak(array("f", [0.1, -0.2]))
    assert max(abs(v) for v in n) == pytest.approx(0.8)  # capped at 4x
    assert normalize_peak(array("f", [0.5, -0.89 / 0.89 * 0.5])).tolist() == pytest.approx([0.89, -0.89])
    loud = array("f", [0.95])
    assert normalize_peak(loud) is loud


def test_mmaudio_windows_and_join() -> None:
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from localflow.worker.adapters.mmaudio import join, windows

    assert windows(5.0) == [(0.0, 5.0)] and windows(10.0) == [(0.0, 10.0)]
    for d in (10.5, 12.0, 13.5, 15.0, 15.08, 23.7, 30.0, 61.0):
        w = windows(d)
        assert w[0][0] == 0 and w[-1][0] + w[-1][1] == pytest.approx(d)
        assert all(s == int(s) and 0 < length <= 8 for s, length in w)  # whole-second starts
        assert all(w[k][0] + w[k][1] - w[k + 1][0] >= 1.0 for k in range(len(w) - 1)), (d, w)
    assert windows(15.0) == [(0.0, 8.0), (4.0, 8.0), (7.0, 8.0)]

    sr = 100
    a, b = np.ones(800, dtype=np.float32), -np.ones(800, dtype=np.float32)
    out = join([(0.0, a), (4.0, b)], 1200, sr)  # overlap 4..8 s, crossfade 5.5..6.5 s
    assert out[:550].tolist() == [1.0] * 550 and out[650:].tolist() == [-1.0] * 550
    mid = np.arange(550, 650)
    t = (mid / sr - 5.5) / 1.0
    assert out[mid] == pytest.approx(np.cos(np.pi / 2 * t) - np.sin(np.pi / 2 * t), abs=1e-5)  # equal power


def test_write_wav(tmp_path: Path) -> None:
    out = tmp_path / "x.wav"
    write_wav(out, array("f", [0.0, 0.5, -2.0, 1.0]), 24000)
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 4)
        assert array("h", w.readframes(4)).tolist() == [0, 16383, -32767, 32767]


# --- API -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider({"voice_design": [DESIGN]})


@pytest.fixture(scope="module")
def c(tmp_path_factory: pytest.TempPathFactory, fake_llm: FakeLLMProvider) -> Iterator[TestClient]:
    store = MemoryKeyStore()
    store.set("openai_api_key", "sk-test-openai-key-1234")

    async def checker(name: str, value: str) -> KeyCheck:
        return KeyCheck(ok=True)

    app = create_app(
        make_settings(tmp_path_factory.mktemp("m6")),
        secrets=SecretStore(store, env=None),
        llm_factory=lambda name, key: fake_llm,
        key_checker=checker,
    )
    with TestClient(app, headers={"X-LocalFlow-Token": TOKEN}) as client:
        wait_worker_idle(client)
        yield client


def voices(c: TestClient) -> dict[str, dict[str, Any]]:
    return {v["native_id"] or v["name"]: v for v in ok(c, "GET", "/api/voices")}


def wav_info(c: TestClient, asset_id: str) -> tuple[float, int, list[int]]:
    """(duration, sample rate, samples) of an audio asset."""
    svc = c.app.state.services  # type: ignore[attr-defined]
    path = svc.assets.file_path(svc.assets.get(asset_id))
    with wave.open(str(path)) as w:
        sr, n = w.getframerate(), w.getnframes()
        data = array("h", w.readframes(n)).tolist() if w.getsampwidth() == 2 and w.getnchannels() == 1 else []
    return n / sr, sr, data


def done(c: TestClient, job: dict[str, Any]) -> dict[str, Any]:
    out = wait_job(c, job["id"])
    assert out["status"] == "completed", out
    return out


def test_builtin_voices_and_suggestions(c: TestClient) -> None:
    v = voices(c)
    assert {"fake_aria", "fake_ben", "fake_cleo", "fake_dev", "fake_emma", "fake_finn"} <= set(v)
    aria = v["fake_aria"]
    assert aria["kind"] == "builtin" and aria["available"] and aria["language"] == "en-US"
    assert aria["gender"] == "female" and aria["quality"] == "A" and aria["preview_url"] is None
    s = ok(
        c, "GET", "/api/voices/suggest", params={"gender": "male", "age": "70s", "tone": "gravelly, gruff"}
    )
    assert s["voice"]["native_id"] == "fake_finn"
    s = ok(c, "GET", "/api/voices/suggest", params={"gender": "woman", "tone": "british, composed"})
    assert s["voice"]["native_id"] == "fake_emma"
    r = c.patch(f"/api/voices/{aria['id']}", json={"name": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "builtin_voice"
    assert c.delete(f"/api/voices/{aria['id']}").status_code == 409
    caps = ok(c, "GET", "/api/characters/capabilities")
    assert caps["voices"] is True


def test_preview_is_cached_and_served_with_a_query_token(c: TestClient) -> None:
    ben = voices(c)["fake_ben"]
    out = ok(c, "POST", f"/api/voices/{ben['id']}/preview", json={})
    assert out["url"] is None and out["job"]["type"] == "tts"
    assert out["job"]["target_ref"]["kind"] == "voice_preview"
    done(c, out["job"])
    url = wait_for(lambda: ok(c, "GET", f"/api/voices/{ben['id']}")["preview_url"])
    again = ok(c, "POST", f"/api/voices/{ben['id']}/preview", json={})
    assert again["job"] is None and again["url"] == url  # cached
    other = ok(c, "POST", f"/api/voices/{ben['id']}/preview", json={"text": "Another line."})
    assert other["job"] is not None
    plain = TestClient(c.app)
    wav = plain.get(f"{url}&token={TOKEN}", headers={"host": "127.0.0.1"})
    assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
    assert plain.get(url, headers={"host": "127.0.0.1"}).status_code == 401


def test_custom_voice_design_audition_and_crud(c: TestClient, fake_llm: FakeLLMProvider) -> None:
    v = voices(c)
    ben = v["fake_ben"]
    d = ok(c, "POST", "/api/voices/design", json={"base_voice_id": ben["id"], "description": "an old sailor"})
    assert d["params"] == {"mix": [{"voice": "fake_finn", "weight": 0.3}], "base_weight": 0.7, "speed": 0.9}
    assert d["notes"] == "Older and gruffer." and "nope" in d["warnings"][0] and d["usage"]["model"]
    call = [x for x in fake_llm.calls if x["purpose"] == "voice_design"][-1]
    user = call["messages"][1]["content"]
    assert "<description>\nan old sailor\n</description>" in user and '"native_id": "fake_finn"' in user
    assert '"native_id": "fake_ben"' in user.split("<voices>")[0]  # the base voice is described separately

    aud = ok(c, "POST", "/api/voices/audition", 201, json={"base_voice_id": ben["id"], "params": d["params"]})
    job = done(c, aud["job"])
    req = job["params"]["request"]
    assert req["voice_id"] == "fake_ben" and req["voice_params"] == d["params"]

    bad = {"name": "X", "base_voice_id": ben["id"], "params": {"mix": [{"voice": "af_heart", "weight": 0.5}]}}
    r = c.post("/api/voices", json=bad)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_voice_params"
    r = c.post("/api/voices", json={"name": "X", "base_voice_id": "nope"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_voice"

    salt = ok(
        c,
        "POST",
        "/api/voices",
        201,
        json={
            "name": "Old salt",
            "base_voice_id": ben["id"],
            "description": "an old sailor",
            "params": d["params"],
        },
    )
    assert salt["kind"] == "custom" and salt["adapter_key"] == "fake" and salt["base_voice_name"] == "Ben"
    assert (
        salt["gender"] == "male" and salt["available"] and ok(c, "GET", "/api/voices")[0]["id"] == salt["id"]
    )
    done(c, ok(c, "POST", f"/api/voices/{salt['id']}/preview", json={})["job"])
    assert wait_for(lambda: ok(c, "GET", f"/api/voices/{salt['id']}")["preview_url"])
    # Renaming keeps the sample; changing the sound drops it.
    assert ok(c, "PATCH", f"/api/voices/{salt['id']}", json={"name": "Salt"})["preview_url"]
    changed = ok(c, "PATCH", f"/api/voices/{salt['id']}", json={"params": {"speed": 1.1}})
    assert changed["preview_url"] is None and changed["params"]["speed"] == 1.1

    # Deleting a custom voice unassigns it from characters.
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Voices"})["id"]
    ing = ok(c, "POST", f"/api/projects/{pid}/ingredients", 201, json={"kind": "character", "name": "Jo"})
    r = c.patch(f"/api/ingredients/{ing['id']}", json={"voice_id": "nope"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_voice"
    assert (
        ok(c, "PATCH", f"/api/ingredients/{ing['id']}", json={"voice_id": salt["id"]})["voice_id"]
        == salt["id"]
    )
    assert c.delete(f"/api/voices/{salt['id']}").status_code == 204
    assert ok(c, "GET", f"/api/ingredients/{ing['id']}")["voice_id"] is None


def test_generate_tts_accepts_localflow_voice_ids(c: TestClient) -> None:
    cleo = voices(c)["fake_cleo"]
    job = ok(c, "POST", "/api/generate/tts", 201, json={"text": "Hello there.", "voice_id": cleo["id"]})
    assert job["params"]["request"]["voice_id"] == "fake_cleo" and job["params"]["adapter_key"] == "fake"
    out = done(c, job)
    dur, sr, _ = wav_info(c, out["result"]["asset_ids"][0])
    assert sr == 24000 and dur == pytest.approx(max(0.6, 2 / 2.5), abs=0.01)


def shot_with_take(
    c: TestClient, seconds: float, lines: list[dict[str, Any]], sfx: str | None = "waves crash on rocks"
) -> dict[str, Any]:
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Audio"})["id"]
    hints = {"gender": "female", "age": "60s", "tone": "warm"}
    mara = ok(
        c,
        "POST",
        f"/api/projects/{pid}/ingredients",
        201,
        json={"kind": "character", "name": "Mara", "voice_hints": hints},
    )
    scene = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "Dock"})
    dialogue = [{"character_ingredient_id": mara["id"], **ln} for ln in lines]
    sh = ok(
        c,
        "POST",
        f"/api/scenes/{scene['id']}/shots",
        201,
        json={"duration_s": seconds, "dialogue": dialogue, "sfx_prompt": sfx},
    )
    take = fake_video(c, pid, seconds)
    ok(c, "POST", f"/api/shots/{sh['id']}/select", json={"slot": "video", "asset_id": take["id"]})
    return {"pid": pid, "shot": sh, "take": take, "mara": mara, "scene": scene}


def selected_audio(c: TestClient, pid: str, shot_id: str) -> list[str]:
    board = ok(c, "GET", f"/api/projects/{pid}/scenes")
    sh = next(s for sc in board["scenes"] for s in sc["shots"] if s["id"] == shot_id)
    return list(sh["selected_audio_asset_ids"])


def test_shot_dialogue_and_sfx_align_to_the_clip_and_export(c: TestClient) -> None:
    """M6 acceptance: one dialogue line + an SFX prompt make two WAVs aligned to the clip; the export
    includes the mixed audio; a designed custom voice is used for the character's dialogue."""
    p = shot_with_take(c, 3.0, [{"line": "The tide is turning, love.", "emotion": "calm"}])
    pid, shot_id, mara = p["pid"], p["shot"]["id"], p["mara"]

    plan = ok(c, "POST", f"/api/shots/{shot_id}/audio-plan", json={})
    assert plan["clip_s"] == pytest.approx(3.0) and plan["has_take"] and plan["fits"]
    line = plan["lines"][0]
    assert line["speaker"] == "Mara" and line["voice_source"] == "suggested" and line["voice_name"] == "Aria"
    assert line["speed"] == 0.97 and any("Mara has no voice yet" in n for n in plan["notes"])
    assert plan["sfx_prompt"] == "waves crash on rocks" and plan["selected"] == {
        "dialogue": None,
        "sfx": None,
    }

    # A custom voice for Mara, designed from a description, then assigned.
    v = voices(c)
    d = ok(c, "POST", "/api/voices/design", json={"base_voice_id": v["fake_emma"]["id"], "description": "x"})
    params = {**d["params"], "mix": []}
    custom = ok(
        c,
        "POST",
        "/api/voices",
        201,
        json={"name": "Mara's voice", "base_voice_id": v["fake_emma"]["id"], "params": params},
    )
    ok(c, "PATCH", f"/api/ingredients/{mara['id']}", json={"voice_id": custom["id"]})
    plan = ok(c, "POST", f"/api/shots/{shot_id}/audio-plan", json={})
    assert plan["lines"][0]["voice_id"] == custom["id"] and plan["lines"][0]["voice_source"] == "character"
    assert not plan["notes"]

    out = ok(c, "POST", f"/api/shots/{shot_id}/generate-audio", 201, json={})
    assert [j["type"] for j in out["jobs"]] == ["tts", "v2a"]
    tts, v2a = (done(c, j) for j in out["jobs"])
    req = tts["params"]["request"]
    assert req["duration_s"] == pytest.approx(3.0) and req["lines"][0]["voice_id"] == "fake_emma"
    assert req["lines"][0]["voice_params"] == params and req["lines"][0]["speed"] == 0.97
    assert v2a["params"]["request"]["prompt"] == "waves crash on rocks"
    assert v2a["params"]["request"]["negative_prompt"] == "speech, talking, voice, music"

    dialogue_id, sfx_id = tts["result"]["asset_ids"][0], v2a["result"]["asset_ids"][0]
    take_s = p["take"]["duration_s"]
    for aid in (dialogue_id, sfx_id):
        dur, _, _ = wav_info(c, aid)
        assert dur == pytest.approx(take_s, abs=0.02)  # aligned to the clip
    asset = ok(c, "GET", f"/api/assets/{dialogue_id}")
    placed = asset["generation"]["output"]["lines"]
    assert placed[0]["key"] == "0" and placed[0]["start_s"] == pytest.approx(0.3)
    assert asset["generation"]["output"]["overflow_s"] == 0
    _, sr, samples = wav_info(c, dialogue_id)
    assert max(abs(x) for x in samples[: int(0.25 * sr)]) == 0  # silent lead-in
    assert max(abs(x) for x in samples[int(0.35 * sr) : int(0.6 * sr)]) > 1000  # then speech
    assert selected_audio(c, pid, shot_id) == [dialogue_id, sfx_id]
    assert ok(c, "POST", f"/api/shots/{shot_id}/audio-plan", json={})["selected"] == {
        "dialogue": dialogue_id,
        "sfx": sfx_id,
    }

    tl = ok(c, "POST", f"/api/projects/{pid}/timeline/auto-assemble", json={})["timeline"]
    tracks = {t["id"]: t["clips"] for t in tl["tracks"]}
    assert [x["asset_id"] for x in tracks["a_dialogue"]] == [dialogue_id]
    assert [x["asset_id"] for x in tracks["a_sfx"]] == [sfx_id]
    assert tracks["a_dialogue"][0]["start_s"] == tracks["v1"][0]["start_s"] == 0

    # Regenerating the dialogue swaps it on the timeline in place (no re-assemble needed).
    again = ok(c, "POST", f"/api/shots/{shot_id}/generate-audio", 201, json={"parts": ["dialogue"]})
    new_id = done(c, again["jobs"][0])["result"]["asset_ids"][0]
    assert selected_audio(c, pid, shot_id) == [sfx_id, new_id]
    tl = ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"]
    assert [x["asset_id"] for x in track(tl, "a_dialogue")] == [new_id]
    # ...and the old take can be picked again.
    ok(c, "POST", f"/api/shots/{shot_id}/select-audio", json={"asset_id": dialogue_id})
    assert selected_audio(c, pid, shot_id) == [sfx_id, dialogue_id]

    job = ok(c, "POST", f"/api/projects/{pid}/export", 201, json={"preset": "720p"})
    exported = done(c, job)
    if FFPROBE:
        svc = c.app.state.services  # type: ignore[attr-defined]
        info = streams(svc.assets.file_path(svc.assets.get(exported["result"]["asset_ids"][0])))
        assert info["audio"] == {"codec": "aac", "rate": 48000, "channels": 2}
        assert info["duration"] == pytest.approx(take_s, abs=0.1)


def test_long_dialogue_speeds_up_then_overflows(c: TestClient) -> None:
    words = "one two three four five six seven eight nine ten"
    p = shot_with_take(c, 2.0, [{"line": words}], sfx=None)
    shot_id = p["shot"]["id"]
    plan = ok(c, "POST", f"/api/shots/{shot_id}/audio-plan", json={})
    assert (
        not plan["fits"] and plan["fit_speed"] > 1.15 and any("extend the shot" in n for n in plan["notes"])
    )
    out = ok(c, "POST", f"/api/shots/{shot_id}/generate-audio", 201, json={"parts": ["dialogue"]})
    res = done(c, out["jobs"][0])
    output = ok(c, "GET", f"/api/assets/{res['result']['asset_ids'][0]}")["generation"]["output"]
    assert output["speed_factor"] == 1.15 and output["overflow_s"] > 0
    # fit="none" leaves the pace alone.
    out = ok(
        c,
        "POST",
        f"/api/shots/{shot_id}/generate-audio",
        201,
        json={"parts": ["dialogue"], "params": {"fit": "none"}},
    )
    res = done(c, out["jobs"][0])
    output = ok(c, "GET", f"/api/assets/{res['result']['asset_ids'][0]}")["generation"]["output"]
    assert output["speed_factor"] == 1.0
    # A shot with no take can still get dialogue, but not SFX.
    scene = p["scene"]["id"]
    bare = ok(c, "POST", f"/api/scenes/{scene}/shots", 201, json={"duration_s": 2})
    r = c.post(f"/api/shots/{bare['id']}/generate-audio", json={"parts": ["sfx"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_video"
    r = c.post(f"/api/shots/{bare['id']}/generate-audio", json={"parts": ["dialogue"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_dialogue"


def test_music_from_cues_is_placed_on_the_music_track(c: TestClient) -> None:
    p = shot_with_take(c, 2.0, [{"line": "Hi."}])
    pid, scene = p["pid"], p["scene"]["id"]
    ok(c, "PATCH", f"/api/shots/{p['shot']['id']}", json={"music_cue": "sparse piano, melancholy"})
    second = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "Two"})
    sh2 = ok(
        c,
        "POST",
        f"/api/scenes/{second['id']}/shots",
        201,
        json={"duration_s": 1.5, "music_cue": "strings swell"},
    )
    take2 = fake_video(c, pid, 1.5)
    ok(c, "POST", f"/api/shots/{sh2['id']}/select", json={"slot": "video", "asset_id": take2["id"]})
    ok(c, "POST", f"/api/projects/{pid}/timeline/auto-assemble", json={})

    film = ok(c, "POST", f"/api/projects/{pid}/generate-music", 201, json={})
    assert film["prompt"] == "sparse piano, melancholy; strings swell"
    total = ok(c, "GET", f"/api/projects/{pid}/timeline")["duration_s"]
    assert film["duration_s"] == pytest.approx(total, abs=0.01)
    film_asset = done(c, film["job"])["result"]["asset_ids"][0]

    two = ok(
        c,
        "POST",
        f"/api/projects/{pid}/generate-music",
        201,
        json={"scope": "scene", "scene_id": second["id"], "prompt": "a lone cello"},
    )
    assert two["prompt"] == "a lone cello" and two["duration_s"] == pytest.approx(
        take2["duration_s"], abs=0.01
    )
    scene_asset = done(c, two["job"])["result"]["asset_ids"][0]
    music = track(ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"], "a_music")
    starts = {x["asset_id"]: x["start_s"] for x in music}
    v1 = track(ok(c, "GET", f"/api/projects/{pid}/timeline")["timeline"], "v1")
    assert starts == {film_asset: 0, scene_asset: pytest.approx(v1[1]["start_s"], abs=0.01)}
    r = c.post(f"/api/projects/{pid}/generate-music", json={"scope": "scene", "scene_id": "nope"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_scene"
    assert scene  # the first scene stays as it was
