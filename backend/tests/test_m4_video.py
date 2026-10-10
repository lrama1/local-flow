"""M4 through the API (FakeEngine worker): shot video from keyframes, fallbacks, variations, extend, chained
extend, generate-all, take selection, hard cancel. Plus units for frames, sizes, camera fragments, concat."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.models.tables import Shot
from localflow.services.media.ffmpeg import concat_args
from localflow.services.videogen import (
    CONTINUE_SUFFIX,
    camera_fragment,
    snap_frames,
    video_negative,
    video_size,
)
from tests.conftest import TOKEN, make_settings, wait_for, wait_job, wait_worker_idle
from tests.test_m3_imagegen import install, ok, uninstall

SIZES = {
    "quality": {"16:9": [1280, 704], "9:16": [704, 1280], "1:1": [960, 960]},
    "balanced": {"16:9": [1280, 704], "9:16": [704, 1280], "1:1": [960, 960]},
    "low": {"16:9": [832, 480], "9:16": [480, 832], "1:1": [640, 640]},
}


# --- units -------------------------------------------------------------------------------------------
def test_frames_snap_to_4k_plus_1_within_the_clip() -> None:
    assert snap_frames(4 * 24, 121) == 97
    assert snap_frames(5 * 24, 121) == 121
    assert snap_frames(9 * 24, 121) == 121  # longer shots are extended, not stretched
    assert snap_frames(1, 121) == 5
    assert snap_frames(80, 81) == 81
    assert snap_frames(200, 120) == 117  # a max that isn't 4k+1 rounds down


def test_video_size_by_profile_and_aspect() -> None:
    d = {"sizes": SIZES}
    assert video_size(d, "balanced", "16:9") == (1280, 704)
    assert video_size(d, "low", "9:16") == (480, 832)
    w, h = video_size(d, "balanced", "4:3")
    assert w % 32 == 0 and h % 32 == 0 and abs(w / h - 4 / 3) < 0.05 and abs(w * h - 1280 * 704) < 60_000
    assert video_size({}, "balanced", "16:9") == (1280, 704)  # no model: 720p area, multiple of 32


def test_camera_fragments() -> None:
    assert camera_fragment("dolly_in") == "The camera dollies in smoothly toward the subject"
    assert camera_fragment("pan_left", "subtle") == "The camera pans slowly and gently to the left"
    assert camera_fragment("handheld", "strong") == "Handheld camera with strong, energetic shake"
    assert camera_fragment("static", "strong").startswith("Static camera")
    assert camera_fragment("nonsense") == ""


def test_static_shots_also_negate_camera_movement() -> None:
    static = Shot(scene_id="s", camera_move="static", negative_prompt="blurry ")
    assert video_negative(static) == "blurry, camera movement, panning, zooming, camera shake"
    assert video_negative(Shot(scene_id="s", camera_move="static", negative_prompt="")) == (
        "camera movement, panning, zooming, camera shake"
    )
    assert video_negative(Shot(scene_id="s", camera_move="dolly_in", negative_prompt="blurry")) == "blurry"


def test_worker_exit_codes_read_like_the_os_reports_them(monkeypatch: pytest.MonkeyPatch) -> None:
    from localflow.services.jobs import supervisor

    monkeypatch.setattr(supervisor.sys, "platform", "win32")
    assert supervisor.exit_code_text(4294967295) == "-1 (0xFFFFFFFF)"  # Stop-Process -Force
    assert supervisor.exit_code_text(3221225477) == "-1073741819 (0xC0000005)"  # access violation
    assert supervisor.exit_code_text(1) == "1" and supervisor.exit_code_text(None) == "unknown"
    monkeypatch.setattr(supervisor.sys, "platform", "linux")
    assert supervisor.exit_code_text(-9) == "-9 (signal 9)"


def test_concat_args_cut_and_crossfade() -> None:
    clips = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
    cut = concat_args(clips, [5.0, 5.0, 5.0], Path("o.mp4"), width=1280, height=704, fps=24)
    graph = cut[cut.index("-filter_complex") + 1]
    assert "concat=n=3:v=1:a=0[out]" in graph and "scale=1280:704" in graph and cut[-1] == "o.mp4"
    fade = concat_args(
        clips, [5.0, 5.0, 5.0], Path("o.mp4"), width=1280, height=704, fps=24, crossfade_frames=6
    )
    graph = fade[fade.index("-filter_complex") + 1]
    assert "xfade=transition=fade:duration=0.2500:offset=4.7500[x1]" in graph
    assert "[x1][v2]xfade=transition=fade:duration=0.2500:offset=9.5000[x2]" in graph
    assert fade[fade.index("-map") + 1] == "[x2]"


# --- integration ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def c(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    with TestClient(
        create_app(make_settings(tmp_path_factory.mktemp("m4"))), headers={"X-LocalFlow-Token": TOKEN}
    ) as client:
        wait_worker_idle(client)
        yield client


def done(c: TestClient, job: dict[str, Any]) -> dict[str, Any]:
    j = wait_job(c, job["id"])
    assert j["status"] == "completed", j
    return j


def shot_of(c: TestClient, shot_id: str) -> dict[str, Any]:
    return ok(c, "GET", f"/api/shots/{shot_id}")  # type: ignore[no-any-return]


def make_project(c: TestClient, **shot: Any) -> dict[str, str]:
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Harbor", "aspect_ratio": "16:9"})["id"]
    ok(
        c,
        "POST",
        f"/api/projects/{pid}/ingredients",
        201,
        json={"kind": "style", "name": "Look", "key": "style", "visual_prompt": "35mm film"},
    )
    keeper = ok(
        c,
        "POST",
        f"/api/projects/{pid}/ingredients",
        201,
        json={
            "kind": "character",
            "name": "Old Keeper",
            "key": "keeper",
            "visual_prompt": "weathered man, grey beard",
        },
    )
    scene = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "Pier"})
    body = {
        "video_prompt": "@keeper walks along the pier",
        "keyframe_prompt": "@keeper on the pier at dawn",
        "camera_move": "dolly_in",
        "duration_s": 4,
        **shot,
    }
    sh = ok(c, "POST", f"/api/scenes/{scene['id']}/shots", 201, json=body)
    return {"pid": pid, "scene": scene["id"], "shot": sh["id"], "keeper": keeper["id"]}


def with_keyframe(c: TestClient, shot_id: str, which: str = "start") -> str:
    out = ok(
        c,
        "POST",
        f"/api/shots/{shot_id}/generate-keyframe",
        201,
        json={"which": which, "count": 1, "mode": "t2i"},
    )
    asset = done(c, out["job"])["result"]["asset_ids"][0]
    slot = "keyframe" if which == "start" else "end_keyframe"
    ok(c, "POST", f"/api/shots/{shot_id}/select", json={"slot": slot, "asset_id": asset})
    return str(asset)


@pytest.fixture(scope="module")
def wan(c: TestClient) -> Iterator[str]:
    entry = install(c, "wan22-ti2v-5b")
    yield entry
    uninstall(c, entry)


def test_i2v_from_the_selected_keyframe(c: TestClient, wan: str) -> None:
    p = make_project(c)
    keyframe = with_keyframe(c, p["shot"])
    plan = ok(c, "POST", f"/api/shots/{p['shot']}/video-plan", json={})
    assert plan["mode"] == "i2v" and plan["model_entry_id"] == wan and plan["note"] is None
    assert (plan["width"], plan["height"], plan["fps"], plan["num_frames"]) == (1280, 704, 24, 97)
    assert plan["max_clip_seconds"] == 5.0
    assert (
        plan["prompt"]
        == "Old Keeper walks along the pier. The camera dollies in smoothly toward the subject."
    )
    queued = [j for j in c.get("/api/jobs").json()["items"] if (j["target_ref"] or {}).get("slot") == "video"]
    assert all(j["target_ref"]["shot_id"] != p["shot"] for j in queued)  # planning queues nothing

    out = ok(c, "POST", f"/api/shots/{p['shot']}/generate-video", 201, json={"camera_intensity": "strong"})
    assert out["mode"] == "i2v" and len(out["jobs"]) == 1
    assert out["prompt"].endswith("The camera dollies in quickly and dramatically toward the subject.")
    req = out["jobs"][0]["params"]["request"]
    assert req["image_asset_id"] == keyframe and req["model_entry_id"] == wan and req["num_frames"] == 97
    assert out["jobs"][0]["params"]["prompt_raw"] == "@keeper walks along the pier"
    assert out["jobs"][0]["target_ref"] == {"kind": "shot", "shot_id": p["shot"], "slot": "video"}

    take = done(c, out["jobs"][0])["result"]["asset_ids"][0]
    asset = ok(c, "GET", f"/api/assets/{take}")
    assert asset["type"] == "video" and asset["fps"] == 24 and keyframe in asset["parent_asset_ids"]
    assert asset["generation"]["prompt_raw"] == "@keeper walks along the pier"
    shot = wait_for(lambda: (s := shot_of(c, p["shot"]))["selected_video_asset_id"] and s)
    assert shot["selected_video_asset_id"] == take and shot["status"] == "video_ready"
    takes = ok(c, "GET", f"/api/projects/{p['pid']}/assets", params={"shot_id": p["shot"], "slot": "video"})
    assert [a["id"] for a in takes["items"]] == [take]


def test_variations_fallbacks_and_long_shots(c: TestClient, wan: str) -> None:
    p = make_project(c, duration_s=9, needs_extend=True, extend_segments=1)
    out = ok(c, "POST", f"/api/shots/{p['shot']}/generate-video", 201, json={"variations": 2})
    assert out["mode"] == "t2v" and len(out["jobs"]) == 2 and out["num_frames"] == 121
    assert "No keyframe is selected" in out["note"] and "use Extend" in out["note"]
    # Text-to-video describes the cast and adds the style; image modes leave that to the keyframe.
    assert out["prompt"].startswith("Old Keeper, weathered man, grey beard walks along the pier")
    assert out["prompt"].rstrip(".").endswith("35mm film")
    seeds = [j["params"]["request"]["seed"] for j in out["jobs"]]
    assert seeds[1] == (seeds[0] + 1) % 2**32
    first, second = (done(c, j)["result"]["asset_ids"][0] for j in out["jobs"])
    shot = wait_for(lambda: (s := shot_of(c, p["shot"]))["selected_video_asset_id"] and s)
    assert shot["selected_video_asset_id"] == first  # the second take doesn't replace the selection

    r = c.post(f"/api/shots/{p['shot']}/generate-video", json={"mode": "i2v"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_keyframe"
    ok(c, "PATCH", f"/api/shots/{p['shot']}", json={"generation_mode": "flf2v"})
    with_keyframe(c, p["shot"])
    plan = ok(c, "POST", f"/api/shots/{p['shot']}/video-plan", json={})
    assert plan["mode"] == "i2v" and plan["note"].startswith("No last keyframe is selected")
    r = c.post(f"/api/shots/{p['shot']}/generate-video", json={"mode": "flf2v"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_keyframe"
    with_keyframe(c, p["shot"], "end")
    flf = ok(c, "POST", f"/api/shots/{p['shot']}/generate-video", 201, json={"mode": "flf2v"})
    req = flf["jobs"][0]["params"]["request"]
    assert flf["mode"] == "flf2v" and req["first_asset_id"] != req["last_asset_id"]
    done(c, flf["jobs"][0])

    ok(c, "PATCH", f"/api/shots/{p['shot']}", json={"video_prompt": "@ghost waves"})
    r = c.post(f"/api/shots/{p['shot']}/generate-video", json={})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_mention"
    assert second


def test_extend_and_generate_plus_extend(c: TestClient, wan: str) -> None:
    p = make_project(c, duration_s=9, needs_extend=True, extend_segments=1)
    with_keyframe(c, p["shot"])
    r = c.post(f"/api/shots/{p['shot']}/extend", json={})
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_video"

    # Generate + extend: the extension is queued when the take arrives, and replaces it as the selection.
    out = ok(c, "POST", f"/api/shots/{p['shot']}/generate-video", 201, json={"extend_segments": 1})
    assert out["extend_segments"] == 1 and out["note"] is None
    take = done(c, out["jobs"][0])["result"]["asset_ids"][0]
    ext_job = wait_for(
        lambda: next(
            (
                j
                for j in c.get("/api/jobs").json()["items"]
                if j["type"] == "extend" and j["params"]["request"]["video_asset_id"] == take
            ),
            None,
        )
    )
    req = ext_job["params"]["request"]
    assert req["segments"] == 1 and req["num_frames"] == 121 and CONTINUE_SUFFIX in req["prompt"]
    extended = done(c, ext_job)["result"]["asset_ids"][0]
    shot = wait_for(lambda: (s := shot_of(c, p["shot"]))["selected_video_asset_id"] == extended and s)
    asset = ok(c, "GET", f"/api/assets/{extended}")
    assert take in asset["parent_asset_ids"] and asset["duration_s"] > 9
    assert shot["status"] == "video_ready"

    # Manual extend of a chosen take, two segments at once.
    two = ok(c, "POST", f"/api/shots/{p['shot']}/extend", 201, json={"asset_id": take, "segments": 2})
    assert two["jobs"][0]["params"]["request"]["segments"] == 2
    assert two["duration_s"] == pytest.approx(121 / 24 + 2 * 5, abs=0.1)  # each continuation drops a frame
    longer = done(c, two["jobs"][0])["result"]["asset_ids"][0]
    assert ok(c, "GET", f"/api/assets/{longer}")["duration_s"] > 14
    # The selection only follows an extension of the selected take.
    assert shot_of(c, p["shot"])["selected_video_asset_id"] == extended


def test_select_clear_and_delete_takes(c: TestClient, wan: str) -> None:
    p = make_project(c)
    with_keyframe(c, p["shot"])
    take = done(c, ok(c, "POST", f"/api/shots/{p['shot']}/generate-video", 201, json={})["jobs"][0])[
        "result"
    ]["asset_ids"][0]
    wait_for(lambda: shot_of(c, p["shot"])["selected_video_asset_id"] == take)
    shot = ok(c, "POST", f"/api/shots/{p['shot']}/select", json={"slot": "video", "asset_id": None})
    assert shot["selected_video_asset_id"] is None and shot["status"] == "keyframe_ready"
    shot = ok(c, "POST", f"/api/shots/{p['shot']}/select", json={"slot": "video", "asset_id": take})
    assert shot["status"] == "video_ready"
    ok(c, "PATCH", f"/api/shots/{p['shot']}", json={"status": "approved"})
    image = shot["selected_keyframe_asset_id"]
    r = c.post(f"/api/shots/{p['shot']}/select", json={"slot": "video", "asset_id": image})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_asset"
    assert c.delete(f"/api/assets/{take}").status_code == 204
    shot = shot_of(c, p["shot"])
    assert shot["selected_video_asset_id"] is None and shot["status"] == "keyframe_ready"


def test_generate_all_with_auto_approved_keyframes(c: TestClient, wan: str) -> None:
    p = make_project(c)  # shot 1: no keyframe yet (i2v)
    s2 = ok(
        c,
        "POST",
        f"/api/scenes/{p['scene']}/shots",
        201,
        json={"video_prompt": "gulls circle", "generation_mode": "t2v"},
    )
    s3 = ok(
        c,
        "POST",
        f"/api/scenes/{p['scene']}/shots",
        201,
        json={"video_prompt": "@keeper waves", "keyframe_prompt": "@keeper waves"},
    )
    with_keyframe(c, s3["id"])
    s4 = ok(c, "POST", f"/api/scenes/{p['scene']}/shots", 201, json={"video_prompt": "empty"})
    with_keyframe(c, s4["id"])
    take = done(c, ok(c, "POST", f"/api/shots/{s4['id']}/generate-video", 201, json={})["jobs"][0])["result"][
        "asset_ids"
    ][0]
    wait_for(lambda: shot_of(c, s4["id"])["selected_video_asset_id"] == take)

    out = ok(c, "POST", f"/api/projects/{p['pid']}/generate-all", 201, json={"auto_approve_keyframes": True})
    assert out["keyframe_jobs"] == 1 and out["video_jobs"] == 2
    assert out["skipped"] == [{"shot_id": s4["id"], "reason": "Already has a take."}]
    again = ok(
        c, "POST", f"/api/projects/{p['pid']}/generate-all", 201, json={"auto_approve_keyframes": True}
    )
    assert again["keyframe_jobs"] == again["video_jobs"] == 0
    assert sum(s["reason"] == "Already generating." for s in again["skipped"]) == 3

    # Shot 1's keyframe is selected automatically and its video follows.
    for sid in (p["shot"], s2["id"], s3["id"]):
        shot = wait_for(lambda sid=sid: (s := shot_of(c, sid))["selected_video_asset_id"] and s, timeout=60)
        assert shot["status"] == "video_ready"
    assert shot_of(c, p["shot"])["selected_keyframe_asset_id"]

    # Without auto-approve, keyframes are queued and the user picks before videos are made.
    q = make_project(c)
    manual = ok(c, "POST", f"/api/projects/{q['pid']}/generate-all", 201, json={})
    assert manual["keyframe_jobs"] == 1 and manual["video_jobs"] == 0
    assert manual["skipped"][0]["reason"].startswith("Pick a keyframe")
    jobs = [
        j for j in c.get("/api/jobs").json()["items"] if (j["target_ref"] or {}).get("shot_id") == q["shot"]
    ]
    assert jobs and jobs[0]["params"]["request"]["num_images"] == 2 and "chain" not in jobs[0]["params"]
    for j in jobs:
        done(c, j)
    time.sleep(0.3)
    assert shot_of(c, q["shot"])["selected_keyframe_asset_id"] is None


def test_camera_moves_endpoint(c: TestClient) -> None:
    moves = ok(c, "GET", "/api/camera-moves")
    assert moves["orbit"]["subtle"] == "The camera orbits slowly and gently around the subject"
    assert len(moves) == 15


# --- hard cancel ---------------------------------------------------------------------------------------
def test_a_job_that_ignores_cancel_gets_its_worker_restarted(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data", cancel_grace_s=1.0)
    with TestClient(create_app(settings), headers={"X-LocalFlow-Token": TOKEN}) as c:
        wait_worker_idle(c)
        before = c.get("/api/system/status").json()["worker"]
        job = ok(
            c, "POST", "/api/generate/t2v", 201, json={"prompt": "x", "extra": {"hang_s": 60, "steps": 3}}
        )
        wait_job(c, job["id"], {"running"})
        time.sleep(0.3)
        t0 = time.monotonic()
        ok(c, "POST", f"/api/jobs/{job['id']}/cancel")
        cancelled = wait_job(c, job["id"], {"cancelled", "failed", "completed"}, timeout=20)
        assert cancelled["status"] == "cancelled" and time.monotonic() - t0 < 6
        status = wait_worker_idle(c)["worker"]
        assert status["restarts"] == before["restarts"] + 1 and status["pid"] != before["pid"]
        after = ok(c, "POST", "/api/generate/t2v", 201, json={"prompt": "y", "extra": {"steps": 2}})
        assert wait_job(c, after["id"])["status"] == "completed"
