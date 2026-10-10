"""End-to-end job lifecycle with the spawned FakeEngine worker (M0 acceptance)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from localflow.services.media.ffmpeg import probe
from tests.conftest import TOKEN, wait_for, wait_job, wait_worker_idle

FAST = {"steps": 3, "step_delay_s": 0.01}


def _submit(client: TestClient, kind: str, **body: object) -> dict:
    r = client.post(f"/api/generate/{kind}", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_fake_t2v_produces_playable_mp4(live_client: TestClient) -> None:
    job = _submit(
        live_client, "t2v", prompt="a red fox in snow", width=320, height=180, duration_s=1.0, extra=FAST
    )
    assert job["status"] == "queued"
    assert job["params"]["request"]["seed"] is not None  # seed resolved and recorded

    done = wait_job(live_client, job["id"])
    assert done["status"] == "completed", done
    (asset_id,) = done["result"]["asset_ids"]

    asset = live_client.get(f"/api/assets/{asset_id}").json()
    assert asset["type"] == "video"
    assert (asset["width"], asset["height"]) == (320, 180)
    assert abs(asset["duration_s"] - 1.0) < 0.2
    assert asset["generation"]["adapter_key"] == "fake"
    assert asset["generation"]["seed"] == job["params"]["request"]["seed"]
    assert asset["thumb_url"]

    # Media elements can't send headers, so files accept ?token=
    plain = TestClient(live_client.app)
    r = plain.get(f"{asset['url']}?token={TOKEN}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert r.content[4:8] == b"ftyp"

    ranged = plain.get(f"{asset['url']}?token={TOKEN}", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206
    assert len(ranged.content) == 100

    assert plain.get(asset["url"]).status_code == 401
    assert plain.get(f"{asset['thumb_url']}?token={TOKEN}").headers["content-type"] == "image/jpeg"


def test_t2i_then_i2v_uses_input_asset(live_client: TestClient, tmp_path: Path) -> None:
    img_job = wait_job(
        live_client,
        _submit(live_client, "t2i", prompt="castle", width=256, height=256, num_images=2, extra=FAST)["id"],
    )
    assert img_job["status"] == "completed", img_job
    first, second = img_job["result"]["asset_ids"]
    a1 = live_client.get(f"/api/assets/{first}").json()
    a2 = live_client.get(f"/api/assets/{second}").json()
    assert a1["generation"]["seed"] + 1 == a2["generation"]["seed"]

    vid = wait_job(
        live_client,
        _submit(live_client, "i2v", image_asset_id=first, width=256, height=144, duration_s=0.5, extra=FAST)[
            "id"
        ],
    )
    assert vid["status"] == "completed", vid
    out = live_client.get(f"/api/assets/{vid['result']['asset_ids'][0]}").json()
    assert out["parent_asset_ids"] == [first]


def test_i2v_rejects_wrong_asset_type(live_client: TestClient) -> None:
    job = wait_job(live_client, _submit(live_client, "tts", text="hello there", extra=FAST)["id"])
    audio_id = job["result"]["asset_ids"][0]
    r = live_client.post("/api/generate/i2v", json={"image_asset_id": audio_id})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "wrong_asset_type"


def test_audio_jobs(live_client: TestClient) -> None:
    tts = wait_job(live_client, _submit(live_client, "tts", text="one two three four five", extra=FAST)["id"])
    music = wait_job(
        live_client, _submit(live_client, "music", prompt="calm", duration_s=1.5, extra=FAST)["id"]
    )
    for job in (tts, music):
        assert job["status"] == "completed", job
        asset = live_client.get(f"/api/assets/{job['result']['asset_ids'][0]}").json()
        assert asset["type"] == "audio"
        assert asset["duration_s"] > 0


def test_progress_and_previews_are_published(live_client: TestClient) -> None:
    plain = TestClient(live_client.app)
    with plain.websocket_connect(f"/ws/events?token={TOKEN}") as ws:
        first = ws.receive_json()
        assert first["type"] == "worker.status"
        job = _submit(
            live_client,
            "t2i",
            prompt="preview me",
            width=128,
            height=128,
            extra={"steps": 6, "step_delay_s": 0.02},
        )
        seen: set[str] = set()
        while "job.completed" not in seen:
            ev = ws.receive_json()
            if ev.get("job", {}).get("id") == job["id"] or ev.get("job_id") == job["id"]:
                seen.add(ev["type"])
        assert {"job.created", "job.progress", "job.preview", "job.completed"} <= seen


def test_failure_is_reported(live_client: TestClient) -> None:
    job = wait_job(
        live_client,
        _submit(live_client, "t2i", prompt="x", width=64, height=64, extra={**FAST, "fail": "boom"})["id"],
    )
    assert job["status"] == "failed"
    assert "boom" in job["error"]

    retried = live_client.post(f"/api/jobs/{job['id']}/retry")
    assert retried.status_code == 201
    assert retried.json()["id"] != job["id"]
    assert retried.json()["params"] == job["params"]
    wait_job(live_client, retried.json()["id"])


def test_cancel_running_and_queued(live_client: TestClient) -> None:
    slow = _submit(
        live_client,
        "t2v",
        prompt="slow",
        width=128,
        height=72,
        duration_s=0.5,
        extra={"steps": 200, "step_delay_s": 0.05},
    )
    queued = _submit(live_client, "t2i", prompt="waiting", width=64, height=64, extra=FAST)
    wait_job(live_client, slow["id"], {"running"})

    r = live_client.post(f"/api/jobs/{queued['id']}/cancel")
    assert r.json()["status"] == "cancelled"

    live_client.post(f"/api/jobs/{slow['id']}/cancel")
    done = wait_job(live_client, slow["id"], timeout=5)
    assert done["status"] == "cancelled"
    assert live_client.post(f"/api/jobs/{slow['id']}/cancel").status_code == 409


def test_worker_crash_marks_job_failed_and_restarts(live_client: TestClient) -> None:
    before = live_client.get("/api/system/status").json()["worker"]
    job = wait_job(
        live_client,
        _submit(live_client, "t2i", prompt="crash", width=64, height=64, extra={**FAST, "crash": True})["id"],
    )
    assert job["status"] == "failed"
    assert "crashed" in job["error"]

    status = wait_worker_idle(live_client)
    assert status["worker"]["restarts"] == before["restarts"] + 1
    assert status["worker"]["pid"] != before["pid"]

    ok = wait_job(
        live_client, _submit(live_client, "t2i", prompt="after", width=64, height=64, extra=FAST)["id"]
    )
    assert ok["status"] == "completed"


def test_list_jobs_and_assets(live_client: TestClient) -> None:
    pid = live_client.get("/api/playground").json()["project_id"]
    jobs = live_client.get("/api/jobs", params={"project_id": pid, "status": ["completed"]}).json()
    assert jobs["total"] >= 1
    assert all(j["status"] == "completed" for j in jobs["items"])

    videos = live_client.get(f"/api/projects/{pid}/assets", params={"type": "video"}).json()
    assert videos["total"] >= 1
    assert all(a["type"] == "video" for a in videos["items"])

    adapters = live_client.get("/api/models/adapters").json()
    fake = next(a for a in adapters if a["adapter_key"] == "fake")
    assert "t2v" in fake["capabilities"]
    assert "steps" in fake["param_schema"]["properties"]


def test_video_file_is_valid_h264(live_client: TestClient) -> None:
    job = wait_job(
        live_client,
        _submit(
            live_client,
            "t2v",
            prompt="probe",
            width=160,
            height=90,
            duration_s=0.5,
            with_audio=True,
            extra=FAST,
        )["id"],
    )
    asset = live_client.get(f"/api/assets/{job['result']['asset_ids'][0]}").json()
    assert asset["has_audio"] is True
    settings = live_client.app.state.settings
    path = settings.projects_dir / asset["project_id"] / "assets" / "videos" / f"{asset['id']}.mp4"
    info = wait_for(lambda: path.exists() and probe(settings.ffprobe, path))
    assert info["has_video"] and info["has_audio"]
