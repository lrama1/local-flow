"""API, security, and job-queue behavior that doesn't need the worker."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localflow.config import load_settings
from localflow.db import Database, run_migrations
from localflow.errors import AppError
from localflow.logging_setup import redact
from localflow.main import create_app
from localflow.models import Job
from localflow.services.assets import AssetService
from localflow.services.events import EventBus
from localflow.services.jobs.service import JobService
from tests.conftest import TOKEN, make_settings


def test_health_is_public(api_client: TestClient) -> None:
    assert TestClient(api_client.app).get("/api/health").json() == {"status": "ok"}


def test_token_required(api_client: TestClient) -> None:
    plain = TestClient(api_client.app)
    r = plain.get("/api/projects")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"
    assert plain.get("/api/projects", headers={"X-LocalFlow-Token": "wrong"}).status_code == 401
    # Query tokens are only accepted for media GETs
    assert plain.get(f"/api/projects?token={TOKEN}").status_code == 401


def test_host_header_checked(api_client: TestClient) -> None:
    r = api_client.get("/api/health", headers={"Host": "evil.example:8765"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_host"


def test_websocket_requires_token(api_client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    plain = TestClient(api_client.app)
    with pytest.raises(WebSocketDisconnect) as exc, plain.websocket_connect("/ws/events") as ws:
        ws.receive_json()
    assert exc.value.code == 4401
    with plain.websocket_connect(f"/ws/events?token={TOKEN}") as ws:
        assert ws.receive_json()["type"] == "worker.status"
    with (
        pytest.raises(WebSocketDisconnect),
        plain.websocket_connect(f"/ws/events?token={TOKEN}", headers={"Origin": "http://evil.example"}) as ws,
    ):
        ws.receive_json()


def test_projects_crud(api_client: TestClient) -> None:
    r = api_client.post("/api/projects", json={"name": "Short film", "aspect_ratio": "9:16"})
    assert r.status_code == 201
    p = r.json()
    assert p["aspect_ratio"] == "9:16" and p["fps"] == 24
    assert p["created_at"].endswith(("Z", "+00:00"))

    assert any(x["id"] == p["id"] for x in api_client.get("/api/projects").json())
    r = api_client.patch(f"/api/projects/{p['id']}", json={"name": "Renamed", "archived": True})
    assert r.json()["name"] == "Renamed"
    assert all(x["id"] != p["id"] for x in api_client.get("/api/projects").json())
    assert any(x["id"] == p["id"] for x in api_client.get("/api/projects?include_archived=true").json())

    project_dir = api_client.app.state.settings.projects_dir / p["id"]
    assert project_dir.is_dir()
    assert api_client.delete(f"/api/projects/{p['id']}").status_code == 204
    assert not project_dir.exists()
    r = api_client.get(f"/api/projects/{p['id']}")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_validation_error_shape(api_client: TestClient) -> None:
    r = api_client.post("/api/projects", json={"name": ""})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    r = api_client.post("/api/generate/t2v", json={"prompt": "x", "unknown_field": 1})
    assert r.status_code == 422


def test_generate_queues_without_worker(api_client: TestClient) -> None:
    r = api_client.post("/api/generate/t2v", json={"prompt": "hello"})
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "queued"
    assert job["params"]["adapter_key"] == "fake"
    assert job["project_id"] == api_client.get("/api/playground").json()["project_id"]


def test_reorder_queued_jobs(api_client: TestClient) -> None:
    ids = [api_client.post("/api/generate/t2i", json={"prompt": str(i)}).json()["id"] for i in range(3)]
    svc = api_client.app.state.services
    assert svc.jobs.next_gpu_job(None, 3).id == ids[0]  # FIFO
    r = api_client.post("/api/jobs/reorder", json={"job_ids": [ids[2], ids[0]]})
    assert r.status_code == 200
    queued = svc.jobs.queued_gpu_jobs(10)
    assert [j.id for j in queued] == [ids[2], ids[0], ids[1]]


def test_batching_prefers_loaded_model(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    run_migrations(db_path)
    jobs = JobService(Database(db_path), EventBus())

    def add(adapter: str, model: str | None, priority: int = 0) -> Job:
        return jobs.create(
            type_="t2i",
            params={"adapter_key": adapter, "request": {"model_entry_id": model}},
            project_id=None,
            priority=priority,
        )

    a = add("flux", "m1")
    b = add("sdxl", "m2")
    c = add("flux", "m1")
    d = add("wan", "m3")
    assert jobs.next_gpu_job(None, 3).id == a.id
    assert jobs.next_gpu_job(("sdxl", "m2"), 3).id == b.id
    assert jobs.next_gpu_job(("wan", "m3"), 3).id == a.id  # d is outside the top K=3
    assert jobs.next_gpu_job(("wan", "m3"), 4).id == d.id
    high = add("sdxl", "m2", priority=10)
    assert jobs.next_gpu_job(("flux", "m1"), 1).id == high.id
    assert c.id  # silence unused


def test_interrupted_jobs_fail_on_startup(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data", start_worker=False)
    run_migrations(settings.db_path)
    db = Database(settings.db_path)
    with db.session() as s:
        s.add(Job(id="01JOBRUNNING0000000000000A", type="t2v", status="running"))
        s.commit()
    db.dispose()
    with TestClient(create_app(settings), headers={"X-LocalFlow-Token": TOKEN}) as client:
        job = client.get("/api/jobs/01JOBRUNNING0000000000000A").json()
        assert job["status"] == "failed"
        assert "Interrupted" in job["error"]


def test_asset_paths_cannot_escape_project(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    assets = AssetService(settings, Database(tmp_path / "x.db"))
    with pytest.raises(AppError):
        assets.resolve_in_project("p1", "../../localflow.db")
    with pytest.raises(AppError):
        assets.project_dir("..")
    assert assets.resolve_in_project("p1", "assets/images/a.png").is_relative_to(settings.projects_dir)


def test_config_toml_and_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text('port = 9000\nvram_profile = "low"\n', encoding="utf-8")
    monkeypatch.setenv("LOCALFLOW_VRAM_PROFILE", "quality")
    s = load_settings(data_dir=tmp_path)
    assert s.port == 9000  # from toml
    assert s.vram_profile == "quality"  # env beats toml
    assert (tmp_path / "projects").is_dir()


def test_default_config_written(tmp_path: Path) -> None:
    load_settings(data_dir=tmp_path / "fresh")
    assert "LOCALFLOW_" in (tmp_path / "fresh" / "config.toml").read_text(encoding="utf-8")


def test_secret_redaction() -> None:
    text = (
        "key sk-abcdefghijklmnop and hf_ABCDEFGHIJKLMNOP; Authorization: Bearer abc.def-ghijklmn; "
        "https://civitai.com/api/download/models/1?token=supersecret123&x=1"
    )
    out = redact(text)
    for secret in ("sk-abcdefghijklmnop", "hf_ABCDEFGHIJKLMNOP", "abc.def-ghijklmn", "supersecret123"):
        assert secret not in out
    assert "&x=1" in out


def test_uvicorn_access_log_is_redacted() -> None:
    import logging

    from localflow.logging_setup import RedactFilter

    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s"',
        ("127.0.0.1", "GET", "/ws/events?token=abcdef123456"),
        None,
    )
    assert RedactFilter().filter(record)
    assert "abcdef123456" not in record.getMessage()
    assert "/ws/events?token=[REDACTED]" in record.getMessage()
