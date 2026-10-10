from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from localflow.config import Settings
from localflow.main import create_app

TOKEN = "test-token-0123456789"
TERMINAL = {"completed", "failed", "cancelled"}


def make_settings(data_dir: Path, **overrides: Any) -> Settings:
    s = Settings(
        data_dir=data_dir,
        api_token=TOKEN,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
        fake_step_delay_s=0.01,
        engine="fake",
        frontend_dist=data_dir / "no-frontend",
        # Never read the developer's real API keys or OS keyring in tests.
        secrets_from_env=False,
        secret_backend="memory",
        **overrides,
    )
    s.ensure_dirs()
    return s


def wait_for(fn: Any, timeout: float = 30.0, interval: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s (last={last!r})")


def wait_job(client: TestClient, job_id: str, statuses: set[str] = TERMINAL, timeout: float = 60.0) -> dict:
    def _check() -> dict | None:
        job = client.get(f"/api/jobs/{job_id}").json()
        return job if job["status"] in statuses else None

    return wait_for(_check, timeout)


def wait_worker_idle(client: TestClient, timeout: float = 60.0) -> dict:
    return wait_for(
        lambda: (s := client.get("/api/system/status").json())["worker"]["state"] == "idle" and s, timeout
    )


@pytest.fixture(scope="module")
def live_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """App with a real (spawned) worker process running the FakeEngine. Shared per module for speed."""
    settings = make_settings(tmp_path_factory.mktemp("data"))
    app = create_app(settings)
    with TestClient(app, headers={"X-LocalFlow-Token": TOKEN}) as client:
        wait_worker_idle(client)
        yield client


@pytest.fixture
def api_client(tmp_path: Path) -> Iterator[TestClient]:
    """App without a worker (fast); for API/DB tests that don't need generation."""
    settings = make_settings(tmp_path / "data", start_worker=False)
    with TestClient(create_app(settings), headers={"X-LocalFlow-Token": TOKEN}) as client:
        yield client
