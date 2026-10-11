"""M7 diagnostics (Section 19): checks, storage, log tails and the redacted bundle."""

from __future__ import annotations

import io
import json
import logging
import zipfile
from typing import Any

from fastapi.testclient import TestClient

from localflow.services.diagnostics import _cuda_check, write_test
from localflow.services.secrets import EncryptedFileStore, MemoryKeyStore, SecretStore
from tests.conftest import wait_worker_idle
from tests.test_m3_imagegen import ok


def checks(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["key"]: c for c in d["checks"]}


def test_diagnostics_report(live_client: TestClient) -> None:
    c = live_client
    wait_worker_idle(c)
    svc = c.app.state.services  # type: ignore[attr-defined]
    (svc.settings.models_dir / "hf" / "blob.bin").write_bytes(b"x" * 2048)

    d = ok(c, "GET", "/api/system/diagnostics", params={"refresh": True})
    by = checks(d)
    assert by["data_dir"]["status"] == "ok" and "writable" in by["data_dir"]["detail"]
    assert by["ffmpeg"]["status"] == "ok" and by["ffprobe"]["status"] == "ok"
    assert by["worker"]["status"] == "ok" and "fake engine" in by["worker"]["detail"]
    assert by["cuda"]["status"] == "skip"  # the FakeEngine never loads torch
    assert by["keyring"]["detail"] and by["llm"]["status"] in ("ok", "warn")
    assert not list((svc.settings.tmp_dir).glob("diag-*"))  # the write test cleans up
    sizes = {f["key"]: f for f in d["storage"]}
    assert set(sizes) == {"models", "projects", "library", "tmp", "logs", "database"}
    assert sizes["models"]["files"] >= 1 and sizes["database"]["size_mb"] > 0
    assert d["versions"]["localflow"] and d["versions"]["python"] and d["versions"]["engine"] == "fake"
    assert d["disk_free_mb"] > 0 and d["disk_total_mb"] >= d["disk_free_mb"]
    names = [f["name"] for f in d["logs"]]
    assert "localflow.log" in names and "worker.log" in names

    status = ok(c, "GET", "/api/system/status")
    assert status["cuda_available"] is None and status["worker"]["runtime"] is None


def test_log_tail_is_readable_and_redacted(live_client: TestClient) -> None:
    c = live_client
    logging.getLogger("localflow.test").warning(
        "probe with key sk-abcdefghijklmnop1234 for job", extra={"job_id": "J1"}
    )
    for h in logging.getLogger().handlers:
        h.flush()
    tail = ok(c, "GET", "/api/system/logs/localflow.log", params={"lines": 50})
    line = next(x for x in tail["lines"] if "probe with key" in x)
    assert "WARNING" in line and "localflow.test:" in line and "[job_id=J1]" in line
    assert "sk-abcdefghijklmnop1234" not in line and "[REDACTED]" in line
    assert tail["total_lines"] >= len(tail["lines"]) and len(tail["lines"]) <= 50
    assert c.get("/api/system/logs/..%2Flocalflow.db").status_code == 404
    assert c.get("/api/system/logs/secrets.key").status_code == 404


def test_bundle_has_no_secrets(live_client: TestClient) -> None:
    c = live_client
    svc = c.app.state.services  # type: ignore[attr-defined]
    (svc.settings.data_dir / "config.toml").write_text(
        'note = "token hf_abcdefghijklmnopqrstu"\n', encoding="utf-8"
    )
    r = c.get("/api/system/diagnostics/bundle")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    files = set(z.namelist())
    assert {"diagnostics.json", "system-status.json", "config.toml", "README.txt"} <= files
    assert "logs/localflow.log.txt" in files and "logs/worker.log.txt" in files
    assert not any(n.endswith((".db", ".enc", ".key")) for n in files)
    assert "hf_abcdefghijklmnopqrstu" not in z.read("config.toml").decode()
    assert json.loads(z.read("diagnostics.json"))["checks"]
    blob = b"".join(z.read(n) for n in files)
    assert b"sk-abcdefghijklmnop1234" not in blob and svc.settings.api_token.encode() not in blob


def test_check_helpers(tmp_path: Any) -> None:
    assert write_test(tmp_path).status == "ok"
    blocked = tmp_path / "file"
    blocked.write_text("not a dir")
    assert write_test(blocked).status == "error"  # data_dir/tmp can't be created under a file

    ok_ = _cuda_check(
        "real", {"torch": "2.9", "cuda": "13.0", "cuda_available": True, "device": "RTX"}, None, "idle"
    )
    assert ok_.status == "ok" and "CUDA 13.0" in ok_.detail and "RTX" in ok_.detail
    bad = _cuda_check("real", {"torch": "2.9+cpu", "cuda": None, "cuda_available": False}, None, "idle")
    assert bad.status == "error" and "can't see a CUDA GPU" in bad.detail
    assert (
        _cuda_check("fake", None, "PyTorch can't see a CUDA GPU", "idle").status == "warn"
    )  # auto fell back
    assert _cuda_check("real", None, None, "starting").status == "warn"

    assert "memory" in SecretStore(MemoryKeyStore(), env=None).backend_name().lower()
    assert "Encrypted file" in SecretStore(EncryptedFileStore(tmp_path), env=None).backend_name()
