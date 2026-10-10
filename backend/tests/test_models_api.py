"""Model management through the API (M2 acceptance, offline): HF + Civitai installs, access states,
cancel/resume, delete, license acknowledgement, and token hygiene."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from localflow.main import create_app
from localflow.services.llm.catalog import KeyCheck
from localflow.services.registry.hf import Access, FetchRequest, HubAccessError, RepoFile, RepoInfo
from localflow.services.secrets import MemoryKeyStore, SecretStore
from tests.conftest import TOKEN, make_settings, wait_for, wait_job

HF_TOKEN = "hf_TESTSECRETTOKEN1234567890"
CIVITAI_KEY = "0123456789abcdef0123456789abcdef"


# --- fakes -----------------------------------------------------------------------------------------
class FakeHub:
    """In-memory Hugging Face: repos with files, gating, and per-repo access state for token holders."""

    def __init__(self) -> None:
        self.repos: dict[str, dict[str, Any]] = {}
        self.fetches: list[tuple[str, list[str], str | None]] = []
        self.hold: asyncio.Event | None = None

    def add(self, repo_id: str, files: dict[str, bytes], gated: str | None = None, **extra: Any) -> None:
        self.repos[repo_id] = {"files": files, "gated": gated, "access": "ok", **extra}

    def repo_info(self, repo_id: str, revision: str | None, token: str | None) -> RepoInfo:
        if repo_id not in self.repos:
            raise HubAccessError("not_found", f"Repository '{repo_id}' was not found.", {"repo_id": repo_id})
        r = self.repos[repo_id]
        return RepoInfo(
            repo_id=repo_id,
            sha="c0ffee",
            gated=r["gated"],
            private=False,
            files=[RepoFile(path=p, size=len(b)) for p, b in r["files"].items()],
            tags=r.get("tags", []),
            license=r.get("license"),
        )

    def access(self, info: RepoInfo, token: str | None, token_type: str | None) -> Access:
        if not info.gated:
            return Access("ok")
        if not token:
            return Access("needs_token", f"{info.repo_id} is gated. Add a Hugging Face token in Settings.")
        state = self.repos[info.repo_id]["access"]
        return Access(state, None if state == "ok" else f"{info.repo_id}: {state}")

    async def fetch(self, req: FetchRequest, on_progress: Any) -> None:
        self.fetches.append((req.repo_id, [f.path for f in req.files], req.token))
        total = sum(f.size or 0 for f in req.files)
        done = 0
        if self.hold is not None:
            on_progress(0, total)
            await self.hold.wait()
        for f in req.files:
            dest = req.local_dir / f.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.repos[req.repo_id]["files"][f.path])
            done += f.size or 0
            on_progress(done, total)


def payload(n: int, seed: int = 1) -> bytes:
    header = b'{"__metadata__":{}}'
    body = bytes((i * 31 + seed) % 256 for i in range(n))
    return len(header).to_bytes(8, "little") + header + body


class FakeCivitai:
    """Civitai API + CDN behind an httpx2.MockTransport."""

    CDN = "https://cdn.example.r2.cloudflarestorage.com"

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.sha = hashlib.sha256(content).hexdigest().upper()
        self.login_required = False
        self.early_access = False
        self.serve_html = False
        self.nsfw = False
        self.bad_sha = False
        self.fail_after: int | None = None  # simulate a dropped connection after N bytes (once)
        self.requests: list[httpx2.Request] = []

    def version(self) -> dict[str, Any]:
        return {
            "id": 200,
            "modelId": 100,
            "name": "v1.0",
            "baseModel": "SDXL 1.0",
            "trainedWords": ["detailed", "sharp"],
            "downloadUrl": "https://civitai.com/api/download/models/200",
            "usageControl": "Download",
            "paidAccess": {"permanent": False, "endsAt": "2026-11-01T00:00:00Z"}
            if self.early_access
            else None,
            "model": {"name": "Detail Tweaker", "type": "LORA", "nsfw": self.nsfw},
            "images": [
                {"url": "https://image.civitai.com/nsfw.jpeg", "nsfwLevel": 8, "type": "image"},
                {"url": "https://image.civitai.com/ok.jpeg", "nsfwLevel": 1, "type": "image"},
            ],
            "files": [
                {
                    "id": 300,
                    "name": "detail-xl.safetensors",
                    "sizeKB": len(self.content) / 1024,
                    "type": "Model",
                    "primary": True,
                    "metadata": {"format": "SafeTensor", "fp": "fp16"},
                    "hashes": {"SHA256": "0" * 64 if self.bad_sha else self.sha},
                    "downloadUrl": "https://civitai.com/api/download/models/200?type=Model&format=SafeTensor",
                    "pickleScanResult": "Success",
                    "virusScanResult": "Success",
                },
                {
                    "id": 301,
                    "name": "detail-xl.ckpt",
                    "sizeKB": 10,
                    "type": "Model",
                    "metadata": {"format": "PickleTensor"},
                    "hashes": {},
                    "downloadUrl": "https://civitai.com/api/download/models/200?type=Model&format=PickleTensor",
                },
            ],
        }

    def model(self) -> dict[str, Any]:
        return {
            "id": 100,
            "name": "Detail Tweaker",
            "type": "LORA",
            "nsfw": self.nsfw,
            "creator": {"username": "bob"},
            "description": "<p>Adds <b>detail</b> &amp; clarity</p>",
            "allowCommercialUse": ["Image"],
            "allowDerivatives": True,
            "allowNoCredit": False,
            "allowDifferentLicense": False,
            "stats": {"downloadCount": 1234, "thumbsUpCount": 99},
            "tags": ["detail"],
            "modelVersions": [self.version()],
        }

    def handler(self, req: httpx2.Request) -> httpx2.Response:
        self.requests.append(req)
        url, path = str(req.url), req.url.path
        if url.startswith(self.CDN):
            assert "authorization" not in req.headers, "token must never be sent to the CDN"
            if self.serve_html:
                return httpx2.Response(200, headers={"content-type": "text/html"}, content=b"<!DOCTYPE html>")
            start = 0
            if rng := req.headers.get("range"):
                start = int(rng.split("=")[1].split("-")[0])
            body = self.content[start:]
            if self.fail_after is not None and start == 0:
                body, self.fail_after = body[: self.fail_after], None
                return httpx2.Response(
                    200, content=body, headers={"content-type": "application/octet-stream"}
                )
            status = 206 if start else 200
            headers = {"content-type": "application/octet-stream", "accept-ranges": "bytes"}
            if start:
                headers["content-range"] = f"bytes {start}-{len(self.content) - 1}/{len(self.content)}"
            return httpx2.Response(status, content=body, headers=headers)
        assert req.url.host == "civitai.com"
        if path == "/api/v1/models":
            items = [self.model(), {**self.model(), "id": 101, "name": "Spicy", "nsfw": True}]
            return httpx2.Response(200, json={"items": items, "metadata": {"nextCursor": "abc|1"}})
        if path == "/api/v1/models/100":
            return httpx2.Response(200, json=self.model())
        if path == "/api/v1/model-versions/200":
            return httpx2.Response(200, json=self.version())
        if path.startswith("/api/download/models/200"):
            if self.login_required and "authorization" not in req.headers:
                return httpx2.Response(401, json={"error": "Unauthorized"})
            if self.early_access:
                return httpx2.Response(
                    403, json={"error": "Early Access", "message": "This asset is in Early Access."}
                )
            return httpx2.Response(
                307,
                headers={"location": f"{self.CDN}/detail-xl.safetensors?X-Amz-Signature=sekret&token=t0k"},
            )
        return httpx2.Response(404, json={"error": "nope"})


class H:
    def __init__(self, client: TestClient, hub: FakeHub, civ: FakeCivitai, store: MemoryKeyStore, data: Path):
        self.client, self.hub, self.civ, self.store, self.data = client, hub, civ, store, data

    def ok(self, method: str, path: str, status: int = 200, **kw: Any) -> Any:
        r = self.client.request(method, path, **kw)
        assert r.status_code == status, r.text
        return r.json()

    def entry(self, key: str) -> dict[str, Any]:
        return next(m for m in self.ok("GET", "/api/models") if m["catalog_key"] == key)


SDXL_FILES = {
    "model_index.json": b"{}",
    "scheduler/scheduler_config.json": b"{}",
    "text_encoder/config.json": b"{}",
    "text_encoder/model.fp16.safetensors": payload(500),
    "text_encoder/model.safetensors": payload(900),
    "text_encoder_2/config.json": b"{}",
    "text_encoder_2/model.fp16.safetensors": payload(600),
    "tokenizer/vocab.json": b"{}",
    "tokenizer_2/vocab.json": b"{}",
    "unet/config.json": b"{}",
    "unet/diffusion_pytorch_model.fp16.safetensors": payload(2000),
    "vae/config.json": b"{}",
    "README.md": b"# sdxl",
}
FLUX_BASE = {
    "model_index.json": b"{}",
    "scheduler/scheduler_config.json": b"{}",
    "text_encoder/model.safetensors": payload(300),
    "text_encoder_2/model.safetensors": payload(400),
    "tokenizer/vocab.json": b"{}",
    "tokenizer_2/spiece.model": b"x",
    "vae/diffusion_pytorch_model.safetensors": payload(100),
    "transformer/config.json": b"{}",
    "flux1-dev.safetensors": payload(5000),
}


@pytest.fixture
def h(tmp_path: Path) -> Iterator[H]:
    hub = FakeHub()
    hub.add("stabilityai/stable-diffusion-xl-base-1.0", SDXL_FILES, license="openrail++")
    hub.add(
        "madebyollin/sdxl-vae-fp16-fix",
        {"config.json": b"{}", "diffusion_pytorch_model.safetensors": payload(150)},
    )
    hub.add("black-forest-labs/FLUX.1-dev", FLUX_BASE, gated="auto")
    hub.add(
        "city96/FLUX.1-dev-gguf",
        {"flux1-dev-Q8_0.gguf": payload(1200), "flux1-dev-Q4_K_S.gguf": payload(700)},
    )
    hub.add("hexgrad/Kokoro-82M", {"config.json": b"{}", "kokoro-v1_0.pth": b"pickle", "voices/af.pt": b"p"})
    hub.add(
        "someone/tiny-lora",
        {"lora.safetensors": payload(64), "README.md": b"hi", "train.py": b"x"},
        tags=["lora"],
    )
    civ = FakeCivitai(payload(300_000))
    store = MemoryKeyStore()

    async def checker(name: str, value: str) -> KeyCheck:
        return KeyCheck(ok=True, account={"username": "lrama1"}, token_type="read")

    data = tmp_path / "data"
    app = create_app(
        make_settings(data, start_worker=False),
        secrets=SecretStore(store, env=None),
        key_checker=checker,
        hf_backend=hub,
        civitai_transport=httpx2.MockTransport(civ.handler),
    )
    with TestClient(app, headers={"X-LocalFlow-Token": TOKEN}) as client:
        yield H(client, hub, civ, store, data)


def _install_and_wait(h: H, entry_id: str) -> dict[str, Any]:
    out = h.ok("POST", f"/api/models/{entry_id}/install", 201)
    job = wait_job(h.client, out["job_id"])
    assert job["status"] == "completed", job
    return job


# --- tests -----------------------------------------------------------------------------------------
def test_catalog_listed_with_options(h: H) -> None:
    models = h.ok("GET", "/api/models")
    keys = [m["catalog_key"] for m in models]
    assert keys[:3] == ["flux1-dev-q8", "flux1-dev-q4", "sdxl-base"]  # catalog order
    sdxl = h.entry("sdxl-base")
    assert sdxl["status"] == "available" and sdxl["license"].startswith("CreativeML")
    assert sdxl["approx_size_bytes"] > 0 and sdxl["recommended"] is True
    kokoro = h.entry("kokoro-82m")
    assert kokoro["requires_pickle"] and kokoro["pickle_blocked"]
    opts = h.ok("GET", "/api/models/options")
    assert {a["key"] for a in opts["adapters"]} >= {"flux", "sdxl", "wan", "kokoro"}
    assert opts["hf"]["configured"] is False and opts["free_bytes"] > 0
    assert h.ok("GET", "/api/models", params={"capability": "t2v", "catalog": True})


def test_install_catalog_model_then_delete_frees_disk(h: H) -> None:
    sdxl = h.entry("sdxl-base")
    job = _install_and_wait(h, sdxl["id"])
    assert job["type"] == "download_model" and job["params"] == {
        "model_entry_id": sdxl["id"],
        "name": sdxl["name"],
    }
    e = h.ok("GET", f"/api/models/{sdxl['id']}")
    assert (
        e["status"] == "installed"
        and e["local_path"] == "models/hf/repos/stabilityai--stable-diffusion-xl-base-1.0"
    )
    base = h.data / e["local_path"]
    assert (base / "unet/diffusion_pytorch_model.fp16.safetensors").is_file()
    assert not (base / "text_encoder/model.safetensors").exists()  # only the fp16 variant was selected
    assert not (base / "README.md").exists()
    assert (
        h.data / "models/hf/repos/madebyollin--sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors"
    ).is_file()
    expected = sum(
        len(v) for k, v in SDXL_FILES.items() if k in {f["path"] for p in e["parts"] for f in p["files"]}
    )
    expected += len(payload(150)) + 2
    assert e["size_bytes"] == expected
    assert all(t is None for _, _, t in h.hub.fetches)  # public repos, no token configured

    out = h.ok("DELETE", f"/api/models/{sdxl['id']}")
    assert out["freed_bytes"] == expected and out["deleted_entry"] is False
    assert out["entry"]["status"] == "available" and out["entry"]["local_path"] is None
    assert not base.exists()


def test_gated_access_states_and_check_again(h: H) -> None:
    flux = h.entry("flux1-dev-q8")
    # 1) no token → needs_token
    r = h.client.post(f"/api/models/{flux['id']}/install")
    assert r.status_code == 409 and r.json()["error"]["code"] == "needs_token"
    assert h.ok("GET", f"/api/models/{flux['id']}")["access"] == "needs_token"
    insp = h.ok(
        "POST",
        "/api/models/hf/inspect",
        json={"repo_id": "https://huggingface.co/black-forest-labs/FLUX.1-dev"},
    )
    assert (
        insp["repo_id"] == "black-forest-labs/FLUX.1-dev"
        and insp["gated"]
        and insp["access"] == "needs_token"
    )
    assert "flux1-dev.safetensors" not in insp["default_selection"]  # root single file vs diffusers layout

    # 2) token, license not accepted → actionable message with the account + a model-page link
    h.store.set("hf_token", HF_TOKEN)
    h.ok("POST", "/api/settings/secrets/hf_token/test")
    h.hub.repos["black-forest-labs/FLUX.1-dev"]["access"] = "license_not_accepted"
    r = h.client.post(f"/api/models/{flux['id']}/install")
    err = r.json()["error"]
    assert r.status_code == 409 and err["code"] == "license_not_accepted"
    assert "signed in as lrama1" in err["message"] and "Check again" in err["message"]
    assert err["details"]["model_url"] == "https://huggingface.co/black-forest-labs/FLUX.1-dev"
    e = h.ok("GET", f"/api/models/{flux['id']}")
    assert e["access"] == "license_not_accepted" and e["gated"] and e["status"] == "available"
    gated_part = next(p for p in e["parts"] if p["repo_id"] == "black-forest-labs/FLUX.1-dev")
    assert gated_part["access"] == "license_not_accepted" and gated_part["gated"]

    # 3) accepted on the website → "Check again" → ok → install completes, token passed explicitly
    h.hub.repos["black-forest-labs/FLUX.1-dev"]["access"] = "ok"
    assert h.ok("POST", f"/api/models/{flux['id']}/check-access")["access"] == "ok"
    _install_and_wait(h, flux["id"])
    assert {t for _, _, t in h.hub.fetches} == {HF_TOKEN}
    e = h.ok("GET", f"/api/models/{flux['id']}")
    assert e["status"] == "installed" and e["access"] == "ok"

    # No token anywhere it could leak (Section 9.6).
    jobs = h.ok("GET", "/api/jobs")["items"]
    blob = json.dumps(jobs) + json.dumps(h.ok("GET", "/api/models"))
    assert HF_TOKEN not in blob
    log = (h.data / "logs" / "localflow.log").read_text(encoding="utf-8")
    assert HF_TOKEN not in log


def test_shared_base_files_survive_deleting_one_entry(h: H) -> None:
    h.store.set("hf_token", HF_TOKEN)
    q8, q4 = h.entry("flux1-dev-q8"), h.entry("flux1-dev-q4")
    _install_and_wait(h, q8["id"])
    _install_and_wait(h, q4["id"])
    repo = h.data / "models/hf/repos"
    out = h.ok("DELETE", f"/api/models/{q8['id']}")
    assert out["freed_bytes"] == len(payload(1200))  # only its own transformer file
    assert not (repo / "city96--FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf").exists()
    assert (repo / "city96--FLUX.1-dev-gguf/flux1-dev-Q4_K_S.gguf").is_file()
    assert (repo / "black-forest-labs--FLUX.1-dev/text_encoder_2/model.safetensors").is_file()
    h.ok("DELETE", f"/api/models/{q4['id']}")
    assert not (repo / "black-forest-labs--FLUX.1-dev").exists()


def test_hf_inspect_and_user_install(h: H) -> None:
    insp = h.ok("POST", "/api/models/hf/inspect", json={"repo_id": "someone/tiny-lora"})
    assert insp["access"] == "ok" and insp["suggestion"]["kind"] == "lora"
    by_path = {f["path"]: f for f in insp["files"]}
    assert by_path["train.py"]["blocked_reason"] and not by_path["train.py"]["default_selected"]
    assert insp["default_selection"] == ["lora.safetensors"]

    r = h.client.post("/api/models/hf/install", json={"repo_id": "someone/tiny-lora", "adapter_key": "nope"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_adapter"
    r = h.client.post(
        "/api/models/hf/install",
        json={"repo_id": "someone/tiny-lora", "adapter_key": "sdxl", "allow_patterns": ["*"]},
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "file_blocked"  # train.py

    out = h.ok(
        "POST", "/api/models/hf/install", 201, json={"repo_id": "someone/tiny-lora", "adapter_key": "sdxl"}
    )
    assert wait_job(h.client, out["job_id"])["status"] == "completed"
    e = h.ok("GET", f"/api/models/{out['entry']['id']}")
    assert (e["source"], e["kind"], e["adapter_key"], e["status"]) == ("hf", "lora", "sdxl", "installed")
    assert (
        h.ok("POST", "/api/models/hf/inspect", json={"repo_id": "someone/tiny-lora"})["existing_entry_id"]
        == e["id"]
    )
    r = h.client.post("/api/models/hf/install", json={"repo_id": "someone/tiny-lora", "adapter_key": "sdxl"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_installed"
    out = h.ok("DELETE", f"/api/models/{e['id']}")
    assert out["deleted_entry"] and out["freed_bytes"] == len(payload(64))
    assert h.client.get(f"/api/models/{e['id']}").status_code == 404

    r = h.client.post("/api/models/hf/inspect", json={"repo_id": "nobody/missing"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_component_install_has_no_capabilities(h: H) -> None:
    body = {"repo_id": "madebyollin/sdxl-vae-fp16-fix", "adapter_key": "sdxl", "kind": "vae"}
    out = h.ok("POST", "/api/models/hf/install", 201, json=body)
    assert out["entry"]["kind"] == "vae" and out["entry"]["capabilities"] == []
    assert wait_job(h.client, out["job_id"])["status"] == "completed"


def test_cancel_download_then_resume(h: H) -> None:
    h.hub.hold = asyncio.Event()
    sdxl = h.entry("sdxl-base")
    out = h.ok("POST", f"/api/models/{sdxl['id']}/install", 201)
    wait_job(h.client, out["job_id"], {"running"})
    assert h.ok("GET", f"/api/models/{sdxl['id']}")["status"] == "downloading"
    r = h.client.delete(f"/api/models/{sdxl['id']}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "download_in_progress"
    h.ok("POST", f"/api/jobs/{out['job_id']}/cancel")
    assert wait_job(h.client, out["job_id"])["status"] == "cancelled"
    e = h.ok("GET", f"/api/models/{sdxl['id']}")
    assert e["status"] == "available" and e["job_id"] is None

    h.hub.hold = None
    _install_and_wait(h, sdxl["id"])
    assert h.ok("GET", f"/api/models/{sdxl['id']}")["status"] == "installed"


def test_pickle_entry_blocked_by_default(h: H) -> None:
    r = h.client.post(f"/api/models/{h.entry('kokoro-82m')['id']}/install")
    assert r.status_code == 409 and r.json()["error"]["code"] == "pickle_weights_blocked"


def test_license_ack_gates_generation(h: H) -> None:
    sdxl = h.entry("sdxl-base")
    body = {"prompt": "a gull", "model_entry_id": sdxl["id"]}
    r = h.client.post("/api/generate/t2i", json=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_not_installed"
    _install_and_wait(h, sdxl["id"])
    r = h.client.post("/api/generate/t2i", json=body)
    err = r.json()["error"]
    assert r.status_code == 409 and err["code"] == "license_not_acknowledged"
    assert err["details"]["license_url"].startswith("https://")
    e = h.ok("POST", f"/api/models/{sdxl['id']}/acknowledge-license")
    assert e["license_acknowledged"] and e["license_acknowledged_at"]
    assert h.ok("POST", "/api/generate/t2i", 201, json=body)["status"] == "queued"
    # LoRAs need their own acknowledgement too.
    r = h.client.post("/api/generate/t2i", json={**body, "loras": [{"model_entry_id": "missing"}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_model"


def test_render_constraints_follow_project_video_model(h: H) -> None:
    from localflow.models import Project

    a14b = h.entry("wan22-i2v-a14b-q4")
    svc = h.client.app.state.services  # type: ignore[attr-defined]
    rc = svc.story.render_constraints(Project(name="p", default_video_model_id=a14b["id"]))
    assert rc.video_model == a14b["name"] and rc.supported_modes == ["i2v"] and rc.max_clip_seconds == 5.0
    default = svc.story.render_constraints(Project(name="p"))
    assert default.supported_modes == ["t2v", "i2v"]


# --- Civitai -------------------------------------------------------------------------------------
def test_civitai_search_hides_nsfw_and_maps_versions(h: H) -> None:
    out = h.ok(
        "GET", "/api/models/civitai/search", params={"query": "detail", "types": ["LORA"], "sort": "Newest"}
    )
    assert out["hidden_nsfw"] == 1 and out["next_cursor"] == "abc|1"
    (m,) = out["items"]
    assert (
        m["kind"] == "lora"
        and m["creator"] == "bob"
        and m["preview_url"] == "https://image.civitai.com/ok.jpeg"
    )
    v = m["versions"][0]
    assert (v["base_model"], v["adapter_key"], v["supported"]) == ("sdxl", "sdxl", True)
    assert v["trained_words"] == ["detailed", "sharp"] and v["downloadable"] and not v["early_access"]
    safe, pickle = v["files"]
    assert safe["blocked_reason"] is None and safe["size_bytes"] == len(h.civ.content)
    assert pickle["blocked_reason"]
    sent = h.civ.requests[-1]
    assert sent.url.params.get("nsfw") == "false" and sent.url.params.get_list("types") == ["LORA"]
    detail = h.ok("GET", "/api/models/civitai/100")
    assert detail["description"] == "Adds detail & clarity"


def test_civitai_install_anonymous(h: H) -> None:
    out = h.ok("POST", "/api/models/civitai/install", 201, json={"version_id": 200})
    assert wait_job(h.client, out["job_id"])["status"] == "completed"
    e = h.ok("GET", f"/api/models/{out['entry']['id']}")
    assert (e["source"], e["source_ref"], e["kind"], e["adapter_key"]) == (
        "civitai",
        "100:200",
        "lora",
        "sdxl",
    )
    assert e["trigger_words"] == ["detailed", "sharp"] and e["capabilities"] == ["t2i"]
    assert e["local_path"] == "models/civitai/100/200/detail-xl.safetensors"
    path = h.data / e["local_path"]
    assert path.read_bytes() == h.civ.content and not path.with_name(path.name + ".part").exists()
    assert e["civitai"]["permissions"]["allowDerivatives"] is True
    blob = json.dumps(h.ok("GET", "/api/models")) + json.dumps(h.ok("GET", "/api/jobs"))
    assert "sekret" not in blob and "X-Amz" not in blob and "token=" not in blob
    out = h.ok("DELETE", f"/api/models/{e['id']}")
    assert out["freed_bytes"] == len(h.civ.content) and not (h.data / "models/civitai/100").exists()


def test_civitai_login_required(h: H) -> None:
    h.civ.login_required = True
    r = h.client.post("/api/models/civitai/install", json={"version_id": 200})
    err = r.json()["error"]
    assert r.status_code == 409 and err["code"] == "needs_token" and "Civitai API key" in err["message"]
    h.store.set("civitai_api_key", CIVITAI_KEY)
    first = len(h.civ.requests)
    out = h.ok("POST", "/api/models/civitai/install", 201, json={"version_id": 200})
    assert wait_job(h.client, out["job_id"])["status"] == "completed"
    auth = [r.headers.get("authorization") for r in h.civ.requests[first:] if r.url.host == "civitai.com"]
    assert auth and all(a == f"Bearer {CIVITAI_KEY}" for a in auth)
    blob = json.dumps(h.ok("GET", "/api/models")) + json.dumps(h.ok("GET", "/api/jobs"))
    assert CIVITAI_KEY not in blob
    assert CIVITAI_KEY not in (h.data / "logs" / "localflow.log").read_text(encoding="utf-8")


def test_civitai_early_access_forbidden(h: H) -> None:
    h.civ.early_access = True
    r = h.client.post("/api/models/civitai/install", json={"version_id": 200})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    assert "Early Access" in r.json()["error"]["message"]


def test_civitai_html_response_is_auth_failure(h: H) -> None:
    h.civ.serve_html = True
    out = h.ok("POST", "/api/models/civitai/install", 201, json={"version_id": 200})
    job = wait_job(h.client, out["job_id"])
    assert job["status"] == "failed" and job["result"]["code"] == "needs_token"
    e = h.ok("GET", f"/api/models/{out['entry']['id']}")
    assert e["status"] == "error" and e["access"] == "needs_token"
    assert not list((h.data / "models/civitai").rglob("*.safetensors*"))


def test_civitai_sha_mismatch_rejected(h: H) -> None:
    h.civ.bad_sha = True
    out = h.ok("POST", "/api/models/civitai/install", 201, json={"version_id": 200})
    job = wait_job(h.client, out["job_id"])
    assert job["status"] == "failed" and job["result"]["code"] == "sha256_mismatch"
    assert not list((h.data / "models/civitai").rglob("*.safetensors*"))


def test_civitai_resume_with_range(h: H) -> None:
    h.civ.fail_after = 100_000  # first attempt is truncated → size check fails, partial file kept
    out = h.ok("POST", "/api/models/civitai/install", 201, json={"version_id": 200})
    job = wait_job(h.client, out["job_id"])
    assert job["status"] == "failed" and job["result"]["code"] == "download_corrupt"
    # Simulate a partial download left behind, then resume: only the rest is fetched with Range.
    part = h.data / "models/civitai/100/200/detail-xl.safetensors.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(h.civ.content[:120_000])
    out = h.ok("POST", f"/api/models/{out['entry']['id']}/install", 201)
    assert wait_job(h.client, out["job_id"])["status"] == "completed"
    cdn = [r for r in h.civ.requests if r.url.host != "civitai.com"]
    assert cdn[-1].headers["range"] == "bytes=120000-"
    assert (h.data / "models/civitai/100/200/detail-xl.safetensors").read_bytes() == h.civ.content


def test_civitai_pickle_file_blocked(h: H) -> None:
    r = h.client.post("/api/models/civitai/install", json={"version_id": 200, "file_id": 301})
    assert r.status_code == 400 and r.json()["error"]["code"] == "file_blocked"


# --- hub accounts --------------------------------------------------------------------------------
def test_import_hf_cli_login(h: H, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "nohome")
    monkeypatch.delenv("HF_HOME", raising=False)
    r = h.client.post("/api/settings/secrets/hf_token/import-cli")
    assert r.status_code == 404 and r.json()["error"]["code"] == "hf_cli_token_not_found"
    token_file = tmp_path / "token"
    token_file.write_text(HF_TOKEN + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_file))
    st = h.ok("POST", "/api/settings/secrets/hf_token/import-cli")
    assert st["configured"] and st["source"] == "memory" and st["last4"] == HF_TOKEN[-4:]
    assert st["account"] == {"username": "lrama1"}
    assert h.store.get("hf_token") == HF_TOKEN
    assert not (h.data / "models" / "hf" / "token").exists()  # never written into HF_HOME
    assert wait_for(lambda: h.ok("GET", "/api/models/options")["hf"]["username"] == "lrama1")
