"""Registry building blocks: file policy, URL/log hygiene, catalog sync, inference, HF access states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from pydantic import ValidationError
from sqlmodel import select

from localflow.db import Database, run_migrations
from localflow.logging_setup import redact
from localflow.models import ModelEntry
from localflow.services.registry import policy
from localflow.services.registry.catalog import CatalogEntry, load_catalog, sync_catalog
from localflow.services.registry.hf import HubClient, RepoInfo, classify_gated_error, worst_access
from localflow.services.registry.infer import civitai_base_model, suggest_civitai, suggest_hf


# --- policy --------------------------------------------------------------------------------------
def test_classify_and_block() -> None:
    assert policy.classify("unet/diffusion_pytorch_model.fp16.safetensors") == "weight"
    assert policy.classify("flux1-dev-Q8_0.gguf") == "weight"
    assert policy.classify("pytorch_model.bin") == "pickle"
    assert policy.classify("model.onnx") == "other_weight"
    assert policy.classify("tokenizer/vocab.json") == "config"
    assert policy.classify("pipeline.py") == "code"
    assert policy.blocked_reason("pipeline.py", True) == "Code files are never downloaded."
    assert policy.blocked_reason("x.ckpt", False) and policy.blocked_reason("x.ckpt", True) is None
    assert policy.blocked_reason("model.safetensors", False) is None


def test_default_selection_prefers_safe_fp16_diffusers_files() -> None:
    files = [
        ".gitattributes",
        "README.md",
        "model_index.json",
        "sd_xl_base_1.0.safetensors",  # root single-file duplicate of the diffusers layout
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder/pytorch_model.bin",  # pickle duplicate
        "onnx/unet/model.onnx",
        "examples/cat.png",
        "custom_pipeline.py",
    ]
    assert policy.default_selection(files, allow_pickle=False) == [
        "model_index.json",
        "text_encoder/model.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
    ]


def test_default_selection_pickle_only_repo() -> None:
    files = ["config.json", "kokoro-v1_0.pth", "voices/af.pt"]
    assert policy.default_selection(files, allow_pickle=False) == ["config.json"]
    assert policy.default_selection(files, allow_pickle=True) == files


def test_match_patterns_and_validate() -> None:
    files = ["a/x.safetensors", "a/y.json", "b/z.safetensors", "run.py"]
    assert policy.match_patterns(files, ["a/"]) == ["a/x.safetensors", "a/y.json"]
    assert policy.match_patterns(files, ["*.safetensors"]) == ["a/x.safetensors", "b/z.safetensors"]
    with pytest.raises(Exception) as exc:
        policy.validate_selection(["run.py"], allow_pickle=True)
    assert getattr(exc.value, "code", None) == "file_blocked"


def test_safe_url_and_redaction() -> None:
    url = "https://civitai.com/api/download/models/1?type=Model&format=SafeTensor&token=abc123"
    assert policy.safe_url(url) == "https://civitai.com/api/download/models/1?type=Model&format=SafeTensor"
    signed = "https://cdn.r2.dev/f.safetensors?X-Amz-Signature=deadbeef&X-Amz-Credential=me"
    assert policy.safe_url(signed) == "https://cdn.r2.dev/f.safetensors"
    assert "deadbeef" not in redact(f"GET {signed} 200")
    assert "0123456789abcdef0123456789abcdef" not in redact("key 0123456789abcdef0123456789abcdef used")
    sha = "a" * 64
    assert sha in redact(f"sha {sha}")  # hashes stay readable


def test_check_disk(tmp_path: Path) -> None:
    policy.check_disk(tmp_path, 1)
    with pytest.raises(Exception) as exc:
        policy.check_disk(tmp_path, 10**18)
    assert getattr(exc.value, "code", None) == "insufficient_disk"


# --- catalog -------------------------------------------------------------------------------------
def test_catalog_loads_and_is_consistent() -> None:
    entries = load_catalog()
    keys = {e.key for e in entries}
    assert {"flux1-dev-q8", "sdxl-base", "wan22-ti2v-5b", "flux1-kontext-dev-q4", "kokoro-82m"} <= keys
    for e in entries:
        assert e.parts and e.license and e.license_url.startswith("https://")
        assert all("/" in p.repo_id for p in e.parts)


def _db(tmp_path: Path) -> Database:
    run_migrations(tmp_path / "t.db")
    return Database(tmp_path / "t.db")


def test_sync_catalog_keeps_install_state(tmp_path: Path) -> None:
    db = _db(tmp_path)
    assert sync_catalog(db) == len(load_catalog())
    with db.session() as s:
        row = s.exec(select(ModelEntry).where(ModelEntry.catalog_key == "sdxl-base")).one()
        row.status, row.local_path, row.license_acknowledged_at = "installed", "models/hf/x", row.created_at
        s.add(row)
        s.commit()
    sync_catalog(db)
    with db.session() as s:
        row = s.exec(select(ModelEntry).where(ModelEntry.catalog_key == "sdxl-base")).one()
        assert row.status == "installed" and row.local_path == "models/hf/x"
        assert row.license_acknowledged_at is not None
        count = len(s.exec(select(ModelEntry)).all())
    assert count == len(load_catalog())

    # Removing entries from the catalog drops uninstalled rows and keeps installed ones.
    only = tuple(e for e in load_catalog() if e.key == "flux1-dev-q8")
    sync_catalog(db, only)
    with db.session() as s:
        keys = {r.catalog_key: r for r in s.exec(select(ModelEntry)).all()}
    assert set(keys) == {"flux1-dev-q8", "sdxl-base"}
    assert keys["sdxl-base"].meta_json.get("catalog_removed") is True
    db.dispose()


def test_catalog_entry_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CatalogEntry.model_validate({"key": "x", "bogus": 1})


# --- inference -----------------------------------------------------------------------------------
def test_civitai_base_model_mapping() -> None:
    assert civitai_base_model("SDXL 1.0") == "sdxl"
    assert civitai_base_model("Pony") == "pony"
    assert civitai_base_model("Illustrious") == "illustrious"
    assert civitai_base_model("Flux.1 D") == "flux1"
    assert civitai_base_model("Wan Video 14B i2v 720p") == "wan2.1-14b-i2v"
    assert civitai_base_model("Wan Video 2.2 T2V-A14B") == "wan2.2-a14b-t2v"
    assert civitai_base_model("SD 1.5") is None
    s = suggest_civitai("LORA", "Illustrious")
    assert (s.kind, s.adapter_key, s.capabilities, s.supported) == ("lora", "sdxl", ["t2i"], True)
    u = suggest_civitai("Checkpoint", "Krea 2")
    assert u.supported is False and u.adapter_key is None and u.base_model == "krea-2"
    assert suggest_civitai("Workflows", "SDXL 1.0").kind == "other"


def test_suggest_hf() -> None:
    s = suggest_hf("black-forest-labs/FLUX.1-dev", {"tags": ["diffusers:FluxPipeline"], "files": []})
    assert (s.base_model, s.adapter_key, s.kind) == ("flux1", "flux", "checkpoint")
    v = suggest_hf("madebyollin/sdxl-vae-fp16-fix", {"files": ["config.json", "x.safetensors"]})
    assert (v.kind, v.base_model) == ("vae", "sdxl")
    w = suggest_hf("Wan-AI/Wan2.2-TI2V-5B-Diffusers", {"files": ["model_index.json"]})
    assert w.adapter_key == "wan" and w.capabilities == ["t2v", "i2v", "extend"]
    lora = suggest_hf(
        "someone/cool-style", {"tags": ["lora"], "base_models": ["black-forest-labs/FLUX.1-dev"]}
    )
    assert (lora.kind, lora.adapter_key) == ("lora", "flux")
    unknown = suggest_hf("someone/thing", {})
    assert unknown.supported is False


# --- Hugging Face access states (Section 9.3) ------------------------------------------------------
def _err(cls: type[Exception], status: int, message: str) -> Exception:
    resp = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/models/x"))
    return cls(message, response=resp)  # type: ignore[call-arg]


class _FakeApi:
    def __init__(self, exc: Exception | None) -> None:
        self.exc = exc
        self.tokens: list[Any] = []

    def auth_check(self, repo_id: str, token: Any = None) -> None:
        self.tokens.append(token)
        if self.exc:
            raise self.exc


def _hub(exc: Exception | None) -> tuple[HubClient, _FakeApi]:
    api = _FakeApi(exc)
    hub = HubClient()
    hub._api = lambda: api  # type: ignore[method-assign]
    return hub, api


GATED = RepoInfo(repo_id="black-forest-labs/FLUX.1-dev", sha="abc", gated="auto", private=False, files=[])


@pytest.mark.parametrize(
    ("exc", "token", "state"),
    [
        (None, None, "needs_token"),
        (None, "hf_tok", "ok"),
        (
            _err(
                GatedRepoError,
                403,
                "Access to model X is restricted and you are not in the authorized list. Visit",
            ),
            "hf_tok",
            "license_not_accepted",
        ),
        (
            _err(
                GatedRepoError,
                403,
                "Your request to access model X is awaiting a review from the repo authors.",
            ),
            "hf_tok",
            "pending_approval",
        ),
        (
            _err(
                GatedRepoError,
                403,
                "Please enable access to public gated repositories in your fine-grained token settings",
            ),
            "hf_tok",
            "token_no_access",
        ),
        (_err(GatedRepoError, 401, "You must be authenticated"), "hf_tok", "needs_token"),
        (_err(RepositoryNotFoundError, 404, "not found"), "hf_tok", "token_no_access"),
        (_err(HfHubHTTPError, 403, "forbidden"), "hf_tok", "token_no_access"),
    ],
)
def test_hf_access_states(exc: Exception | None, token: str | None, state: str) -> None:
    hub, api = _hub(exc)
    assert hub.access(GATED, token, "read").state == state
    assert all(t == token for t in api.tokens)  # token always passed explicitly


def test_public_repo_needs_no_check() -> None:
    hub, api = _hub(None)
    public = RepoInfo(repo_id="a/b", sha=None, gated=None, private=False, files=[])
    assert hub.access(public, None, None).state == "ok" and api.tokens == []


def test_classify_gated_without_token_and_worst() -> None:
    assert classify_gated_error("a/b", 401, "x", False, None).state == "needs_token"
    assert worst_access(["ok", "license_not_accepted", None]) == "license_not_accepted"
    assert worst_access(["ok", "needs_token", "license_not_accepted"]) == "needs_token"
    assert worst_access([None]) is None


def test_repo_info_token_false_when_missing() -> None:
    seen: dict[str, Any] = {}

    class Api:
        def model_info(self, repo_id: str, **kw: Any) -> Any:
            seen.update(kw)
            raise _err(RepositoryNotFoundError, 401, "nope")

    hub = HubClient()
    hub._api = lambda: Api()  # type: ignore[method-assign]
    with pytest.raises(Exception) as exc:
        hub.repo_info("a/b", None, None)
    assert seen["token"] is False  # never implicit token discovery
    assert getattr(exc.value, "code", None) == "not_found"
