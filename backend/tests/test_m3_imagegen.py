"""M3 flows through the API: ingredient candidates, keyframe composition, model resolution (FakeEngine)."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from localflow.main import create_app
from localflow.models import ModelEntry, utcnow
from tests.conftest import TOKEN, make_settings, wait_job, wait_worker_idle


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = make_settings(tmp_path_factory.mktemp("m3"))
    with TestClient(create_app(settings), headers={"X-LocalFlow-Token": TOKEN}) as c:
        wait_worker_idle(c)
        yield c


def ok(c: TestClient, method: str, path: str, status: int = 200, **kw: Any) -> Any:
    r = c.request(method, path, **kw)
    assert r.status_code == status, r.text
    return r.json() if r.content else None


def install(c: TestClient, catalog_key: str, *, ack: bool = True) -> str:
    """Marks a catalog entry installed (no download) so jobs can name it."""
    svc = c.app.state.services  # type: ignore[attr-defined]
    with svc.db.session() as s:
        from sqlmodel import select

        e = s.exec(select(ModelEntry).where(ModelEntry.catalog_key == catalog_key)).one()
        e.status = "installed"
        e.license_acknowledged_at = utcnow() if ack else None
        s.add(e)
        s.commit()
        return e.id


def uninstall(c: TestClient, entry_id: str) -> None:
    svc = c.app.state.services  # type: ignore[attr-defined]
    with svc.db.session() as s:
        e = s.get(ModelEntry, entry_id)
        e.status, e.license_acknowledged_at = "available", None
        s.add(e)
        s.commit()


def project_with_cast(c: TestClient) -> dict[str, str]:
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Lighthouse", "aspect_ratio": "16:9"})["id"]
    style = ok(
        c,
        "POST",
        f"/api/projects/{pid}/ingredients",
        201,
        json={"kind": "style", "name": "Style", "key": "style", "visual_prompt": "moody 35mm film"},
    )
    keeper = ok(
        c,
        "POST",
        f"/api/projects/{pid}/ingredients",
        201,
        json={
            "kind": "character",
            "name": "Old Keeper",
            "visual_prompt": "weathered man, grey beard, yellow raincoat",
        },
    )
    dog = ok(
        c,
        "POST",
        f"/api/projects/{pid}/ingredients",
        201,
        json={"kind": "character", "name": "Pip", "visual_prompt": "small white terrier, red scarf"},
    )
    shore = ok(
        c,
        "POST",
        f"/api/projects/{pid}/ingredients",
        201,
        json={"kind": "location", "name": "Shore", "visual_prompt": "rocky shore at dusk"},
    )
    scene = ok(
        c,
        "POST",
        f"/api/projects/{pid}/scenes",
        201,
        json={"title": "Arrival", "location_ingredient_id": shore["id"]},
    )
    shot = ok(
        c,
        "POST",
        f"/api/scenes/{scene['id']}/shots",
        201,
        json={
            "keyframe_prompt": "The keeper and Pip walk along the shore",
            "video_prompt": "they walk",
            "ingredient_ids": [keeper["id"], dog["id"]],
        },
    )
    return {
        "pid": pid,
        "style": style["id"],
        "keeper": keeper["id"],
        "dog": dog["id"],
        "shore": shore["id"],
        "shot": shot["id"],
    }


def test_ingredient_candidates_primary_and_upload(client: TestClient) -> None:
    ids = project_with_cast(client)
    job = ok(client, "POST", f"/api/ingredients/{ids['keeper']}/generate", 201, json={"count": 3})
    assert job["type"] == "t2i" and job["target_ref"] == {
        "kind": "ingredient",
        "ingredient_id": ids["keeper"],
    }
    prompt = job["params"]["request"]["prompt"]
    assert prompt.startswith("weathered man, grey beard, yellow raincoat. character reference portrait")
    assert "moody 35mm film" in prompt and job["params"]["request"]["num_images"] == 3
    assert wait_job(client, job["id"])["status"] == "completed"
    cands = ok(client, "GET", f"/api/projects/{ids['pid']}/assets", params={"ingredient_id": ids["keeper"]})[
        "items"
    ]
    assert len(cands) == 3 and all(
        a["generation"]["target_ref"]["ingredient_id"] == ids["keeper"] for a in cands
    )
    ing = ok(client, "GET", f"/api/ingredients/{ids['keeper']}")
    assert ing["primary_asset_id"] in {a["id"] for a in cands}  # first candidate auto-selected
    other = next(a["id"] for a in cands if a["id"] != ing["primary_asset_id"])
    assert (
        ok(client, "PATCH", f"/api/ingredients/{ids['keeper']}/primary", json={"asset_id": other})[
            "primary_asset_id"
        ]
        == other
    )

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "orange").save(buf, "PNG")
    up = ok(
        client,
        "POST",
        f"/api/ingredients/{ids['dog']}/assets",
        201,
        files={"file": ("pip.png", buf.getvalue(), "image/png")},
    )
    assert up["type"] == "image" and up["source"] == "uploaded" and (up["width"], up["height"]) == (64, 48)
    assert ok(client, "GET", f"/api/ingredients/{ids['dog']}")["primary_asset_id"] == up["id"]
    assert (
        ok(client, "GET", f"/api/projects/{ids['pid']}/assets", params={"ingredient_id": ids["dog"]})["total"]
        == 1
    )

    r = client.post(
        f"/api/ingredients/{ids['dog']}/assets", files={"file": ("x.png", b"not an image", "image/png")}
    )
    assert r.status_code == 415
    r = client.patch(f"/api/ingredients/{ids['dog']}/primary", json={"asset_id": "missing"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_asset"


def test_deleting_an_asset_clears_primary_and_selected_keyframe(client: TestClient) -> None:
    ids = project_with_cast(client)
    job = ok(client, "POST", f"/api/ingredients/{ids['keeper']}/generate", 201, json={"count": 1})
    assert wait_job(client, job["id"])["status"] == "completed"
    primary = ok(client, "GET", f"/api/ingredients/{ids['keeper']}")["primary_asset_id"]
    assert primary
    shot = ok(
        client, "POST", f"/api/shots/{ids['shot']}/select", json={"slot": "keyframe", "asset_id": primary}
    )
    assert shot["status"] == "keyframe_ready"

    wanted = {"asset.deleted", "ingredients.updated", "board.updated"}
    with client.websocket_connect(f"/ws/events?token={TOKEN}") as ws:
        assert client.delete(f"/api/assets/{primary}").status_code == 204
        seen: set[str] = set()
        for _ in range(20):  # published before the 204 returns; skips the worker.status snapshot etc.
            seen.add(ws.receive_json()["type"])
            if wanted <= seen:
                break
    assert wanted <= seen
    assert ok(client, "GET", f"/api/ingredients/{ids['keeper']}")["primary_asset_id"] is None
    board = ok(client, "GET", f"/api/projects/{ids['pid']}/scenes")
    s = next(sh for sc in board["scenes"] for sh in sc["shots"] if sh["id"] == ids["shot"])
    assert s["selected_keyframe_asset_id"] is None and s["status"] == "planned"


def test_keyframe_falls_back_to_t2i_without_edit_model(client: TestClient) -> None:
    ids = project_with_cast(client)
    out = ok(client, "POST", f"/api/shots/{ids['shot']}/generate-keyframe", 201, json={"count": 1})
    assert out["mode"] == "t2i" and "no primary images" in out["note"]
    assert (
        out["job"]["params"]["request"]["width"] == 1280 and out["job"]["params"]["request"]["height"] == 720
    )
    assert "Old Keeper: weathered man" in out["prompt"] and "Shore: rocky shore" in out["prompt"]
    assert wait_job(client, out["job"]["id"])["status"] == "completed"


def test_keyframe_composed_from_references(client: TestClient) -> None:
    ids = project_with_cast(client)
    for key in ("keeper", "dog"):
        j = ok(client, "POST", f"/api/ingredients/{ids[key]}/generate", 201, json={"count": 1})
        assert wait_job(client, j["id"])["status"] == "completed"
    # Without an installed edit model: text-to-image with a note.
    out = ok(client, "POST", f"/api/shots/{ids['shot']}/generate-keyframe", 201, json={"count": 1})
    assert out["mode"] == "t2i" and "No image-edit model" in out["note"]
    r = client.post(
        f"/api/shots/{ids['shot']}/generate-keyframe", json={"mode": "edit", "model_entry_id": "nope"}
    )
    assert r.status_code == 400

    kontext = install(client, "flux1-kontext-dev-q4")
    try:
        out = ok(client, "POST", f"/api/shots/{ids['shot']}/generate-keyframe", 201, json={"count": 2})
        assert out["mode"] == "edit" and out["reference_names"] == ["Old Keeper", "Pip"]
        req = out["job"]["params"]["request"]
        assert req["model_entry_id"] == kontext and len(req["reference_asset_ids"]) == 2
        assert (
            "Old Keeper (left), Pip (right)" in out["prompt"] and "Style: moody 35mm film." in out["prompt"]
        )
        job = wait_job(client, out["job"]["id"])
        assert job["status"] == "completed" and len(job["result"]["asset_ids"]) == 2
        cands = ok(
            client,
            "GET",
            f"/api/projects/{ids['pid']}/assets",
            params={"shot_id": ids["shot"], "slot": "keyframe"},
        )
        # 1 text-to-image fallback from above + 2 composed from the references (newest first).
        assert cands["total"] == 3
        composed = [a for a in cands["items"] if a["parent_asset_ids"]]
        assert len(composed) == 2
        assert all(set(a["parent_asset_ids"]) == set(req["reference_asset_ids"]) for a in composed)
        shot = ok(
            client,
            "POST",
            f"/api/shots/{ids['shot']}/select",
            json={"slot": "keyframe", "asset_id": cands["items"][0]["id"]},
        )
        assert (
            shot["selected_keyframe_asset_id"] == cands["items"][0]["id"]
            and shot["status"] == "keyframe_ready"
        )
        shot = ok(
            client, "POST", f"/api/shots/{ids['shot']}/select", json={"slot": "keyframe", "asset_id": None}
        )
        assert shot["status"] == "planned"
        end = ok(
            client,
            "POST",
            f"/api/shots/{ids['shot']}/generate-keyframe",
            201,
            json={"which": "end", "count": 1},
        )
        assert end["job"]["target_ref"]["slot"] == "end_keyframe"
    finally:
        uninstall(client, kontext)


def test_license_ack_required_for_named_model(client: TestClient) -> None:
    sdxl = install(client, "sdxl-base", ack=False)
    try:
        r = client.post("/api/generate/t2i", json={"prompt": "x", "model_entry_id": sdxl})
        assert r.status_code == 409 and r.json()["error"]["code"] == "license_not_acknowledged"
        ok(client, "POST", f"/api/models/{sdxl}/acknowledge-license")
        job = ok(client, "POST", "/api/generate/t2i", 201, json={"prompt": "x", "model_entry_id": sdxl})
        assert job["params"]["adapter_key"] == "fake"  # the FakeEngine stands in for real adapters
    finally:
        uninstall(client, sdxl)


# --- real-engine model resolution (no worker) --------------------------------------------------------
@pytest.fixture
def real_api(tmp_path: Path) -> Iterator[TestClient]:
    settings = make_settings(tmp_path / "data", start_worker=False)
    settings.engine = "real"
    with TestClient(create_app(settings), headers={"X-LocalFlow-Token": TOKEN}) as c:
        yield c


def test_default_model_and_worker_spec(real_api: TestClient) -> None:
    r = real_api.post("/api/generate/t2i", json={"prompt": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_model_installed"
    sdxl = install(real_api, "sdxl-base")
    flux = install(real_api, "flux1-dev-q4")
    job = ok(real_api, "POST", "/api/generate/t2i", 201, json={"prompt": "x"})
    # Catalog order: FLUX Q4 comes before SDXL.
    assert job["params"]["adapter_key"] == "flux" and job["params"]["request"]["model_entry_id"] == flux
    pid = ok(real_api, "POST", "/api/projects", 201, json={"name": "P", "default_image_model_id": sdxl})["id"]
    job = ok(real_api, "POST", "/api/generate/t2i", 201, json={"prompt": "x", "project_id": pid})
    assert job["params"]["adapter_key"] == "sdxl"

    svc = real_api.app.state.services  # type: ignore[attr-defined]
    req = svc.generation.build_request(svc.jobs.get(job["id"]))
    assert req.adapter_key == "sdxl" and req.model["entry_id"] == sdxl
    roles = {p["role"]: p for p in req.model["parts"]}
    assert set(roles) == {"base", "vae"}
    assert roles["base"]["local_dir"].endswith(
        str(Path("models/hf/repos/stabilityai--stable-diffusion-xl-base-1.0"))
    )
    assert req.model["default_params"]["steps"] == 30

    # A FLUX LoRA can't be used with an SDXL model.
    with svc.db.session() as s:
        lora = ModelEntry(
            source="civitai",
            source_ref="1:2",
            name="Flux LoRA",
            kind="lora",
            adapter_key="flux",
            base_model="flux1",
            status="installed",
            license_acknowledged_at=utcnow(),
            capabilities_json=["t2i"],
            files_json=[
                {"role": "weights", "source": "civitai", "local_path": "models/civitai/1/2/l.safetensors"}
            ],
        )
        s.add(lora)
        s.commit()
        lora_id = lora.id
    r = real_api.post(
        "/api/generate/t2i",
        json={"prompt": "x", "model_entry_id": sdxl, "loras": [{"model_entry_id": lora_id}]},
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "lora_incompatible"
    job = ok(
        real_api,
        "POST",
        "/api/generate/t2i",
        201,
        json={"prompt": "x", "model_entry_id": flux, "loras": [{"model_entry_id": lora_id, "weight": 0.7}]},
    )
    req = svc.generation.build_request(svc.jobs.get(job["id"]))
    assert req.loras[0]["weight"] == 0.7 and req.loras[0]["path"].endswith("l.safetensors")


def test_upload_rejects_non_media(real_api: TestClient) -> None:
    pid = ok(real_api, "POST", "/api/projects", 201, json={"name": "U"})["id"]
    r = real_api.post(
        f"/api/projects/{pid}/assets/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media"
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "JPEG")
    a = ok(
        real_api,
        "POST",
        f"/api/projects/{pid}/assets/upload",
        201,
        files={"file": ("photo.jpeg", buf.getvalue(), "image/jpeg")},
    )
    assert a["type"] == "image" and a["thumb_url"] and a["generation"]["uploaded_name"] == "photo.jpeg"
    svc = real_api.app.state.services  # type: ignore[attr-defined]
    assert svc.assets.get(a["id"]).path.endswith(".jpg")  # extension from the content, not the name
