"""M7: the asset library: collections, filters and facets, bulk edits, frame extraction, upload normalization
(Section 13)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from localflow.services.media.ffmpeg import probe
from tests.test_m3_imagegen import ok

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="ffmpeg not installed")


def svc(c: TestClient) -> Any:
    return c.app.state.services  # type: ignore[attr-defined]


def image(c: TestClient, pid: str, tmp: Path, name: str, generation: dict[str, Any], **kw: Any) -> str:
    """An image asset with a given generation record (as if a job had made it)."""
    path = tmp / f"{name}.png"
    Image.new("RGB", (32, 32), "teal").save(path)
    return str(
        svc(c)
        .assets.ingest_file(pid, path, "image", source=kw.get("source", "generated"), generation=generation)
        .id
    )


def lavfi(tmp: Path, name: str, args: list[str]) -> Path:
    assert FFMPEG
    out = tmp / name
    subprocess.run([FFMPEG, "-v", "error", "-y", *args, str(out)], check=True)
    return out


def upload(c: TestClient, pid: str, path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        r = c.post(
            f"/api/projects/{pid}/assets/upload", files={"file": (path.name, fh, "application/octet-stream")}
        )
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


@pytest.fixture
def project(api_client: TestClient, tmp_path: Path) -> dict[str, Any]:
    """A project with a character, a scene with two shots, and assets made in different ways."""
    c = api_client
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Library"})["id"]
    mara = ok(c, "POST", f"/api/projects/{pid}/ingredients", 201, json={"kind": "character", "name": "Mara"})
    scene = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "Dock"})
    other = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "Night"})
    sh1 = ok(c, "POST", f"/api/scenes/{scene['id']}/shots", 201, json={"ingredient_ids": [mara["id"]]})
    sh2 = ok(c, "POST", f"/api/scenes/{other['id']}/shots", 201, json={})
    flux = {"model_entry_id": "m-flux", "model_name": "FLUX.1 [dev]", "job_type": "t2i"}
    ids = {
        "portrait": image(
            c,
            pid,
            tmp_path,
            "portrait",
            {**flux, "target_ref": {"ingredient_id": mara["id"]}, "params": {"prompt": "a sailor"}},
        ),
        "mention": image(
            c,
            pid,
            tmp_path,
            "mention",
            {
                **flux,
                "mentions": ["mara"],
                "prompt_raw": "@mara on the pier",
                "params": {"prompt": "Mara on the pier"},
            },
        ),
        "keyframe": image(
            c,
            pid,
            tmp_path,
            "keyframe",
            {
                "model_name": "SDXL",
                "target_ref": {"shot_id": sh1["id"], "slot": "keyframe"},
                "params": {"prompt": "wide dock"},
            },
        ),
        "night": image(
            c, pid, tmp_path, "night", {"target_ref": {"shot_id": sh2["id"]}, "params": {"prompt": "stars"}}
        ),
        "upload": image(
            c, pid, tmp_path, "upload", {"uploaded_name": "Harbor sketch.jpg"}, source="uploaded"
        ),
    }
    return {"c": c, "pid": pid, "mara": mara, "scene": scene, "sh1": sh1, "ids": ids}


def listed(c: TestClient, pid: str, **params: Any) -> list[str]:
    return [a["id"] for a in ok(c, "GET", f"/api/projects/{pid}/assets", params=params)["items"]]


def test_filters_facets_and_bulk_edits(project: dict[str, Any]) -> None:
    c, pid, ids = project["c"], project["pid"], project["ids"]
    everything = listed(c, pid)
    assert len(everything) == 5 and everything[0] == ids["upload"]  # newest first
    assert listed(c, pid, sort="oldest")[0] == ids["portrait"]
    assert set(listed(c, pid, model="m-flux")) == {ids["portrait"], ids["mention"]}
    assert listed(c, pid, model="SDXL") == [ids["keyframe"]]  # older assets: by name
    assert listed(c, pid, scene_id=project["scene"]["id"]) == [ids["keyframe"]]
    # Featuring Mara: her candidate, a prompt that @mentions her, and the shot she's in.
    assert set(listed(c, pid, featuring=project["mara"]["id"])) == {
        ids["portrait"],
        ids["mention"],
        ids["keyframe"],
    }
    assert listed(c, pid, featuring="nope") == []
    assert listed(c, pid, q="PIER") == [ids["mention"]]  # resolved prompt, case-insensitive
    assert listed(c, pid, q="@mara") == [ids["mention"]]  # the raw prompt too
    assert listed(c, pid, q="harbor sketch") == [ids["upload"]]  # uploaded file names
    assert listed(c, pid, source="uploaded") == [ids["upload"]]

    out = ok(
        c,
        "POST",
        f"/api/projects/{pid}/assets/bulk",
        json={
            "asset_ids": [ids["portrait"], ids["night"]],
            "add_tags": ["hero", " moody "],
            "favorite": True,
        },
    )
    assert out == {"updated": 2, "deleted": 0}
    assert set(listed(c, pid, tag="moody")) == {ids["portrait"], ids["night"]}
    assert listed(c, pid, q="hero") == listed(c, pid, tag="hero")  # tags are searchable
    ok(
        c,
        "POST",
        f"/api/projects/{pid}/assets/bulk",
        json={"asset_ids": [ids["night"]], "remove_tags": ["hero"]},
    )
    assert ok(c, "GET", f"/api/assets/{ids['night']}")["tags"] == ["moody"]

    facets = ok(c, "GET", f"/api/projects/{pid}/assets/facets")
    assert facets["total"] == 5 and facets["favorites"] == 2
    assert facets["types"] == {"image": 5} and facets["sources"] == {"generated": 4, "uploaded": 1}
    assert facets["tags"] == [{"tag": "moody", "count": 2}, {"tag": "hero", "count": 1}]
    assert facets["models"][0] == {"model": "m-flux", "name": "FLUX.1 [dev]", "count": 2}

    other = ok(c, "POST", "/api/projects", 201, json={"name": "Other"})["id"]
    r = c.post(f"/api/projects/{other}/assets/bulk", json={"asset_ids": [ids["night"]], "delete": True})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_asset"
    out = ok(
        c, "POST", f"/api/projects/{pid}/assets/bulk", json={"asset_ids": [ids["night"]], "delete": True}
    )
    assert out == {"updated": 0, "deleted": 1}
    assert ids["night"] not in listed(c, pid)


def test_collections(project: dict[str, Any]) -> None:
    c, pid, ids = project["c"], project["pid"], project["ids"]
    hero = ok(
        c,
        "POST",
        f"/api/projects/{pid}/collections",
        201,
        json={"name": " Hero shots ", "asset_ids": [ids["portrait"], ids["keyframe"], ids["portrait"]]},
    )
    assert hero["name"] == "Hero shots" and hero["asset_count"] == 2
    assert hero["cover_thumb_url"] == f"/api/thumbs/{pid}/{ids['keyframe']}"
    r = c.post(f"/api/projects/{pid}/collections", json={"name": "Hero shots"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "collection_exists"
    refs = ok(c, "POST", f"/api/projects/{pid}/collections", 201, json={"name": "Refs"})
    assert refs["asset_count"] == 0 and refs["cover_thumb_url"] is None
    assert [x["name"] for x in ok(c, "GET", f"/api/projects/{pid}/collections")] == ["Hero shots", "Refs"]

    out = ok(
        c,
        "POST",
        f"/api/collections/{hero['id']}/assets",
        json={"asset_ids": [ids["upload"], ids["portrait"]]},
    )
    assert out["asset_count"] == 3
    assert set(listed(c, pid, collection=hero["id"])) == {ids["portrait"], ids["keyframe"], ids["upload"]}
    assert ok(c, "GET", f"/api/assets/{ids['portrait']}/collections") == [hero["id"]]
    out = c.request(
        "DELETE", f"/api/collections/{hero['id']}/assets", json={"asset_ids": [ids["upload"]]}
    ).json()
    assert out["asset_count"] == 2

    other = ok(c, "POST", "/api/projects", 201, json={"name": "Other"})["id"]
    foreign = ok(c, "POST", f"/api/projects/{other}/collections", 201, json={"name": "X"})
    r = c.post(f"/api/collections/{foreign['id']}/assets", json={"asset_ids": [ids["portrait"]]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_asset"
    renamed = ok(
        c, "PATCH", f"/api/collections/{refs['id']}", json={"name": "References", "description": "looks"}
    )
    assert renamed["name"] == "References" and renamed["description"] == "looks"
    r = c.patch(f"/api/collections/{refs['id']}", json={"name": "Hero shots"})
    assert r.status_code == 409

    # Deleting an asset drops it from collections; deleting a collection keeps its assets.
    assert c.delete(f"/api/assets/{ids['keyframe']}").status_code == 204
    assert ok(c, "GET", f"/api/projects/{pid}/collections")[0]["asset_count"] == 1
    assert c.delete(f"/api/collections/{hero['id']}").status_code == 204
    assert ids["portrait"] in listed(c, pid)
    assert ok(c, "GET", f"/api/assets/{ids['portrait']}/collections") == []
    assert c.delete(f"/api/collections/{hero['id']}").status_code == 404


@needs_ffmpeg
def test_extract_frame(api_client: TestClient, tmp_path: Path) -> None:
    c = api_client
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Frames"})["id"]
    clip = lavfi(
        tmp_path,
        "clip.mp4",
        ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=2", "-pix_fmt", "yuv420p"],
    )
    video = upload(c, pid, clip)
    still = ok(c, "POST", f"/api/assets/{video['id']}/extract-frame", 201, json={"time_s": 1.01})
    assert (
        still["type"] == "image"
        and still["source"] == "extracted"
        and still["parent_asset_ids"] == [video["id"]]
    )
    assert (still["width"], still["height"]) == (160, 90) and still["thumb_url"]
    assert still["generation"] == {
        "extracted_from": video["id"],
        "time_s": 1.0,
        "frame": 24,
    }  # on the frame grid
    end = ok(c, "POST", f"/api/assets/{video['id']}/extract-frame", 201, json={"time_s": 99})
    assert end["generation"]["frame"] == 47  # clamped to the last frame
    r = c.post(f"/api/assets/{still['id']}/extract-frame", json={"time_s": 0})
    assert r.status_code == 400 and r.json()["error"]["code"] == "not_a_video"
    assert listed(c, pid, source="extracted") == [end["id"], still["id"]]


@needs_ffmpeg
def test_extracted_frames_are_the_frames_they_name(api_client: TestClient, tmp_path: Path) -> None:
    """Frame k at 24/30 fps: k/fps rounded to milliseconds can land after the frame (1/24 → 0.042)."""
    assert FFMPEG
    c = api_client
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Exact"})["id"]
    for fps in (24, 30):
        frames = tmp_path / f"f{fps}"
        frames.mkdir()
        for k in range(8):  # frame k is solid red k*32
            Image.new("RGB", (64, 64), (k * 32, 0, 0)).save(frames / f"{k:02d}.png")
        clip = tmp_path / f"steps{fps}.mp4"
        pattern = str(frames / "%02d.png")
        lossless = ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p"]
        subprocess.run(
            [FFMPEG, "-v", "error", "-y", "-framerate", str(fps), "-i", pattern, *lossless, str(clip)],
            check=True,
        )
        video = upload(c, pid, clip)
        for k in (1, 2, 4, 5, 7):
            still = ok(c, "POST", f"/api/assets/{video['id']}/extract-frame", 201, json={"time_s": k / fps})
            assert still["generation"]["frame"] == k
            s = svc(c)
            with Image.open(s.assets.file_path(s.assets.get(still["id"]))) as im:
                red = im.convert("RGB").getpixel((32, 32))[0]
            assert abs(red - k * 32) <= 6, (fps, k, red)


@needs_ffmpeg
def test_uploads_are_normalized(api_client: TestClient, tmp_path: Path) -> None:
    c = api_client
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Uploads"})["id"]
    src = ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=1"]
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]

    def stored(a: dict[str, Any]) -> dict[str, Any]:
        s = svc(c)
        path = s.assets.file_path(s.assets.get(a["id"]))
        return {"ext": path.suffix, **probe(s.settings.ffprobe, path)}

    webm = upload(
        c, pid, lavfi(tmp_path, "clip.webm", [*src, *tone, "-c:v", "libvpx-vp9", "-c:a", "libopus"])
    )
    info = stored(webm)
    assert webm["type"] == "video" and info["ext"] == ".mp4"
    assert (info["video_codec"], info["pix_fmt"], info["audio_codec"]) == ("h264", "yuv420p", "aac")
    assert webm["generation"]["normalized"] == "webm/vp9 → mp4/h264" and webm["has_audio"]
    assert webm["generation"]["uploaded_name"] == "clip.webm" and webm["thumb_url"]

    mov = upload(c, pid, lavfi(tmp_path, "clip.mov", [*src, "-c:v", "libx264", "-pix_fmt", "yuv420p"]))
    assert stored(mov)["ext"] == ".mp4" and mov["generation"]["normalized"] == "mov/h264 remuxed to mp4"

    mp4 = upload(c, pid, lavfi(tmp_path, "ok.mp4", [*src, "-c:v", "libx264", "-pix_fmt", "yuv420p"]))
    assert "normalized" not in mp4["generation"]  # already H.264 MP4: kept as uploaded

    flac = upload(c, pid, lavfi(tmp_path, "tone.flac", tone))
    info = stored(flac)
    assert flac["type"] == "audio" and info["ext"] == ".wav" and info["audio_codec"] == "pcm_s16le"
    assert flac["generation"]["normalized"] == "flac → wav" and flac["duration_s"] == pytest.approx(
        1.0, abs=0.05
    )
    wav = upload(c, pid, lavfi(tmp_path, "tone.wav", tone))
    assert "normalized" not in wav["generation"]

    # libx264 needs even sizes: an odd-sized VP9 screen capture loses its last row and column.
    odd_src = ["-f", "lavfi", "-i", "testsrc2=size=161x91:rate=24:duration=1", "-c:v", "libvpx-vp9"]
    odd = upload(c, pid, lavfi(tmp_path, "odd.webm", odd_src))
    assert (odd["width"], odd["height"]) == (160, 90) and stored(odd)["video_codec"] == "h264"
    # Nothing is left behind in the upload area.
    assert not any((svc(c).settings.tmp_dir / "uploads").glob("*"))


@needs_ffmpeg
def test_a_failed_conversion_is_a_clear_error_and_leaves_nothing(
    api_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localflow.services import assets as assets_mod
    from localflow.services.media.ffmpeg import FFmpegError

    c = api_client
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Broken"})["id"]
    clip = lavfi(
        tmp_path,
        "clip.webm",
        ["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=1", "-c:v", "libvpx-vp9"],
    )

    def fail(ffmpeg: Any, args: list[str], **kw: Any) -> None:
        Path(args[-1]).write_bytes(b"partial")  # a half-written output, like a real failure leaves
        raise FFmpegError("Conversion failed: invalid data")

    monkeypatch.setattr(assets_mod, "run_ffmpeg", fail)
    with clip.open("rb") as fh:
        r = c.post(f"/api/projects/{pid}/assets/upload", files={"file": (clip.name, fh, "video/webm")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "convert_failed"
    assert "invalid data" in r.json()["error"]["message"]
    assert not any((svc(c).settings.tmp_dir / "uploads").glob("*"))
    assert ok(c, "GET", f"/api/projects/{pid}/assets")["total"] == 0


@needs_ffmpeg
def test_deleting_audio_clears_the_shot_selection(api_client: TestClient, tmp_path: Path) -> None:
    c = api_client
    pid = ok(c, "POST", "/api/projects", 201, json={"name": "Audio"})["id"]
    scene = ok(c, "POST", f"/api/projects/{pid}/scenes", 201, json={"title": "S"})
    sh = ok(c, "POST", f"/api/scenes/{scene['id']}/shots", 201, json={})
    tone = upload(c, pid, lavfi(tmp_path, "tone.wav", ["-f", "lavfi", "-i", "sine=duration=1"]))
    ok(c, "PATCH", f"/api/shots/{sh['id']}", json={"selected_audio_asset_ids": [tone["id"]]})
    assert c.delete(f"/api/assets/{tone['id']}").status_code == 204
    shots = ok(c, "GET", f"/api/projects/{pid}/scenes")["scenes"][0]["shots"]
    assert shots[0]["selected_audio_asset_ids"] == []
