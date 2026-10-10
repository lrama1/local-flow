"""Ingredients, scenes, and shots CRUD + ordering."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _setup(c: TestClient) -> tuple[str, dict[str, str]]:
    pid = c.post("/api/projects", json={"name": "Board"}).json()["id"]
    ids = {}
    for kind, name in (("character", "Mara Quinn"), ("location", "Pier"), ("prop", "Lamp")):
        r = c.post(f"/api/projects/{pid}/ingredients", json={"kind": kind, "name": name})
        assert r.status_code == 201
        ids[name] = r.json()["id"]
    return pid, ids


def test_ingredient_crud(api_client: TestClient) -> None:
    pid, ids = _setup(api_client)
    ing = api_client.get(f"/api/ingredients/{ids['Mara Quinn']}").json()
    assert ing["key"] == "mara-quinn" and ing["origin"] == "user" and ing["locked"] is True
    dup = api_client.post(f"/api/projects/{pid}/ingredients", json={"kind": "prop", "name": "mara quinn"})
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "key_taken"
    r = api_client.patch(
        f"/api/ingredients/{ids['Pier']}", json={"visual_prompt": "wooden pier", "locked": False}
    )
    assert r.json()["visual_prompt"] == "wooden pier" and r.json()["locked"] is False
    r = api_client.patch(f"/api/ingredients/{ids['Pier']}", json={"key": "nope"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "key_immutable"  # only drafts can


def test_scenes_and_shots(api_client: TestClient) -> None:
    pid, ids = _setup(api_client)
    s1 = api_client.post(
        f"/api/projects/{pid}/scenes", json={"title": "One", "location_ingredient_id": ids["Pier"]}
    )
    s2 = api_client.post(f"/api/projects/{pid}/scenes", json={"title": "Two"})
    s0 = api_client.post(f"/api/projects/{pid}/scenes", json={"title": "Zero", "index": 0})
    assert [s.status_code for s in (s1, s2, s0)] == [201, 201, 201]
    bad = api_client.post(f"/api/projects/{pid}/scenes", json={"location_ingredient_id": ids["Lamp"]})
    assert bad.status_code == 400

    a = api_client.post(
        f"/api/scenes/{s1.json()['id']}/shots", json={"keyframe_prompt": "a", "duration_s": 3}
    ).json()
    b = api_client.post(
        f"/api/scenes/{s1.json()['id']}/shots",
        json={
            "keyframe_prompt": "b",
            "ingredient_ids": [ids["Mara Quinn"], ids["Lamp"]],
            "dialogue": [{"character_ingredient_id": ids["Mara Quinn"], "line": "Hi"}],
        },
    ).json()
    c = api_client.post(f"/api/scenes/{s2.json()['id']}/shots", json={"keyframe_prompt": "c"}).json()
    assert (
        api_client.post(f"/api/scenes/{s1.json()['id']}/shots", json={"ingredient_ids": ["nope"]}).status_code
        == 400
    )

    board = api_client.get(f"/api/projects/{pid}/scenes").json()
    assert [s["title"] for s in board["scenes"]] == ["Zero", "One", "Two"]
    assert board["shot_count"] == 3 and board["total_duration_s"] == 3 + 4 + 4

    # reorder within a scene, then move a shot across scenes
    assert (
        api_client.post(
            f"/api/scenes/{s1.json()['id']}/shots/reorder", json={"ids": [b["id"], a["id"]]}
        ).status_code
        == 204
    )
    assert (
        api_client.post(f"/api/scenes/{s1.json()['id']}/shots/reorder", json={"ids": [b["id"]]}).status_code
        == 400
    )  # must include every current shot
    assert (
        api_client.post(
            f"/api/scenes/{s1.json()['id']}/shots/reorder", json={"ids": [b["id"], c["id"], a["id"]]}
        ).status_code
        == 204
    )
    board = api_client.get(f"/api/projects/{pid}/scenes").json()
    one = next(s for s in board["scenes"] if s["title"] == "One")
    two = next(s for s in board["scenes"] if s["title"] == "Two")
    assert [sh["keyframe_prompt"] for sh in one["shots"]] == ["b", "c", "a"]
    assert [sh["index"] for sh in one["shots"]] == [0, 1, 2] and two["shots"] == []

    order = [s2.json()["id"], s1.json()["id"], s0.json()["id"]]
    assert api_client.post(f"/api/projects/{pid}/scenes/reorder", json={"ids": order}).status_code == 204
    assert api_client.post(f"/api/projects/{pid}/scenes/reorder", json={"ids": order[:2]}).status_code == 400
    board = api_client.get(f"/api/projects/{pid}/scenes").json()
    assert [s["title"] for s in board["scenes"]] == ["Two", "One", "Zero"]

    r = api_client.patch(f"/api/shots/{a['id']}", json={"camera_move": "orbit", "duration_s": 2.5})
    assert r.json()["camera_move"] == "orbit" and r.json()["duration_s"] == 2.5
    assert api_client.patch(f"/api/shots/{a['id']}", json={"camera_move": "spin"}).status_code == 422

    # deleting an ingredient cleans references
    assert api_client.delete(f"/api/ingredients/{ids['Mara Quinn']}").status_code == 204
    assert api_client.delete(f"/api/ingredients/{ids['Pier']}").status_code == 204
    shot_b = api_client.get(f"/api/shots/{b['id']}").json()
    assert shot_b["ingredient_ids"] == [ids["Lamp"]]
    assert shot_b["dialogue"][0]["character_ingredient_id"] is None
    board = api_client.get(f"/api/projects/{pid}/scenes").json()
    assert all(s["location_ingredient_id"] is None for s in board["scenes"])

    assert api_client.delete(f"/api/shots/{b['id']}").status_code == 204
    assert api_client.delete(f"/api/scenes/{s0.json()['id']}").status_code == 204
    board = api_client.get(f"/api/projects/{pid}/scenes").json()
    assert [s["index"] for s in board["scenes"]] == [0, 1]


def test_project_delete_cascades(api_client: TestClient) -> None:
    pid, _ = _setup(api_client)
    sc = api_client.post(f"/api/projects/{pid}/scenes", json={}).json()
    shot = api_client.post(f"/api/scenes/{sc['id']}/shots", json={}).json()
    assert api_client.delete(f"/api/projects/{pid}").status_code == 204
    assert api_client.get(f"/api/shots/{shot['id']}").status_code == 404
