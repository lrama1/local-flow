# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pillow>=10"]
# ///
"""Renders the built-in character template thumbnails with the running app (the default image model).

Start the built app first (`npm start`), then:

    uv run scripts/characters/make_template_thumbnails.py [--base-url http://127.0.0.1:8765] [--only a,b] [--force]

Each template becomes a throwaway wizard draft in the Playground project; its portrait is cropped to a
384 px JPEG in backend/localflow/services/characters/thumbnails/<key>.jpg, then the draft is deleted.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

OUT = Path(__file__).resolve().parents[2] / "backend/localflow/services/characters/thumbnails"
SIZE = 384
SEED = 20261010
# Templates describe appearance only (a project's style ingredient sets the look). Thumbnails have no
# project, so they get a look that matches their category.
LOOK = {
    "realistic": "photorealistic portrait photograph, natural soft light",
    "historical": "photorealistic period film still, natural soft light",
    "sci-fi": "photorealistic science-fiction film still",
    "fantastical": "photorealistic fantasy film still",
    "creature": "photorealistic fantasy film still",
    "anime": "anime illustration, clean line art",
    "cartoon": "colorful cartoon illustration",
    "animal": "storybook illustration",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--only", default="", help="comma-separated template keys")
    ap.add_argument("--force", action="store_true", help="re-render existing thumbnails")
    args = ap.parse_args()

    html = httpx.get(args.base_url).text
    m = re.search(r'name="localflow-token" content="([^"]+)"', html)
    if not m:
        sys.exit("No API token in the page: serve the built app with `npm start`.")
    c = httpx.Client(base_url=args.base_url, headers={"X-LocalFlow-Token": m.group(1)}, timeout=60)

    playground = c.get("/api/playground").raise_for_status().json()["project_id"]
    only = {k for k in args.only.split(",") if k}
    templates = [t for t in c.get("/api/character-templates").raise_for_status().json() if t["builtin"]]
    OUT.mkdir(parents=True, exist_ok=True)

    for t in templates:
        key, path = t["key"], OUT / f"{t['key']}.jpg"
        if (only and key not in only) or (path.exists() and not args.force):
            continue
        started = time.monotonic()
        draft = c.post(
            f"/api/projects/{playground}/characters", json={"source": "template", "template_key": key}
        ).raise_for_status().json()
        try:
            if look := LOOK.get(t["category"]):
                visual = f"{t['visual_prompt'].rstrip('. ')}. {look}"
                c.patch(f"/api/ingredients/{draft['id']}", json={"visual_prompt": visual}).raise_for_status()
            job = c.post(
                f"/api/ingredients/{draft['id']}/generate", json={"count": 1, "params": {"seed": SEED}}
            ).raise_for_status().json()
            while job["status"] not in {"completed", "failed", "cancelled"}:
                time.sleep(2)
                job = c.get(f"/api/jobs/{job['id']}").raise_for_status().json()
            if job["status"] != "completed":
                sys.exit(f"{key}: {job['status']}: {job.get('error')}")
            asset_id = job["result"]["asset_ids"][0]
            data = c.get(f"/api/files/{playground}/{asset_id}").raise_for_status().content
            img = Image.open(io.BytesIO(data)).convert("RGB")
            side = min(img.size)
            left, top = (img.width - side) // 2, (img.height - side) // 4  # keep the face in frame
            img = img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            img.save(path, "JPEG", quality=88, optimize=True)
            print(f"{key}: {path.name} ({time.monotonic() - started:.0f} s)")
        finally:
            c.delete(f"/api/ingredients/{draft['id']}")


if __name__ == "__main__":
    main()
