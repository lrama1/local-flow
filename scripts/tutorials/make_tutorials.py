# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.50", "edge-tts>=7.0"]
# ///
"""Records the narrated in-app tutorials (Tutorials page) by driving a running LocalFlow.

    uv run scripts/tutorials/make_tutorials.py [--base-url http://127.0.0.1:8765] [--only story,images]

LocalFlow must already be running (`npm start` is best: no dev overlays). The images tutorial uses the
project the story tutorial creates, a FLUX.1 [dev] model and FLUX Kontext; it runs real jobs on the GPU.
The story tutorial calls your default LLM (about $0.03 with gpt-5.4-mini).

How it works:
- Narration is synthesized up front with edge-tts (Microsoft Edge's online neural voices), so each step
  knows how long its line takes; sentence boundaries become WebVTT captions.
- Chrome is driven with Playwright and captured with the DevTools screencast (JPEG frames with
  timestamps, at 1.5x device pixel ratio for a sharp 1080p video). A drawn cursor shows the clicks.
- Waits for the LLM or GPU are time-lapsed: a badge shows the real elapsed time while ffmpeg speeds the
  segment up. Narration never overlaps a time-lapse, so the audio keeps its timing.
- ffmpeg builds the MP4 (H.264 + AAC), a poster frame and captions into frontend/public/tutorials/, and
  index.json lists them for the app.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import edge_tts
from playwright.async_api import BrowserContext, Locator, Page, async_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "tutorials"
WORK = Path(tempfile.gettempdir()) / "localflow-tutorials"
VIEW = {"width": 1280, "height": 720}
SCALE = 1.5  # device pixel ratio: the 1280x720 layout is captured at 1920x1080
FPS = 30
VOICE = "en-US-AndrewMultilingualNeural"
STORY_PROJECT = "The Lighthouse Gull"


# --- narration ---------------------------------------------------------------------------------------
@dataclass
class Clip:
    text: str
    path: Path
    duration: float
    sentences: list[tuple[float, float, str]]  # (offset, duration, text) within the clip


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


async def synthesize(text: str, voice: str, rate: str) -> Clip:
    cache = WORK / "tts"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
    mp3, meta = cache / f"{key}.mp3", cache / f"{key}.json"
    if not (mp3.exists() and meta.exists()):
        sentences: list[tuple[float, float, str]] = []
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="SentenceBoundary")
        with mp3.open("wb") as f:
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
                    sentences.append((chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"]))
        meta.write_text(json.dumps(sentences))
    sentences = [(float(o), float(d), str(t)) for o, d, t in json.loads(meta.read_text())]
    return Clip(text, mp3, ffprobe_duration(mp3), sentences)


# --- screen capture ----------------------------------------------------------------------------------
class Screencast:
    """DevTools screencast: a frame per compositor update, each with its wall-clock timestamp."""

    def __init__(self, ctx: BrowserContext, page: Page, frames_dir: Path) -> None:
        self.ctx, self.page, self.dir = ctx, page, frames_dir
        self.frames: list[tuple[float, Path]] = []
        self.cdp: Any = None

    async def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cdp = await self.ctx.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", self._frame)
        await self.cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 92,
                "maxWidth": int(VIEW["width"] * SCALE),
                "maxHeight": int(VIEW["height"] * SCALE),
                "everyNthFrame": 1,
            },
        )

    async def _frame(self, params: dict[str, Any]) -> None:
        path = self.dir / f"{len(self.frames):06d}.jpg"
        path.write_bytes(base64.b64decode(params["data"]))
        self.frames.append((float(params["metadata"]["timestamp"]), path))
        with contextlib.suppress(Exception):  # the session may be closing
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

    async def stop(self) -> None:
        await self.cdp.send("Page.stopScreencast")


CURSOR_JS = """
(() => {
  const install = () => {
    if (document.getElementById('__lf_cursor')) return;
    const c = document.createElement('div');
    c.id = '__lf_cursor';
    c.style.cssText = 'position:fixed;left:-40px;top:-40px;width:20px;height:20px;margin:-10px 0 0 -10px;' +
      'border-radius:50%;background:rgba(255,255,255,.9);border:2px solid rgba(15,15,25,.75);' +
      'box-shadow:0 0 0 5px rgba(124,108,255,.35),0 2px 8px rgba(0,0,0,.4);pointer-events:none;' +
      'z-index:2147483647;transition:transform .12s ease,background .12s ease';
    document.documentElement.appendChild(c);
    const look = (scale, bg) => { c.style.transform = scale; c.style.background = bg; };
    addEventListener('mousemove', (e) => {
      c.style.left = e.clientX + 'px';
      c.style.top = e.clientY + 'px';
    }, true);
    addEventListener('mousedown', () => look('scale(.65)', 'rgba(124,108,255,.95)'), true);
    addEventListener('mouseup', () => look('', 'rgba(255,255,255,.9)'), true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install);
  else install();
})();
"""

BADGE_SHOW_JS = """
(label) => {
  let b = document.getElementById('__lf_ff');
  if (!b) {
    b = document.createElement('div');
    b.id = '__lf_ff';
    b.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:2147483646;' +
      'background:rgba(20,18,36,.92);color:#fff;font:600 14px/1.2 system-ui,sans-serif;padding:8px 16px;' +
      'border-radius:999px;box-shadow:0 6px 24px rgba(0,0,0,.35);pointer-events:none;letter-spacing:.2px';
    document.documentElement.appendChild(b);
  }
  const t0 = Date.now();
  const tick = () => {
    if (!b.isConnected) return;
    const s = Math.floor((Date.now() - t0) / 1000);
    const clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    b.textContent = `\u23e9  ${label} \u00b7 ${clock} real time`;
    setTimeout(tick, 1000);  // one repaint a second: long waits stay cheap to capture
  };
  tick();
}
"""

CARD_HTML = """<!doctype html><html><head><style>
html,body{{margin:0;height:100%;font-family:'Segoe UI',system-ui,sans-serif;color:#fff;overflow:hidden}}
body{{display:flex;align-items:center;justify-content:center;
background:radial-gradient(1200px 700px at 20% 10%,#3b2f7a 0%,transparent 60%),
radial-gradient(900px 600px at 90% 90%,#0f5a6b 0%,transparent 55%),#0d0c16}}
.box{{max-width:900px;padding:0 60px;animation:in .8s ease both}}
.kicker{{display:flex;align-items:center;gap:12px;font-size:20px;font-weight:600;opacity:.85}}
.kicker img{{width:34px;height:34px}}
h1{{font-size:56px;line-height:1.08;margin:18px 0 14px;letter-spacing:-.5px}}
p{{font-size:22px;line-height:1.45;opacity:.78;margin:0}}
@keyframes in{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}
</style></head><body><div class="box">
<div class="kicker"><img src="{base}/favicon.svg" alt="">LocalFlow · {kicker}</div>
<h1>{title}</h1><p>{subtitle}</p></div></body></html>"""


# --- a recording session -----------------------------------------------------------------------------
@dataclass
class Session:
    page: Page
    clips: dict[str, Clip]
    base: str
    token: str = ""
    t0: float = field(default_factory=time.time)
    placed: list[tuple[float, Clip]] = field(default_factory=list)
    fast: list[tuple[float, float]] = field(default_factory=list)
    chapters: list[tuple[float, str]] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)
    mouse: tuple[float, float] = (VIEW["width"] / 2, VIEW["height"] / 2)
    speech_end: float = 0.0
    # Media whose own sound belongs in the tutorial (the finished film playing): (start, file, seconds).
    # The screencast has no audio, so it's mixed in from the file.
    sounds: list[tuple[float, Path, float]] = field(default_factory=list)

    def now(self) -> float:
        return time.time() - self.t0

    def chapter(self, title: str) -> None:
        self.chapters.append((self.now(), title))

    def mark(self, name: str) -> None:
        self.marks[name] = self.now()

    # narration
    async def say(self, key: str, *, wait: bool = True) -> None:
        await self.quiet(0.0)
        clip = self.clips[key]
        t = self.now()
        self.placed.append((t, clip))
        self.speech_end = t + clip.duration
        if wait:
            await self.quiet()

    async def quiet(self, gap: float = 0.45) -> None:
        remaining = self.speech_end - self.now()
        if remaining > 0:
            await asyncio.sleep(remaining)
        await asyncio.sleep(gap)

    async def fast_forward(self, until: Callable[[], Awaitable[Any]], label: str) -> None:
        """Time-lapses a wait (LLM, GPU). Narration must have finished first."""
        await self.quiet(0.3)
        await self.page.evaluate(BADGE_SHOW_JS, label)
        start = self.now()
        await until()
        await asyncio.sleep(0.6)
        self.fast.append((start, self.now()))
        await self.page.evaluate("document.getElementById('__lf_ff')?.remove()")
        await asyncio.sleep(0.5)

    # pointer and keyboard
    async def move(self, loc: Locator, fx: float = 0.5, fy: float = 0.5) -> None:
        await loc.wait_for(state="visible")  # e.g. the shot panel remounts when its shot updates
        await loc.scroll_into_view_if_needed()
        box = await loc.bounding_box()
        assert box is not None, f"{loc} has no box"
        x, y = box["x"] + box["width"] * fx, box["y"] + box["height"] * fy
        x0, y0 = self.mouse
        dist = math.hypot(x - x0, y - y0)
        n = max(10, min(45, int(dist / 14)))
        for i in range(1, n + 1):
            p = i / n
            e = p * p * (3 - 2 * p)  # smoothstep
            await self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            await asyncio.sleep(1 / 60)
        self.mouse = (x, y)
        await asyncio.sleep(0.12)

    async def click(self, loc: Locator, *, after: float = 0.35) -> None:
        await self.move(loc)
        await self.page.mouse.down()
        await asyncio.sleep(0.08)
        await self.page.mouse.up()
        await asyncio.sleep(after)

    async def type(self, loc: Locator, text: str, delay: int = 22) -> None:
        await self.click(loc, after=0.15)
        await loc.press_sequentially(text, delay=delay)
        await asyncio.sleep(0.3)

    async def select(self, loc: Locator, *, label: str) -> None:
        await self.move(loc)
        await loc.select_option(label=label)
        await asyncio.sleep(0.5)

    async def scroll_to(self, loc: Locator, block: str = "center") -> None:
        await loc.evaluate(f"(el) => el.scrollIntoView({{behavior: 'smooth', block: '{block}'}})")
        await asyncio.sleep(0.9)

    async def card(self, kicker: str, title: str, subtitle: str) -> None:
        await self.page.set_content(
            CARD_HTML.format(base=self.base, kicker=kicker, title=title, subtitle=subtitle)
        )
        self.marks.setdefault("start", self.now())  # the video begins at the first title card
        await asyncio.sleep(0.9)

    # API helpers (same origin, per-launch token)
    async def api(self, method: str, path: str, body: Any = None) -> Any:
        for attempt in range(6):
            try:
                r = await self.page.request.fetch(
                    f"{self.base}{path}",
                    method=method,
                    headers={"X-LocalFlow-Token": self.token, "Content-Type": "application/json"},
                    data=json.dumps(body) if body is not None else None,
                )
                break
            except Exception as e:  # a dropped connection during an hour of polling isn't fatal
                if method != "GET" or attempt == 5:
                    raise
                print(f"  retrying {path}: {e}", flush=True)
                await asyncio.sleep(3)
        if not r.ok:
            raise RuntimeError(f"{method} {path} → {r.status}: {await r.text()}")
        return await r.json() if r.status != 204 else None


# --- tutorials ---------------------------------------------------------------------------------------
@dataclass
class Tutorial:
    id: str
    title: str
    summary: str
    lines: dict[str, str]
    run: Callable[[Session], Awaitable[None]]


STORY_LINES = {
    "intro": "Welcome to LocalFlow. In this tutorial, we'll turn a short story idea into a complete shot "
    "list, ready for image and video generation.",
    "project": "Everything lives in a project. Click New project, give it a name, and choose an aspect "
    "ratio and frame rate for the film.",
    "idea": "The project opens on the Story tab, which works in four steps. Step one is the idea. "
    "A few sentences are enough.",
    "options": "You can also steer the result with a genre, a tone, a target length in seconds, and "
    "style notes for the look of the film.",
    "refine": "Click Refine. Your language model, from OpenAI, Anthropic, or OpenRouter, rewrites the idea "
    "as a filmable story with a clear beginning, middle, and end.",
    "refined": "Here's the refined story. You can edit it right here. Every change is saved as a "
    "revision, so you can compare versions and restore an older one at any time.",
    "bible": "Step three is the story bible. It describes the visual style, and every character, "
    "location, and prop in the story.",
    "bible_done": "Each part of the bible becomes an ingredient. Ingredients keep characters and places "
    "consistent from shot to shot, and you can lock any of them, so a rebuild never overwrites your "
    "changes.",
    "breakdown": "Finally, break the story down into shots. LocalFlow sizes every shot to the clip length "
    "of your video model, and writes a keyframe prompt, a video prompt, camera direction, and dialogue "
    "for each one.",
    "board": "Open the Shots board. Each scene is a row, and each shot is a card. Drag the cards to "
    "reorder them, or to move a shot to another scene.",
    "panel": "Click a shot to open its details. You can edit every prompt, change the shot type and "
    "camera move, choose who appears in the shot, and add lines of dialogue.",
    "rewrite": "Not quite right? Describe the change you want, and the AI rewrites the shot's prompts.",
    "diff": "You see exactly what would change, and decide whether to apply it.",
    "outro": "That's the whole story pipeline. In the next tutorial, we'll create reference images for "
    "the ingredients, and compose a keyframe for this shot.",
}


async def story_tutorial(s: Session) -> None:
    page = s.page
    for p in await s.api("GET", "/api/projects"):
        if p["name"] == STORY_PROJECT:
            await s.api("DELETE", f"/api/projects/{p['id']}")

    await s.card(
        "Tutorial 1",
        "From story idea to shot list",
        "Write an idea, refine it, build a story bible, and break it into scenes and shots.",
    )
    s.chapter("Introduction")
    await s.say("intro")

    s.chapter("Create a project")
    await page.goto(f"{s.base}/projects")
    await page.get_by_role("heading", name="Projects", exact=True).wait_for()
    await asyncio.sleep(0.4)
    await s.say("project", wait=False)
    await s.click(page.get_by_role("button", name="New project").first)
    dialog = page.get_by_role("dialog", name="New project")
    await s.type(dialog.get_by_role("textbox", name="Name"), STORY_PROJECT, delay=45)
    await s.move(dialog.get_by_role("combobox", name="Aspect ratio"))
    await s.quiet(0.2)
    await s.click(dialog.get_by_role("button", name="Create project"), after=1.0)
    await page.get_by_role("textbox", name="Story idea").wait_for()

    s.chapter("Write the idea")
    await s.say("idea", wait=False)
    await s.type(
        page.get_by_role("textbox", name="Story idea"),
        "An old lighthouse keeper finds a mechanical seagull washed up on the rocks after a storm. "
        "He repairs it, and the gull leads him to a message in a bottle from his long-lost brother.",
        delay=14,
    )
    await s.quiet(0.2)
    await s.say("options", wait=False)
    await s.type(page.locator("#opt-genre"), "fantasy")
    await s.type(page.locator("#opt-tone"), "wistful, hopeful")
    await s.type(page.locator("#opt-length"), "30")
    await s.type(page.locator("#opt-style"), "hand-painted watercolor, muted colors")
    await s.quiet()

    s.chapter("Refine the story")
    await s.say("refine", wait=False)
    await s.move(page.get_by_role("button", name="Refine", exact=True))
    await s.quiet(0.2)
    await s.click(page.get_by_role("button", name="Refine", exact=True))
    refined = page.get_by_test_id("refined-editor")
    await s.fast_forward(lambda: refined.wait_for(timeout=240_000), "Refining with your LLM")
    await s.scroll_to(refined, "start")
    s.mark("poster")
    await s.say("refined")

    s.chapter("Build the story bible")
    build = page.get_by_role("button", name="Build bible")
    await s.scroll_to(build)
    await s.say("bible", wait=False)
    await s.move(build)
    await s.quiet(0.2)
    await s.click(build)
    bible = page.get_by_test_id("bible-view")
    await s.fast_forward(lambda: bible.wait_for(timeout=300_000), "Building the story bible")
    await s.scroll_to(bible, "start")
    await s.say("bible_done", wait=False)
    await asyncio.sleep(3)
    await page.locator("main").evaluate("(m) => m.scrollBy({top: 260, behavior: 'smooth'})")
    await s.quiet()

    s.chapter("Break down into shots")
    breakdown = page.get_by_role("button", name="Break down into shots")
    await s.scroll_to(breakdown)
    await s.say("breakdown", wait=False)
    await s.move(breakdown)
    await s.quiet(0.2)
    await s.click(breakdown)
    summary = page.get_by_test_id("board-summary")
    await s.fast_forward(lambda: summary.wait_for(timeout=300_000), "Breaking the story into shots")
    await s.move(summary)
    await asyncio.sleep(1.2)

    s.chapter("The Shots board")
    await s.click(page.get_by_role("tab", name="Shots board"), after=1.0)
    await s.say("board", wait=False)
    first = page.get_by_test_id("shot-1.1")
    await first.wait_for()
    await s.move(first)
    second = page.get_by_test_id("shot-1.2")
    if await second.count():
        await asyncio.sleep(0.8)
        await s.move(second)
    await s.quiet()

    s.chapter("Edit a shot")
    await s.say("panel", wait=False)
    await s.click(first, after=0.8)
    panel = page.get_by_role("dialog")
    await s.move(panel.locator("#sh-kf"))
    await asyncio.sleep(1.5)
    await s.move(panel.locator("#sh-cam"))
    await asyncio.sleep(1.0)
    await s.quiet()
    rw = panel.locator("#sh-rw")
    await s.scroll_to(rw)
    await s.say("rewrite", wait=False)
    await s.type(rw, "make it golden hour, with warmer light", delay=35)
    await s.quiet(0.2)
    await s.click(panel.get_by_role("button", name="Rewrite"))
    discard = panel.get_by_role("button", name="Discard")
    await s.fast_forward(lambda: discard.wait_for(timeout=120_000), "Rewriting the prompts")
    await s.scroll_to(rw, "start")
    await s.say("diff", wait=False)
    await asyncio.sleep(1.2)
    await s.click(discard)
    await s.quiet()
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.6)

    s.chapter("Next steps")
    await s.card(
        "Up next",
        "Reference images and keyframes",
        "Give every character and place a consistent look, then compose each shot's first frame.",
    )
    await s.say("outro")
    await asyncio.sleep(0.8)


IMAGE_LINES = {
    "intro": "In this tutorial, we'll give our story a look: reference images for its characters and "
    "places, and a keyframe for the first shot.",
    "ingredients": "Open the Ingredients tab. These came from the story bible: characters, locations, "
    "props, and the overall style. Each card has a visual prompt that describes how it looks.",
    "model": "At the top, pick an image model, and how many candidates to make per run. FLUX dev, "
    "quantized to fit a sixteen gigabyte GPU, is a great default.",
    "generate": "Click Generate on a character. The job runs on your own GPU. You can follow its "
    "progress right on the card, with a live preview of each image as it forms.",
    "primary": "The first image becomes the character's primary reference. If you prefer another "
    "candidate, just click it. You can also upload a picture of your own.",
    "location": "Let's make a reference for the location too, so the keyframe can put the character in "
    "the right setting.",
    "board": "Now open the Shots board, and click the first shot.",
    "keyframe": "The Keyframe section lists the ingredient images it will use as references: characters "
    "first, then props, then the scene's location.",
    "compose": "Click Compose keyframe. LocalFlow places the references side by side on a reference "
    "sheet, and FLUX Kontext paints a single new frame that keeps each one's look.",
    "select": "Click the candidate to make it the shot's keyframe. The shot is now marked keyframe "
    "ready, and the board shows the image.",
    "playground": "For experiments outside the story, the Playground runs any installed model directly. "
    "Pick a model, add LoRAs with their weights, and set a seed to reproduce a result exactly.",
    "outro": "That's it. Every shot can now start from a consistent keyframe, ready to be turned into video.",
}


async def images_tutorial(s: Session) -> None:
    page = s.page
    project = next((p for p in await s.api("GET", "/api/projects") if p["name"] == STORY_PROJECT), None)
    if project is None:
        raise SystemExit(f'Project "{STORY_PROJECT}" not found: record the story tutorial first.')
    pid = project["id"]
    board = await s.api("GET", f"/api/projects/{pid}/scenes")
    ingredients = {i["id"]: i for i in await s.api("GET", f"/api/projects/{pid}/ingredients")}
    hero = shot_no = location = None
    for si, scene in enumerate(board["scenes"]):
        for hi, shot in enumerate(scene["shots"]):
            chars = [i for i in shot["ingredient_ids"] if ingredients.get(i, {}).get("kind") == "character"]
            if chars:
                hero, shot_no = ingredients[chars[0]], (f"{si + 1}.{hi + 1}", shot["id"])
                location = ingredients.get(scene.get("location_ingredient_id") or "")
                break
        if hero:
            break
    if hero is None or shot_no is None:
        raise SystemExit("No shot with a character: re-run the story tutorial.")
    # Start clean, so the first candidate visibly becomes the primary.
    for ing in [hero, location]:
        if ing:
            await clear_candidates(s, pid, {"ingredient_id": ing["id"]})
            await s.api("PATCH", f"/api/ingredients/{ing['id']}/primary", {"asset_id": None})
    await s.api("POST", f"/api/shots/{shot_no[1]}/select", {"slot": "keyframe", "asset_id": None})
    await clear_candidates(s, pid, {"shot_id": shot_no[1], "slot": "keyframe"})

    await s.card(
        "Tutorial 2",
        "Reference images and keyframes",
        "Generate consistent references for characters and places, then compose a shot's keyframe from them.",
    )
    s.chapter("Introduction")
    await s.say("intro")

    s.chapter("Ingredients")
    await page.goto(f"{s.base}/projects/{pid}/ingredients")
    card = page.get_by_test_id(f"ingredient-{hero['key']}")
    await card.wait_for()
    await asyncio.sleep(0.5)
    await s.say("ingredients", wait=False)
    await s.move(card.get_by_role("textbox", name="Visual prompt"))
    await s.quiet()

    s.chapter("Pick a model")
    await s.say("model", wait=False)
    await s.select(page.get_by_role("combobox", name="Model"), label="FLUX.1 [dev] · GGUF Q4_K_S")
    await s.select(page.get_by_role("combobox", name="Images per run"), label="2 per run")
    await s.quiet()

    s.chapter("Generate a character")
    gen = card.get_by_role("button", name="Generate 2")
    await s.scroll_to(card)
    await s.say("generate", wait=False)
    await s.click(gen)
    progress = card.get_by_test_id("job-progress")
    await progress.wait_for(timeout=30_000)
    await s.move(progress)
    await s.quiet()
    candidates = card.get_by_role("button", name="Use as primary image")
    await s.fast_forward(lambda: candidates.first.wait_for(timeout=600_000), "Generating on your GPU")
    await page.get_by_alt_text(f"{hero['name']} reference").wait_for(timeout=30_000)
    s.mark("poster")

    s.chapter("Choose the primary")
    await s.say("primary", wait=False)
    await s.move(page.get_by_alt_text(f"{hero['name']} reference"))
    await asyncio.sleep(1.2)
    await s.click(candidates.first, after=1.0)
    await s.move(card.get_by_role("button", name="Upload"))
    await s.quiet()

    if location:
        s.chapter("A location reference")
        loc_card = page.get_by_test_id(f"ingredient-{location['key']}")
        await s.scroll_to(loc_card)
        await s.say("location", wait=False)
        await s.click(loc_card.get_by_role("button", name="Generate 2"))
        await s.quiet()
        await s.fast_forward(
            lambda: page.get_by_alt_text(f"{location['name']} reference").wait_for(timeout=600_000),
            "Generating on your GPU",
        )
        await asyncio.sleep(0.8)

    s.chapter("Compose a keyframe")
    await s.say("board", wait=False)
    await s.click(page.get_by_role("tab", name="Shots board"), after=0.8)
    shot_card = page.get_by_test_id(f"shot-{shot_no[0]}")
    await shot_card.wait_for()
    await s.click(shot_card, after=0.8)
    await s.quiet()
    panel = page.get_by_role("dialog")
    section = panel.get_by_test_id("keyframe-section")
    await s.scroll_to(section, "start")
    await s.say("keyframe", wait=False)
    await s.move(section.get_by_test_id("keyframe-references"))
    await s.quiet()
    await s.select(section.get_by_role("combobox", name="Keyframe candidates"), label="1 image")
    compose = section.get_by_role("button", name="Compose keyframe")
    await s.say("compose", wait=False)
    await s.move(compose)
    await s.quiet(0.2)
    await s.click(compose)
    await section.get_by_test_id("job-progress").wait_for(timeout=30_000)
    pick = section.get_by_role("button", name="Use as keyframe")
    await s.fast_forward(lambda: pick.first.wait_for(timeout=600_000), "Composing with FLUX Kontext")
    await s.scroll_to(section.get_by_test_id("keyframe-candidates"))

    s.chapter("Select the keyframe")
    await s.say("select", wait=False)
    await s.click(pick.first, after=1.2)
    await s.scroll_to(panel.get_by_alt_text("Selected keyframe"))
    await asyncio.sleep(1.0)
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.6)
    await s.move(page.get_by_test_id(f"shot-{shot_no[0]}"))
    await s.quiet()

    s.chapter("The Playground")
    await s.say("playground", wait=False)
    await s.click(page.get_by_role("link", name="Playground"), after=0.6)
    await s.click(page.get_by_role("tab", name="Text → image"), after=0.6)
    form = page.get_by_role("form", name="Generation form")
    await s.select(form.get_by_role("combobox", name="Model"), label="SDXL 1.0 base (fp16)")
    lora = form.get_by_role("combobox", name="Add LoRA")
    if await lora.count():
        await s.move(lora)
        await lora.select_option(index=1)
        await asyncio.sleep(0.8)
    await s.type(form.locator("#pg-seed"), "1234", delay=80)
    await s.quiet()

    s.chapter("Wrap-up")
    await s.card(
        "Tutorial 2",
        "Consistent looks, shot by shot",
        "Ingredients give every shot the same characters and places. Next, keyframes become video.",
    )
    await s.say("outro")
    await asyncio.sleep(0.8)


FILM_PROJECT = "The Last Rose"
FILM_LINES = {
    "intro": "In this tutorial, we'll make a complete short film with LocalFlow: from a two-sentence idea "
    "to a finished video, with keyframes, motion, voices, sound effects, and music. Apart from the story "
    "writing, everything runs on your own GPU.",
    "project": "Start with a new project, and give it a name.",
    "idea": "Write the idea, and set a target length. Fifteen seconds gives us a few shots: enough to see "
    "the whole pipeline.",
    "story": "Now refine the idea into a story, build the story bible, and break it down into shots. Your "
    "language model takes a few seconds for each step.",
    "story_done": "We have scenes and shots, each sized to fit one video clip, and an ingredient for every "
    "character, place, and prop.",
    "ingredients": "Next, give the main character a look. On the Ingredients tab, pick the FLUX dev model, "
    "and click Generate.",
    "ingredients_done": "The first candidate becomes the character's primary reference. The other "
    "characters, places, and props get theirs the same way, so let's fast-forward through them.",
    "generate_all": "Now the big step. On the Shots board, Generate all composes a keyframe for every shot "
    "from its references, then animates each keyframe with Wan 2.2. With Use each new keyframe right away "
    "ticked, it runs from start to finish on its own.",
    "generate_wait": "This is the longest part: a few minutes of GPU time per shot. You can keep working, "
    "or step away. Progress shows in the Jobs drawer.",
    "takes": "Every shot now has a keyframe and a video take. Open a shot to see them.",
    "review": "Check consistency asks a vision model to compare the take with the reference images. It "
    "lists anything that drifted, like a different hair color, and suggests a corrected prompt.",
    # True whichever way the check goes: the take may match, or drift in a detail or two.
    "review_done": "Here's the verdict, with anything that drifted. Apply fix puts the corrected prompt "
    "into the shot, ready to regenerate.",
    "audio": "Below the video is the Audio section. Generate dialogue speaks the shot's lines in each "
    "character's voice, timed to the clip. Generate sound effects watches the picture, and makes matching "
    "sound with MMAudio.",
    "audio_rest": "The other shots get their audio the same way.",
    "timeline": "On the Timeline tab, Auto-assemble lays out every take in order, with its dialogue and "
    "sound effects on their own tracks.",
    "preview": "Press play to preview the cut. You can trim clips, reorder them, and change the transitions "
    "right here.",
    "music": "For a score, click plus on the Music track. MusicGen composes from the shots' music cues, and "
    "the result is placed under the whole film.",
    "export": "Finally, export. Pick a size, and LocalFlow renders an MP4 with every track mixed together.",
    "film": "And here's our film.",
    "library": "Everything you made is in the Assets tab. Filter by type, model, or scene, group things into "
    "collections, or reuse any result's settings in the Playground.",
    "outro": "That's the whole journey, from an idea to a finished film, on your own machine. The other "
    "tutorials go deeper into each step.",
}


async def wait_idle(s: Session, pid: str, limit_s: float) -> None:
    """Until the project has no queued or running jobs (they're queued before this is called)."""
    deadline = time.time() + limit_s
    while time.time() < deadline:
        page = await s.api("GET", f"/api/jobs?status=queued&status=running&project_id={pid}&limit=1")
        if page["total"] == 0:
            return
        await asyncio.sleep(2)
    raise RuntimeError(f"jobs still running after {limit_s:.0f} s")


def all_shots(board: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"{si + 1}.{hi + 1}", sh)
        for si, sc in enumerate(board["scenes"])
        for hi, sh in enumerate(sc["shots"])
    ]


async def film_tutorial(s: Session) -> None:
    page = s.page
    for p in await s.api("GET", "/api/projects"):
        if p["name"] == FILM_PROJECT:
            await s.api("DELETE", f"/api/projects/{p['id']}")
    # Against a FakeEngine server (a quick rehearsal: placeholder media, no model picks) or the real thing.
    rehearsal = (await s.api("GET", "/api/system/status"))["worker"]["engine"] == "fake"
    flux = next(
        (m for m in await s.api("GET", "/api/models") if m.get("catalog_key") == "flux1-dev-q4"), None
    )
    if not rehearsal and (not flux or flux["status"] != "installed"):
        raise SystemExit("Install FLUX.1 [dev] Q4_K_S (plus Kontext, Wan 2.2 5B, Kokoro, MMAudio, MusicGen).")

    await s.card(
        "Tutorial",
        "Make a film, start to finish",
        "Idea, story, references, keyframes, video, voices, sound, music, and export, in one pass.",
    )
    s.chapter("Introduction")
    await s.say("intro")

    # --- story ---------------------------------------------------------------------------------
    s.chapter("Create a project")
    await page.goto(f"{s.base}/projects")
    await page.get_by_role("heading", name="Projects", exact=True).wait_for()
    await s.say("project", wait=False)
    await s.click(page.get_by_role("button", name="New project").first)
    dialog = page.get_by_role("dialog", name="New project")
    await s.type(dialog.get_by_role("textbox", name="Name"), FILM_PROJECT, delay=45)
    await s.quiet(0.2)
    await s.click(dialog.get_by_role("button", name="Create project"), after=1.0)
    idea = page.get_by_role("textbox", name="Story idea")
    await idea.wait_for()
    pid = page.url.split("/projects/")[1].split("/")[0]

    s.chapter("Idea and story")
    await s.say("idea", wait=False)
    await s.type(
        idea,
        "A rusty robot gardener tends the last rose in an abandoned greenhouse. One morning a little girl "
        "climbs in through a broken pane, and the robot shows her the rose.",
        delay=12,
    )
    await s.type(page.locator("#opt-length"), "15")
    await s.type(page.locator("#opt-style"), "warm cinematic film, soft morning light")
    await s.quiet()
    await s.say("story", wait=False)
    refine = page.get_by_role("button", name="Refine", exact=True)
    await s.move(refine)
    await s.quiet(0.2)
    await s.click(refine)
    refined = page.get_by_test_id("refined-editor")
    await s.fast_forward(lambda: refined.wait_for(timeout=240_000), "Refining with your LLM")
    build = page.get_by_role("button", name="Build bible")
    await s.scroll_to(build)
    await s.click(build)
    bible = page.get_by_test_id("bible-view")
    await s.fast_forward(lambda: bible.wait_for(timeout=300_000), "Building the story bible")
    breakdown = page.get_by_role("button", name="Break down into shots")
    await s.scroll_to(breakdown)
    await s.click(breakdown)
    summary = page.get_by_test_id("board-summary")
    await s.fast_forward(lambda: summary.wait_for(timeout=300_000), "Breaking the story into shots")
    await s.move(summary)
    await s.say("story_done")

    # --- references ------------------------------------------------------------------------------
    board = await s.api("GET", f"/api/projects/{pid}/scenes")
    ingredients = {i["id"]: i for i in await s.api("GET", f"/api/projects/{pid}/ingredients")}
    shots = all_shots(board)
    hero = next(
        (
            ingredients[i]
            for _, sh in shots
            for i in sh["ingredient_ids"]
            if ingredients.get(i, {}).get("kind") == "character"
        ),
        None,
    ) or next((i for i in ingredients.values() if i["kind"] == "character"), None)
    if hero is None:
        raise SystemExit("The breakdown has no character; record again.")

    s.chapter("A reference for the hero")
    await s.click(page.get_by_role("tab", name="Ingredients"), after=0.8)
    card = page.get_by_test_id(f"ingredient-{hero['key']}")
    await card.wait_for()
    await s.say("ingredients", wait=False)
    if not rehearsal:
        await s.select(page.get_by_role("combobox", name="Model"), label="FLUX.1 [dev] · GGUF Q4_K_S")
    await s.select(page.get_by_role("combobox", name="Images per run"), label="2 per run")
    await s.scroll_to(card)
    await s.click(card.get_by_role("button", name="Generate 2"))
    await s.quiet()
    candidates = card.get_by_role("button", name="Use as primary image")
    await s.fast_forward(lambda: candidates.first.wait_for(timeout=900_000), "Generating on your GPU")
    await page.get_by_alt_text(f"{hero['name']} reference").wait_for(timeout=60_000)
    await s.move(page.get_by_alt_text(f"{hero['name']} reference"))
    await s.say("ingredients_done")
    others = [
        i
        for i in ingredients.values()
        if i["kind"] != "style" and i["id"] != hero["id"] and not i["primary_asset_id"]
    ]
    model_id = None if rehearsal or not flux else flux["id"]
    for ing in others:
        await s.api(
            "POST", f"/api/ingredients/{ing['id']}/generate", {"count": 1, "model_entry_id": model_id}
        )
    if others:
        await s.fast_forward(lambda: wait_idle(s, pid, 3600), f"Generating {len(others)} more references")
        await page.locator("main").evaluate("(m) => m.scrollTo({top: 0, behavior: 'smooth'})")
        await asyncio.sleep(1.5)

    # --- keyframes and video -------------------------------------------------------------------------
    s.chapter("Generate every shot")
    await s.click(page.get_by_role("tab", name="Shots board"), after=0.8)
    await page.get_by_test_id("shot-1.1").wait_for()
    await s.say("generate_all", wait=False)
    await s.click(page.get_by_role("button", name="Generate all"), after=0.6)
    gen = page.get_by_role("dialog", name="Generate all")
    await s.click(gen.get_by_role("checkbox", name=re.compile("Use each new keyframe right away")))
    await s.quiet(0.2)
    await s.click(gen.get_by_role("button", name="Queue"), after=0.8)
    await gen.get_by_test_id("generate-all-result").wait_for()
    await asyncio.sleep(1.0)
    await s.click(gen.get_by_role("button", name="Done"), after=0.5)
    await s.click(page.get_by_role("button", name="Toggle jobs drawer"), after=0.6)
    await s.say("generate_wait")
    await s.click(page.get_by_role("button", name="Toggle jobs drawer"), after=0.4)

    async def all_takes() -> None:
        deadline = time.time() + 4 * 3600
        while time.time() < deadline:
            b = await s.api("GET", f"/api/projects/{pid}/scenes")
            if all(sh["selected_video_asset_id"] for _, sh in all_shots(b)):
                return
            await asyncio.sleep(5)
        raise RuntimeError("shots still without takes")

    await s.fast_forward(all_takes, f"Keyframes and video for {len(shots)} shots")
    await wait_idle(s, pid, 600)
    await asyncio.sleep(1.5)  # the board updates over the WebSocket
    await s.say("takes", wait=False)
    for no, _ in shots[:3]:
        await s.move(page.get_by_test_id(f"shot-{no}"))
        await asyncio.sleep(0.6)
    await s.quiet()

    # --- review and audio on one shot ------------------------------------------------------------------
    plans = {sh["id"]: await s.api("POST", f"/api/shots/{sh['id']}/audio-plan", {}) for _, sh in shots}
    speaking = [(no, sh) for no, sh in shots if plans[sh["id"]]["lines"]]
    shot_no, shot = (speaking or shots)[0]
    s.chapter("Check a take")
    await s.click(page.get_by_test_id(f"shot-{shot_no}"), after=0.8)
    panel = page.get_by_role("dialog")
    video = panel.get_by_test_id("video-section")
    await s.scroll_to(video, "start")
    await panel.get_by_test_id("selected-take").evaluate("(v) => { v.muted = true; v.play(); }")
    review = panel.get_by_test_id("review-take")
    if await review.count():
        await s.say("review", wait=False)
        await s.move(review.get_by_role("button", name="Check consistency"))
        await s.quiet(0.2)
        await s.click(review.get_by_role("button", name="Check consistency"))
        verdict = review.get_by_role("region", name="Consistency check")
        await s.fast_forward(lambda: verdict.wait_for(timeout=180_000), "Asking the vision model")
        await s.scroll_to(verdict)
        await s.say("review_done", wait=False)
        await asyncio.sleep(1.5)
        dismiss = verdict.get_by_role("button", name="Dismiss")
        await s.quiet()
        if await dismiss.count():
            await s.click(dismiss)

    s.chapter("Dialogue and sound effects")
    audio = panel.get_by_test_id("audio-section")
    await s.scroll_to(audio, "start")
    await s.say("audio", wait=False)
    dialogue = audio.get_by_role("button", name="Generate dialogue")
    if await dialogue.is_enabled():
        await s.click(dialogue, after=1.5)
    await s.click(audio.get_by_role("button", name="Generate sound effects"), after=1.0)
    await s.scroll_to(audio, "start")  # the panel remounts as the shot updates, scrolled to the top
    await s.quiet()
    await s.fast_forward(lambda: wait_idle(s, pid, 1800), "Voices and sound on your GPU")
    await asyncio.sleep(1.0)
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.5)
    await s.say("audio_rest")
    rest = [sh for _, sh in shots if sh["id"] != shot["id"]]
    for sh in rest:
        parts = ["sfx", *(["dialogue"] if plans[sh["id"]]["lines"] else [])]
        await s.api("POST", f"/api/shots/{sh['id']}/generate-audio", {"parts": parts})
    if rest:
        await s.fast_forward(lambda: wait_idle(s, pid, 3600), f"Audio for {len(rest)} more shots")

    # --- timeline, music, export -------------------------------------------------------------------------
    s.chapter("Assemble the timeline")
    await s.click(page.get_by_role("tab", name="Timeline"), after=1.0)
    await s.say("timeline", wait=False)
    await s.click(page.get_by_role("button", name="Auto-assemble").first, after=1.5)
    await page.get_by_test_id("timeline-view").wait_for()
    await s.quiet()
    await s.say("preview", wait=False)
    await s.click(page.get_by_role("button", name="Play", exact=True))
    await asyncio.sleep(5)
    await s.quiet()
    await s.click(page.get_by_role("button", name="Pause", exact=True))

    s.chapter("Music")
    await s.say("music", wait=False)
    await s.click(page.get_by_role("button", name="Add audio to Music"), after=0.8)
    composer = page.get_by_test_id("music-generator")
    await s.click(composer.get_by_role("button", name="Generate", exact=True))
    await s.quiet()
    await s.fast_forward(lambda: wait_idle(s, pid, 1800), "Composing with MusicGen")
    await asyncio.sleep(1.0)
    await page.keyboard.press("Escape")
    await asyncio.sleep(1.0)
    await s.move(page.get_by_test_id("track-a_music"))
    await asyncio.sleep(1.0)

    s.chapter("Export")
    panel = page.get_by_test_id("export-panel")
    await s.scroll_to(panel)
    await s.say("export", wait=False)
    await s.move(panel.get_by_role("combobox", name="Export size"))
    await s.click(panel.get_by_role("button", name="Export MP4"))
    await s.quiet()
    history = panel.get_by_test_id("export-history").get_by_role("listitem")
    await s.fast_forward(lambda: history.first.wait_for(timeout=1800_000), "Rendering the MP4")

    s.chapter("The finished film")
    exports = await s.api("GET", f"/api/projects/{pid}/assets?source=export&limit=1")
    film = exports["items"][0]
    local = WORK / "film.mp4"
    r = await page.request.get(
        f"{s.base}/api/files/{pid}/{film['id']}", headers={"X-LocalFlow-Token": s.token}
    )
    local.write_bytes(await r.body())
    await s.say("film")
    await s.click(panel.get_by_role("button", name="Play export"), after=0.3)
    player = page.get_by_role("dialog").locator("video")
    await player.wait_for()
    await player.evaluate(
        "(v) => new Promise((done) => { v.loop = false; v.currentTime = 0; v.muted = true;"
        " v.addEventListener('playing', () => done(), {once: true}); v.play().catch(() => done()); })"
    )
    seconds = float(film["duration_s"] or ffprobe_duration(local))
    await page.mouse.move(VIEW["width"] - 6, VIEW["height"] - 6)  # keep the cursor off the picture
    s.mouse = (VIEW["width"] - 6, VIEW["height"] - 6)
    s.sounds.append((s.now(), local, seconds))
    s.marks["poster"] = s.now() + min(seconds * 0.6, 9.0)  # past the opening fade
    await asyncio.sleep(seconds + 0.5)
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.6)

    s.chapter("The asset library")
    await s.click(page.get_by_role("tab", name="Assets"), after=1.0)
    await s.say("library", wait=False)
    source = page.get_by_role("combobox", name="Source")
    await s.move(source)
    await source.select_option(value="export")
    await asyncio.sleep(1.5)
    await source.select_option(value="")
    await s.move(page.get_by_role("navigation", name="Collections"))
    await s.quiet()

    s.chapter("Wrap-up")
    await s.card(
        "Tutorial",
        "From idea to film, on your GPU",
        "Story, references, keyframes, video, voices, sound, music, and export, all in one project.",
    )
    await s.say("outro")
    await asyncio.sleep(0.8)


SMOKE_LINES = {
    "a": "This is a quick check of the recording pipeline.",
    "b": "The wait you just saw was sped up, and the narration stayed in sync.",
}


async def smoke_tutorial(s: Session) -> None:
    """Pipeline self-test (`--only smoke --out <dir>`): no LLM or GPU work."""
    await s.card("Self-test", "Recording pipeline", "Narration, cursor, time-lapse and captions.")
    s.chapter("Start")
    await s.say("a", wait=False)
    await s.page.goto(f"{s.base}/models")
    await s.click(s.page.get_by_role("tab", name="Catalog"))
    await s.quiet()
    await s.fast_forward(lambda: asyncio.sleep(8), "Waiting")
    s.chapter("After the wait")
    await s.say("b")


async def clear_candidates(s: Session, pid: str, query: dict[str, str]) -> None:
    qs = "&".join(f"{k}={v}" for k, v in query.items())
    page = await s.api("GET", f"/api/projects/{pid}/assets?type=image&limit=200&{qs}")
    for a in page["items"]:
        await s.api("DELETE", f"/api/assets/{a['id']}")


TUTORIALS = [
    Tutorial(
        "full-film",
        "Make a film, start to finish",
        "The whole pipeline in one project: a two-sentence idea becomes a story and shots, reference images, "
        "keyframes and video on your GPU, voices, sound effects and music, a timeline, and an MP4 export.",
        FILM_LINES,
        film_tutorial,
    ),
    Tutorial(
        "story-to-shots",
        "From story idea to shot list",
        "Create a project, turn a few sentences into a refined story and a story bible with your LLM, and "
        "break it into scenes and shots on the Shots board.",
        STORY_LINES,
        story_tutorial,
    ),
    Tutorial(
        "images-and-keyframes",
        "Reference images and keyframes",
        "Generate consistent reference images for characters and locations on your GPU, then compose a "
        "shot's keyframe from them with FLUX Kontext.",
        IMAGE_LINES,
        images_tutorial,
    ),
]
EXTRA = [Tutorial("smoke", "Recording self-test", "Pipeline check.", SMOKE_LINES, smoke_tutorial)]


# --- assembly ----------------------------------------------------------------------------------------
@dataclass
class Segment:
    start: float
    end: float
    speed: float


def segments(fast: list[tuple[float, float]], total: float) -> list[Segment]:
    out, cursor = [], 0.0
    for a, b in sorted(fast):
        if a > cursor:
            out.append(Segment(cursor, a, 1.0))
        length = b - a
        target = min(6.0, max(2.5, length / 10))  # a long wait still reads as "it took a while"
        out.append(Segment(a, b, max(1.0, length / target)))
        cursor = b
    if total > cursor:
        out.append(Segment(cursor, total, 1.0))
    return out


def remap(t: float, segs: list[Segment]) -> float:
    acc = 0.0
    for sg in segs:
        if t <= sg.start:
            break
        acc += (min(t, sg.end) - sg.start) / sg.speed
    return acc


def vtt_time(t: float) -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def captions(placed: list[tuple[float, Clip]], segs: list[Segment]) -> str:
    cues = []
    for t, clip in placed:
        parts = clip.sentences or [(0.0, clip.duration, clip.text)]
        for off, dur, text in parts:
            for a, b, chunk in split_cue(off, dur, text):
                cues.append((remap(t + a, segs), remap(t + b, segs), chunk))
    lines = ["WEBVTT", ""]
    for i, (a, b, text) in enumerate(cues, 1):
        lines += [str(i), f"{vtt_time(a)} --> {vtt_time(b)}", text, ""]
    return "\n".join(lines)


def split_cue(off: float, dur: float, text: str, limit: int = 84) -> list[tuple[float, float, str]]:
    """Long sentences become several cues, timed in proportion to their length."""
    words, chunks, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    total = sum(len(c) for c in chunks) or 1
    out, t = [], off
    for c in chunks:
        d = dur * len(c) / total
        out.append((t, t + d, c))
        t += d
    return out


def run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-3000:])


def assemble(
    tut: Tutorial, s: Session, cast: Screencast, end: float, work: Path, voice: str
) -> dict[str, Any]:
    if not cast.frames:
        raise RuntimeError("no frames captured")
    begin = s.marks.get("start", 0.0)  # the video starts at the first title card
    offset = begin
    segs = segments([(a - begin, b - begin) for a, b in s.fast], end - begin)
    length = remap(end - begin, segs)

    # Each frame is shown from its own timestamp until the next one, on the edited timeline (time-lapse
    # stretches are compressed by remapping). One concat list, one pass, no trims. The screencast only
    # sends frames on change, so the last frame is held until the end.
    times = [ts - s.t0 for ts, _ in cast.frames]
    listing: list[str] = []
    for i, (_, path) in enumerate(cast.frames):
        a = max(times[i], begin)
        b = times[i + 1] if i + 1 < len(times) else end + 1.0
        if b <= begin:
            continue
        d = remap(b - begin, segs) - remap(a - begin, segs)
        if d > 0:
            listing.append(f"file '{path.as_posix()}'\nduration {d:.5f}")
    listing.append(f"file '{cast.frames[-1][1].as_posix()}'")
    (work / "frames.txt").write_text("\n".join(listing))

    # Frames are 1920x1080 give or take a pixel of rounding: fit without resampling, pad the rest.
    vfilter = (
        f"[0:v]fps={FPS},scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p[vout]"
    )

    inputs = ["-f", "concat", "-safe", "0", "-i", str(work / "frames.txt")]
    aparts = []
    for k, (t, clip) in enumerate(s.placed):
        inputs += ["-i", str(clip.path)]
        ms = int(remap(t - offset, segs) * 1000)
        aparts.append(f"[{k + 1}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}:all=1[a{k}]")
    labels = [f"[a{k}]" for k in range(len(s.placed))]
    for j, (t, path, seconds) in enumerate(s.sounds):
        n = len(s.placed) + 1 + j
        inputs += ["-i", str(path)]
        ms = int(remap(t - offset, segs) * 1000)
        aparts.append(
            f"[{n}:a]atrim=0:{seconds:.3f},aresample=48000,aformat=channel_layouts=stereo,"
            f"afade=t=out:st={max(0.0, seconds - 0.6):.3f}:d=0.6,adelay={ms}:all=1[s{j}]"
        )
        labels.append(f"[s{j}]")
    afilter = ";".join(aparts) + ";" + "".join(labels)
    afilter += f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0,apad,atrim=0:{length:.3f}[aout]"

    mp4 = OUT / f"{tut.id}.mp4"
    run_ffmpeg(
        [
            *inputs,
            "-filter_complex",
            f"{vfilter};{afilter}",
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            "-t",
            f"{length:.3f}",
            str(mp4),
        ]
    )
    poster_at = remap(s.marks.get("poster", end * 0.4) - offset, segs)
    run_ffmpeg(
        ["-ss", f"{poster_at:.2f}", "-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(OUT / f"{tut.id}.jpg")]
    )
    shifted = [(t - offset, clip) for t, clip in s.placed]
    (OUT / f"{tut.id}.vtt").write_text(captions(shifted, segs), encoding="utf-8")
    return {
        "id": tut.id,
        "title": tut.title,
        "summary": tut.summary,
        "duration_s": round(ffprobe_duration(mp4), 1),
        "video": f"/tutorials/{tut.id}.mp4",
        "poster": f"/tutorials/{tut.id}.jpg",
        "captions": f"/tutorials/{tut.id}.vtt",
        "chapters": [
            {"time_s": round(remap(t - offset, segs), 1), "title": title}
            for t, title in s.chapters
            if t >= offset
        ],
        "voice": voice,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def write_manifest(entries: dict[str, dict[str, Any]]) -> None:
    path = OUT / "index.json"
    existing = json.loads(path.read_text(encoding="utf-8")).get("tutorials", []) if path.exists() else []
    merged = {e["id"]: e for e in existing} | entries
    order = [t.id for t in TUTORIALS]
    items = sorted(merged.values(), key=lambda e: order.index(e["id"]) if e["id"] in order else 99)
    path.write_text(json.dumps({"version": 1, "tutorials": items}, indent=2) + "\n", encoding="utf-8")


# --- main --------------------------------------------------------------------------------------------
async def fit_viewport(ctx: BrowserContext, page: Page) -> None:
    """Resizes the window until the page viewport is exactly VIEW (the window frame takes some room)."""
    cdp = await ctx.new_cdp_session(page)
    win = (await cdp.send("Browser.getWindowForTarget"))["windowId"]

    def close(w: int, h: int) -> bool:  # fractional scaling can leave a pixel of rounding
        return abs(w - VIEW["width"]) <= 2 and abs(h - VIEW["height"]) <= 2

    for _ in range(4):
        iw, ih = await page.evaluate("[innerWidth, innerHeight]")
        if close(iw, ih):
            break
        bounds = (await cdp.send("Browser.getWindowBounds", {"windowId": win}))["bounds"]
        await cdp.send(
            "Browser.setWindowBounds",
            {
                "windowId": win,
                "bounds": {
                    "width": bounds["width"] + VIEW["width"] - iw,
                    "height": bounds["height"] + VIEW["height"] - ih,
                },
            },
        )
        await asyncio.sleep(0.4)
    size = await page.evaluate("[innerWidth, innerHeight, devicePixelRatio]")
    if not close(size[0], size[1]):
        raise SystemExit(f"Couldn't size the browser viewport to {VIEW}: got {size}")
    await cdp.detach()


async def record(tut: Tutorial, args: argparse.Namespace) -> dict[str, Any]:
    print(f"[{tut.id}] synthesizing {len(tut.lines)} narration lines with {args.voice}...", flush=True)
    clips = {k: await synthesize(v, args.voice, args.rate) for k, v in tut.lines.items()}
    work = WORK / tut.id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    async with async_playwright() as p:
        # A real device scale factor (not Playwright's emulated one, which the screencast ignores) makes
        # the compositor, and so the captured frames, render at 1920x1080 for a 1280x720 layout.
        browser = await p.chromium.launch(
            channel=args.channel,
            headless=not args.headed,
            args=[
                f"--force-device-scale-factor={SCALE}",
                f"--window-size={VIEW['width']},{VIEW['height']}",
                # Software rendering: Chrome's GPU compositing takes VRAM that the generation jobs plan to
                # use, and Wan (swap mode) then ran about 4x slower while recording.
                "--disable-gpu",
            ],
        )
        ctx = await browser.new_context(no_viewport=True)
        await ctx.add_init_script(CURSOR_JS)
        page = await ctx.new_page()
        page.set_default_timeout(20_000)
        await page.goto(f"{args.base_url}/projects")
        await fit_viewport(ctx, page)
        token = await page.locator('meta[name="localflow-token"]').get_attribute("content") or args.token
        if not token:
            raise SystemExit("No API token: serve the built app (`npm start`) or pass --token.")
        s = Session(page=page, clips=clips, base=args.base_url, token=token)
        cast = Screencast(ctx, page, work / "frames")
        s.t0 = time.time()
        await cast.start()
        try:
            print(f"[{tut.id}] recording...", flush=True)
            await tut.run(s)
            end = s.now()
        finally:
            await cast.stop()
            await asyncio.sleep(0.3)
            await ctx.close()
            await browser.close()
    print(f"[{tut.id}] {len(cast.frames)} frames, {len(s.fast)} time-lapses; encoding...", flush=True)
    entry = assemble(tut, s, cast, end, work, args.voice)
    if not args.keep_work:
        shutil.rmtree(work / "frames", ignore_errors=True)
    print(f"[{tut.id}] done: {entry['duration_s']} s -> {OUT / (tut.id + '.mp4')}", flush=True)
    return entry


async def main() -> None:
    global OUT
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]  # Windows consoles are often cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--only", help="comma-separated tutorial ids: " + ", ".join(t.id for t in TUTORIALS))
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--rate", default="+0%", help="edge-tts speaking rate, e.g. -5%%")
    ap.add_argument("--channel", default="chrome", help="installed browser: chrome or msedge")
    ap.add_argument("--token", default="", help="API token (only needed against `npm run dev`)")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument(
        "--out", type=Path, default=OUT, help="output directory (default: frontend/public/tutorials)"
    )
    args = ap.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg must be on PATH")
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else {t.id for t in TUTORIALS}
    entries = {}
    for tut in [*TUTORIALS, *EXTRA]:
        if tut.id in wanted:
            entries[tut.id] = await record(tut, args)
            write_manifest(entries)
    print(f"manifest: {OUT / 'index.json'}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
