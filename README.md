# LocalFlow

A locally-running, single-user AI filmmaking studio inspired by Google Flow. React frontend + Python (FastAPI)
backend; image/video/audio generation runs on your GPU. See [SPEC.md](SPEC.md) for the full design and
[docs/DECISIONS.md](docs/DECISIONS.md) for implementation decisions.

**Status:** milestones **M0** through **M5** are done.

- **M0:** projects, a SQLite-backed job queue, a supervised GPU worker process running the **FakeEngine**
  (placeholder outputs, no GPU needed), live progress over WebSocket, the Playground, and the app shell.
- **M1:** API keys, the LLM layer (OpenAI, Anthropic, OpenRouter via LiteLLM), and the story pipeline:
  idea → refined story → story bible (ingredients) → scenes and shots, with revisions, diffs, AI prompt
  rewrites, and a drag-and-drop shots board.
- **M2:** model management. The **Models** page has a curated catalog (FLUX.1-dev, SDXL, FLUX Kontext,
  Wan 2.2/2.1, Kokoro, MMAudio, ACE-Step, MusicGen) plus install from any Hugging Face repo or Civitai. It
  covers gated-repo access checks, resumable downloads with progress/cancel, license acknowledgement, and
  delete.
- **M3:** real image generation on the GPU with FLUX.1 [dev] (GGUF), FLUX.1 Kontext and SDXL, plus LoRAs.
  Ingredients get reference images (generate candidates, star a primary, or upload your own). Shot
  keyframes are composed from the shot's ingredient images with Kontext, or generated from text. The GPU
  chip shows the loaded model and VRAM.
- **M3.5:** the character creator. A wizard builds a character from a template, a description (expanded by
  the LLM) or a photo, adds consistent extra views (full body, turnaround, side, back, face), and saves it to
  the project and/or the **Cast library** for reuse across projects. Prompts can `@mention` characters,
  locations and props.
- **M4:** real video generation with Wan 2.2 TI2V-5B (and Wan 2.2 A14B / Wan 2.1 first-last-frame entries).
  Shots animate their keyframe (or generate from text), with camera presets, several takes, **Extend** for
  shots longer than one clip, and **Generate all** for the whole board. Cancelling a stuck job restarts the
  worker.
- **M5:** the timeline. **Auto-assemble** lays out every shot's selected take. You can reorder, trim, add
  crossfades and fades, place audio, preview, and **Export** a 720p or 1080p MP4.

Audio generation arrives in M6; until then the Playground's audio tabs use FakeEngine placeholders, and the
timeline takes uploaded audio.

## Timeline and export (M5)

- **Timeline tab → Auto-assemble** puts each shot's selected take on the video track, in board order.
  Shots in the same scene are cut together, a new scene crossfades in (0.5 s), and the film fades in from
  black. Shots without a take are skipped. Running it again rebuilds the video and keeps the music track.
- **Editing:**
  - Drag a clip to reorder it, or drag its edges to trim. Trims snap to clip edges and the playhead (an
    amber line shows the snap) and to whole frames.
  - Click a clip to open its panel: exact in/out, transition (cut, crossfade, fade from black) with its
    length, a different take of the same shot, moving it earlier or later, and the shot's prompt with
    **new takes / extend**.
- **Audio:** use **+** on Dialogue, Sound effects or Music to add a project audio file at the playhead, or
  upload one. Drag audio freely; the panel sets start, trim, gain and track. The speaker buttons mute a
  track. On Video, that mutes the clips' own sound.
- **Preview:** Space plays and pauses, ←/→ step a frame (Shift: a second), Home/End jump, and Delete
  removes the selected clip. The preview is approximate; the export is exact.
- **Export:**
  - Pick **720p** or **1080p** (in the project's aspect ratio) and optionally **Normalize loudness**.
  - The MP4 (H.264 + AAC) is saved under `projects/<id>/exports/` and listed under the button, with play
    and download.
  - Clips with a slightly different aspect ratio (Wan's 1280×704) are cropped to fill the frame; others
    are letterboxed.
  - A 30 s 720p film exports in a few seconds.

## Video (M4)

- Install **Wan 2.2 TI2V-5B** from **Models → Catalog** (about 20 GB). It makes 5 s clips at 1280×704 and
  24 fps, or 832×480 in the `low` VRAM profile.
- **Shot panel → Video:**
  - **Generate video** animates the selected keyframe (image → video). Without a keyframe, the shot is
    generated from text and the panel says so. The line above the button shows what will run: mode, model,
    size, frames and duration. **prompt** shows the exact prompt, including the camera move.
  - Choose the mode, the model, how strongly the camera move is described, and 1–4 takes (consecutive
    seeds).
  - Shots longer than one clip get **Generate + extend ×N**. **Extend take** continues any take from its
    last frame, adding about 5 s per segment.
  - The first take is selected automatically; click another take to switch. **Approve take** marks the
    shot as final. The board card shows the selected take.
  - The square button cancels a running job. If a job doesn't stop within 15 s (`cancel_grace_s`), the
    worker is restarted.
- **Shots tab → Generate all** composes missing keyframes and queues a video for every shot without a take.
  With **Use each new keyframe right away**, it doesn't wait for you to pick keyframes.
- **Speed on a 16 GB GPU** (RTX 5080): a 5 s 720p clip takes about 10 minutes at the default 30 steps,
  and a 480p clip about 2.5 minutes. Pick fewer **steps** in the Video section for quick drafts. Peak VRAM
  is about 10 GB.
- **Playground → Text → video / Image → video / Extend** can pick any installed video model.
- First/last-frame shots need the **Wan 2.1 FLF2V** entry: TI2V-5B can only start from a first frame.

## Characters (M3.5)

- **Ingredients → New character** opens the wizard:
  1. **Look:** start from one of 12 templates, describe the character and **Expand with AI**, or upload a
     photo (**Describe with AI** writes the visual prompt; **Clean up reference** re-renders it as a plain
     front-facing portrait). Generate portraits and click the one to build on.
  2. **Views:** describe the body and outfit, tick the views, and **Generate views**. With FLUX Kontext
     installed, each view is an edit of the portrait, so the face and outfit stay the same (about 90 s per
     view on a 16 GB card). Without it, views use text-to-image and are less consistent.
  3. **Identity:** name, `@key` (fixed once saved) and character info for the story LLM.
  4. **Voice:** describe it (gender, age, tone). Picking a real voice arrives with text-to-speech in M6.
  5. **Save**, optionally also to the Cast library.
- **`@mentions`:** type `@` in a shot, ingredient or Playground prompt to pick a character, location or prop.
  The first mention becomes its name plus visual prompt, and later ones just the name. Mentioning a
  character adds it to the shot. Unknown keys are flagged, and generating with one is refused. Write `@@`
  for a literal `@`. The Playground mentions Cast library characters.
- **Keyframe references follow the shot type:** wide shots use a character's full-body view, close-ups the
  face close-up (or the portrait). **Choose** in the shot panel overrides them per shot.
- **Cast library** (sidebar): characters shared across projects. Cast one into a project from the library,
  from **Ingredients → Import**, or from the Story tab's cast list. Projects get a copy; when the library
  version changes, the card offers **Update from library**, and **Push to library** sends a project's
  edits back. **Save as template** turns any character into a reusable template.
- The built-in template thumbnails are rendered with FLUX by
  `uv run scripts/characters/make_template_thumbnails.py` against the running app (`npm start`).

## Images (M3)

- **Ingredients tab:** pick a model and a count, then **Generate** on a card. The first image becomes the
  primary; click any candidate to make it the primary instead, or **Upload** your own image.
- **Shot panel → Keyframe:** **Compose keyframe** puts up to 3 primary images (characters, then props, then
  the scene's location) on a reference sheet. FLUX Kontext then renders one frame that keeps their look.
  With no references, or no image-edit model installed, it falls back to text-to-image and says so.
  Click a candidate to select it (first/last frames separately for first/last-frame shots). The selected
  keyframe shows on the board.
- **Playground → Text → image / Image edit:** choose any installed model and up to 3 LoRAs with weights.
- A 16 GB GPU runs FLUX Q4 at about 40 s per 1024² image (28 steps) and SDXL at about 6 s. Models load on
  first use (5–16 s) and stay loaded until a different one is needed. SDXL's CLIP encoders read only the
  first 77 tokens of a prompt; Flux ignores negative prompts.

## Models

**Models → Catalog** lists recommended models with their size, license and VRAM notes. **Hugging Face**
inspects any repo (`org/name` or URL): it shows the files, size, license and your access, lets you pick
files, and you confirm the adapter. **Civitai** searches checkpoints and LoRAs, with base-model mapping,
trigger words and permissions.

- **Gated Hugging Face repos** (e.g. FLUX.1-dev) need a token (`HF_TOKEN`, or Settings → API keys →
  *Import existing Hugging Face login* to reuse `hf auth login`). If the license isn't accepted yet, the card
  links to the model page; accept it there, then click **Check again**.
- **Civitai:** public files download without an account. Login-required versions need `CIVITAI_API_TOKEN`.
  Early-access versions must be unlocked on civitai.com.
- Files go to `models/hf/repos/<org>--<name>/` and `models/civitai/<modelId>/<versionId>/` in the data dir.
  Only `.safetensors`/`.gguf` are accepted by default. Kokoro and MMAudio ship PyTorch pickles and need
  `allow_pickle_weights = true` in `config.toml`.
- Before a model's first use, acknowledge its license (**Review license** on the card).

## API keys

Keys are read from **environment variables first**, so if these are set you don't need to do anything:

| Provider | Environment variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Hugging Face | `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) |
| Civitai | `CIVITAI_API_TOKEN` (or `CIVITAI_API_KEY`) |

On Windows, variables set with `setx` or System Properties are picked up even if the terminal was opened
before they were set. You can also paste keys in **Settings**; they're stored in the OS keyring (Windows
Credential Manager), and an environment variable always takes precedence. At least one LLM key is needed for
the Story tab.

Default models (override in Settings → Default models, or in `config.toml`):
`openai/gpt-5.4-mini`, `anthropic/claude-sonnet-5`, `openrouter/anthropic/claude-sonnet-5`. A 30-second
story (refine + bible + breakdown) costs about $0.02–0.03 with `gpt-5.4-mini`.

## Requirements

- Windows 11 or Linux
- [uv](https://docs.astral.sh/uv/) (manages Python 3.12 for the backend)
- Node.js 20+ and npm
- `ffmpeg` and `ffprobe` on `PATH` (or set `ffmpeg_path` / `ffprobe_path` in `config.toml`)
- For generation: an NVIDIA GPU with 16 GB+ VRAM (developed on an RTX 5080) and a driver that supports
  CUDA 13. Without one, the app still runs and uses the FakeEngine.

On Windows: `winget install astral-sh.uv Gyan.FFmpeg OpenJS.NodeJS.LTS`

If a command then reports `'uv' is not recognized` (or `ffmpeg`), the terminal still has the PATH from before
the install. Fully quit and reopen VS Code (a new terminal tab inherits VS Code's old environment), or refresh
the current PowerShell session:
`$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`

## Setup

```powershell
npm run setup        # uv sync (backend) + npm install (frontend)
```

`uv sync` installs the GPU stack (PyTorch CUDA 13 build, diffusers, transformers; about 3.5 GB). On a machine
without an NVIDIA GPU, skip it with `cd backend; uv sync --no-group gpu`. `engine = "auto"` (the default)
uses the GPU when it's usable and the FakeEngine otherwise; set `engine = "fake"` to force placeholders.

## Run

```powershell
npm run dev          # backend on :8765 + Vite on :5173, backend restarts on .py changes
```

Open <http://127.0.0.1:5173>. Create a project, write a few sentences on the **Story** tab, then **Refine** →
**Build bible** → **Break down into shots**, and open the **Shots board**. Install an image model from
**Models → Catalog** (FLUX.1 [dev] Q4_K_S is recommended for 16 GB GPUs; FLUX Kontext enables keyframe
composition), then generate ingredient images on the **Ingredients** tab and keyframes from the shot panel.
The **Playground** runs any generation type directly.

New to LocalFlow? Open **Tutorials** in the sidebar for narrated walkthroughs (with captions and chapters)
of the story pipeline and of reference images and keyframes.

Production-style single process (FastAPI serves the built frontend):

```powershell
npm start            # builds frontend/dist, then serves everything on http://127.0.0.1:8765
```

## Data and configuration

Everything lives in the data directory: `%USERPROFILE%\LocalFlow` on Windows, `~/LocalFlow` elsewhere
(override with `LOCALFLOW_DATA_DIR`, e.g. to put the models on a bigger drive; on Windows a user variable
set with `setx` is honored even in terminals opened before it). It holds `localflow.db`, `config.toml`
(created with commented defaults on first run), `projects/`, `models/`, `library/`, `logs/`, and `tmp/`.

Every `config.toml` key can be overridden with an env var `LOCALFLOW_<KEY>`, e.g. `LOCALFLOW_PORT=9000`.
The server only binds to `127.0.0.1`, and every API/WebSocket call needs the per-launch token.

## Develop

```powershell
npm test             # backend pytest (spawns the FakeEngine worker) + frontend vitest
npm run lint         # ruff, ruff format --check, mypy, eslint, prettier, tsc
npm run typegen      # regenerate frontend API types from the backend OpenAPI schema
```

Real-GPU regression test (Flux → Kontext → SDXL, then unload to baseline; about 2 minutes, uses the models
installed in your data dir and skips missing ones): `cd backend; uv run pytest -m gpu`.

Database migrations live in `backend/alembic/versions/` and run automatically at startup.

**Re-recording the tutorials** (after UI changes): start the built app (`npm start`), then run
`uv run scripts/tutorials/make_tutorials.py` (or `--only story-to-shots` / `--only images-and-keyframes`).
The script drives Chrome through the real app, narrates with edge-tts (Microsoft's online neural voices; the
narration text is sent to that service), time-lapses LLM and GPU waits, and writes MP4 + captions + poster +
`index.json` to `frontend/public/tutorials/`. The story tutorial recreates the "The Lighthouse Gull"
project and calls your default LLM (a few cents). The images tutorial runs FLUX and FLUX Kontext jobs on that
project, so record the story tutorial first. `--only smoke --out <dir>` checks the pipeline without the LLM
or GPU.

```
backend/localflow/   FastAPI app (routers, services, models) and worker/ (GPU process + adapters)
frontend/src/        React app (api/, app/, features/, components/)
scripts/dev.mjs      dev runner
scripts/tutorials/   tutorial recorder (Playwright + edge-tts + ffmpeg)
```
