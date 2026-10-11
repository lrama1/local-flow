# LocalFlow

A locally-running, single-user AI filmmaking studio inspired by Google Flow. React frontend + Python (FastAPI)
backend; image/video/audio generation runs on your GPU. See [SPEC.md](SPEC.md) for the full design and
[docs/DECISIONS.md](docs/DECISIONS.md) for implementation decisions.

**Status:** milestones **M0** through **M7** are done. New here? Start with [Quick start](#quick-start-windows)
and [Your first film](#your-first-film).

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
- **M6:** audio. Characters get **voices** (Kokoro's built-ins, or custom voices designed from a text
  description), shots get **dialogue** and **sound effects** made to fit the clip (MMAudio reads the take's
  picture), and scenes or the whole film get **music** (MusicGen). Everything lands on the timeline and in
  the export.
- **M7:** polish. The **Assets** tab is a full library (filters, search, collections, bulk actions, upload,
  reuse in the Playground, frame extraction). **Check consistency** has a vision LLM compare a keyframe or
  take with the references. **Settings → Diagnostics** runs health checks and downloads a support bundle.
  A Playwright smoke test covers idea → export.

## Quick start (Windows)

1. **Install the tools** in PowerShell:

   ```powershell
   winget install astral-sh.uv Gyan.FFmpeg OpenJS.NodeJS.LTS Git.Git
   ```

   Then **close and reopen the terminal** (and VS Code, if you use its terminal) so it picks up the new
   `PATH`. Check with `uv --version; node --version; ffmpeg -version`.
2. **GPU driver** (for real generation): an NVIDIA GPU with 16 GB+ VRAM and a driver new enough for CUDA 13,
   meaning the header of `nvidia-smi` says `CUDA Version: 13.0` or higher. Update it from nvidia.com or the
   NVIDIA app if not. No NVIDIA GPU? Everything still runs and makes placeholder images and videos (the
   FakeEngine).
3. **Clone and install:**

   ```powershell
   git clone https://github.com/lrama1/local-flow.git
   cd local-flow
   npm run setup
   ```

   This runs `uv sync` in `backend/` (uv fetches Python 3.12 and the packages, including the CUDA 13 build of
   PyTorch from download.pytorch.org: about 4 GB) and `npm install` in `frontend/` (about 250 MB). Without an
   NVIDIA GPU, skip the GPU packages: `cd backend; uv sync --no-group gpu; cd ..\frontend; npm install`.
4. **Check that PyTorch sees the GPU:**

   ```powershell
   cd backend; uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; cd ..
   ```

   Expect something like `2.14.1+cu130 True`.
5. **Choose where data goes** (optional). Models need 20–80 GB, and everything lives in
   `%USERPROFILE%\LocalFlow` unless you set another folder before the first start, e.g.
   `setx LOCALFLOW_DATA_DIR D:\LocalFlow`.
6. **Add an LLM key** for the story tools, e.g. `setx OPENAI_API_KEY "sk-..."` (or `ANTHROPIC_API_KEY`,
   `OPENROUTER_API_KEY`), or paste one later in **Settings → API keys**. See [API keys](#api-keys).
7. **Start:**

   ```powershell
   npm start            # builds the frontend, then serves everything on http://127.0.0.1:8765
   ```

   Open <http://127.0.0.1:8765>. For development, `npm run dev` runs the backend on :8765 plus Vite with hot
   reload on <http://127.0.0.1:5173> (the backend restarts on `.py` changes).
8. **Open Settings → Diagnostics.** Every check should be green: data folder writable, disk space, ffmpeg,
   NVIDIA GPU, PyTorch + CUDA, GPU worker, secret storage (Windows Credential Manager) and LLM providers.
   Amber or red rows say what to fix.

## Your first film

1. **Models → Catalog:** install what you need. Shared parts (text encoders, VAEs) download once, so the
   total is less than the sum.

   | Model | Used for | Download |
   |---|---|---|
   | FLUX.1 [dev] · GGUF Q4_K_S | ingredient images, keyframes from text | 16.7 GB |
   | FLUX.1 Kontext [dev] · GGUF Q4_K_M | keyframes composed from references, character views | 16.9 GB |
   | Wan 2.2 TI2V-5B · GGUF Q8_0 | video (5 s clips at 1280×704) | 19.6 GB |
   | Kokoro-82M | dialogue | 0.4 GB |
   | MMAudio large 44.1 kHz v2 | sound effects synced to the picture | 10.7 GB |
   | MusicGen small | music | 2.4 GB |

   FLUX.1 [dev] is gated: accept its license on Hugging Face and add an `HF_TOKEN` (Settings → API keys can
   import an existing `hf auth login`). Kokoro and MMAudio need `allow_pickle_weights = true` in
   `config.toml` ([Audio](#audio-m6)). Acknowledge each model's license on its card before first use.
2. **Projects → New project.** On **Story**, write a few sentences, then **Refine** → **Build bible** →
   **Break down into shots**. A 30-second story costs about $0.02–0.03 with `gpt-5.4-mini`.
3. **Ingredients:** generate images for the characters, locations and props and pick a primary for each, or
   use **New character** for the wizard with consistent extra views.
4. **Shots board:** open a shot, **Compose keyframe**, pick one, then **Generate video**. Or use
   **Generate all** for every shot. **Check consistency** compares a keyframe or take with the references.
5. **Audio** (shot panel): dialogue and sound effects. Music is generated from the timeline.
6. **Timeline → Auto-assemble**, adjust trims, order and transitions, then **Export MP4**. Exports also
   appear on the **Assets** tab.

The **Tutorials** page (sidebar) has narrated walkthroughs of steps 2–4.

## Troubleshooting

- **`'uv' is not recognized`** (or `node`, `ffmpeg`): the terminal is older than the install. Fully quit and
  reopen VS Code or the terminal (a new VS Code terminal tab inherits VS Code's old environment), or refresh
  the current PowerShell session:
  `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`
- **Diagnostics shows "PyTorch + CUDA" in red, or outputs are placeholders:** update the NVIDIA driver until
  `nvidia-smi` reports CUDA 13.0+, then rerun `cd backend; uv sync`. If `torch.__version__` ends in `+cpu`,
  the CUDA build wasn't installed: `uv sync --reinstall-package torch`. The GPU worker check shows why the
  FakeEngine was chosen.
- **ffmpeg not found:** `winget install Gyan.FFmpeg` and reopen the terminal, or set `ffmpeg_path` and
  `ffprobe_path` in `config.toml`.
- **Out of GPU memory:** close other GPU apps and set `vram_profile = "low"` in `config.toml` (video drops to
  832×480). For video, fewer steps also help.
- **A gated model won't download:** accept the license on its Hugging Face page, then press **Check again**
  on the card.
- **Port 8765 is in use:** `$env:LOCALFLOW_PORT = 9000; npm start`.
- **Kokoro or MMAudio won't install:** set `allow_pickle_weights = true` in `config.toml` and restart.
- **Reporting a bug:** **Settings → Diagnostics → Support bundle** downloads a zip with the checks,
  versions, `config.toml` and recent logs. Keys and tokens are redacted, and no media or database is
  included. The logs (`localflow.log`, `worker.log` in `<data dir>\logs\`) can also be read in Diagnostics.

## Linux

Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`), Node.js 20+, ffmpeg (e.g.
`sudo apt install ffmpeg`) and an NVIDIA driver with CUDA 13 support, then follow the same steps from
`npm run setup`. Data goes to `~/LocalFlow`. Keys come from your shell environment, or Settings stores them
in the Secret Service keyring (GNOME Keyring, KWallet). Without a keyring, they go to an encrypted file in
the data dir; Diagnostics shows which.

## Draft mode

Three optional **(draft)** models in **Models → Catalog** trade some quality for speed (about 6 GB extra,
because they share files with the models above):

| Draft model | Replaces | Steps | Time on an RTX 5080 |
|---|---|---|---|
| FLUX.1 [dev] · Q4_K_S + Turbo-Alpha | FLUX.1 [dev] | 8 instead of 28 | image 17 s instead of 42 s |
| FLUX.1 Kontext [dev] · Q4_K_M + Turbo-Alpha | FLUX.1 Kontext | 8 instead of 28 | keyframe 29 s instead of 80 s |
| Wan 2.2 TI2V-5B Turbo · Q8_0 | Wan 2.2 TI2V-5B | 4 instead of 30 | 3 s clip 52 s instead of 295 s |

Use the **Final / Draft** switch at the top of a project. In Draft mode, anything you don't pick a model for
(Generate all, keyframes, character views, new images and videos) uses the installed draft models, then falls
back to the normal ones. A model you choose in a picker is always used. The switch shows which draft models
are still missing. Switch back to **Final** to regenerate the keepers.

The FLUX drafts look nearly the same as the full models. The video draft stays faithful to the keyframe but
is a little softer. Turbo-Alpha uses the FLUX.1 [dev] non-commercial license. The Wan Turbo GGUF is a
community repack (Apache 2.0); the original distillation states no license.

## Library, review and diagnostics (M7)

- **Assets tab (the project's library):**
  - Filter by type, source (generated, uploaded, extracted frames, exports), model, tag, scene and ingredient,
    and search prompts, dialogue lines and file names. Sort newest or oldest first.
  - **Collections** (left) group assets, e.g. "Storyboard" or "Hero looks"; **Favorites** has starred ones.
  - Tick the box on a tile (it appears on hover) to select several. Then add them to a collection, tag,
    favorite or delete them. While selecting, clicking a tile adds it to the selection.
  - **Upload** (or drop files on the library) adds images, videos and audio. Videos that aren't H.264 MP4
    (MOV, WebM, VP9…) are converted (odd sizes lose a pixel row or column), and MP3/FLAC/OGG become WAV.
- **Asset details** (click a tile):
  - Preview, size, model, seed, the full generation settings, tags and collections, and the assets it was
    made from.
  - **Reuse settings** opens the Playground with the same kind, prompt, seed, size, model, LoRAs and inputs,
    even if they're in another project. Playground outputs keep their @mentions; a project's shots and
    ingredients reuse the full prompt they were generated with.
  - **Use as keyframe** (pick a shot, first or last frame) and **Use as ingredient reference** (the primary
    image, or a character view). For a video, both use the frame under the playhead.
  - Videos also have **Extract frame** (a new image at the playhead, on the frame grid) and **Extend**: a
    shot's take is extended on its shot, and other videos open in Playground → Extend.
- **Check consistency** (shot panel, under the keyframe and the take): a vision-capable LLM compares the
  image (or two frames of the take) with the shot's reference images. It lists concrete mismatches and
  proposes a revised prompt; **Apply fix** puts it in the panel for you to save or regenerate. It needs a
  vision model for the `vision_review` purpose (gpt-5.4-mini works) and costs about $0.003–0.005 a check.
- **Settings → Diagnostics:** checks with fixes, versions (torch, CUDA, cuDNN, diffusers…), disk use per
  data folder, log viewer and the redacted **Support bundle**.

## Audio (M6)

- **Install:** set `allow_pickle_weights = true` in `config.toml` (Kokoro and MMAudio only publish PyTorch
  pickles; LocalFlow loads them with `torch.load(weights_only=True)`), then install **Kokoro-82M**
  (0.4 GB), **MMAudio large 44.1 kHz v2** (10.7 GB including its CLIP and vocoder) and **MusicGen small**
  (2.4 GB) from **Models → Catalog**.
- **Voices** (**Cast library → Voices**):
  - Kokoro has 41 built-in voices in English (US/UK), Spanish, French, Hindi, Italian and Portuguese.
    Press ▶ to hear a sample; samples are cached.
  - **New voice** makes a custom voice: a base voice blended with up to two others at your pace. Describe
    it ("a weathered old fisherman; gravelly, slow and kind") and **Design with AI** picks the settings.
    **Audition** before saving.
  - Give a character its voice in the character wizard's **Voice** step or on its card in **Ingredients**.
    Without one, a voice is suggested from the character's voice hints.
- **Shot audio** (shot panel → **Audio**):
  - The plan shows who says what, with which voice, and whether the dialogue fits the clip.
  - **Generate dialogue** speaks all lines into one track as long as the clip. Lines are sped up to at
    most 1.15× to fit (or keep the natural pace); a warning shows if they still run long, so extend the
    shot or shorten them.
  - **Generate sound effects** makes audio synchronized to the selected take from the shot's SFX prompt.
  - The newest take of each is used, and **Use** switches back to an older one. If the timeline already
    has the shot, its Dialogue and SFX clips update in place.
- **Music:** on the timeline, press **+** on Music → **Generate music** for the whole film or one scene.
  It uses the shots' music cues unless you write a prompt (MusicGen makes up to 30 s) and is placed on the
  Music track when done.
- **Playground → Speech** has a voice picker too.
- **Speed:** Kokoro takes about 2 s per shot (12 s for the first job). MMAudio takes about 30 s to load,
  then about 3 s per 5 s take. Takes over 10 s are made in overlapping 8 s windows.

## Narration (M8)

A film-level voice-over. The words are written and edited per shot, where you can see the picture; the voice
is made on the Timeline, as one continuous track.

- **Write it:** **Timeline tab → Narration → Write narration.** The LLM reads the story and the shots in
  order and writes one line per shot (or leaves a shot silent), sized to the shot (about 2.3 words per
  second). The optional **Style** box sets perspective and tone ("a hushed storybook"). A whole film costs
  a fraction of a cent. Or type the lines yourself in the panel or in a shot's **Narration** field.
  **Rewrite existing lines** replaces lines that already exist. The step has its own model picker in
  Settings → LLM ("Narration").
- **Choose the narrator** (one voice per project, from the Voices library) and the **Pace**. Without a
  voice, the panel uses a suggested one and says so.
- **Voice narration** voices every line that's missing or out of date (it shows how many), one Kokoro take
  per shot, about 2 s each. **Voice** or **Re-voice** on a row redoes one line; older takes are under the
  shot's audio candidates (**Use** switches back).
- **On the timeline:** each line starts a quarter second into its shot's clip, on the **Narration** track
  (violet). It isn't squeezed to the clip: it may run into the next shot, and a **+N s over** badge warns
  if it would run into the next line (shorten the line, speed it up or lengthen the shot). Editing a line
  marks its take **out of date** until you voice it again.
- **It follows the picture:** reorder or trim clips and the narration moves with its shots. A line you
  dragged somewhere stays where you put it; **Re-sync** puts every line back on its shot.
- **Export:** **Lower music under narration** (on by default) dips the music, the sound effects and the
  clips' own sound by about 9–11 dB while the narrator speaks, then brings them back. The preview doesn't
  duck; the export does. Dialogue is never ducked.

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
- **Audio:** use **+** on Narration, Dialogue, Sound effects or Music to add a project audio file at the
  playhead, or upload one. Drag audio freely; the panel sets start, trim, gain and track. The speaker
  buttons mute a track. On Video, that mutes the clips' own sound.
- **Preview:** Space plays and pauses, ←/→ step a frame (Shift: a second), Home/End jump, and Delete
  removes the selected clip. The preview is approximate; the export is exact.
- **Export:**
  - Pick **720p** or **1080p** (in the project's aspect ratio), optionally **Normalize loudness**, and
    leave **Lower music under narration** on (see [Narration](#narration-m8)).
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
  and a 480p clip about 2.5 minutes. Pick fewer **steps** in the Video section, or use [Draft mode](#draft-mode),
  for quick drafts. Peak VRAM is about 10 GB.
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

## Data and configuration

Everything lives in the data directory: `%USERPROFILE%\LocalFlow` on Windows, `~/LocalFlow` elsewhere
(override with `LOCALFLOW_DATA_DIR`, e.g. to put the models on a bigger drive; on Windows a user variable
set with `setx` is honored even in terminals opened before it). It holds `localflow.db`, `config.toml`
(created with commented defaults on first run), `projects/`, `models/`, `library/`, `logs/`, and `tmp/`.

Every `config.toml` key can be overridden with an env var `LOCALFLOW_<KEY>`, e.g. `LOCALFLOW_PORT=9000`.
The server only binds to `127.0.0.1`, and every API/WebSocket call needs the per-launch token.
`engine = "auto"` (the default) uses the GPU when it's usable and the FakeEngine otherwise; set
`engine = "fake"` to force placeholders. `fake_llm = true` answers every LLM call with a canned demo story
(no key or network needed; for offline demos and the smoke test).

## Develop

```powershell
npm test             # backend pytest (spawns the FakeEngine worker) + frontend vitest
npm run test:e2e     # Playwright smoke test: build, then idea → story → generate all → timeline → export
npm run lint         # ruff, ruff format --check, mypy, eslint, prettier, tsc
npm run typegen      # regenerate frontend API types from the backend OpenAPI schema
```

The smoke test (`frontend/e2e/`) starts its own server on port 8799 with a fresh data dir in the temp
folder, the FakeEngine and `fake_llm`, so it needs no GPU, keys or network, and never touches your data or
keyring. It takes about 15 s. Run `cd frontend; npx playwright install chromium` once first. A failed run
keeps a trace and video in `frontend/test-results/` (`npx playwright show-report`).

Real-GPU regression test (Flux → Kontext → SDXL, then unload to baseline; about 2 minutes, uses the models
installed in your data dir and skips missing ones): `cd backend; uv run pytest -m gpu`.

Database migrations live in `backend/alembic/versions/` and run automatically at startup.

**Re-recording the tutorials** (after UI changes): start the built app (`npm start`), then run
`uv run scripts/tutorials/make_tutorials.py` (or `--only full-film` / `--only story-to-shots` /
`--only images-and-keyframes`).
The script drives Chrome through the real app, narrates with edge-tts (Microsoft's online neural voices; the
narration text is sent to that service), time-lapses LLM and GPU waits, and writes MP4 + captions + poster +
`index.json` to `frontend/public/tutorials/`. The story tutorial recreates the "The Lighthouse Gull"
project and calls your default LLM (a few cents). The images tutorial runs FLUX and FLUX Kontext jobs on that
project, so record the story tutorial first. The full-film tutorial makes "The Last Rose" from scratch and
needs FLUX, FLUX Kontext, Wan 2.2 5B, Kokoro, MMAudio and MusicGen installed; it takes about an hour of GPU
time. `--only smoke --out <dir>` checks the pipeline without the LLM
or GPU.

```
backend/localflow/   FastAPI app (routers, services, models) and worker/ (GPU process + adapters)
frontend/src/        React app (api/, app/, features/, components/)
scripts/dev.mjs      dev runner
scripts/tutorials/   tutorial recorder (Playwright + edge-tts + ffmpeg)
```
