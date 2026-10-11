# LocalFlow — Implementation Specification

> A locally-running, single-user AI filmmaking studio inspired by Google Flow.
> React frontend + Python (FastAPI) backend. Image/video/audio generation runs locally
> via Hugging Face `diffusers` (and a few standalone audio libraries) using models from
> Hugging Face and Civitai. Story refinement and scene breakdown use cloud LLMs
> (OpenAI, Anthropic, OpenRouter) via user-supplied API keys.

---

## 0. Instructions for the implementing agent

1. **Work milestone by milestone** (Section 20). Do not start a milestone until the previous one's acceptance criteria pass.
2. **Verify before hardcoding.** Open-model names, `diffusers` pipeline classes and arguments, and Civitai API fields change frequently. Before writing an adapter, check the current Hugging Face model card and the installed `diffusers` version's documentation and source. If a model listed here is unavailable or unsupported, choose the closest working alternative, record the decision in `docs/DECISIONS.md`, and continue.
3. **Keep the GPU layer behind interfaces.** All model-specific code lives in adapters (Section 10). The rest of the app must not import `torch` or `diffusers`.
4. **Everything must be testable without a GPU.** Provide a `FakeEngine` adapter and a `FakeLLMProvider` so the API, job queue, and UI can be developed and tested on any machine.
5. **No secrets in code, logs, or the frontend.**
6. Prefer boring, well-maintained libraries. Don't add dependencies that aren't justified here without recording why in `docs/DECISIONS.md`.

---

## 1. Product overview

### 1.1 Goal
Let a user turn a rough story idea into a short film:

1. Write or paste a story idea.
2. An LLM refines it, extracts a **story bible** (characters, locations, props, style), and breaks it into **scenes → shots** sized to what the local video model can render.
3. The user generates and approves **ingredients**: reference images for each character, location, and prop. Characters can also be built in the **character creator** or cast from a cross-project **Cast library** (Section 11.7).
4. Each shot is generated as a **keyframe image** (using ingredients), then animated to **video** (image-to-video, or first+last-frame-to-video).
5. Shots can be **extended**, regenerated, or edited. Audio (native, or dialogue/SFX/music via fallback) is added.
6. Shots are arranged on a **timeline** (SceneBuilder) and **exported** to MP4.

### 1.2 Feature parity map (Google Flow → LocalFlow)

| Google Flow | LocalFlow |
|---|---|
| Text-to-video (Veo) | Text-to-video via local video model adapter |
| Frames-to-video | First/last-frame-to-video (FLF2V) adapter |
| Ingredients-to-video | Ingredient reference images → keyframe composition → image-to-video |
| Characters (templates, from scratch, upload/import, extra views, voice, character info) | Character creator (Section 11.7) |
| Casting characters in prompts | `@key` mentions in every prompt editor (Section 11.7.3) |
| Reusing characters across projects | Cast library (Section 11.7.2) |
| Custom voices | Preset + description-designed custom voices (Section 12.2) |
| Avatars ("Me") | Not in v1 (Section 1.3) |
| Imagen image generation | Local text-to-image / image-edit adapters (Flux, SDXL, Civitai checkpoints + LoRAs) |
| Extend | Last-frame continuation + concatenation |
| SceneBuilder | Timeline editor + FFmpeg export |
| Camera controls | Camera presets → prompt fragments and optional camera/motion LoRAs |
| Native audio | Native-audio video adapter when available; otherwise TTS + video-to-audio SFX + music, mixed with FFmpeg |
| Flow agent / Gemini planning | Cloud LLM story pipeline (OpenAI / Anthropic / OpenRouter) |
| Asset management / collections | Project asset library with tags and collections |

### 1.3 Non-goals (v1)
- Multi-user, authentication, remote access, cloud deployment.
- Training LoRAs.
- Real-time preview while generating, beyond step-progress thumbnails.
- Mobile layout.
- Personal avatars ("Me": capturing the user's own face and voice) and voice cloning from uploaded audio.
- Commercial-use license compliance tooling. The app is for **personal, non-commercial use**, but every model's license is still displayed (Section 9.5).

---

## 2. Target environment & constraints

| Item | Value |
|---|---|
| OS | Windows 11 (must also run on Linux; avoid Windows-only APIs except optional keyring backend) |
| GPU | NVIDIA RTX 5080, **16 GB VRAM**, Blackwell (sm_120) → requires PyTorch built with **CUDA 12.8+** |
| System RAM | **~23 GB** (tight; CPU offload of 14B models in bf16 will not fit — use quantized weights) |
| Python | **3.12** in a dedicated venv managed by `uv` (do not rely on system Python 3.14; ML wheels may lag) |
| Node | 20 LTS or newer |
| External binaries | `ffmpeg` and `ffprobe` on PATH (or configured path) |
| Network binding | Backend listens on `127.0.0.1` only |

### 2.1 Hardware implications (design requirements)
- **One heavy model resident at a time.** The model manager must unload (move to `meta`, delete, `gc.collect()`, `torch.cuda.empty_cache()`) before loading a different heavy model.
- **Quantization is mandatory for 14B-class video models.** Support GGUF (via `diffusers` `GGUFQuantizationConfig`) and/or FP8/NF4 (via `torchao` / `bitsandbytes`, whichever the installed `diffusers` supports) per adapter.
- **Offloading strategies** are configurable per model: `none`, `model_cpu_offload`, `sequential_cpu_offload`, `group_offload` (if supported by installed `diffusers`). Also enable VAE tiling and slicing where available.
- Provide **VRAM profiles** (`low` / `balanced` / `quality`) that set resolution, frame count, quantization, and offload defaults per adapter.
- Expect generation times measured in **minutes**. The UI is queue-centric, not interactive-instant.

---

## 3. Tech stack

### 3.1 Backend
- **FastAPI** + **Uvicorn** (HTTP + WebSocket).
- **Pydantic v2** for all schemas; `pydantic-settings` for config.
- **SQLModel** (or SQLAlchemy 2.x) + **SQLite** (WAL mode) for persistence; **Alembic** for migrations.
- **diffusers**, **transformers**, **accelerate**, **safetensors**, **huggingface_hub**, **gguf**, and **torchao** and/or **bitsandbytes** (whichever the chosen quantization path needs).
- **torch** (CUDA 12.8+ build).
- **LiteLLM** for a unified OpenAI / Anthropic / OpenRouter interface (fallback: thin adapters over the official `openai` and `anthropic` SDKs, using `openai` with `base_url` for OpenRouter).
- **keyring** for secret storage (Windows Credential Manager on Windows).
- **httpx** for Civitai API and downloads.
- **Pillow**, **imageio[ffmpeg]** or direct `ffmpeg` subprocess for media I/O.
- Audio: a TTS library (default **Kokoro**; alternatives: Chatterbox, F5-TTS), **MMAudio** for video-to-audio SFX (standalone repo/package; not part of diffusers), optional music model (e.g., ACE-Step or MusicGen via `transformers`). Verify install paths on Windows; isolate each behind an adapter.
- **structlog** or stdlib `logging` with JSON formatter.
- **pytest**, **pytest-asyncio**, **httpx** test client, **ruff**, **mypy** (or pyright).

### 3.2 Frontend
- **Vite** + **React 18+** + **TypeScript** (strict).
- **TanStack Query** for server state; **Zustand** for local UI state.
- **React Router**.
- **Tailwind CSS** + **shadcn/ui** (Radix) components.
- **dnd-kit** for drag-and-drop (shot board and timeline).
- Native `<video>` and `<audio>` elements for playback; the timeline is a custom component (Section 15).
- API client generated from the backend OpenAPI schema (**openapi-typescript** + `openapi-fetch`) so types stay in sync.
- **Vitest** + **React Testing Library**; **Playwright** for E2E smoke tests.
- **ESLint** + **Prettier**.

### 3.3 Dev tooling
- Monorepo with `backend/` and `frontend/`.
- `uv` for Python and `npm` (or `pnpm`) for JS.
- Root scripts: `dev` (runs backend + frontend), `test`, `lint`, `typegen` (regenerates TS types from OpenAPI).
- In production mode, FastAPI serves the built frontend from `frontend/dist`, so the user runs a single command.

---

## 4. Architecture

```
┌──────────────────────────── Browser (React) ────────────────────────────┐
│ Projects │ Story │ Ingredients │ Shots board │ Timeline │ Models │ Settings│
└───────▲──────────────────────────────▲──────────────────────────────────┘
        │ REST (JSON, multipart)        │ WebSocket (/ws/events)
┌───────┴───────────────────────────────┴──────────────────────────────────┐
│ API process (FastAPI, async)                                             │
│  ├─ routers: projects, story, ingredients, shots, timeline, assets,      │
│  │           jobs, models, settings, files                               │
│  ├─ services: StoryService (LLM), ModelRegistry, Downloader,             │
│  │            TimelineService, ExportService (FFmpeg), SecretStore       │
│  ├─ JobQueue (SQLite-backed)  ──── submit/cancel ────┐                   │
│  └─ EventBus (in-proc pub/sub → WebSocket fan-out) ◄─┼─ progress events  │
└──────────────────────────────────────────────────────┼───────────────────┘
                                                       │ multiprocessing Queues
┌──────────────────────────────────────────────────────┴───────────────────┐
│ GPU Worker process (single, spawned by API process on startup)           │
│  ├─ Engine: ModelCache (one heavy model resident), VRAM profiles         │
│  ├─ Adapters: T2I, ImageEdit, T2V, I2V, FLF2V, Ref2V, TTS, V2A, Music    │
│  └─ Writes outputs to data/projects/<id>/... and reports progress        │
└──────────────────────────────────────────────────────────────────────────┘
          │ downloads                              │ LLM calls (from API process)
   Hugging Face Hub / Civitai API          OpenAI / Anthropic / OpenRouter
```

### 4.1 Process model
- **API process:** async FastAPI. Never imports torch. Owns the DB, LLM calls, downloads, FFmpeg exports (run in a thread pool or subprocess), and WebSocket fan-out.
- **GPU worker process:** a single long-lived `multiprocessing` process started with the `spawn` context. It receives `JobRequest` messages over a `multiprocessing.Queue` and sends `JobEvent` messages back (`started`, `progress`, `preview`, `completed`, `failed`, `cancelled`). It processes one GPU job at a time.
- **Crash isolation:** if the worker dies (CUDA OOM, segfault), the API marks the running job `failed` with the reason, restarts the worker, and continues with the queue.
- **Cancellation:** the API sets a shared `multiprocessing.Event` per running job. Adapters check it inside the diffusers `callback_on_step_end` and raise `JobCancelled`.
- **Downloads** run in the API process (asyncio + httpx / `huggingface_hub`), not the GPU worker, so the GPU stays busy.

### 4.2 Data directory layout
Configurable root (default `%USERPROFILE%\LocalFlow` on Windows, `~/LocalFlow` elsewhere):

```
LocalFlow/
  localflow.db
  config.toml
  models/
    hf/                     # HF_HOME / hub cache
    civitai/<modelId>/<versionId>/<file>.safetensors
    loras/
  projects/<project_id>/
    assets/images/<asset_id>.png
    assets/videos/<asset_id>.mp4
    assets/audio/<asset_id>.wav
    thumbs/<asset_id>.jpg
    exports/<export_id>.mp4
  library/
    characters/<character_id>/<n>.png    # Cast library images (Section 11.7.2)
    voices/<voice_id>/preview.wav
  logs/
  tmp/
```

---

## 5. Repository structure

```
local-flow/
  SPEC.md
  README.md
  docs/DECISIONS.md
  scripts/                    # dev.ps1, dev.sh, setup scripts
  backend/
    pyproject.toml
    alembic/
    localflow/
      main.py                 # FastAPI app factory, lifespan (starts worker)
      config.py
      db.py
      models/                 # SQLModel tables
      schemas/                # Pydantic request/response + LLM output schemas
      routers/
      services/
        llm/                  # provider abstraction, prompts/, story pipeline
        registry/             # model registry, HF + Civitai clients, downloader
        jobs/                 # queue, event bus, worker supervisor
        media/                # ffmpeg helpers, thumbnails, frame extraction
        timeline/             # timeline model + export
        characters/           # templates.yaml, mentions.py, library.py, reference selection
        secrets.py
      worker/
        main.py               # worker loop
        engine.py             # ModelCache, memory mgmt, VRAM profiles
        adapters/
          base.py             # capability interfaces
          fake.py
          image/  flux.py  sdxl.py  image_edit.py
          video/  wan.py  ltx.py  hunyuan.py
          audio/  tts_kokoro.py  v2a_mmaudio.py  music.py
    tests/
  frontend/
    package.json
    src/
      api/                    # generated types + client
      app/                    # routes, layout
      features/
        projects/ story/ ingredients/ characters/ cast-library/ voices/ shots/ timeline/ models/ settings/ jobs/
      components/             # shared UI
      lib/
    tests/
```

---

## 6. Domain model (SQLite)

All IDs are ULIDs (strings). All tables have `created_at` and `updated_at`. JSON columns are stored as TEXT and validated through Pydantic.

### 6.1 Tables

**projects**
`id, name, description, aspect_ratio ("16:9"|"9:16"|"1:1"), fps (int, default 24), default_video_model_id, default_image_model_id, style_notes (text), archived (bool)`

**stories** (one active per project; keep revisions)
`id, project_id, revision (int), raw_idea (text), refined_story (text), bible_json (StoryBible), llm_provider, llm_model, status ("draft"|"refined"|"broken_down")`

**ingredients**
`id, project_id, kind ("character"|"location"|"prop"|"style"), key (slug, unique per project, immutable once saved), name, description, visual_prompt, negative_prompt, voice_id (nullable → voices.id), primary_asset_id (nullable), lora_ids_json (list), locked (bool), origin ("bible"|"user"|"library"|"import"), character_info (text, nullable; characters only), body_outfit_prompt (text, nullable; characters only), template_key (nullable), library_character_id (nullable), library_synced_at (nullable)`

**ingredient_assets** (many reference images per ingredient)
`ingredient_id, asset_id, is_primary, view ("primary"|"full_body"|"three_view"|"side"|"back"|"face_closeup"|"other")`

**character_templates** (user-saved templates; built-ins come from `templates.yaml`, Section 11.7.1)
`id, key, name, category, visual_prompt, negative_prompt, character_info, voice_hints_json, thumbnail_path (nullable)`

**library_characters** (global Cast library, Section 11.7.2)
`id, key, name, description, visual_prompt, negative_prompt, character_info, body_outfit_prompt, voice_id (nullable), lora_ids_json, template_key (nullable), images_json (list of {file, view, is_primary}; files relative to library/characters/<id>/), source_project_id (nullable)`

**voices** (global, Section 12.2)
`id, kind ("builtin"|"custom"), adapter_key, native_id (builtin: the adapter's voice name, e.g. "af_heart"), base_voice_id (custom: a builtin voices.id), name, description, language, gender, params_json, meta_json (builtin: quality grade, traits), preview_path (nullable; library/voices/<id>/preview.wav), preview_text (nullable)`; unique `(adapter_key, kind, native_id)`

**scenes**
`id, project_id, story_id, index, title, summary, location_ingredient_id (nullable)`

**shots**
`id, scene_id, index, duration_s (float), shot_type ("wide"|"medium"|"close"|"extreme_close"|"insert"|"pov"|"over_shoulder"), camera_move (enum, Section 13.5), video_prompt, negative_prompt, keyframe_prompt, end_keyframe_prompt (nullable), generation_mode ("t2v"|"i2v"|"flf2v"|"ref2v"), ingredient_ids_json, dialogue_json (list of {character_ingredient_id, line, emotion}), sfx_prompt, music_cue, status ("planned"|"keyframe_ready"|"video_ready"|"approved"), selected_keyframe_asset_id, selected_end_keyframe_asset_id, selected_video_asset_id, selected_audio_asset_ids_json`

**assets**
`id, project_id, type ("image"|"video"|"audio"), path (relative to project dir), thumb_path, width, height, duration_s, fps, has_audio (bool), source ("generated"|"uploaded"|"extracted"|"export"|"imported"), parent_asset_ids_json, generation_json (full params incl. model id, seed, prompts, adapter version), tags_json, favorite (bool)`

**collections**, **collection_assets**
User-defined groupings of assets: `collections (id, project_id, name (unique per project), description)` and `collection_assets (collection_id, asset_id)`, a unique pair. An asset can be in several collections; deleting either side removes the membership.

**timelines** (one per project in v1)
`id, project_id, tracks_json (Timeline, Section 15.1)`

**jobs**
`id, project_id (nullable), type (Section 14.1), status ("queued"|"running"|"completed"|"failed"|"cancelled"), priority (int), params_json, result_json, error (text), progress (0..1), progress_message, started_at, finished_at, target_ref_json (e.g., {"shot_id": "...", "slot": "keyframe"})`

**model_entries** (registry; Section 9)
`id, source ("hf"|"civitai"|"local"), source_ref (repo id or civitai modelId:versionId), name, kind ("checkpoint"|"lora"|"vae"|"text_encoder"|"audio"), capabilities_json, adapter_key, base_model (e.g., "flux1", "sdxl", "wan2.2-5b"), license, license_url, nsfw (bool), files_json, local_path, size_bytes, status ("available"|"downloading"|"installed"|"error"), default_params_json, vram_profiles_json`

**llm_usage**
`id, provider, model, purpose, prompt_tokens, completion_tokens, cost_usd (nullable), created_at`

**settings** (key/value, non-secret only)

### 6.2 Invariants
- Deleting a project deletes its directory (after confirmation in the UI).
- Assets are immutable files; edits create new assets with `parent_asset_ids`.
- `generation_json` must contain everything needed to reproduce the asset (model entry id, adapter key, adapter version, all params, seed). For prompts with `@key` mentions, it stores both the raw and the resolved prompt.
- Library characters and voices are not deleted with a project. Project ingredients cast from the library are independent copies.

---

## 7. Configuration & secrets

### 7.1 Config (`config.toml` + env overrides, prefix `LOCALFLOW_`)
- `data_dir`, `ffmpeg_path`, `ffprobe_path`, `host` (fixed default `127.0.0.1`), `port` (default `8765`)
- `vram_profile` (`low`|`balanced`|`quality`, default `balanced`)
- `default_llm_provider`, `default_llm_model` (per purpose overrides: `refine`, `bible`, `breakdown`, `prompt_rewrite`, `vision_review`)
- `hf_home` (defaults to `<data_dir>/models/hf`)
- `allow_nsfw_models` (default `false`; hides NSFW Civitai results)
- `allow_pickle_weights` (default `false`; see Section 18)
- `fake_llm` (default `false`; canned LLM responses for demos and the E2E test, Section 19)

### 7.2 Secrets
Secrets: `openai_api_key`, `anthropic_api_key`, `openrouter_api_key`, `hf_token`, `civitai_api_key`.
- **Resolution order: environment variable first, then the keyring.** Because the app runs locally, keys already in the user's environment are used directly, without being copied anywhere:

  | Secret | Environment variables (first match wins) |
  |---|---|
  | `openai_api_key` | `OPENAI_API_KEY` |
  | `anthropic_api_key` | `ANTHROPIC_API_KEY` |
  | `openrouter_api_key` | `OPENROUTER_API_KEY` |
  | `hf_token` | `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` |
  | `civitai_api_key` | `CIVITAI_API_TOKEN`, `CIVITAI_API_KEY` |

  On Windows, if a variable isn't in the process environment, also read the user and system environment from the registry (`HKCU\Environment`, then `HKLM\...\Session Manager\Environment`). This picks up variables set after the launching terminal or IDE started.
- **Keyring fallback:** keys pasted in Settings are stored with **`keyring`** under service name `LocalFlow`. If no keyring backend exists, use a Fernet-encrypted file in `data_dir`, keyed by a machine-local key file with restricted permissions, and log a warning. An environment variable always overrides a stored key; Settings says so when both exist.
- The API **never returns secret values**. Settings endpoints return `{name, configured: bool, source: "env"|"keyring"|"file"|null, env_var, stored: bool, last4, validated_at, ok, account, warnings}`.
- `POST /api/settings/secrets/{name}/test` validates a key with a cheap call (e.g., list models).
- **At least one LLM key is required.** Until one is configured and validated, the Story features show a blocking setup prompt. Image/video features still work.

### 7.3 Model hub accounts (Hugging Face, Civitai)
Access to gated or login-required models uses **personal access tokens**, not browser OAuth (Section 22). Tokens come from the environment or Settings, per Section 7.2.
- **Token resolution order:** environment variable (`HF_TOKEN`, `CIVITAI_API_TOKEN`; read-only, shown as "from environment") → keyring → none.
- **Import existing Hugging Face login:** because `HF_HOME` points into `data_dir` (Section 7.1), a token saved earlier with `hf auth login` / `huggingface-cli login` is not picked up automatically. Settings offers "Import existing Hugging Face login", which reads the default token file (`$HF_TOKEN_PATH`, else `~/.cache/huggingface/token`) and copies it into the keyring. Never write tokens into `HF_HOME`.
- Always pass the token explicitly to `huggingface_hub` calls (`token=`). Don't rely on implicit token discovery.
- **Account status:** the test endpoint returns the account, not just validity.
  - Hugging Face: `HfApi().whoami(token)` → username, avatar, token type/role. Warn if the token has write scope (recommend a read token). For fine-grained tokens, warn if they lack read access to public gated repos. Verify field names against the installed `huggingface_hub` (Section 0.2).
  - Civitai: a cheap authenticated call (verify the current endpoint) → username if available, else `configured + validated`.
  - Status shape: `{configured, source: "keyring"|"env", last4, validated_at, account: {username, avatar_url?} | null, token_type?, warnings: [str]}`.
- Settings shows "Signed in as <username>" per hub, the token source, and links to each hub's token page.

---

## 8. LLM layer & story pipeline

### 8.1 Provider abstraction
```python
class LLMProvider(Protocol):
    name: Literal["openai", "anthropic", "openrouter"]
    async def list_models(self) -> list[LLMModelInfo]: ...
    async def complete_structured(
        self, *, model: str, system: str, messages: list[Msg],
        schema: type[BaseModel], temperature: float = 0.7,
        max_tokens: int | None = None, images: list[ImageInput] | None = None,
    ) -> StructuredResult[BaseModel]: ...   # parsed object + usage
    async def complete_text(...) -> TextResult: ...
    async def stream_text(...) -> AsyncIterator[str]: ...
```
- Implement with LiteLLM (model strings such as `openai/<model>`, `anthropic/<model>`, `openrouter/<vendor>/<model>`).
- **Structured output strategy:** use native JSON-schema / tool-calling structured outputs when the provider and model support them. Otherwise instruct "return only JSON matching this schema". Always validate with Pydantic. On validation failure, retry up to 2 times, sending the validation errors back to the model ("repair" turn). Then fail with a clear error.
- Record every call in `llm_usage`. Estimate cost from LiteLLM's pricing table when available.
- Timeouts: 120 s default. Retry on 429/5xx with exponential backoff (max 3 attempts).
- Don't hardcode model IDs in code. Default model per provider comes from config, and the UI model picker is populated from `list_models()`. Ship sensible defaults in `config.toml` and document them in the README.
- Provide a `FakeLLMProvider` returning deterministic fixtures for tests.

### 8.2 Model capability context
Every story-pipeline call includes a **render constraints block** generated from the project's selected video and image models:

```json
{
  "video_model": "wan2.2-ti2v-5b",
  "max_clip_seconds": 5,
  "fps": 24,
  "supported_modes": ["t2v", "i2v"],
  "supports_native_audio": false,
  "supports_first_last_frame": false,
  "prompt_style_guide": "Describe subject, action, setting, lighting, camera in 1-3 sentences. Avoid lists and on-screen text.",
  "aspect_ratio": "16:9"
}
```
The breakdown step must never produce a shot longer than `max_clip_seconds`. A longer beat becomes multiple shots, or one shot flagged `needs_extend: true` with `extend_segments`.

### 8.3 Pipeline steps (each is a separate endpoint and a separate user-reviewable step)

**Step 1 — Refine** (`purpose=refine`)
Input: `raw_idea`, optional `{genre, tone, target_length_seconds, audience, style_notes}`.
Output schema `RefinedStory`:
```json
{ "title": "str", "logline": "str", "synopsis": "str (<=200 words)",
  "story": "str (prose, beat-ordered)", "estimated_runtime_s": 60,
  "tone": "str", "visual_style": "str" }
```

**Step 2 — Story bible** (`purpose=bible`)
Input: refined story + visual style + `existing_cast` (Section 8.5).
Output schema `StoryBible`:
```json
{
  "style": { "look": "str", "palette": "str", "lighting": "str", "lens": "str",
             "era": "str", "negative": "str" },
  "characters": [{ "key": "slug", "name": "str", "description": "str",
                   "visual_prompt": "str (appearance, wardrobe, age, distinctive features; no actions)",
                   "voice": { "gender": "str", "age": "str", "tone": "str" } }],
  "locations":  [{ "key": "slug", "name": "str", "visual_prompt": "str" }],
  "props":      [{ "key": "slug", "name": "str", "visual_prompt": "str" }]
}
```
The backend upserts **ingredients** from the bible, matching by `key`. It never overwrites `locked` ingredients.

**Step 3 — Breakdown** (`purpose=breakdown`)
Input: refined story + bible + cast block (Section 8.5) + render constraints + `target_length_seconds`.
Output schema `Breakdown`:
```json
{
  "scenes": [{
    "title": "str", "summary": "str", "location_key": "slug|null",
    "shots": [{
      "duration_s": 4.0,
      "shot_type": "wide|medium|close|extreme_close|insert|pov|over_shoulder",
      "camera_move": "static|pan_left|pan_right|tilt_up|tilt_down|dolly_in|dolly_out|truck_left|truck_right|orbit|handheld|crane_up|crane_down|zoom_in|zoom_out",
      "ingredient_keys": ["slug"],
      "generation_mode": "t2v|i2v|flf2v|ref2v",
      "keyframe_prompt": "str (single still image: composition, subjects, setting, lighting, style)",
      "end_keyframe_prompt": "str|null (only for flf2v)",
      "video_prompt": "str (motion/action over time, in the video model's prompt style)",
      "negative_prompt": "str",
      "dialogue": [{ "character_key": "slug", "line": "str", "emotion": "str" }],
      "sfx_prompt": "str|null",
      "music_cue": "str|null",
      "needs_extend": false,
      "continuity_notes": "str"
    }]
  }]
}
```
Rules embedded in the system prompt:
- Each shot shows one clear action. Duration ≤ `max_clip_seconds`.
- Reference ingredients only by `key`. Repeat each character's key visual traits in every prompt that features them (consistency).
- Prefer `i2v` (keyframe-first) when the shot has character ingredients. Use `flf2v` for transformations or precise transitions, but only if supported.
- Dialogue lines are short enough to fit the shot duration (≈ 2.5 words/second).
- The total of shot durations is within ±15% of the target length.

**Step 4 — Shot prompt rewrite (on demand)** (`purpose=prompt_rewrite`)
Rewrites one shot's prompts based on a user instruction ("make it night", "more dramatic"), keeping ingredient consistency.

**Step 5 — Vision review (optional; needs a vision-capable model)** (`purpose=vision_review`)
Sends a keyframe (or two frames of a take, at 30% and 70%) and the shot's reference images (the ones keyframe composition uses, Section 11.7) to a vision-capable model. Returns `{consistent: bool, issues: [str], suggested_prompt_fix: str}`. The fix is the complete revised keyframe, end-frame or video prompt (whichever the image came from), keeping its @mentions. The UI applies it to the shot panel's draft.

### 8.4 Prompt templates
- Stored as versioned Jinja2 templates in `services/llm/prompts/` (`refine.v1.j2`, etc.). The template version is recorded with each story revision.
- Each template contains a role, task, constraints, the render constraints block, and the JSON schema. The user's text is wrapped in delimiters and treated as data (basic prompt-injection hygiene).

### 8.5 Cast context
- Every story-pipeline call (refine, bible, breakdown, prompt_rewrite) includes a **cast block** listing the project's character ingredients: `{key, name, visual_prompt, character_info, voice: {name, description}}`. `character_info` is delimited user data (Section 8.4).
- **Refine** also receives the characters the user picked in the Story tab's "Cast" picker, marked "must appear".
- **Bible:** characters with `origin` other than `bible`, or with `locked=true`, are passed as `existing_cast`. The LLM must reuse their keys unchanged and must not alter their visual traits. It may add new characters.
- **Breakdown / prompt rewrite:** actions, expressions, and dialogue `emotion` must be consistent with each character's `character_info`.
- `character_info` is never sent to image or video models.
- Additional purposes (each a versioned template, recorded in `llm_usage`):
  - `character_expand`: short description (+ optional template) → `{visual_prompt, character_info, voice_hints}`.
  - `character_describe`: image(s) → `{visual_prompt}`. Requires a vision-capable model; the action is hidden otherwise.
  - `voice_design`: voice description + the TTS adapter's voice-param JSON schema → params (Section 12.2).

---

## 9. Model management

### 9.1 Registry
- `model_entries` is the single source of truth.
- A curated **catalog file** (`backend/localflow/services/registry/catalog.yaml`) ships with recommended defaults. Each entry has: name, source ref, adapter key, capabilities, license, approximate size, VRAM profiles, and default params. The implementing agent must **verify every catalog entry** against Hugging Face before shipping (Section 0.2).
- Users can add any HF repo or Civitai model version. The app infers `base_model` and `kind` from metadata, but the user must confirm which **adapter** handles it.

### 9.2 Initial catalog (verify each; replace if unavailable)
| Purpose | Candidate | Notes |
|---|---|---|
| Text-to-image | FLUX.1-dev (GGUF/FP8 transformer) | Non-commercial license OK for personal use |
| Text-to-image (Civitai ecosystem) | SDXL base + Civitai SDXL/Pony/Illustrious checkpoints | Loaded via `from_single_file` |
| Image edit / ingredient composition | FLUX.1 Kontext-dev or Qwen-Image-Edit | Composes keyframes from ingredient references |
| T2V + I2V (low VRAM default) | Wan 2.2 TI2V-5B | 720p, fits 16 GB with offload |
| T2V / I2V (quality) | Wan 2.2 A14B (T2V, I2V), GGUF Q4/Q5 | Requires quantization + offload given 23 GB RAM |
| First/last frame | Wan 2.1 FLF2V-14B (or newer Wan FLF variant), GGUF | |
| Reference-to-video | Wan VACE (2.1/2.2) | Optional, later milestone |
| Native audio+video | LTX-2 (or newer) / HunyuanVideo 1.5 | **Include only if supported by installed diffusers**; otherwise mark "experimental/unsupported" |
| TTS | Kokoro-82M | Fast, small |
| Video-to-audio SFX | MMAudio | Standalone package |
| Music (optional) | ACE-Step or MusicGen | |
| Camera/motion LoRAs | Wan-compatible LoRAs from HF/Civitai | Optional |

### 9.3 Hugging Face integration
- Use `huggingface_hub` (`snapshot_download` with `allow_patterns` to fetch only needed files, `hf_hub_download` for single files). Pass `hf_token` from the secret store for gated repos.
- Show repo size before download, using `HfApi().model_info(files_metadata=True)`.
- **Access check** (part of inspect, and before every install): read `gated` from `model_info`, then check access for the current token (e.g., `HfApi().auth_check()` if available in the installed version, else `get_hf_file_metadata` on one required file). Map the result to an `access` state:

  | `access` | Cause | UI |
  |---|---|---|
  | `ok` | Public, or token has access | Install enabled |
  | `needs_token` | Gated/private and no token configured | "Add a Hugging Face token" → Settings |
  | `license_not_accepted` | Gated, token valid, terms not accepted (`GatedRepoError`) | "Open model page" (`https://huggingface.co/<repo_id>`) + "Check again" |
  | `pending_approval` | Manually gated, request not yet approved | Same as above, with "awaiting approval by the repo owner" |
  | `token_no_access` | Fine-grained token lacks gated-repo permission, or private repo not shared | "Update token permissions" link + "Check again" |
  | `not_found` | Wrong id, or private repo hidden (`RepositoryNotFoundError`) | Error with the repo id |

- Catalog and installed model cards show a lock badge for gated repos, with the current `access` state. Access is re-checked on "Check again" and when the token changes.

### 9.4 Civitai integration
- REST API base `https://civitai.com/api/v1`. Endpoints: `GET /models` (search: query, types, baseModels, sort, nsfw), `GET /models/{id}`, `GET /model-versions/{id}`. Download via the version's `downloadUrl` with `Authorization: Bearer <civitai_api_key>` (or the `token` query param if required). **Verify current API behavior.**
- **Login-required and early-access downloads:** some creators require login to download, and some versions are early-access. Show a lock badge in the browser when version metadata indicates either (verify field names). Map failures to clear states: no key configured → `needs_token`; 401 → `invalid_token`; 403 → `forbidden` (early access, or account restrictions). If a download responds with HTML (e.g., a login page) instead of a file, treat it as an auth failure and never save it as a model.
- Model browser UI: search, filter by type (Checkpoint, LORA, VAE) and base model, show preview images, trigger words, license/permissions fields, and NSFW flag (hidden unless `allow_nsfw_models`).
- Map Civitai `baseModel` strings (e.g., "SDXL 1.0", "Flux.1 D", "Wan Video 14B i2v 720p") to internal `base_model` and compatible adapters. Unknown → "unsupported" (still downloadable).
- Store trigger words. Offer them in the prompt editor when a LoRA is attached.

### 9.5 Licenses
- Every entry stores `license` and `license_url`. The UI shows a license badge on the model card and in the shot generation panel.
- First use of a model shows a one-time "I have reviewed this license" acknowledgement, persisted per model.

### 9.6 Downloader
- Async, resumable (HTTP Range) for Civitai. HF handles its own resume.
- Progress events go over WebSocket. Downloads can be cancelled. SHA256 is verified when the source provides a hash.
- Only `.safetensors` and `.gguf` are accepted by default (Section 18).
- Disk-space check before download.
- Tokens never appear in logs, `jobs.params_json`, `model_entries.files_json`, or events. Strip `token` query params and signed redirect URLs (e.g., Civitai CDN redirects) before persisting or logging any URL.

---

## 10. Inference engine (GPU worker)

### 10.1 Capability interfaces
```python
class TextToImage(Protocol):
    def generate(self, req: T2IRequest, ctx: JobContext) -> list[ImageResult]: ...
class ImageEdit(Protocol):           # reference-conditioned composition
    def edit(self, req: ImageEditRequest, ctx: JobContext) -> list[ImageResult]: ...
class TextToVideo(Protocol):
    def generate(self, req: T2VRequest, ctx: JobContext) -> VideoResult: ...
class ImageToVideo(Protocol):
    def generate(self, req: I2VRequest, ctx: JobContext) -> VideoResult: ...
class FirstLastFrameToVideo(Protocol):
    def generate(self, req: FLF2VRequest, ctx: JobContext) -> VideoResult: ...
class ReferenceToVideo(Protocol):
    def generate(self, req: Ref2VRequest, ctx: JobContext) -> VideoResult: ...
class TextToSpeech(Protocol):
    def synthesize(self, req: TTSRequest, ctx: JobContext) -> AudioResult: ...
class VideoToAudio(Protocol):
    def generate(self, req: V2ARequest, ctx: JobContext) -> AudioResult: ...
class TextToMusic(Protocol):
    def generate(self, req: MusicRequest, ctx: JobContext) -> AudioResult: ...
```
- `JobContext` provides: `report_progress(fraction, message)`, `report_preview(image)`, `cancel_event`, `seed`, `output_dir`, `vram_profile`, and a logger.
- An adapter declares: `adapter_key`, `adapter_version`, `capabilities`, `supported_base_models`, `param_schema` (Pydantic → JSON Schema, used by the UI for an "Advanced" form), `vram_profiles`, `supports_lora`, `supports_native_audio`.
- `VideoResult` contains: path to MP4 (H.264, yuv420p), fps, frame count, `has_audio`, and the seed used.

### 10.2 Common request fields
`model_entry_id, prompt, negative_prompt, width, height, num_frames | duration_s, fps, steps, guidance_scale, seed (int | null → random, recorded), loras: [{model_entry_id, weight}], scheduler (optional), extra (adapter-specific dict validated by param_schema)`.
- Width and height are snapped to the multiple required by the model (e.g., 16 or 32). Frame counts are snapped to the model's valid set (e.g., Wan requires `4k+1`).

### 10.3 Model cache & memory management
- `ModelCache` holds at most **one heavy pipeline** plus optional small models (TTS) when the profile allows.
- Key = `(model_entry_id, adapter_key, quantization, frozenset(loras))`. LoRA changes use `load_lora_weights` / `set_adapters` / `unload_lora_weights` without reloading the base model when possible.
- On switch: delete the pipeline, `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`. Log VRAM before and after.
- On `torch.cuda.OutOfMemoryError`: free memory, retry **once** with the next-lower VRAM profile (more offload, smaller resolution only if the user allowed auto-downgrade). Otherwise fail with an actionable message.
- `torch.backends.cuda.matmul.allow_tf32 = True`. Use bf16 where supported. Optionally use SDPA / attention backends supported by installed diffusers. **No** `torch.compile` by default (Windows support is fragile; make it an opt-in setting).

### 10.4 Adapter notes (verify against installed diffusers)
- **Flux:** `FluxPipeline`. Transformer loaded from GGUF via `FluxTransformer2DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16))`, or quantized T5 (text encoder). LoRA via `load_lora_weights`.
- **SDXL / Civitai checkpoints:** `StableDiffusionXLPipeline.from_single_file(path)`. Civitai LoRAs via `load_lora_weights(path)`. Support an IP-Adapter for ingredient conditioning when no image-edit model is installed.
- **Image edit (ingredient composition):** Kontext/Qwen-Image-Edit pipeline. Input is 1–N ingredient reference images plus a keyframe prompt. If the pipeline supports only one reference image, build a reference sheet: a horizontal collage of the references, documented in `DECISIONS.md`.
- **Wan:** `WanPipeline` (T2V) and `WanImageToVideoPipeline` (I2V, and FLF2V via `last_image` if supported). `AutoencoderKLWan` in float32 when the model card recommends it. Transformer GGUF loading via `from_single_file` + `GGUFQuantizationConfig` if supported. Default 81 frames @ 16 fps for 2.1-era models, or the model card's recommended fps/frames for 2.2. Use `enable_model_cpu_offload()` or group offloading.
- **LTX / Hunyuan:** implement only if the installed diffusers has working pipelines. Otherwise register the adapter as `unavailable` with a reason, and the UI hides it.
- **Progress:** use `callback_on_step_end` to report `step/total`. Every N steps (configurable, default off for video), decode a low-res preview. Check `cancel_event`.
- **Output encoding:** frames → MP4 via ffmpeg (`libx264`, `-pix_fmt yuv420p`, `-crf 18`). Generate a JPEG thumbnail (first frame) and store `ffprobe` metadata.

### 10.5 Seeds & reproducibility
- Always resolve and record a concrete seed. Use a `torch.Generator(device="cpu").manual_seed(seed)` for portability.

---

## 11. Generation workflows

### 11.1 Ingredient generation
- For each ingredient: T2I with `style.look` + `visual_prompt` → N candidates (default 4) → the user picks the primary and can upload their own images.
- Characters use the character creator (Section 11.7) for reference shots, extra views (full body, 3-view sheet), voice, and character info.
- `locked` ingredients aren't modified by bible regeneration.

### 11.2 Keyframe-first shot generation (default path for `i2v`)
1. Compose the keyframe: ImageEdit with references = the shot's ingredient assets (primary assets; for characters, views chosen per Section 11.7 "Reference selection"); prompt = `keyframe_prompt` (mentions resolved, Section 11.7.3) + style. Fallback: T2I (with IP-Adapter if SDXL) when no edit model exists.
2. The user selects a keyframe (or regenerates or uploads one).
3. I2V from the selected keyframe with `video_prompt` + camera fragment (Section 11.5).

### 11.3 FLF2V
Generate or select start and end keyframes the same way as 11.2, then run the FLF2V adapter.

### 11.4 Extend
- Input: a video asset and a continuation prompt (default: the shot's `video_prompt` + "continues").
- Extract the last frame (ffmpeg `-sseof`), run I2V, then concatenate. Optional: drop the first frame of the continuation to avoid a duplicate frame. Optional 4–8 frame crossfade.
- The result is a new asset with both parents recorded. The shot's `selected_video_asset_id` can point to the extended asset.
- `needs_extend` shots from the breakdown offer one-click "Generate + extend N times".

### 11.5 Camera controls
- `camera_move` enum maps to a prompt fragment per adapter. Example for Wan: `dolly_in` → "the camera slowly dollies in toward the subject".
- Optional: attach a camera-motion LoRA registered with a `camera_move` tag. When present, it replaces or augments the prompt fragment.
- UI: a camera preset dropdown with icons and an intensity slider (maps to adverbs/LoRA weight).

### 11.6 Variations & batch
- "Generate N variations" queues N jobs with different seeds.
- "Generate all planned shots" queues keyframes first, then videos for shots whose keyframes are auto-approved (setting) or approved by the user.

### 11.7 Character creator
LocalFlow's version of Google Flow's Characters. A character is still an ingredient with `kind="character"`. This section adds a dedicated creation flow, extra views, personality ("character info"), voices, a cross-project library, and prompt mentions.

**Creation wizard** (Ingredients tab → "New character", or Cast library → "New character"):
1. **Visual base.** Choose one source:
   - **Template:** a built-in or user template (Section 11.7.1) prefills `visual_prompt`, `negative_prompt`, `character_info`, and voice hints. Everything stays editable.
   - **From scratch:** write a prompt. Optional "Expand with AI" (`purpose=character_expand`) turns a short description into the same fields.
   - **Upload:** 1–N images. Optional "Describe from image" (`purpose=character_describe`) fills `visual_prompt`. Optional "Clean reference" uses ImageEdit to turn an upload into a neutral-pose, plain-background reference.
   - **Import:** from the Cast library, or from a character in another project (Section 11.7.2).

   Template and from-scratch run T2I with `style.look` (when inside a project) + `visual_prompt` + a reference-shot suffix (front view, neutral expression, head and shoulders, plain background, even lighting) → N candidates (default 4). The user picks the primary.
2. **Extra views.** The user describes body and outfit (`body_outfit_prompt`, e.g., "dark navy sweater and cargo pants"). Views are generated with ImageEdit, using the primary as the reference: `full_body` and `three_view` (front/side/back sheet) by default; `side`, `back`, and `face_closeup` are optional. Fallbacks when no edit model is installed: SDXL + IP-Adapter, else T2I from `visual_prompt` + `body_outfit_prompt` with a "lower consistency" warning. Each view can be regenerated, uploaded, or deleted.
3. **Identity.** Name, `key` (slug derived from the name; editable until saved, then immutable), and `character_info`: baseline personality, quirks, mannerisms, typical expressions, and backstory. This feeds the LLM pipeline (Section 8.5) only.
4. **Voice.** Preset or custom (Section 12.2), with audition.
5. **Save.** Creates the ingredient with `origin="user"` and `locked=true`. Optional "Also save to Cast library".

The same fields are editable later in the character detail drawer. Bible-created characters can also be opened in the wizard to generate views and voices.

**Reference selection** (used by Section 11.2 step 1, implemented in `services/characters/`): for each character in a shot, pick reference images by `shot_type`:

| shot_type | References |
|---|---|
| `close`, `extreme_close`, `over_shoulder` | primary (+ `face_closeup` if present) |
| `medium`, `pov`, `insert` | primary + `full_body` |
| `wide` | `full_body` + primary |

The total respects the edit model's maximum reference count, falling back to the reference-sheet collage (Section 10.4). The user can override references per shot. Attached character LoRAs are applied as usual.

#### 11.7.1 Templates
- Built-ins live in `services/characters/templates.yaml` (loaded with `yaml.safe_load`, read-only): about 12 templates across categories (realistic, fantastical, sci-fi, historical, anime, cartoon, creature, animal). Each has `key, name, category, visual_prompt, negative_prompt, character_info, voice_hints {gender, age, tone}` and a bundled thumbnail.
- "Save as template" on any character stores a user template in `character_templates`. User templates can be edited and deleted. Built-ins can't.

#### 11.7.2 Cast library (cross-project)
- Global (not project-scoped). Rows in `library_characters`; images in `library/characters/<id>/` (Section 4.2).
- **Save to library** copies the character's fields, primary image, and views into the library.
- **Cast into project** copies them back as a new project ingredient (`origin="library"`, `library_character_id` and `library_synced_at` set). Images become new project assets with `source="imported"`.
- Copies are snapshots. Edits never propagate automatically. If the library entry is newer than `library_synced_at`, the ingredient shows "Update from library". "Push to library" overwrites the library entry after confirmation.
- **Import from another project** copies the ingredient and its assets the same way (`origin="import"`).
- Key collisions on import get a numeric suffix (`mara-2`). LoRAs and voices are global and referenced by id, not copied.

#### 11.7.3 Mentions (`@key`)
- Every prompt editor (shot prompts, Playground, ingredient prompts) offers `@` autocomplete over the project's ingredients (characters first, then locations and props). In the Playground, mentions resolve against the selected project or the Cast library.
- Mentions are stored as plain text `@key`. `@@` is a literal `@`. Unknown keys are a validation error, shown inline and returned as a 422 by generation endpoints.
- Resolution at generation time (`services/characters/mentions.py`):
  - Keyframe, T2I, and T2V prompts: the first mention of a character becomes `"{name}, {visual_prompt}"`; later mentions become `name`.
  - I2V, FLF2V, and extend prompts: `name` only (appearance comes from the keyframe).
  - Mentioned ingredients are added to the shot's `ingredient_ids_json`, so they become keyframe references.
- `POST /prompts/resolve` previews the resolved prompt and the referenced ingredients.

---

## 12. Audio

### 12.1 Strategy
1. If the shot's video adapter `supports_native_audio` and the user enables it, generate video with audio, using the dialogue and SFX folded into the prompt in that model's documented format.
2. Otherwise, fallback pipeline per shot:
   - **Dialogue:** TTS per line using the character ingredient's `voice_id` → WAV. Fit the timing to the shot (warn if too long; offer "speed up ≤ 1.15×" or "extend shot"). *As built (M6):* one TTS job per shot with `lines` (each with its voice and params) produces one WAV as long as the clip; a 0.3 s lead-in and 0.35 s gaps shrink first, then the pace rises to `max_speed`; dialogue still too long is kept and reported (`generation.output.overflow_s`). Speakers without a usable voice get one suggested from their voice hints. Emotions only adjust the pace (Kokoro has no emotion control).
   - **SFX/ambience:** video-to-audio (MMAudio) on the generated clip with `sfx_prompt` → WAV matching the clip duration. *As built:* takes over 10 s are made in overlapping ~8 s windows joined with equal-power crossfades; the default negative prompt keeps speech and music out.
   - **Music:** per scene or whole film from `music_cue`s (optional model). The user can also upload music. *As built:* MusicGen small (up to 30 s per job), placed on the Music track at the film's or scene's start.
   - New dialogue and SFX takes become the shot's selected audio for their part and, if the timeline holds the shot's take, replace its clip on the Dialogue/SFX track in place.
3. Lip-sync is **out of scope for v1**. Record it as a future adapter type (`LipSync`).

### 12.2 Voices
- The `voices` table holds built-in voices (synced from the installed TTS adapters when the worker starts, `kind="builtin"`) and custom voices. A character's `voice_id` references `voices.id`. Voice hints from the bible or template auto-suggest a voice.
- **Preset voices:** listed with audition (a sample line, preview cached as WAV).
- **Custom voices:** pick a base voice, name it, and describe the sound ("fast-paced, slightly raspy"). If the TTS adapter declares `supports_voice_design` (text-described voices), the description is passed directly. Otherwise `purpose=voice_design` maps the description onto the adapter's voice-param JSON schema (e.g., base-voice blend weights, speed, pitch where supported), validated with Pydantic. The user can tweak the params and audition before saving. Custom voices are global, so library characters keep them across projects. *As built (M6):* Kokoro's params (`VoiceParams`) are up to two blend voices of the same language with weights, a base weight, and a pace (0.7–1.4); no adapter has `supports_voice_design` yet. Built-ins sync lazily when the set of usable TTS adapters changes; Japanese and Mandarin Kokoro voices are omitted (extra G2P packages).
- Voice cloning from uploaded audio is out of scope for v1 (Section 1.3).

---

## 13. Asset library
- Grid view with filters: type, source, ingredient, scene/shot, favorite, tags, collection, and model used. Search covers prompts (raw and resolved), dialogue, file names, model names, and tags. Newest or oldest first, paginated.
- Collections sidebar (all, favorites, each collection with its count; create, rename, delete). Multi-select for bulk tag, favorite, add to / remove from a collection, and delete.
- Asset detail drawer: preview, full `generation_json`, "reuse settings" (opens the Playground with the same kind, prompt, seed, size, model, LoRAs and inputs), "use as keyframe" (shot + first/last frame), "use as ingredient reference" (primary or a character view), "extend" (video: on its shot if it's a take, else in the Playground), "extract frame at time" (video → image asset at the playhead, on the frame grid), tags, collections, source assets ("made from"), download. For a video, "use as keyframe/reference" uses the frame under the playhead.
- Upload (button or drag-and-drop): images (png/jpg/webp), videos (non-H.264 → transcoded to H.264 MP4; H.264 in MOV/MKV → remuxed), audio (mp3/flac/ogg → WAV; wav/m4a/aac kept). `generation_json.normalized` records the conversion.

---

## 14. Job system

### 14.1 Job types
`t2i, image_edit, t2v, i2v, flf2v, ref2v, extend, tts, v2a, music, export, download_model` (download_model and export run in the API process; the rest in the GPU worker).

### 14.2 Behavior
- Persisted in SQLite. On startup, jobs left `running` are marked `failed` ("interrupted") and the user can retry.
- Queue order: priority desc, then FIFO. The user can reorder queued jobs and cancel queued or running jobs.
- **Batching by model:** when picking the next GPU job, prefer one that uses the currently loaded model if it is within the top K (default 3) of the queue. This minimizes reloads.
- On completion, the worker writes files. The API creates the `assets` row(s) and links them to `target_ref` (e.g., adds them to a shot's candidate list).

### 14.3 Events (WebSocket `/ws/events`)
JSON messages: `{type, ...}` where `type ∈ job.created | job.updated | job.progress | job.preview | job.completed | job.failed | download.progress | model.status | worker.status | story.updated | board.updated | ingredients.updated | timeline.updated | asset.created | asset.deleted | assets.updated | collections.updated | project.deleted` (plus `library.updated` and `voices.updated`, Sections 11.7 and 12.2).
- The frontend invalidates the relevant TanStack Query caches on events.
- Heartbeat every 15 s. The client reconnects automatically with backoff.

---

## 15. Timeline (SceneBuilder) & export

### 15.1 Timeline model
```json
{
  "fps": 24, "width": 1280, "height": 720,
  "tracks": [
    { "id": "v1", "kind": "video", "clips": [
      { "id": "c1", "asset_id": "...", "start_s": 0.0, "in_s": 0.0, "out_s": 4.0,
        "transition_in": { "type": "cut|crossfade|fade_from_black", "duration_s": 0.5 },
        "shot_id": "..." } ] },
    { "id": "a_dialogue", "kind": "audio", "clips": [ { "id": "...", "asset_id": "...", "start_s": 0.0, "in_s": 0.0, "out_s": 2.1, "gain_db": 0 } ] },
    { "id": "a_sfx", "kind": "audio", "clips": [] },
    { "id": "a_music", "kind": "audio", "clips": [] }
  ]
}
```
- v1: one video track (sequential, no overlaps except during transitions) and up to 3 audio tracks.
  - Tracks also carry `name`, `muted` and `gain_db`. On the video track, `muted` mutes the clips' own sound.
  - The server derives video clips' `start_s` from their order, trims and crossfades, on the timeline's frame grid.
  - A crossfade is at most half the shorter neighbouring clip. A first clip can't crossfade, so it fades in from black instead.
  - Audio clips keep the `start_s` they're given.
- "Auto-assemble" button: builds the timeline from scenes/shots in order using each shot's selected video and audio assets.
  - Shots within a scene are cut. Scenes are joined with a 0.5 s crossfade by default, and the first shot fades in from black.
  - Shots without a take are skipped and reported. The music track is kept.

### 15.2 Timeline UI
- Horizontal zoomable timeline with time ruler, playhead, and snapping.
- Drag to reorder video clips. Trim handles for in/out. Drag audio clips freely on audio tracks.
- Preview player: plays the timeline by switching `<video>` sources at clip boundaries and syncing `<audio>` elements (approximate preview is acceptable; the export is authoritative).
- Clicking a clip selects its shot. A side panel shows the shot's prompts and actions (regenerate, extend, swap take).

### 15.3 Export
- FFmpeg `filter_complex`: scale/pad every clip to project resolution and fps, apply trims, `xfade` for crossfades, `adelay` + `amix` for audio tracks with per-clip gain, loudness normalize (`loudnorm`, optional).
  - Clips whose aspect ratio is within 3% of the frame's are scaled and center-cropped instead of padded, so Wan's 1280×704 fills a 1280×720 frame.
  - Audio is mixed over silence of exactly the video's length.
- Output: MP4 H.264 + AAC. Presets: 720p/1080p (upscale is a plain scale in v1). Progress is parsed from `-progress pipe:1`.
- Export creates an asset with `source="export"`.

---

## 16. REST API (prefix `/api`)

All request and response bodies are Pydantic models. Errors use `{error: {code, message, details}}` with proper HTTP status codes.

**Projects**
- `GET/POST /projects`, `GET/PATCH/DELETE /projects/{id}`

**Story**
- `GET /projects/{id}/story` (latest revision) · `GET /projects/{id}/story/revisions`
- `POST /projects/{id}/story/refine` `{raw_idea, options, llm?: {provider, model}}`
- `POST /projects/{id}/story/bible`
- `POST /projects/{id}/story/breakdown` `{target_length_seconds, replace_existing: bool}`
- `PATCH /projects/{id}/story` (manual edits)
- `POST /shots/{id}/rewrite-prompt` `{instruction}`
- `POST /shots/{id}/review` `{asset_id? (default: the selected keyframe, else the take), llm?}` → `{asset_id, target: "keyframe"|"end_keyframe"|"video", prompt_field, current_prompt, consistent, issues, suggested_prompt_fix, references, frames_s, usage}` (vision review, Section 8.3; 400 `nothing_to_review` / `no_references`, 409 `llm_no_vision`)
- `GET /llm/vision?purpose=` → `{available, model, reason}` (whether the purpose's model can read images)

LLM endpoints are synchronous request/response with a 120 s timeout. The UI shows a spinner and allows cancel (AbortController).

**Ingredients**
- `GET/POST /projects/{id}/ingredients`, `PATCH/DELETE /ingredients/{id}`
- `POST /ingredients/{id}/generate` `{count, model_entry_id?, params?}` → one T2I job producing `count` candidates (the first one becomes primary if none is set)
- `POST /ingredients/{id}/assets` (multipart upload; form fields `set_primary`, optional `view` from Section 11.7), `PATCH /ingredients/{id}/primary` `{asset_id | null}`

**Characters** (Section 11.7)
- `GET /character-templates` (built-in + user) · `POST /character-templates` `{from_ingredient_id? | fields}` · `PATCH/DELETE /character-templates/{id}` (user templates only; built-ins → 409 `builtin_template`) · `GET /character-templates/{key}/thumbnail`
- `GET /characters/capabilities?project_id=` → `{expand, describe, edit_model, voices}` with reasons (what the wizard can offer)
- `POST /projects/{id}/characters` `{source: "template"|"prompt"|"upload", template_key?, fields?}` → creates a draft character ingredient (`draft: true`, provisional key). Drafts reuse the ingredient endpoints (generate, upload, candidates) and are saved with `PATCH /ingredients/{id}` `{name, key, …, draft: false}`; `DELETE` cancels and removes the draft's candidates. `GET /projects/{id}/ingredients?include_drafts=true` lists them.
- `POST /characters/expand?project_id=` `{description, template_key?}` · `POST /characters/describe` `{asset_ids, name?}` (LLM, synchronous; describe needs a vision model → 409 `llm_no_vision`)
- `POST /ingredients/{id}/generate-views` `{body_outfit_prompt, views: [...], count, model_entry_id?, seed?}` → `{jobs, views, mode: "edit"|"t2i", note}`
- `POST /ingredients/{id}/clean-reference` `{asset_id}` → job (409 `no_edit_model` without an image-edit model)
- `PUT /ingredients/{id}/views/{view}` `{asset_id | null}` chooses (or clears) a view's image; uploads take a form field `view` (default `primary`); `GET /projects/{id}/assets?ingredient_id=&view=` lists a view's candidates
- `POST /projects/{id}/ingredients/import` `{library_character_id? | ingredient_id?}`
- `POST /ingredients/{id}/save-to-library` · `POST /ingredients/{id}/update-from-library` · `POST /ingredients/{id}/push-to-library`
- `GET /library/characters?q=` · `GET/PATCH/DELETE /library/characters/{id}` · `GET /library/characters/{id}/images/{n}?thumb=1` (accepts `?token=`)
- `POST /prompts/resolve` `{project_id?, text, mode}` → `{resolved_text, ingredient_ids, mentions, errors}`
- `GET /shots/{id}/references?shot_type=&ingredient_ids=` → `{automatic, selected, overridden, max, options}`; the override is the shot field `reference_asset_ids` (null = automatic)
- Event `library.updated` `{library_character_id}`

**Voices** (Section 12.2)
- `GET /voices?adapter_key=&available=` · `POST /voices` `{name, base_voice_id, description, params?}` · `GET/PATCH/DELETE /voices/{id}` (PATCH/DELETE custom only; deleting unassigns characters)
- `GET /voices/suggest?gender=&age=&tone=&language=` → `{voice, reason}` (best available built-in for voice hints)
- `POST /voices/design` `{base_voice_id, description, llm?}` → `{params, notes, warnings, usage}` (LLM, synchronous)
- `POST /voices/{id}/preview` `{text?, refresh?}` → `{url}` when a sample is cached, else `{job}` (tts job; cached when done) · `GET /voices/{id}/preview.wav` (`?token=` allowed)
- `POST /voices/audition` `{base_voice_id, params, text?}` → `{job}` (unsaved settings)
- `POST /generate/tts` accepts LocalFlow voice ids in `voice_id` (and per line in `lines`); the server picks the voice's model
- Event `voices.updated`

**Scenes & shots**
- `GET /projects/{id}/scenes` (with nested shots) · `POST/PATCH/DELETE /scenes/{id}` · reorder `POST /projects/{id}/scenes/reorder`
- `POST/PATCH/DELETE /shots/{id}` · reorder `POST /scenes/{id}/shots/reorder`
- `POST /shots/{id}/generate-keyframe` `{which: "start"|"end", count, mode?: "auto"|"edit"|"t2i", model_entry_id?, params?}` → `{job, mode, prompt, reference_asset_ids, reference_names, note?}` (`note` explains a text-to-image fallback)
- `POST /shots/{id}/generate-video` `{mode?: "auto"|"t2v"|"i2v"|"flf2v", model_entry_id?, params?, variations? (1–4, consecutive seeds), camera_intensity?: "subtle"|"normal"|"strong", extend_segments? (0–4)}` → `{jobs, mode, prompt, model_entry_id, model_name, width, height, fps, num_frames, duration_s, max_clip_seconds, note?, extend_segments}`. Each take is a video asset with `target_ref {kind: "shot", shot_id, slot: "video"}`; the first one becomes the selected take, and `extend_segments` extends it when it's done
- `POST /shots/{id}/video-plan` (same body) → the same fields without `jobs`; queues nothing
- `POST /shots/{id}/extend` `{asset_id? (default: the selected take), segments (1–4), prompt?, model_entry_id?, camera_intensity?, crossfade_frames? (0–8), params?}`. The result replaces the selected take if that was the source
- `GET /camera-moves` → `{move: {intensity: prompt fragment}}`
- `POST /shots/{id}/generate-audio` `{parts: ["dialogue","sfx"], params?: {speed?, fit?: "speed"|"none", max_speed?, tts_model_entry_id?, sfx_prompt?, sfx_negative_prompt?, sfx_steps?, v2a_model_entry_id?, seed?}}` → the plan plus `jobs`. Takes have `target_ref {kind: "shot", shot_id, slot: "dialogue"|"sfx"}`
- `POST /shots/{id}/audio-plan` (the params) → `{clip_s, has_take, lines: [{speaker, line, voice_id, voice_name, voice_source, speed, estimated_s}], speech_s, fit_speed, fits, tts_model, sfx_prompt, v2a_model, selected, notes}`; queues nothing
- `POST /shots/{id}/select-audio` `{asset_id, slot?}` → `{selected_audio_asset_ids}` (one take per part)
- `POST /projects/{id}/generate-music` `{scope: "film"|"scene", scene_id?, prompt?, duration_s?, model_entry_id?, place? (default true)}` → `{job, prompt, duration_s, model_name, notes}`
- `POST /shots/{id}/select` `{slot: "keyframe"|"end_keyframe"|"video"|"audio", asset_id}`
- `POST /projects/{id}/generate-all` `{keyframes?, videos?, auto_approve_keyframes? (default: config), extend?, camera_intensity?}` → `{keyframe_jobs, video_jobs, skipped: [{shot_id, reason}]}`. Shots with a take or a running job are skipped; with auto-approve, each new keyframe is selected and its video queued (audio is generated per shot, from the shot panel)

**Generic generation** (Playground, outside the story flow)
- `POST /generate/{t2i|image_edit|t2v|i2v|flf2v|extend|tts|v2a|music}` → job

**Assets & collections**
- `GET /projects/{id}/assets?type=&source=&tag=&favorite=&collection=&model=&scene_id=&featuring=&q=&sort=newest|oldest` (paginated); `ingredient_id=` or `shot_id=&slot=` lists the candidates generated for that target. `model` is a model entry id (or name); `featuring` is an ingredient id (its candidates, prompts that @mention it, assets made for shots it's in)
- `GET /projects/{id}/assets/facets` → `{total, favorites, types, sources, tags: [{tag, count}], models: [{model, name, count}]}`
- `POST /projects/{id}/assets/upload` (multipart; 415 `unsupported_media`, 422 `convert_failed`) · `PATCH/DELETE /assets/{id}`
- `POST /projects/{id}/assets/bulk` `{asset_ids, add_tags?, remove_tags?, favorite?, delete?}` → `{updated, deleted}`
- `POST /assets/{id}/extract-frame` `{time_s}` → the new image asset
- `GET /assets/{id}/collections` → collection ids
- `GET/POST /projects/{id}/collections` (`{name, description?, asset_ids?}`; 409 `collection_exists`) · `PATCH/DELETE /collections/{id}` · `POST/DELETE /collections/{id}/assets` `{asset_ids}`
- Events `collections.updated`, `assets.updated` `{project_id}`

**Files**
- `GET /files/{project_id}/{asset_id}` with Range support (video seeking). `GET /thumbs/{project_id}/{asset_id}`. Paths are resolved only via DB lookup (Section 18).

**Timeline & export**
- `GET/PUT /projects/{id}/timeline` → `{project_id, timeline, duration_s, updated_at, notes}`. `PUT` takes the Section 15.1 object and returns it normalized: video clip `start_s` is derived, trims and transitions are clamped, and `notes` says what changed. Clips must use assets of this project, videos on the video track and audio on audio tracks (`400 invalid_asset`)
- `POST /projects/{id}/timeline/auto-assemble` `{scene_transition?: "cut"|"crossfade", transition_s?, fade_in?, keep_music?}` → the timeline plus `skipped: [{shot_id, reason}]`
- `POST /projects/{id}/export` `{preset: "720p"|"1080p", loudnorm?}` → job (`400 empty_timeline`). The job snapshots the timeline; the result is an asset with `source="export"` under `exports/`
- Event `timeline.updated` `{project_id}`

**Jobs**
- `GET /jobs?status=&project_id=` · `GET /jobs/{id}` · `POST /jobs/{id}/cancel` · `POST /jobs/{id}/retry` · `POST /jobs/reorder`

**Models**
- `GET /models` (installed + catalog) · `GET /models/{id}` · `DELETE /models/{id}` (deletes files)
- `GET /models/adapters` (adapter keys, capabilities, param JSON schemas, availability)
- `POST /models/hf/inspect` `{repo_id}` → files, size, gated, `access` (Section 9.3), license · `POST /models/hf/install` `{repo_id, allow_patterns?, adapter_key}`
- `GET /models/civitai/search?...` · `GET /models/civitai/{model_id}` · `POST /models/civitai/install` `{version_id, adapter_key?}`
- `POST /models/{id}/acknowledge-license`
- `POST /models/{id}/install` (catalog or previously added entry) · `POST /models/{id}/check-access` ("Check again", Section 9.3) · `GET /models/options` (adapter choices, hub accounts, free disk)

**Settings & system**
- `GET/PATCH /settings` (non-secret) · `GET /settings/secrets` (status only) · `PUT /settings/secrets/{name}` `{value}` · `DELETE /settings/secrets/{name}` · `POST /settings/secrets/{name}/test` (returns account status for hub tokens, Section 7.3) · `POST /settings/secrets/hf_token/import-cli`
- `GET /llm/models?provider=`
- `GET /system/status` → GPU name, VRAM used/total, RAM, disk free, worker state, loaded model, ffmpeg found, torch/CUDA versions (as the worker sees them), CUDA availability, diffusers version
- `GET /system/diagnostics?refresh=` → `{checks: [{key, label, status: "ok"|"warn"|"error"|"skip", detail}], versions, storage: [{key, path, size_mb, files}], disk_free_mb, disk_total_mb, logs}` (Section 19)
- `GET /system/logs/{name}?lines=` → the last lines of a listed log file · `GET /system/diagnostics/bundle` → a redacted zip
- `GET /llm/usage?since=`

---

## 17. Frontend

### 17.1 Layout
Left sidebar: Projects, Cast library, Playground, Models, Settings. Top bar: current project, GPU status chip (VRAM bar, loaded model), jobs indicator. A right-side **Jobs drawer** shows the queue, progress bars, previews, and cancel/retry buttons.

### 17.2 Pages
1. **Projects:** grid of projects with a cover thumbnail, create dialog (name, aspect ratio, fps, default models).
2. **Project workspace** (tabs):
   - **Story:** idea textarea plus options (genre, tone, target length) and a "Cast" picker (project or library characters that must appear) → "Refine" → editable refined story → "Build bible" → "Break down into shots". Each step shows the LLM provider/model picker, token usage, and cost estimate. A revision history dropdown is available. Show a diff when re-running a step would overwrite edits; the user confirms.
   - **Ingredients:** cards grouped by kind; each has a name, editable visual prompt, candidate images, a primary selector, lock toggle, voice picker (characters), and attached LoRAs. The Characters group has "New character" (wizard, Section 11.7) and "Import" (library or another project) buttons. A character detail drawer shows views, character info, voice with audition, library sync status, and "Save as template".
   - **Shots board:** scenes as rows, shots as cards (drag to reorder within or across scenes). A card shows the selected keyframe/video thumbnail, status badge, duration, camera icon, and ingredient avatars. The shot detail panel has prompts (editable, with "AI rewrite"), mode selector, model selector with license badge, advanced params (auto-generated form from the adapter's JSON schema), camera preset, candidates gallery (keyframes, videos, audio) with select/compare, and actions: generate keyframe / video / audio, extend, variations.
   - **Timeline:** Section 15.2, plus an Export button and export history.
   - **Assets:** library (Section 13).
3. **Cast library:** grid of library characters (primary image, name, voice), search, "New character", edit, delete, and "Cast into project…". Also a Voices tab to manage custom voices.
4. **Playground:** standalone generation forms for every capability (useful for testing models).
5. **Models:** tabs for Installed, Catalog (recommended), Hugging Face (paste repo id → inspect → install), Civitai (search/browse/install). Each card shows size, license, capabilities, adapter, status, download progress, and delete.
6. **Settings:** API keys (masked inputs, test button, status), model hub accounts (Section 7.3: "Signed in as", token source, warnings, "Import existing Hugging Face login", links to token pages), default LLM provider/model per purpose, VRAM profile, data directory (read-only display + instructions), ffmpeg path, NSFW toggle, advanced (torch.compile opt-in, preview frequency).

### 17.3 UX requirements
- Every long action becomes a job with visible progress. No UI blocking.
- Optimistic updates for reorder/select. Toasts for job completion and failure, with "view" links.
- Keyboard shortcuts: space (play/pause in timeline), ←/→ (frame step), delete (remove clip).
- Empty states guide the user: "No LLM key configured → Settings", "No video model installed → Models".
- Dark theme by default.

---

## 18. Security & safety
- Bind to `127.0.0.1`. CORS allows only the Vite dev origin in dev mode. In production mode, frontend and API share one origin.
- Set a random per-launch `X-LocalFlow-Token` header, injected into the served `index.html` and required on all API and WebSocket calls (mitigates CSRF / DNS-rebinding from other sites in the browser).
- **File serving:** only by asset id → DB path → resolved path must be inside the project dir (`Path.resolve()` + `is_relative_to`). Library images and voice previews follow the same rule against `library/`.
- **Model weights:** accept only `.safetensors` / `.gguf` by default. `.ckpt` / `.pt` / `.bin` (pickle) require `allow_pickle_weights=true`, a scary warning, and then loading with `torch.load(weights_only=True)` where possible. Never pass `trust_remote_code=True` unless the user explicitly enables it per model.
- Secrets: Sections 7.2–7.3. Redact `sk-...`, `hf_...`, `token=` query params, `Bearer ...`, and key-like strings in logs.
- Upload limits: size caps (configurable). Validate types by content (Pillow / ffprobe), not just by extension.
- LLM prompt-injection hygiene: user text is delimited; LLM output is parsed as data only, never executed.

---

## 19. Error handling, logging, observability
- Structured logs to `logs/localflow.log` (rotating) and the console. Include `job_id` and `project_id` context.
- The worker logs VRAM / RAM usage at model load, job start, and job end.
- User-facing errors are actionable, e.g.: "Out of GPU memory generating 81 frames at 1280×720 with Wan 2.2 A14B. Try the 'low' VRAM profile, fewer frames, or the 5B model." / "black-forest-labs/FLUX.1-dev is gated. You're signed in as <user>, but haven't accepted its license. Open the model page, accept, then click Check again."
- `GET /system/status` and a **Diagnostics** panel in Settings: versions, CUDA availability, ffmpeg check, write test for data dir, keyring backend name. Also: disk space, GPU (nvidia-smi), worker state, LLM providers, disk use per data folder, a log viewer, and a **support bundle** (checks, versions, `config.toml`, recent logs; secrets redacted, no media or database) for bug reports.
- `fake_llm = true` (config) answers every LLM purpose with canned demo responses and needs no key: for offline demos and the E2E smoke test.

---

## 20. Milestones & acceptance criteria

**M0 — Skeleton**
- Monorepo, backend app factory, SQLite + Alembic, frontend shell with routing, theme, and generated API client. `dev` script runs both.
- Fake worker process with the `FakeEngine` (produces placeholder images and videos via ffmpeg test patterns), job queue, WebSocket events, Jobs drawer.
- ✅ Creating a "fake t2v" job from the Playground shows live progress and results in a playable MP4 asset. `pytest` and `vitest` pass. Lint passes.

**M1 — Settings, secrets, LLM story pipeline**
- Keyring secret store, settings UI, key test endpoints, LLM provider layer (LiteLLM), usage tracking.
- Refine → Bible → Breakdown with Pydantic validation and repair retries. Story tab, ingredients created from the bible (no images yet), shots board populated.
- ✅ With a real key for any one provider, a 3-sentence idea produces a refined story, at least 2 characters, and scenes/shots that respect `max_clip_seconds`. Tests cover the pipeline with `FakeLLMProvider` fixtures, including a malformed-JSON repair case.

**M2 — Model management**
- Registry, catalog, HF inspect/install, Civitai search/install, downloads with progress/cancel/resume, license acknowledgement, Models page.
- Hub accounts (Section 7.3), HF access states (Section 9.3), Civitai login-required handling (Section 9.4).
- ✅ Can install one HF model and one Civitai LoRA. Files land in the expected dirs. Registry entries are correct. Deleting a model frees disk.
- ✅ Gated HF repo (e.g., FLUX.1-dev): with no token, inspect reports `needs_token`; with a valid token but unaccepted terms, `license_not_accepted` with a working model-page link; after accepting on the website, "Check again" reports `ok` and the install completes. Settings shows the HF username. An existing CLI login can be imported. A login-required Civitai version downloads with a key and fails with a clear message without one. No token appears in logs, job params, or registry rows.

**M3 — Real image generation**
- GPU worker with ModelCache, VRAM profiles, OOM handling. Flux (GGUF) + SDXL/Civitai checkpoint adapters with LoRA support. Image-edit adapter (Kontext or Qwen-Image-Edit).
- Ingredient generation, keyframe composition from ingredients.
- ✅ On the RTX 5080: generate 4 ingredient candidates at 1024² in a reasonable time without OOM in `balanced`. A keyframe composed from 2 ingredient references is visually recognizable. Switching from Flux to SDXL unloads cleanly (VRAM returns to baseline ±500 MB).

**M3.5 — Character creator**
- Creation wizard (template / from scratch / upload / import), built-in + user templates, extra views, character info in the cast block, reference selection by shot type, Cast library with save/cast/update/push, `@key` mentions with autocomplete and resolution, Cast library page. (Voices come in M6; until then the voice step shows built-in voices only, or is skipped if no TTS adapter is installed.)
- ✅ A character created from a template and saved with `full_body` and `three_view` views is recognizably the same person across all three images. A keyframe for a `wide` shot uses the `full_body` view as a reference. Writing `@key` in a shot prompt adds that character to the shot's references, and the resolved prompt appears in `generation_json`. A character saved to the library and cast into a second project brings its images, character info, and voice. Re-running the bible step with an existing user character keeps its key and visual prompt unchanged (FakeLLMProvider test asserting the cast block contents).
**M4 — Real video generation**
- Wan T2V/I2V (5B default; A14B GGUF in `quality`), FLF2V adapter, camera presets, extend, variations, generate-all.
- ✅ I2V from a keyframe produces a ~5 s 720p (or 480p in `low`) MP4. Extend produces a seamless ~10 s clip. Cancel stops a running job within a few seconds. Worker crash recovery verified by killing the worker mid-job.

**M5 — Timeline & export**
- Timeline model, auto-assemble, editor UI, preview, FFmpeg export with crossfades and audio mixing.
- ✅ A 6-shot project auto-assembles and exports to a valid MP4 (`ffprobe` passes) with correct total duration (±1 frame per transition).

**M6 — Audio**
- TTS adapter + voice picker, `voices` sync, custom voice design with audition (Section 12.2), V2A (MMAudio) adapter, optional music adapter, native-audio video adapter if supported. Per-shot audio generation, audio tracks on the timeline.
- ✅ A shot with one dialogue line and an SFX prompt produces two WAVs aligned to the clip. Export includes mixed audio. A custom voice designed from a text description can be auditioned, assigned to a character, and is used for that character's dialogue.

**M7 — Polish**
- Asset library + collections, prompt rewrite, vision review (optional), diagnostics panel, README with setup steps (CUDA PyTorch install on Windows, ffmpeg, first-run walkthrough), Playwright smoke test.
- ✅ A fresh clone can be set up by following the README. The E2E smoke test (fake engine + fake LLM) runs idea → breakdown → generate → assemble → export.

---

## 21. Testing strategy
- **Unit:** schemas, prompt rendering, breakdown validation rules (duration caps, ingredient key references), timeline → ffmpeg command builder (snapshot tests), Civitai baseModel mapping, path-safety helpers, secret redaction (incl. URLs with `token=` and redirect URLs), hub error → `access` state mapping, token resolution order, mention parsing/resolution (incl. `@@` escape and unknown keys), reference selection by shot type, template loading, library key-collision handling.
- **Integration (no GPU):** API + SQLite + FakeEngine + FakeLLMProvider. Job lifecycle including cancel, failure, and worker restart. Gated/login-required downloads against mocked HF and Civitai HTTP responses (401, 403, gated, HTML login page). Character wizard flows, save/cast/update library round-trip, cross-project import.
- **GPU tests:** marked `@pytest.mark.gpu`, skipped by default. Smoke test each real adapter at the smallest settings.
- **Frontend:** component tests for shot card, timeline math (time ↔ px, snapping), settings secret form, `@` mention autocomplete, character wizard steps. Playwright E2E with the fake backend mode.

---

## 22. Assumptions & open items
| # | Assumption / open item | Default chosen |
|---|---|---|
| 1 | Usage is personal, non-commercial | Licenses displayed, not enforced |
| 2 | Single user, localhost, no login | Per-launch token header only |
| 3 | Inference via diffusers (no ComfyUI) | Adapters isolate this choice; a ComfyUI adapter could be added later |
| 4 | Audio: native when available, else TTS + V2A + music | Lip-sync deferred |
| 5 | Exact open model versions | Verified at implementation time (Section 0.2) |
| 6 | 23 GB system RAM | Quantized 14B models only. Document that 64 GB RAM unlocks higher quality. |
| 7 | Projects are not shared or synced | Export/import of projects (zip) is a post-v1 feature |
| 8 | Upscaling / frame interpolation | Post-v1 (e.g., Real-ESRGAN, RIFE adapters) |
| 9 | Cast library sync model | Snapshot copies with explicit "Update from library" / "Push to library"; no live linking |
| 10 | Custom voice quality depends on the TTS model | Text voice design if the adapter supports it; otherwise LLM-mapped params (e.g., Kokoro voice blending + speed). Verify params at implementation time (Section 0.2) |
| 11 | Avatars / voice cloning | Out of scope for v1 (Section 1.3) |
| 12 | Hub sign-in method | Personal access tokens in v1 (Section 7.3). "Sign in with Hugging Face" (OAuth) is post-v1; Civitai offers API keys only, so it stays token-based |
