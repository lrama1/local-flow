import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { AudioPlan } from "@/api/audio";
import type { Asset, Shot } from "@/api/client";
import { handleServerEvent } from "@/api/events";
import { groupVoices, voiceLabel, vok, type Voice } from "@/api/voices";
import { VoicePicker } from "@/components/VoicePicker";
import { buildBody, DEFAULT_FORM } from "@/features/playground/kinds";
import { AudioSection } from "@/features/shots/AudioSection";
import { shotAudioJobs } from "@/features/shots/video";
import { assetLabel } from "@/features/timeline/math";
import { MusicGenerator } from "@/features/timeline/MusicGenerator";
import { VoiceDesigner } from "@/features/voices/VoiceDesigner";
import { VoicesPanel } from "@/features/voices/VoicesPanel";
import { jsonResponse, makeJob, makeQueryClient, renderWithProviders } from "./helpers";

type Handler = (req: Request) => Response | Promise<Response>;
type Call = { method: string; path: string; search: string; body: unknown };

function mockApi(routes: Record<string, Handler>) {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (req: Request) => {
      const url = new URL(req.url);
      const key = `${req.method} ${url.pathname}`;
      const body =
        req.method === "GET"
          ? undefined
          : await req
              .clone()
              .json()
              .catch(() => undefined);
      calls.push({ method: req.method, path: url.pathname, search: url.search, body });
      const match = Object.entries(routes).find(([pattern]) => new RegExp(`^${pattern}$`).test(key));
      if (!match) return jsonResponse({ error: { code: "not_mocked", message: key } }, 404);
      return match[1](req);
    }),
  );
  return calls;
}

/** jsdom can't play media: a fake Audio that records what was played. */
const played: string[] = [];
class FakeAudio {
  onplay: (() => void) | null = null;
  onended: (() => void) | null = null;
  onpause: (() => void) | null = null;
  constructor(public src: string) {}
  play() {
    played.push(this.src);
    this.onplay?.();
    return Promise.resolve();
  }
  pause() {
    this.onpause?.();
  }
}

beforeEach(() => {
  played.length = 0;
  vi.stubGlobal("Audio", FakeAudio);
});
afterEach(() => vi.unstubAllGlobals());

const PID = "01PROJ00000000000000000001";
const now = new Date().toISOString();

function voice(o: Partial<Voice>): Voice {
  return {
    id: "v-" + (o.native_id ?? o.name),
    kind: "builtin",
    adapter_key: "kokoro",
    native_id: null,
    base_voice_id: null,
    base_voice_name: null,
    name: "Voice",
    description: "",
    language: "en-US",
    gender: "female",
    params: {},
    quality: "",
    traits: "",
    available: true,
    preview_url: null,
    preview_text: null,
    created_at: now,
    updated_at: now,
    ...o,
  };
}

const VOICES: Voice[] = [
  voice({
    id: "v-tom",
    kind: "custom",
    name: "Old Tom",
    base_voice_id: "v-bm_george",
    base_voice_name: "George",
    language: "en-GB",
    gender: "male",
    params: { mix: [{ voice: "bm_lewis", weight: 0.3 }], base_weight: 0.7, speed: 0.85 },
  }),
  voice({ native_id: "af_heart", name: "Heart", quality: "A", traits: "warm, expressive, natural" }),
  voice({ native_id: "am_michael", name: "Michael", gender: "male", quality: "C+", traits: "warm, steady" }),
  voice({ native_id: "bm_george", name: "George", gender: "male", language: "en-GB", quality: "C" }),
  voice({ native_id: "bm_lewis", name: "Lewis", gender: "male", language: "en-GB", quality: "D+" }),
  voice({ native_id: "ef_dora", name: "Dora", language: "es" }),
  voice({ native_id: "jf_x", name: "Gone", available: false }),
];

const page = <T,>(items: T[]) => jsonResponse({ items, total: items.length, limit: 50, offset: 0 });

// --- units ----------------------------------------------------------------------------------------
test("voices group custom first, then built-ins by language; unavailable ones are left out", () => {
  const groups = groupVoices(VOICES);
  expect(groups.map((g) => g.label)).toEqual(["Custom voices", "English (US)", "English (UK)", "Spanish"]);
  expect(groups[1]!.voices.map((v) => v.name)).toEqual(["Heart", "Michael"]);
  expect(groupVoices(VOICES, true).map((g) => g.label)).not.toContain("Custom voices");
  expect(voiceLabel(VOICES[0]!)).toBe("Old Tom · custom (George)");
  expect(voiceLabel(VOICES[1]!)).toBe("Heart · female · A");
});

test("audio jobs per shot and part, with the newest failure", () => {
  const ref = (slot: string) => ({ kind: "shot", shot_id: "shot-1", slot });
  const jobs = [
    makeJob({ id: "a", status: "running", target_ref: ref("dialogue"), created_at: "2026-01-02" }),
    makeJob({ id: "b", status: "failed", target_ref: ref("sfx"), created_at: "2026-01-03" }),
    makeJob({ id: "c", status: "completed", target_ref: ref("sfx"), created_at: "2026-01-01" }),
    makeJob({ id: "d", status: "queued", target_ref: { kind: "shot", shot_id: "other", slot: "sfx" } }),
  ];
  expect(shotAudioJobs(jobs, "shot-1", "dialogue").active.map((j) => j.id)).toEqual(["a"]);
  const sfx = shotAudioJobs(jobs, "shot-1", "sfx");
  expect(sfx.active).toEqual([]);
  expect(sfx.failed?.id).toBe("b");
});

test("playground speech sends the LocalFlow voice id and lets the server pick its model", () => {
  const out = buildBody(
    "tts",
    { ...DEFAULT_FORM, prompt: "Hello", voiceId: "v-tom", adapterKey: "fake" },
    PID,
  );
  expect(out).toEqual({ body: { project_id: PID, extra: {}, text: "Hello", voice_id: "v-tom" } });
  const plain = buildBody("tts", { ...DEFAULT_FORM, prompt: "Hello" }, PID);
  expect(plain).toEqual({ body: { project_id: PID, extra: {}, text: "Hello" } });
});

test("audio assets are labelled by their dialogue, prompt, or kind", () => {
  const a = (generation: Record<string, unknown>) => ({ generation }) as unknown as Asset;
  expect(
    assetLabel(a({ job_type: "tts", params: { text: "", lines: [{ text: "Hi." }, { text: "Bye." }] } })),
  ).toBe("Dialogue: Hi. / Bye.");
  expect(assetLabel(a({ job_type: "v2a", params: { prompt: "" } }))).toBe("Sound effects");
  expect(assetLabel(a({ job_type: "music", params: { prompt: "cello" } }))).toBe("cello");
  expect(assetLabel(a({ uploaded_name: "tone.wav" }))).toBe("tone.wav");
  expect(assetLabel(undefined)).toBe("Audio");
});

test("voices.updated and model changes refresh the voice list", () => {
  const qc = makeQueryClient();
  qc.setQueryData(vok.voices, VOICES);
  handleServerEvent(qc, { type: "voices.updated" });
  expect(qc.getQueryState(vok.voices)?.isInvalidated).toBe(true);
});

// --- voice picker ---------------------------------------------------------------------------------
test("the voice picker suggests a voice from the hints and plays a cached sample", async () => {
  const calls = mockApi({
    "GET /api/voices": () => jsonResponse(VOICES),
    "GET /api/voices/suggest": () => jsonResponse({ voice: VOICES[2], reason: "male, warm, steady" }),
    "POST /api/voices/v-am_michael/preview": () =>
      jsonResponse({ url: "/api/voices/v-am_michael/preview.wav?v=1", job: null }),
  });
  const onChange = vi.fn();
  renderWithProviders(
    <VoicePicker value={null} onChange={onChange} hints={{ gender: "male", age: "50s", tone: "warm" }} />,
  );
  const hint = await screen.findByTestId("voice-suggestion");
  expect(hint).toHaveTextContent("Suggested: Michael (male, warm, steady)");
  const search = calls.find((c) => c.path === "/api/voices/suggest")!.search;
  expect(search).toContain("gender=male");
  expect(search).toContain("tone=warm");
  await userEvent.click(within(hint).getByRole("button", { name: "Play suggested voice" }));
  await waitFor(() => expect(played).toHaveLength(1));
  expect(played[0]).toContain("/api/voices/v-am_michael/preview.wav?v=1&token=");
  await userEvent.click(within(hint).getByRole("button", { name: "Use" }));
  expect(onChange).toHaveBeenCalledWith("v-am_michael");

  const select = screen.getByRole("combobox", { name: "Voice" });
  expect(
    within(select)
      .getAllByRole("option")
      .map((o) => o.textContent),
  ).toContain("Old Tom · custom (George)");
  expect(within(select).queryByText(/Gone/)).toBeNull();
});

test("the voice picker points to the Models page when no TTS model is installed", async () => {
  mockApi({ "GET /api/voices": () => jsonResponse([]) });
  renderWithProviders(<VoicePicker value={null} onChange={() => undefined} />);
  expect(await screen.findByTestId("no-voices")).toHaveTextContent(/install a text-to-speech model/);
  expect(screen.getByRole("link", { name: "Models page" })).toHaveAttribute("href", "/models");
});

test("an uncached sample plays when its TTS job finishes", async () => {
  const job = makeJob({ id: "job-prev", type: "tts", status: "queued", project_id: "pg" });
  mockApi({
    "GET /api/voices": () => jsonResponse(VOICES),
    "POST /api/voices/v-af_heart/preview": () => jsonResponse({ url: null, job }),
    "GET /api/jobs/job-prev": () => jsonResponse(job),
  });
  const { qc } = renderWithProviders(<VoicePicker value="v-af_heart" onChange={() => undefined} />);
  await screen.findByText("warm, expressive, natural");
  await userEvent.click(screen.getByRole("button", { name: "Play sample" }));
  await waitFor(() => expect(qc.getQueryData(["job", "job-prev"])).toBeTruthy());
  expect(played).toEqual([]);
  act(() =>
    handleServerEvent(qc, {
      type: "job.completed",
      job: { ...job, status: "completed", result: { asset_ids: ["aud-1"] } },
    }),
  );
  await waitFor(() => expect(played).toHaveLength(1));
  expect(played[0]).toContain("/api/files/pg/aud-1?token=");
});

// --- voice designer -------------------------------------------------------------------------------
test("designing a voice with AI fills the blend and pace, auditions it, and saves it", async () => {
  const design = {
    params: { mix: [{ voice: "bm_lewis", weight: 0.3 }], base_weight: 0.7, speed: 0.85 },
    notes: "George with a graver edge.",
    warnings: ["Ignored unknown voice 'nope' suggested by the model."],
    usage: null,
  };
  const calls = mockApi({
    "GET /api/voices": () => jsonResponse(VOICES),
    "GET /api/llm/providers": () => jsonResponse([]),
    "POST /api/voices/design": () => jsonResponse(design),
    "POST /api/voices/audition": () =>
      jsonResponse({ url: null, job: makeJob({ id: "j-aud", type: "tts", status: "queued" }) }),
    "GET /api/jobs/j-aud": () => jsonResponse(makeJob({ id: "j-aud", type: "tts", status: "running" })),
    "POST /api/voices": async (req) =>
      jsonResponse(voice({ id: "v-new", kind: "custom", ...(await req.json()) }), 201),
  });
  const onSaved = vi.fn();
  renderWithProviders(
    <VoiceDesigner
      open
      onOpenChange={() => undefined}
      initialName="Old Tom"
      initialDescription="male, 70s, gravelly"
      initialBaseId="v-bm_george"
      onSaved={onSaved}
    />,
  );
  const dialog = await screen.findByRole("dialog");
  expect(await within(dialog).findByLabelText("Base voice")).toHaveValue("v-bm_george");
  // Only voices of the same language can be blended in (any English accent; not Spanish, not George).
  await userEvent.click(within(dialog).getByRole("button", { name: /Blend in a voice/ }));
  const blend = within(dialog).getByRole("combobox", { name: "Blend voice 1" });
  expect(
    within(blend)
      .getAllByRole("option")
      .map((o) => o.textContent),
  ).toEqual(["Heart · female · A", "Michael · male · C+", "Lewis · male · D+"]);
  await userEvent.click(within(dialog).getByRole("button", { name: "Remove blend voice" }));

  await userEvent.click(within(dialog).getByRole("button", { name: /Design with AI/ }));
  expect(await within(dialog).findByTestId("design-notes")).toHaveTextContent("George with a graver edge.");
  expect(within(dialog).getByText(/Ignored unknown voice/)).toBeInTheDocument();
  expect(calls.find((c) => c.path === "/api/voices/design")?.body).toEqual({
    base_voice_id: "v-bm_george",
    description: "male, 70s, gravelly",
    llm: null,
  });
  expect(within(dialog).getByLabelText("Pace")).toHaveValue("0.85");
  expect(within(dialog).getByLabelText("George weight")).toHaveValue("0.7");
  expect(within(dialog).getByRole("combobox", { name: "Blend voice 1" })).toHaveValue("bm_lewis");

  await userEvent.click(within(dialog).getByRole("button", { name: /Audition/ }));
  await waitFor(() => expect(calls.some((c) => c.path === "/api/voices/audition")).toBe(true));
  expect(calls.find((c) => c.path === "/api/voices/audition")?.body).toEqual({
    base_voice_id: "v-bm_george",
    params: design.params,
    text: null,
  });

  await userEvent.click(within(dialog).getByRole("button", { name: "Create voice" }));
  await waitFor(() => expect(onSaved).toHaveBeenCalled());
  expect(calls.find((c) => c.method === "POST" && c.path === "/api/voices")?.body).toEqual({
    name: "Old Tom",
    base_voice_id: "v-bm_george",
    description: "male, 70s, gravelly",
    params: design.params,
  });
});

test("the Voices panel lists custom and built-in voices by language", async () => {
  mockApi({ "GET /api/voices": () => jsonResponse(VOICES) });
  renderWithProviders(<VoicesPanel />);
  const tom = await screen.findByTestId("voice-Old Tom");
  expect(tom).toHaveTextContent("from George · English (UK)");
  expect(screen.getByTestId("builtin-af_heart")).toHaveTextContent("female · warm, expressive, natural");
  expect(screen.queryByTestId("builtin-bm_george")).toBeNull();
  await userEvent.selectOptions(screen.getByRole("combobox", { name: "Language" }), "en-GB");
  expect(screen.getByTestId("builtin-bm_george")).toBeInTheDocument();
});

// --- shot audio -----------------------------------------------------------------------------------
function shot(o: Partial<Shot> = {}): Shot {
  return {
    id: "shot-1",
    scene_id: "scene-1",
    index: 0,
    duration_s: 5,
    status: "video_ready",
    selected_video_asset_id: "take-1",
    selected_audio_asset_ids: ["aud-d1"],
    dialogue: [],
    updated_at: now,
    ...o,
  } as Shot;
}

function plan(o: Partial<AudioPlan> = {}): AudioPlan {
  return {
    clip_s: 5.042,
    has_take: true,
    lines: [
      {
        index: 0,
        character_ingredient_id: "ing-tom",
        speaker: "Old Tom",
        line: "Nets don't mend themselves, lad.",
        emotion: "calm",
        voice_id: "v-tom",
        voice_name: "Old Tom",
        voice_source: "character",
        speed: 0.97,
        estimated_s: 2.3,
      },
    ],
    speech_s: 2.6,
    fit_speed: 1,
    fits: true,
    tts_model: "Kokoro-82M",
    sfx_prompt: "waves lapping",
    v2a_model: "MMAudio",
    selected: { dialogue: "aud-d1", sfx: null },
    notes: [],
    ...o,
  };
}

function audio(id: string, slot: "dialogue" | "sfx", output: Record<string, unknown> = {}): Asset {
  return {
    id,
    project_id: PID,
    type: "audio",
    source: "generated",
    url: `/api/files/${PID}/${id}`,
    thumb_url: null,
    duration_s: 5.042,
    generation: { job_type: slot === "sfx" ? "v2a" : "tts", target_ref: { slot }, output },
    created_at: now,
  } as unknown as Asset;
}

test("the shot's Audio section shows the plan, generates each part and switches takes", async () => {
  const calls = mockApi({
    "POST /api/shots/shot-1/audio-plan": () => jsonResponse(plan()),
    [`GET /api/projects/${PID}/assets`]: (req) =>
      new URL(req.url).searchParams.get("slot") === "dialogue"
        ? page([audio("aud-d2", "dialogue", { overflow_s: 0.6 }), audio("aud-d1", "dialogue")])
        : page([audio("aud-s1", "sfx")]),
    "GET /api/jobs": () => page([]),
    "POST /api/shots/shot-1/generate-audio": () =>
      jsonResponse({ ...plan(), jobs: [makeJob({ id: "j1", type: "tts" })] }, 201),
    "POST /api/shots/shot-1/select-audio": () => jsonResponse({ selected_audio_asset_ids: ["aud-d2"] }),
    "GET /api/jobs/j1": () => jsonResponse(makeJob({ id: "j1", type: "tts", status: "running" })),
  });
  const save = vi.fn(() => Promise.resolve());
  renderWithProviders(<AudioSection shot={shot()} projectId={PID} beforeGenerate={save} />);
  const section = await screen.findByTestId("audio-section");
  const lines = await within(section).findByRole("list", { name: "Dialogue plan" });
  expect(lines).toHaveTextContent("Old Tom (Old Tom): “Nets don't mend themselves, lad.”~2.3 s");
  expect(within(section).getByTestId("audio-fit")).toHaveTextContent("Dialogue ~2.6 s of 5.0 s");

  await userEvent.selectOptions(within(section).getByRole("combobox", { name: "Fit to the clip" }), "none");
  await userEvent.click(within(section).getByRole("button", { name: /Generate dialogue/ }));
  await waitFor(() => expect(calls.some((c) => c.path.endsWith("/generate-audio"))).toBe(true));
  expect(save).toHaveBeenCalled();
  expect(calls.find((c) => c.path.endsWith("/generate-audio"))?.body).toEqual({
    parts: ["dialogue"],
    params: { fit: "none" },
  });

  const used = await within(section).findByTestId("audio-candidate-aud-d1");
  expect(used).toHaveTextContent("In use");
  const longer = within(section).getByTestId("audio-candidate-aud-d2");
  expect(longer).toHaveTextContent("+0.6 s");
  await userEvent.click(within(longer).getByRole("button", { name: "Use" }));
  await waitFor(() => expect(calls.some((c) => c.path.endsWith("/select-audio"))).toBe(true));
  expect(calls.find((c) => c.path.endsWith("/select-audio"))?.body).toEqual({
    asset_id: "aud-d2",
    slot: null,
  });
  expect(within(section).getByTestId("audio-candidate-aud-s1")).toBeInTheDocument();
});

test("a too-long line warns, and sound effects need a take", async () => {
  mockApi({
    "POST /api/shots/shot-1/audio-plan": () =>
      jsonResponse(
        plan({
          has_take: false,
          fits: false,
          fit_speed: 1.6,
          speech_s: 6.2,
          clip_s: 4,
          notes: [
            "No take is selected: dialogue is timed to the planned 4 s, and sound effects need a take.",
          ],
        }),
      ),
    [`GET /api/projects/${PID}/assets`]: () => page([]),
    "GET /api/jobs": () => page([]),
  });
  renderWithProviders(<AudioSection shot={shot({ selected_video_asset_id: null })} projectId={PID} />);
  const fit = await screen.findByTestId("audio-fit");
  expect(fit).toHaveTextContent("too long for the clip. Extend the shot or shorten the lines.");
  expect(screen.getByRole("button", { name: /Generate sound effects/ })).toBeDisabled();
  expect(screen.getByText(/select one first/)).toBeInTheDocument();
});

test("music for one scene is generated from the board's cues", async () => {
  const board = {
    scenes: [
      { id: "sc-1", title: "Harbor", shots: [{ id: "s1", music_cue: "sparse cello" }] },
      { id: "sc-2", title: "Night", shots: [{ id: "s2", music_cue: "strings swell" }] },
    ],
  };
  const calls = mockApi({
    [`GET /api/projects/${PID}/scenes`]: () => jsonResponse(board),
    [`POST /api/projects/${PID}/generate-music`]: () =>
      jsonResponse(
        {
          job: makeJob({ id: "jm", type: "music" }),
          prompt: "strings swell",
          duration_s: 8,
          model_name: "MusicGen small",
          notes: [],
        },
        201,
      ),
    "GET /api/jobs/jm": () => jsonResponse(makeJob({ id: "jm", type: "music", status: "running" })),
  });
  renderWithProviders(<MusicGenerator projectId={PID} />);
  const box = await screen.findByTestId("music-generator");
  await waitFor(() => expect(within(box).getAllByRole("option")).toHaveLength(3));
  expect(within(box).getByRole("textbox", { name: "Music prompt" })).toHaveAttribute(
    "placeholder",
    "sparse cello; strings swell",
  );
  await userEvent.selectOptions(within(box).getByRole("combobox", { name: "Music for" }), "sc-2");
  await userEvent.click(within(box).getByRole("button", { name: /Generate/ }));
  expect(await within(box).findByText(/8.0 s of “strings swell” with MusicGen small/)).toBeInTheDocument();
  expect(calls.find((c) => c.path.endsWith("/generate-music"))?.body).toEqual({
    scope: "scene",
    scene_id: "sc-2",
    prompt: null,
  });
});
