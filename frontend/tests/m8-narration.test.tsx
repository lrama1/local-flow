import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { Schemas } from "@/api/client";
import type { NarrationPlan, NarrationShot } from "@/api/narration";
import { ExportPanel } from "@/features/timeline/ExportPanel";
import { budgetWords, needsVoice, statusLabel, wordCount } from "@/features/timeline/narration";
import { NarrationPanel } from "@/features/timeline/NarrationPanel";
import type { Timeline } from "@/features/timeline/math";
import { useUi } from "@/stores/ui";
import { jsonResponse, makeJob, renderWithProviders } from "./helpers";

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

afterEach(() => {
  vi.unstubAllGlobals();
  useUi.setState({ licensePrompt: null, previews: {} });
});

const PID = "01PROJ00000000000000000001";
const now = new Date().toISOString();

function project(o: Partial<Schemas["ProjectOut"]> = {}): Schemas["ProjectOut"] {
  return {
    id: PID,
    name: "The Last Rose",
    description: "",
    aspect_ratio: "16:9",
    fps: 24,
    default_video_model_id: null,
    default_image_model_id: null,
    draft_mode: false,
    narrator_voice_id: null,
    style_notes: "",
    archived: false,
    created_at: now,
    updated_at: now,
    ...o,
  };
}

function row(o: Partial<NarrationShot> = {}): NarrationShot {
  return {
    shot_id: "s1",
    number: "1.1",
    scene_title: "Dusk",
    duration_s: 4,
    text: "",
    words: 0,
    budget_words: 7,
    estimated_s: 0,
    status: "empty",
    selected_asset_id: null,
    asset_duration_s: null,
    clip_start_s: 0,
    start_s: 0.25,
    overflow_s: 0,
    ...o,
  };
}

function plan(shots: NarrationShot[], o: Partial<NarrationPlan> = {}): NarrationPlan {
  return {
    narrator: { voice_id: "v1", voice_name: "Heart", source: "suggested" },
    tts_model: "Kokoro-82M",
    speed: 1,
    shots,
    total_words: shots.reduce((n, s) => n + s.words, 0),
    notes: [],
    ...o,
  };
}

function voice(id: string, name: string) {
  return {
    id,
    kind: "builtin",
    adapter_key: "kokoro",
    native_id: `af_${name.toLowerCase()}`,
    base_voice_id: null,
    name,
    description: "",
    language: "en-US",
    gender: "female",
    params: {},
    quality: "A",
    traits: "",
    available: true,
    preview_url: null,
    preview_text: null,
    created_at: now,
    updated_at: now,
  };
}

const SHOTS = [
  row({ text: "The tide was turning.", words: 4, status: "text" }),
  row({ shot_id: "s2", number: "1.2", duration_s: 3, budget_words: 5 }),
  row({
    shot_id: "s3",
    number: "2.1",
    duration_s: 5,
    budget_words: 10,
    text: "By morning, nothing would be the same.",
    words: 7,
    status: "stale",
    selected_asset_id: "a3",
    asset_duration_s: 3.1,
    overflow_s: 1.4,
  }),
];

function routes(p: NarrationPlan, extra: Record<string, Handler> = {}): Record<string, Handler> {
  return {
    [`GET /api/projects/${PID}`]: () => jsonResponse(project()),
    [`GET /api/projects/${PID}/narration`]: () => jsonResponse(p),
    "GET /api/voices": () => jsonResponse([voice("v1", "Heart"), voice("v2", "Adam")]),
    "GET /api/jobs": () => jsonResponse({ items: [], total: 0 }),
    ...extra,
  };
}

// --- units ----------------------------------------------------------------------------------------
test("budget, word count, status labels and the lines that still need a voice", () => {
  expect([0.5, 3, 4, 5, 10].map(budgetWords)).toEqual([0, 5, 7, 10, 21]);
  expect(wordCount("  Two   words ")).toBe(2);
  expect(wordCount("")).toBe(0);
  expect(statusLabel("stale")).toBe("out of date");
  expect(needsVoice(SHOTS).map((s) => s.shot_id)).toEqual(["s1", "s3"]);
  expect(needsVoice(undefined)).toEqual([]);
});

// --- the panel ------------------------------------------------------------------------------------
test("shows a row per shot with its status, word budget and overflow", async () => {
  mockApi(routes(plan(SHOTS, { notes: ["Shot 2.1 runs into the next narrated line."] })));
  renderWithProviders(<NarrationPanel projectId={PID} />);
  const r1 = await screen.findByTestId("narration-row-s1");
  expect(within(r1).getByText("not voiced")).toBeInTheDocument();
  expect(within(r1).getByText("4 / 7 words")).toBeInTheDocument();
  expect(within(r1).getByRole("button", { name: /Voice/ })).toBeEnabled();
  const r2 = screen.getByTestId("narration-row-s2");
  expect(within(r2).getByText("silent")).toBeInTheDocument();
  expect(within(r2).getByRole("button", { name: /Voice/ })).toBeDisabled(); // nothing to say
  const r3 = screen.getByTestId("narration-row-s3");
  expect(within(r3).getByText("out of date")).toBeInTheDocument();
  expect(within(r3).getByText("+1.4 s over")).toBeInTheDocument();
  expect(within(r3).getByRole("button", { name: "Re-voice" })).toBeInTheDocument();
  expect(r3.querySelector("audio")).toHaveAttribute("src", expect.stringContaining("a3"));
  expect(screen.getByText(/Shot 2.1 runs into the next narrated line/)).toBeInTheDocument();
  expect(screen.getByText("Using Heart (suggested).")).toBeInTheDocument();
});

test("Voice narration voices the lines that are missing or out of date", async () => {
  const calls = mockApi(
    routes(plan(SHOTS), {
      [`POST /api/projects/${PID}/narration/generate`]: () =>
        jsonResponse({ jobs: [], skipped: [], plan: plan(SHOTS) }, 201),
    }),
  );
  renderWithProviders(<NarrationPanel projectId={PID} />);
  await userEvent.click(await screen.findByRole("button", { name: "Voice narration (2)" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/narration/generate"))?.body).toEqual({
      speed: 1,
      shot_ids: null,
      only_missing: true,
    }),
  );
});

test("a single line can be re-voiced at the chosen pace", async () => {
  const calls = mockApi(
    routes(plan(SHOTS), {
      [`POST /api/projects/${PID}/narration/generate`]: () =>
        jsonResponse({ jobs: [], skipped: [], plan: plan(SHOTS) }, 201),
    }),
  );
  renderWithProviders(<NarrationPanel projectId={PID} />);
  await userEvent.selectOptions(await screen.findByLabelText("Pace"), "1.1");
  await userEvent.click(
    within(screen.getByTestId("narration-row-s3")).getByRole("button", { name: "Re-voice" }),
  );
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/narration/generate"))?.body).toEqual({
      speed: 1.1,
      shot_ids: ["s3"],
      only_missing: false,
    }),
  );
  expect(calls.some((c) => c.path.endsWith("/narration") && c.search.includes("speed=1.1"))).toBe(true);
});

test("a running narration job disables voicing and says how many lines are in progress", async () => {
  const job = makeJob({
    type: "tts",
    status: "running",
    project_id: PID,
    target_ref: { kind: "shot", shot_id: "s1", slot: "narration" },
  });
  mockApi(routes(plan(SHOTS), { "GET /api/jobs": () => jsonResponse({ items: [job], total: 1 }) }));
  renderWithProviders(<NarrationPanel projectId={PID} />);
  const button = await screen.findByRole("button", { name: /Voicing 1 line/ });
  expect(button).toBeDisabled();
  expect(
    within(screen.getByTestId("narration-row-s1")).getByRole("button", { name: /Voice/ }),
  ).toBeDisabled();
});

test("editing a line saves it to the shot when the field loses focus", async () => {
  const calls = mockApi(
    routes(plan(SHOTS), { "PATCH /api/shots/s1": () => jsonResponse({ id: "s1", narration: "x" }) }),
  );
  renderWithProviders(<NarrationPanel projectId={PID} />);
  const box = await screen.findByLabelText("Narration for shot 1.1");
  await userEvent.clear(box);
  await userEvent.type(box, "A new line.");
  expect(screen.getByText("3 / 7 words")).toBeInTheDocument();
  fireEvent.blur(box);
  await waitFor(() =>
    expect(calls.find((c) => c.method === "PATCH" && c.path === "/api/shots/s1")?.body).toEqual({
      narration: "A new line.",
    }),
  );
  // An untouched field isn't saved again.
  const before = calls.filter((c) => c.method === "PATCH").length;
  fireEvent.blur(await screen.findByLabelText("Narration for shot 2.1"));
  expect(calls.filter((c) => c.method === "PATCH").length).toBe(before);
});

test("Write narration sends the style and the rewrite choice, then reports what it wrote", async () => {
  const usage = {
    provider: "openai",
    model: "gpt",
    prompt_tokens: 1,
    completion_tokens: 1,
    cost_usd: 0,
    attempts: 1,
  };
  const calls = mockApi(
    routes(plan(SHOTS), {
      [`POST /api/projects/${PID}/narration/write`]: () =>
        jsonResponse({
          updated: [{ shot_id: "s1", number: "1.1", text: "x", words: 1, budget_words: 7 }],
          skipped: 2,
          usage,
          warnings: ["Shot 2.1: 12 words for a 5 s shot (budget 10)."],
        }),
    }),
  );
  renderWithProviders(<NarrationPanel projectId={PID} />);
  await userEvent.type(await screen.findByLabelText("Narration style"), "a hushed storybook");
  await userEvent.click(screen.getByLabelText("Rewrite existing lines"));
  await userEvent.click(screen.getByRole("button", { name: "Write narration" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/narration/write"))?.body).toEqual({
      overwrite: true,
      style: "a hushed storybook",
    }),
  );
  expect(await screen.findByTestId("narration-written")).toHaveTextContent(
    "Wrote 1 line. Shot 2.1: 12 words for a 5 s shot (budget 10).",
  );
});

test("an LLM failure is shown instead of the result", async () => {
  mockApi(
    routes(plan(SHOTS), {
      [`POST /api/projects/${PID}/narration/write`]: () =>
        jsonResponse(
          { error: { code: "nothing_to_write", message: "Every shot already has narration." } },
          400,
        ),
    }),
  );
  renderWithProviders(<NarrationPanel projectId={PID} />);
  await userEvent.click(await screen.findByRole("button", { name: "Write narration" }));
  expect(await screen.findByText("Every shot already has narration.")).toBeInTheDocument();
});

test("the narrator voice is a project setting", async () => {
  const calls = mockApi(
    routes(plan(SHOTS), {
      [`PATCH /api/projects/${PID}`]: () => jsonResponse(project({ narrator_voice_id: "v2" })),
    }),
  );
  renderWithProviders(<NarrationPanel projectId={PID} />);
  const picker = await screen.findByLabelText("Narrator voice");
  await waitFor(() => expect(within(picker).getAllByRole("option").length).toBeGreaterThan(1));
  await userEvent.selectOptions(picker, "v2");
  await waitFor(() =>
    expect(calls.find((c) => c.method === "PATCH" && c.path === `/api/projects/${PID}`)?.body).toEqual({
      narrator_voice_id: "v2",
    }),
  );
});

test("Re-sync is only offered once a line is voiced, and calls the resync endpoint", async () => {
  const calls = mockApi(
    routes(plan(SHOTS), {
      [`POST /api/projects/${PID}/narration/resync`]: () => jsonResponse({ project_id: PID, timeline: {} }),
    }),
  );
  renderWithProviders(<NarrationPanel projectId={PID} />);
  const resync = await screen.findByRole("button", { name: "Re-sync" });
  await waitFor(() => expect(resync).toBeEnabled()); // shot 2.1 has a take
  await userEvent.click(resync);
  await waitFor(() => expect(calls.some((c) => c.path.endsWith("/narration/resync"))).toBe(true));
});

test("with nothing voiced or written the panel explains itself", async () => {
  mockApi(routes(plan([])));
  renderWithProviders(<NarrationPanel projectId={PID} />);
  expect(await screen.findByText("No shots yet. Break the story into shots first.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Re-sync" })).toBeDisabled();
});

// --- export ---------------------------------------------------------------------------------------
test("the export lowers music under narration by default and can turn that off", async () => {
  const calls = mockApi({
    [`POST /api/projects/${PID}/export`]: () => jsonResponse(makeJob({ type: "export" }), 201),
  });
  const tl: Timeline = {
    fps: 24,
    width: 1280,
    height: 720,
    tracks: [
      {
        id: "v1",
        kind: "video",
        name: "Video",
        muted: false,
        gain_db: 0,
        clips: [
          {
            id: "c1",
            asset_id: "a1",
            start_s: 0,
            in_s: 0,
            out_s: 2,
            transition_in: { type: "cut", duration_s: 0 },
            gain_db: 0,
            shot_id: null,
          },
        ],
      },
    ],
  };
  renderWithProviders(<ExportPanel projectId={PID} timeline={tl} beforeExport={async () => {}} />);
  const duck = screen.getByLabelText("Lower music under narration");
  expect(duck).toBeChecked();
  await userEvent.click(screen.getByRole("button", { name: "Export MP4" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/export"))?.body).toEqual({
      preset: "720p",
      loudnorm: false,
      duck: true,
    }),
  );
  await userEvent.click(duck);
  expect(duck).not.toBeChecked();
});
