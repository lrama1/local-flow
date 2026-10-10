import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { Asset } from "@/api/client";
import { dragPreview, type Drag } from "@/features/timeline/drag";
import {
  clipEnd,
  complete,
  dropIndex,
  fillsFrame,
  moveClip,
  pxToTime,
  sequence,
  sizeFor,
  snapTime,
  tickStep,
  timecode,
  timeToPx,
  totalDuration,
  videoClipAt,
  type Clip,
  type Timeline,
} from "@/features/timeline/math";
import { TimelineTab } from "@/features/timeline/TimelineTab";
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

beforeAll(() => {
  // jsdom has no media playback.
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

afterEach(() => {
  vi.unstubAllGlobals();
  useUi.setState({ licensePrompt: null, previews: {} });
});

const PID = "01PROJ00000000000000000001";
const now = new Date().toISOString();

function clip(o: Partial<Clip>): Clip {
  return {
    id: "c1",
    asset_id: "a1",
    start_s: 0,
    in_s: 0,
    out_s: 2,
    transition_in: { type: "cut", duration_s: 0 },
    gain_db: 0,
    shot_id: null,
    ...o,
  };
}

function timeline(video: Clip[], audio: Clip[] = []): Timeline {
  return {
    fps: 24,
    width: 1280,
    height: 720,
    tracks: [
      { id: "v1", kind: "video", name: "Video", muted: false, gain_db: 0, clips: video },
      { id: "a_dialogue", kind: "audio", name: "Dialogue", muted: false, gain_db: 0, clips: audio },
      { id: "a_sfx", kind: "audio", name: "Sound effects", muted: false, gain_db: 0, clips: [] },
      { id: "a_music", kind: "audio", name: "Music", muted: false, gain_db: 0, clips: [] },
    ],
  };
}

function asset(id: string, o: Partial<Asset> = {}): Asset {
  return {
    id,
    project_id: PID,
    type: "video",
    source: "generated",
    url: `/api/files/${PID}/${id}`,
    thumb_url: `/api/thumbs/${PID}/${id}`,
    width: 1280,
    height: 704,
    duration_s: 5.041667,
    fps: 24,
    has_audio: false,
    favorite: false,
    tags: [],
    parent_asset_ids: [],
    generation: {},
    created_at: now,
    updated_at: now,
    ...o,
  } as unknown as Asset;
}

// --- math -------------------------------------------------------------------------------------------
test("time and pixels convert both ways; ticks stay readable", () => {
  expect(timeToPx(2.5, 40)).toBe(100);
  expect(pxToTime(100, 40)).toBe(2.5);
  expect(tickStep(60)).toBe(1);
  expect(tickStep(24)).toBe(5);
  expect(tickStep(200)).toBe(0.5);
  expect(timecode(61.5, 24)).toBe("1:01.12");
  expect(sizeFor(1280, 720, 1080)).toEqual([1920, 1080]);
  expect(sizeFor(720, 1280, 720)).toEqual([720, 1280]);
});

test("snapping picks the nearest candidate within the pixel threshold at this zoom", () => {
  expect(snapTime(4.1, [0, 4, 4.3], 40)).toEqual({ t: 4, to: 4 }); // 4 px away
  expect(snapTime(4.1, [0, 4, 4.3], 100)).toEqual({ t: 4.1, to: null }); // 10 px away at this zoom
  expect(snapTime(4.25, [4, 4.3], 100).t).toBe(4.3);
});

test("the video track is sequenced like the backend: frame grid, crossfades, first-clip rule", () => {
  const tl = sequence(
    timeline([
      clip({ id: "a", out_s: 2, transition_in: { type: "crossfade", duration_s: 0.5 } }),
      clip({ id: "b", in_s: 0.51, out_s: 3, transition_in: { type: "crossfade", duration_s: 2 } }),
      clip({ id: "c", out_s: 1, transition_in: { type: "cut", duration_s: 1 } }),
    ]),
  );
  const [a, b, c] = tl.tracks[0]!.clips;
  expect(a!.transition_in).toEqual({ type: "fade_from_black", duration_s: 0.5 });
  expect(b!.in_s).toBe(0.5); // snapped to frame 12
  expect(b!.transition_in.duration_s).toBe(1); // half the shorter neighbour (48 frames)
  expect(b!.start_s).toBe(1);
  expect(c!.start_s).toBeCloseTo(3.5);
  expect(c!.transition_in.duration_s).toBe(0);
  expect(totalDuration(tl)).toBeCloseTo(4.5);
  // During the crossfade the incoming clip is the one showing.
  expect(videoClipAt(tl, 1.5)?.id).toBe("b");
  expect(videoClipAt(tl, 0.5)?.id).toBe("a");
  expect(videoClipAt(tl, 9)).toBeUndefined();
});

test("drop index and moves reorder by clip centres", () => {
  const tl = sequence(timeline([clip({ id: "a" }), clip({ id: "b" }), clip({ id: "c" })]));
  const clips = tl.tracks[0]!.clips;
  expect(dropIndex(clips, "a", 5.5)).toBe(2); // past the centres of b (3) and c (5)
  expect(dropIndex(clips, "c", 0.2)).toBe(0);
  const moved = sequence(moveClip(tl, "a", 2));
  expect(moved.tracks[0]!.clips.map((c) => c.id)).toEqual(["b", "c", "a"]);
  expect(clipEnd(moved.tracks[0]!.clips[2]!)).toBe(6);
});

test("drag previews snap trims and audio moves, and keep trims inside the source", () => {
  const tl = sequence(
    timeline(
      [clip({ id: "a" }), clip({ id: "b", asset_id: "b" })],
      [clip({ id: "s", start_s: 0.3, out_s: 1 })],
    ),
  );
  const b = tl.tracks[0]!.clips[1]!;
  const base: Omit<Drag, "mode" | "dt" | "orig" | "kind" | "clipId"> = { x0: 0, moved: true };
  // Trimming B's end to near the playhead (3.0 s) snaps there; it can't pass the 2.5 s source.
  const trim = dragPreview(
    { ...base, clipId: "b", kind: "video", mode: "trim-end", orig: b, dt: -0.95 },
    tl,
    2.5,
    40,
    3,
  );
  expect(trim.patch.out_s).toBe(1);
  expect(trim.snapTo).toBe(3);
  const long = dragPreview(
    { ...base, clipId: "b", kind: "video", mode: "trim-end", orig: b, dt: 5 },
    tl,
    2.5,
    40,
    0,
  );
  expect(long.patch.out_s).toBe(2.5);
  // An audio clip dragged so its end is near the A/B cut (2.0 s) snaps its end there.
  const s = tl.tracks[1]!.clips[0]!;
  const moved = dragPreview(
    { ...base, clipId: "s", kind: "audio", mode: "move", orig: s, dt: 0.65 },
    tl,
    1,
    40,
    9,
  );
  expect(moved.patch.start_s).toBe(1);
  expect(moved.snapTo).toBe(2);
  // Trimming the start of a clip that starts 0.2 s in (with 0.5 s of audio before its in point) stops at
  // the timeline's start: in 0.3, start 0.
  const late = clip({ id: "t", start_s: 0.2, in_s: 0.5, out_s: 1.5 });
  const early = dragPreview(
    { ...base, clipId: "t", kind: "audio", mode: "trim-start", orig: late, dt: -2 },
    tl,
    2,
    40,
    9,
  );
  expect(early.patch).toEqual({ in_s: 0.3, start_s: 0 });
});

test("fill or letterbox like the export", () => {
  expect(fillsFrame(asset("x"), 1280, 720)).toBe(true);
  expect(fillsFrame(asset("x", { width: 832, height: 480 }), 1280, 720)).toBe(true);
  expect(fillsFrame(asset("x", { width: 704, height: 1280 }), 1280, 720)).toBe(false);
});

// --- the tab ------------------------------------------------------------------------------------------
function board(withTakes = true) {
  const shot = (id: string, index: number, take: string | null) => ({
    id,
    scene_id: "s1",
    index,
    duration_s: 5,
    status: take ? "video_ready" : "planned",
    generation_mode: "i2v",
    video_prompt: `prompt ${id}`,
    keyframe_prompt: "",
    ingredient_ids: [],
    dialogue: [],
    needs_extend: false,
    selected_keyframe_asset_id: null,
    selected_end_keyframe_asset_id: null,
    selected_video_asset_id: take,
    updated_at: now,
  });
  return {
    project_id: PID,
    scenes: [
      {
        id: "s1",
        index: 0,
        title: "Pier",
        shots: [shot("sh1", 0, withTakes ? "v1" : null), shot("sh2", 1, withTakes ? "v2" : null)],
      },
    ],
    shot_count: 2,
    total_duration_s: 10,
    max_clip_seconds: 5,
  };
}

function routes(o: { timeline: Timeline; exports?: Asset[]; jobs?: ReturnType<typeof makeJob>[] }) {
  let current = o.timeline;
  const out = () => ({ project_id: PID, timeline: current, duration_s: totalDuration(current), notes: [] });
  return {
    [`GET /api/projects/${PID}/timeline`]: () => jsonResponse(out()),
    [`PUT /api/projects/${PID}/timeline`]: async (req: Request) => {
      current = sequence(complete(await req.json()));
      return jsonResponse(out());
    },
    [`POST /api/projects/${PID}/timeline/auto-assemble`]: () => {
      current = sequence(
        timeline([
          clip({ id: "x1", asset_id: "v1", out_s: 5.041667, shot_id: "sh1" }),
          clip({ id: "x2", asset_id: "v2", out_s: 5.041667, shot_id: "sh2" }),
        ]),
      );
      return jsonResponse({ ...out(), skipped: [] });
    },
    [`POST /api/projects/${PID}/export`]: () =>
      jsonResponse(makeJob({ id: "ex1", type: "export", project_id: PID }), 201),
    [`GET /api/projects/${PID}/scenes`]: () => jsonResponse(board()),
    [`GET /api/projects/${PID}/assets`]: (req: Request) => {
      const q = new URL(req.url).searchParams;
      const items =
        q.get("source") === "export"
          ? (o.exports ?? [])
          : q.get("type") === "video"
            ? [asset("v1"), asset("v2")]
            : [];
      return jsonResponse({ items, total: items.length, offset: 0, limit: 50 });
    },
    "GET /api/jobs": () => jsonResponse({ items: o.jobs ?? [], total: 0, offset: 0, limit: 50 }),
    "GET /api/jobs/[^/]+": () => jsonResponse(makeJob({ id: "ex1", type: "export", status: "running" })),
  } satisfies Record<string, Handler>;
}

test("auto-assembles an empty timeline from the board's takes", async () => {
  const calls = mockApi(routes({ timeline: timeline([]) }));
  renderWithProviders(<TimelineTab projectId={PID} />);
  expect(await screen.findByText("The timeline is empty")).toBeInTheDocument();
  const button = await screen.findByRole("button", { name: "Auto-assemble" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  expect(await screen.findByRole("button", { name: "Clip 1: Shot 1.1, 5.0s" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Clip 2: Shot 1.2, 5.0s" })).toBeInTheDocument();
  expect(screen.getByTestId("timecode")).toHaveTextContent("0:00.00 / 0:10.02");
  expect(screen.getByText("2 clips placed.")).toBeInTheDocument();
  expect(calls.filter((c) => c.path.endsWith("/auto-assemble"))).toHaveLength(1);
});

test("inspector edits and Delete are saved (debounced) with the sequenced timeline", async () => {
  const tl = sequence(
    timeline([
      clip({ id: "x1", asset_id: "v1", out_s: 5.041667, shot_id: "sh1" }),
      clip({ id: "x2", asset_id: "v2", out_s: 5.041667, shot_id: "sh2" }),
    ]),
  );
  const calls = mockApi(routes({ timeline: tl }));
  renderWithProviders(<TimelineTab projectId={PID} />);
  const second = await screen.findByRole("button", { name: "Clip 2: Shot 1.2, 5.0s" });
  fireEvent.pointerDown(second, { button: 0, clientX: 300, pointerId: 1 });
  fireEvent.pointerUp(second, { button: 0, clientX: 300, pointerId: 1 });
  expect(await screen.findByText("Clip 2 of 2 · 5.0s of a 5.0s take")).toBeInTheDocument();

  await userEvent.selectOptions(screen.getByLabelText("Transition in"), "crossfade");
  const out = screen.getByLabelText("Out (s)");
  await userEvent.clear(out);
  await userEvent.type(out, "3{Enter}");
  await waitFor(() => expect(calls.filter((c) => c.method === "PUT").length).toBeGreaterThan(0), {
    timeout: 2000,
  });
  await waitFor(() => {
    const body = calls.filter((c) => c.method === "PUT").at(-1)!.body as Timeline;
    const b = body.tracks[0]!.clips[1]!;
    expect(b.transition_in).toEqual({ type: "crossfade", duration_s: 0.5 });
    expect(b.out_s).toBe(3);
    expect(b.start_s).toBeCloseTo(5.041667 - 0.5);
  });
  expect(screen.getByTestId("timecode")).toHaveTextContent("/ 0:07.13");

  // Delete removes the selected clip (keyboard, Section 17.3).
  (document.activeElement as HTMLElement | null)?.blur();
  fireEvent.keyDown(window, { key: "Delete" });
  await waitFor(
    () => {
      const body = calls.filter((c) => c.method === "PUT").at(-1)!.body as Timeline;
      expect(body.tracks[0]!.clips.map((c) => c.id)).toEqual(["x1"]);
    },
    { timeout: 2000 },
  );
});

test("space plays and pauses; arrows step a frame", async () => {
  const tl = sequence(timeline([clip({ id: "x1", asset_id: "v1", out_s: 5.041667, shot_id: "sh1" })]));
  mockApi(routes({ timeline: tl }));
  renderWithProviders(<TimelineTab projectId={PID} />);
  await screen.findByRole("button", { name: "Clip 1: Shot 1.1, 5.0s" });
  fireEvent.keyDown(window, { key: "ArrowRight" });
  fireEvent.keyDown(window, { key: "ArrowRight" });
  expect(screen.getByTestId("timecode")).toHaveTextContent("0:00.02 /");
  fireEvent.keyDown(window, { key: "ArrowLeft", shiftKey: true });
  expect(screen.getByTestId("timecode")).toHaveTextContent("0:00.00 /");
  fireEvent.keyDown(window, { key: " " });
  expect(await screen.findByRole("button", { name: "Pause" })).toBeInTheDocument();
  fireEvent.keyDown(window, { key: " " });
  expect(await screen.findByRole("button", { name: "Play" })).toBeInTheDocument();
});

test("exports with the chosen preset and lists finished exports", async () => {
  const tl = sequence(timeline([clip({ id: "x1", asset_id: "v1", out_s: 5.041667, shot_id: "sh1" })]));
  const done = asset("e1", { source: "export", width: 1920, height: 1080, duration_s: 5.041667 });
  const calls = mockApi(routes({ timeline: tl, exports: [done] }));
  renderWithProviders(<TimelineTab projectId={PID} />);
  expect(await screen.findByTestId("export-history")).toHaveTextContent("1920×1080 · 5.0s");
  await userEvent.selectOptions(screen.getByLabelText("Export size"), "1080p");
  await userEvent.click(screen.getByLabelText("Normalize loudness"));
  await userEvent.click(screen.getByRole("button", { name: "Export MP4" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/export"))?.body).toEqual({ preset: "1080p", loudnorm: true }),
  );
});
