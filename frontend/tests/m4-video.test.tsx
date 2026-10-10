import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { Asset, Shot } from "@/api/client";
import type { VideoPlan } from "@/api/video";
import { GenerateAllDialog } from "@/features/shots/GenerateAllDialog";
import { segmentsFor, shotVideoJobs } from "@/features/shots/video";
import { VideoSection } from "@/features/shots/VideoSection";
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

function shot(o: Partial<Shot> = {}): Shot {
  return {
    id: "shot-1",
    scene_id: "scene-1",
    index: 0,
    duration_s: 4,
    status: "keyframe_ready",
    generation_mode: "i2v",
    video_prompt: "Old Tom pulls the net taut",
    keyframe_prompt: "Old Tom on the harbor wall",
    ingredient_ids: [],
    needs_extend: false,
    extend_segments: null,
    selected_keyframe_asset_id: "kf-1",
    selected_end_keyframe_asset_id: null,
    selected_video_asset_id: null,
    updated_at: now,
    ...o,
  } as Shot;
}

function plan(o: Partial<VideoPlan> = {}): VideoPlan {
  return {
    mode: "i2v",
    prompt: "Old Tom pulls the net taut. Static camera, a locked-off shot with no camera movement.",
    model_entry_id: "m-wan",
    model_name: "Wan 2.2 TI2V-5B",
    width: 1280,
    height: 704,
    fps: 24,
    num_frames: 121,
    duration_s: 5.042,
    max_clip_seconds: 5,
    note: null,
    ...o,
  };
}

function take(id: string, duration = 5.04): Asset {
  return {
    id,
    project_id: PID,
    type: "video",
    source: "generated",
    url: `/api/files/${PID}/${id}`,
    thumb_url: `/api/thumbs/${PID}/${id}`,
    width: 1280,
    height: 704,
    duration_s: duration,
    fps: 24,
    has_audio: false,
    favorite: false,
    created_at: now,
  } as unknown as Asset;
}

const page = <T,>(items: T[]) => jsonResponse({ items, total: items.length, limit: 50, offset: 0 });

function videoRoutes(o: { takes?: Asset[]; jobs?: ReturnType<typeof makeJob>[]; plan?: VideoPlan } = {}) {
  return {
    "POST /api/shots/shot-1/video-plan": () => jsonResponse(o.plan ?? plan()),
    [`GET /api/projects/${PID}/assets`]: () => page(o.takes ?? []),
    "GET /api/jobs": () => page(o.jobs ?? []),
    "GET /api/jobs/[^/]+": (req: Request) => {
      const id = new URL(req.url).pathname.split("/").pop();
      return jsonResponse((o.jobs ?? []).find((j) => j.id === id) ?? makeJob({ id }));
    },
    "GET /api/models": () => jsonResponse([]),
  } satisfies Record<string, Handler>;
}

// --- units ----------------------------------------------------------------------------------------
test("extend segments: none when one clip covers the shot, else the breakdown's count or enough to fit", () => {
  const p = plan({ duration_s: 5.04, max_clip_seconds: 5 });
  expect(segmentsFor(shot({ duration_s: 4 }), p)).toBe(0);
  expect(segmentsFor(shot({ duration_s: 5 }), p)).toBe(0);
  expect(segmentsFor(shot({ duration_s: 9 }), p)).toBe(1);
  expect(segmentsFor(shot({ duration_s: 14 }), p)).toBe(2);
  expect(segmentsFor(shot({ duration_s: 9, extend_segments: 3 }), p)).toBe(3);
  expect(segmentsFor(shot({ duration_s: 60 }), p)).toBe(4);
  expect(segmentsFor(shot({ duration_s: 9 }), undefined)).toBe(0);
});

test("a shot's video jobs: active ones, and the newest only if it failed", () => {
  const ref = { kind: "shot", shot_id: "shot-1", slot: "video" };
  const jobs = [
    makeJob({ id: "a", status: "running", target_ref: ref, created_at: "2026-01-01T00:00:03Z" }),
    makeJob({ id: "b", status: "queued", target_ref: ref, created_at: "2026-01-01T00:00:02Z" }),
    makeJob({ id: "c", status: "failed", target_ref: ref, created_at: "2026-01-01T00:00:01Z" }),
    makeJob({ id: "d", status: "running", target_ref: { ...ref, slot: "keyframe" } }),
    makeJob({ id: "e", status: "running", target_ref: { ...ref, shot_id: "other" } }),
  ];
  const { active, failed } = shotVideoJobs(jobs, "shot-1");
  expect(active.map((j) => j.id)).toEqual(["a", "b"]);
  expect(failed).toBeUndefined();
  const later = shotVideoJobs(
    [makeJob({ id: "f", status: "failed", target_ref: ref, created_at: "2026-01-01T00:00:09Z" }), ...jobs],
    "shot-1",
  );
  expect(later.failed?.id).toBe("f");
});

// --- VideoSection -----------------------------------------------------------------------------------
test("shows the plan and generates a take, extending a shot longer than one clip", async () => {
  const calls = mockApi({
    ...videoRoutes(),
    "POST /api/shots/shot-1/generate-video": () =>
      jsonResponse({ ...plan(), jobs: [makeJob({ id: "v1", type: "i2v" })], extend_segments: 1 }, 201),
  });
  renderWithProviders(<VideoSection shot={shot({ duration_s: 9, needs_extend: true })} projectId={PID} />);

  expect(await screen.findByTestId("video-plan")).toHaveTextContent(
    "i2v · Wan 2.2 TI2V-5B · 1280×704 · 121 frames @ 24 fps (5.0s)",
  );
  expect(screen.getByText("No take yet")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "prompt" }));
  expect(screen.getByText(/locked-off shot/)).toBeInTheDocument();

  await userEvent.selectOptions(screen.getByLabelText("Camera intensity"), "strong");
  await userEvent.selectOptions(screen.getByLabelText("Steps"), "15");
  await userEvent.click(await screen.findByRole("button", { name: "Generate + extend ×1" }));
  await waitFor(() => expect(calls.some((c) => c.path.endsWith("/generate-video"))).toBe(true));
  const body = calls.find((c) => c.path.endsWith("/generate-video"))!.body;
  expect(body).toMatchObject({
    mode: "auto",
    camera_intensity: "strong",
    variations: 1,
    extend_segments: 1,
    params: { steps: 15 },
  });
  // The plan was re-requested with the new intensity.
  expect(calls.filter((c) => c.path.endsWith("/video-plan")).at(-1)!.body).toMatchObject({
    camera_intensity: "strong",
  });
});

test("saves unsaved panel edits before generating", async () => {
  const order: string[] = [];
  mockApi({
    ...videoRoutes(),
    "POST /api/shots/shot-1/generate-video": () => {
      order.push("generate");
      return jsonResponse({ ...plan(), jobs: [makeJob({ id: "v1" })], extend_segments: 0 }, 201);
    },
  });
  const save = vi.fn(async () => {
    order.push("save");
  });
  renderWithProviders(<VideoSection shot={shot()} projectId={PID} beforeGenerate={save} />);
  await userEvent.click(await screen.findByRole("button", { name: "Generate video" }));
  await waitFor(() => expect(order).toEqual(["save", "generate"]));
});

test("lists takes, plays the selected one, and switches, extends, approves and cancels", async () => {
  const running = makeJob({
    id: "v-run",
    type: "extend",
    status: "running",
    progress: 0.4,
    progress_message: "Segment 1/1 · step 12/30",
    target_ref: { kind: "shot", shot_id: "shot-1", slot: "video" },
  });
  const calls = mockApi({
    ...videoRoutes({ takes: [take("t2"), take("t1", 10.04)], jobs: [running] }),
    "POST /api/shots/shot-1/select": () => jsonResponse(shot({ selected_video_asset_id: "t2" })),
    "POST /api/shots/shot-1/extend": () =>
      jsonResponse({ ...plan(), jobs: [makeJob({ id: "x1", type: "extend" })], extend_segments: 2 }, 201),
    "PATCH /api/shots/shot-1": () => jsonResponse(shot({ status: "approved" })),
    "POST /api/jobs/v-run/cancel": () => jsonResponse({ ...running, status: "cancelled" }),
  });
  renderWithProviders(
    <VideoSection shot={shot({ selected_video_asset_id: "t1", status: "video_ready" })} projectId={PID} />,
  );

  expect(screen.getByTestId("selected-take").getAttribute("src")).toContain(`/api/files/${PID}/t1`);
  expect(await screen.findByRole("button", { name: "Selected take" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(screen.getByText("10s")).toBeInTheDocument(); // take durations
  expect(calls.find((c) => c.path.endsWith("/assets"))!.search).toContain("slot=video");
  expect(calls.find((c) => c.path.endsWith("/assets"))!.search).toContain("type=video");

  await userEvent.click(screen.getByRole("button", { name: "Use this take" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/select"))?.body).toEqual({ slot: "video", asset_id: "t2" }),
  );

  await userEvent.selectOptions(screen.getByLabelText("Extend segments"), "2");
  await userEvent.click(screen.getByRole("button", { name: "Extend take" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/extend"))?.body).toMatchObject({ segments: 2 }),
  );

  await userEvent.click(screen.getByRole("button", { name: "Approve take" }));
  await waitFor(() => expect(calls.find((c) => c.method === "PATCH")?.body).toEqual({ status: "approved" }));

  expect(await screen.findByText("Segment 1/1 · step 12/30")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Cancel video job" }));
  await waitFor(() => expect(calls.some((c) => c.path === "/api/jobs/v-run/cancel")).toBe(true));
});

test("shows the last failure and plan notes", async () => {
  mockApi(
    videoRoutes({
      plan: plan({ mode: "t2v", note: "No keyframe is selected, so this take is text-to-video." }),
      jobs: [
        makeJob({
          id: "bad",
          status: "failed",
          error: "Ran out of GPU memory at 121 frames.",
          target_ref: { kind: "shot", shot_id: "shot-1", slot: "video" },
        }),
      ],
    }),
  );
  renderWithProviders(<VideoSection shot={shot({ selected_keyframe_asset_id: null })} projectId={PID} />);
  expect(await screen.findByText(/this take is text-to-video/)).toBeInTheDocument();
  expect(await screen.findByText(/Last video job failed: Ran out of GPU memory/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Extend take" })).not.toBeInTheDocument();
});

// --- Generate all ------------------------------------------------------------------------------------
test("generate all sends the options and lists skipped shots by number", async () => {
  const calls = mockApi({
    [`POST /api/projects/${PID}/generate-all`]: () =>
      jsonResponse(
        {
          keyframe_jobs: 1,
          video_jobs: 1,
          skipped: [{ shot_id: "s2", reason: "Pick a keyframe, then generate the video." }],
        },
        201,
      ),
  });
  const shots = [
    shot({ id: "s1" }),
    shot({ id: "s2", selected_keyframe_asset_id: null, needs_extend: true }),
    shot({ id: "s3", selected_video_asset_id: "t" }),
  ];
  const onClose = vi.fn();
  renderWithProviders(
    <GenerateAllDialog
      projectId={PID}
      shots={shots}
      numbers={
        new Map([
          ["s1", "1.1"],
          ["s2", "1.2"],
          ["s3", "2.1"],
        ])
      }
      onClose={onClose}
    />,
  );
  expect(screen.getByText(/2 shots without a take, 1 of them without a keyframe/)).toBeInTheDocument();
  expect(screen.getByText("Extend long shots (1)")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("checkbox", { name: /Use each new keyframe right away/ }));
  await userEvent.click(screen.getByRole("button", { name: "Queue" }));
  expect(await screen.findByTestId("generate-all-result")).toHaveTextContent(
    "Queued 1 keyframe job and 1 video job.",
  );
  expect(screen.getByText("Shot 1.2: Pick a keyframe, then generate the video.")).toBeInTheDocument();
  expect(calls[0]!.body).toEqual({
    keyframes: true,
    videos: true,
    auto_approve_keyframes: true,
    extend: true,
    camera_intensity: "normal",
  });
  await userEvent.click(screen.getByRole("button", { name: "Done" }));
  expect(onClose).toHaveBeenCalled();
});
