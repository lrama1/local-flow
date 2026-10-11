import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { Asset } from "@/api/client";
import type { AssetFacets, Collection } from "@/api/library";
import { AssetLibrary } from "@/features/assets/AssetLibrary";
import { nearestPreset, reuseSettings } from "@/features/assets/reuse";
import { PlaygroundPage } from "@/features/playground/PlaygroundPage";
import { useUi } from "@/stores/ui";
import { jsonResponse, renderWithProviders } from "./helpers";

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
  vi.restoreAllMocks();
  useUi.setState({ playgroundPrefill: null, toasts: [] });
});

const PID = "01PROJ00000000000000000001";
const PLAYGROUND = "01PLAY00000000000000000001";
const now = new Date().toISOString();

function asset(o: Partial<Asset> & { id: string }): Asset {
  return {
    project_id: PID,
    type: "image",
    width: 1344,
    height: 768,
    duration_s: null,
    fps: null,
    has_audio: false,
    source: "generated",
    parent_asset_ids: [],
    generation: {},
    tags: [],
    favorite: false,
    created_at: now,
    updated_at: now,
    url: `/api/files/${PID}/${o.id}`,
    thumb_url: `/api/thumbs/${PID}/${o.id}`,
    ...o,
  };
}

const IMG = asset({
  id: "img1",
  generation: {
    job_type: "t2i",
    model_name: "FLUX.1 schnell",
    seed: 77,
    prompt_raw: "@mara on the pier",
    params: {
      prompt: "a woman with red hair on the pier",
      width: 768,
      height: 1344,
      num_images: 2,
      model_entry_id: "m-flux",
      loras: [{ model_entry_id: "l-film", weight: 0.6 }],
    },
  },
});
const VID = asset({
  id: "vid1",
  type: "video",
  width: 1280,
  height: 720,
  duration_s: 5,
  fps: 16,
  generation: {
    job_type: "i2v",
    params: {
      prompt: "waves crash",
      image_asset_id: "img1",
      num_frames: 81,
      fps: 16,
      width: 1280,
      height: 720,
    },
  },
  parent_asset_ids: ["img1"],
});
const UPLOAD = asset({ id: "up1", source: "uploaded", generation: { uploaded_name: "ref.png" } });

test("reuseSettings maps a generation back to Playground settings", () => {
  expect(nearestPreset(768, 1344)).toBe("9:16");
  expect(nearestPreset(1024, 1024)).toBe("1:1");
  expect(nearestPreset(1920, 1080)).toBe("16:9");

  expect(reuseSettings(IMG)).toEqual({
    kind: "t2i",
    form: {
      prompt: "@mara on the pier", // the raw prompt keeps its @mentions
      negative: "",
      seed: "77",
      preset: "9:16",
      numImages: 2,
      modelId: "m-flux",
      loras: [{ id: "l-film", weight: 0.6 }],
    },
  });
  const vid = reuseSettings(VID);
  expect(vid?.kind).toBe("i2v");
  expect(vid?.form).toMatchObject({
    prompt: "waves crash",
    preset: "16:9",
    duration: 5.06,
    fps: 16,
    assets: { image_asset_id: ["img1"] },
  });
  const tts = asset({
    id: "a1",
    type: "audio",
    generation: { job_type: "tts", params: { lines: [{ text: "Hello." }, { text: "Hi!" }] } },
  });
  expect(reuseSettings(tts)?.form.prompt).toBe("Hello.\nHi!");
  // A project's shot take: its @mentions are project keys the Playground can't resolve, and the sent
  // prompt carries the camera move, so the resolved prompt is reused.
  const take = asset({
    id: "t1",
    type: "video",
    generation: {
      job_type: "i2v",
      prompt_raw: "@mara walks the rocks",
      target_ref: { kind: "shot", shot_id: "s1", slot: "video" },
      params: { prompt: "Mara walks the rocks. The camera dollies in.", image_asset_id: "img1" },
    },
  });
  expect(reuseSettings(take)?.form.prompt).toBe("Mara walks the rocks. The camera dollies in.");
  expect(reuseSettings(UPLOAD)).toBeNull();
  expect(
    reuseSettings(asset({ id: "e1", type: "video", source: "export", generation: { job_type: "export" } })),
  ).toBeNull();
});

const FACETS: AssetFacets = {
  total: 3,
  favorites: 1,
  types: { image: 2, video: 1 },
  sources: { generated: 2, uploaded: 1 },
  tags: [{ tag: "pier", count: 1 }],
  models: [{ model: "m-flux", name: "FLUX.1 schnell", count: 1 }],
};
const STORYBOARD: Collection = {
  id: "c1",
  project_id: PID,
  name: "Storyboard",
  description: "",
  asset_count: 1,
  cover_thumb_url: null,
  created_at: now,
  updated_at: now,
};

function libraryApi(extra: Record<string, Handler> = {}) {
  return mockApi({
    [`GET /api/projects/${PID}/assets/facets`]: () => jsonResponse(FACETS),
    [`GET /api/projects/${PID}/collections`]: () => jsonResponse([STORYBOARD]),
    [`GET /api/projects/${PID}/scenes`]: () =>
      jsonResponse({ scenes: [], total_duration_s: 0, shot_count: 0, max_clip_seconds: 5 }),
    [`GET /api/projects/${PID}/ingredients`]: () => jsonResponse([]),
    [`GET /api/projects/${PID}/assets`]: (req) => {
      const q = new URL(req.url).searchParams;
      let items = [IMG, VID, UPLOAD];
      if (q.get("type")) items = items.filter((a) => a.type === q.get("type"));
      if (q.get("collection")) items = items.filter((a) => a.id === "img1");
      return jsonResponse({ items, total: items.length, offset: 0, limit: 60 });
    },
    "GET /api/assets/img1": () => jsonResponse(IMG),
    "GET /api/assets/vid1": () => jsonResponse(VID),
    "GET /api/assets/[^/]+/collections": () => jsonResponse([]),
    ...extra,
  });
}

test("library: filters, collections and bulk actions", async () => {
  const calls = libraryApi({
    [`POST /api/projects/${PID}/assets/bulk`]: () => jsonResponse({ updated: 2, deleted: 0 }),
    "POST /api/collections/c1/assets": () => jsonResponse(STORYBOARD),
    [`POST /api/projects/${PID}/collections`]: () =>
      jsonResponse({ ...STORYBOARD, id: "c2", name: "Looks" }, 201),
  });
  renderWithProviders(<AssetLibrary projectId={PID} />);

  const grid = await screen.findByTestId("asset-grid");
  expect(within(grid).getAllByRole("button", { name: /Open .* asset/ })).toHaveLength(3);
  expect(screen.getByText("3 assets")).toBeInTheDocument();

  // Facet-driven filters go to the server as query params.
  await userEvent.selectOptions(screen.getByLabelText("Type"), "video");
  await waitFor(() =>
    expect(within(screen.getByTestId("asset-grid")).getAllByRole("checkbox")).toHaveLength(1),
  );
  expect(calls.some((c) => c.path.endsWith("/assets") && c.search.includes("type=video"))).toBe(true);
  expect(screen.getByRole("option", { name: "FLUX.1 schnell (1)" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Clear" }));
  await waitFor(() =>
    expect(within(screen.getByTestId("asset-grid")).getAllByRole("checkbox")).toHaveLength(3),
  );

  // Select two, tag them, add them to a collection.
  const [first, second] = within(screen.getByTestId("asset-grid")).getAllByRole("checkbox");
  await userEvent.click(first!);
  await userEvent.click(second!);
  const bar = screen.getByTestId("selection-bar");
  expect(within(bar).getByText("2 selected")).toBeInTheDocument();
  await userEvent.type(within(bar).getByLabelText("Tag selected"), "Hero, night{Enter}");
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/assets/bulk"))?.body).toEqual({
      asset_ids: ["img1", "vid1"],
      add_tags: ["hero", "night"],
    }),
  );
  await userEvent.selectOptions(within(bar).getByLabelText("Add to collection"), "c1");
  await waitFor(() =>
    expect(calls.find((c) => c.path === "/api/collections/c1/assets")?.body).toEqual({
      asset_ids: ["img1", "vid1"],
    }),
  );
  vi.spyOn(window, "confirm").mockReturnValue(true);
  await userEvent.click(within(bar).getByRole("button", { name: "Delete" }));
  await waitFor(() =>
    expect(calls.filter((c) => c.path.endsWith("/assets/bulk")).at(-1)?.body).toEqual({
      asset_ids: ["img1", "vid1"],
      delete: true,
    }),
  );

  // Collections: open one (filters the grid), create another.
  const nav = screen.getByRole("navigation", { name: "Collections" });
  await userEvent.click(within(nav).getByRole("button", { name: /Storyboard/ }));
  await waitFor(() => expect(calls.some((c) => c.search.includes("collection=c1"))).toBe(true));
  await userEvent.click(within(nav).getByRole("button", { name: "New collection" }));
  await userEvent.type(within(nav).getByLabelText("Collection name"), "Looks{Enter}");
  await waitFor(() =>
    expect(
      calls.find((c) => c.method === "POST" && c.path === `/api/projects/${PID}/collections`)?.body,
    ).toEqual({
      name: "Looks",
    }),
  );
});

test("library upload: the button and drag-and-drop send each file", async () => {
  const calls = libraryApi({
    [`POST /api/projects/${PID}/assets/upload`]: () => jsonResponse(UPLOAD, 201),
  });
  renderWithProviders(<AssetLibrary projectId={PID} />);
  await screen.findByTestId("asset-grid");

  const files = [
    new File(["a"], "a.png", { type: "image/png" }),
    new File(["b"], "b.mp4", { type: "video/mp4" }),
  ];
  await userEvent.upload(screen.getByLabelText("Upload files"), files);
  await waitFor(() => expect(calls.filter((c) => c.path.endsWith("/assets/upload"))).toHaveLength(2));
  await waitFor(() => expect(useUi.getState().toasts.map((t) => t.title)).toContain("Uploaded 2 files"));

  const drop = new Event("drop", { bubbles: true, cancelable: true });
  Object.defineProperty(drop, "dataTransfer", {
    value: { types: ["Files"], files: [new File(["c"], "c.wav", { type: "audio/wav" })] },
  });
  screen.getByTestId("asset-library").dispatchEvent(drop);
  await waitFor(() => expect(calls.filter((c) => c.path.endsWith("/assets/upload"))).toHaveLength(3));
});

test("asset drawer: reuse settings, extract a frame, tags", async () => {
  const frame = asset({ id: "f1", source: "extracted", generation: { extracted_from: "vid1", time_s: 0 } });
  const calls = libraryApi({
    "POST /api/assets/vid1/extract-frame": () => jsonResponse(frame, 201),
    "PATCH /api/assets/img1": () => jsonResponse({ ...IMG, tags: ["pier"] }),
  });
  renderWithProviders(<AssetLibrary projectId={PID} />);
  const grid = await screen.findByTestId("asset-grid");

  await userEvent.click(within(grid).getByRole("button", { name: "Open video asset" }));
  let drawer = await screen.findByTestId("asset-drawer");
  expect(within(drawer).getByText("waves crash")).toBeInTheDocument();
  expect(within(drawer).getByRole("button", { name: "Extend in Playground" })).toBeInTheDocument();
  await userEvent.click(within(drawer).getByRole("button", { name: "Extract frame" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/extract-frame"))?.body).toEqual({ time_s: 0 }),
  );
  expect(await within(drawer).findByRole("button", { name: /Open frame at/ })).toBeInTheDocument();

  // The source image is linked from the video ("Made from").
  await userEvent.click(within(drawer).getByRole("button", { name: "Open source asset" }));
  drawer = await screen.findByTestId("asset-drawer");
  expect(await within(drawer).findByText("@mara on the pier")).toBeInTheDocument();
  await userEvent.type(within(drawer).getByLabelText("Tags"), "Pier{Enter}");
  await waitFor(() => expect(calls.find((c) => c.method === "PATCH")?.body).toEqual({ tags: ["pier"] }));

  await userEvent.click(within(drawer).getByRole("button", { name: "Reuse settings" }));
  expect(useUi.getState().playgroundPrefill).toMatchObject({
    kind: "t2i",
    form: { seed: "77", preset: "9:16" },
  });
});

test("the Playground opens with reused settings, including inputs from another project", async () => {
  useUi.getState().setPlaygroundPrefill({
    kind: "i2v",
    form: { prompt: "waves crash", seed: "5", assets: { image_asset_id: ["img1"] } },
  });
  mockApi({
    "GET /api/playground": () => jsonResponse({ project_id: PLAYGROUND }),
    "GET /api/adapters": () => jsonResponse([]),
    "GET /api/models": () => jsonResponse([]),
    [`GET /api/projects/${PLAYGROUND}/assets`]: () =>
      jsonResponse({ items: [], total: 0, offset: 0, limit: 60 }),
    "GET /api/assets/img1": () => jsonResponse(IMG),
  });
  renderWithProviders(<PlaygroundPage />);

  expect(await screen.findByDisplayValue("waves crash")).toBeInTheDocument();
  expect(screen.getByDisplayValue("5")).toBeInTheDocument();
  expect(await screen.findByTestId("foreign-image_asset_id")).toBeInTheDocument();
  await waitFor(() => expect(useUi.getState().playgroundPrefill).toBeNull()); // read once
});
