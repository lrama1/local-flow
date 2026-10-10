import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { Asset, Ingredient, Shot } from "@/api/client";
import { installedFor, type ModelEntry } from "@/api/models";
import { IngredientsTab } from "@/features/ingredients/IngredientsTab";
import { LicenseGate } from "@/features/models/LicenseGate";
import { buildBody, DEFAULT_FORM } from "@/features/playground/kinds";
import { KeyframeSection } from "@/features/shots/KeyframeSection";
import { referenceIngredients } from "@/features/shots/references";
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

function ingredient(o: Partial<Ingredient>): Ingredient {
  return {
    id: "ing-ada",
    project_id: PID,
    kind: "character",
    key: "ada",
    name: "Ada",
    description: "",
    visual_prompt: "a tall woman in a red coat",
    negative_prompt: "",
    character_info: null,
    primary_asset_id: null,
    origin: "bible",
    locked: false,
    created_at: now,
    updated_at: now,
    ...o,
  } as Ingredient;
}

function asset(id: string): Asset {
  return {
    id,
    project_id: PID,
    type: "image",
    source: "generated",
    url: `/api/files/${PID}/${id}`,
    thumb_url: `/api/thumbs/${PID}/${id}`,
    width: 1024,
    height: 1024,
    favorite: false,
    generation: { job_type: "t2i", params: { prompt: "a tall woman" } },
    created_at: now,
  } as unknown as Asset;
}

function model(o: Partial<ModelEntry>): ModelEntry {
  return {
    id: "m-flux",
    name: "FLUX.1 [dev] Q4",
    kind: "checkpoint",
    status: "installed",
    capabilities: ["t2i"],
    adapter_key: "flux",
    license: "FLUX.1 [dev] Non-Commercial License",
    license_url: "https://example.com/license",
    license_acknowledged: false,
    parts: [],
    ...o,
  } as ModelEntry;
}

// --- units ----------------------------------------------------------------------------------------
test("references follow the backend order: characters, props, then the scene location", () => {
  const all = [
    ingredient({ id: "loc", kind: "location", name: "Harbor" }),
    ingredient({ id: "prop", kind: "prop", name: "Lantern" }),
    ingredient({ id: "ada", name: "Ada" }),
    ingredient({ id: "style", kind: "style", name: "Look" }),
  ];
  const refs = referenceIngredients(["prop", "ada", "style"], "loc", all);
  expect(refs.map((r) => r.id)).toEqual(["ada", "prop", "loc"]);
  expect(referenceIngredients(["ada", "loc"], "loc", all).map((r) => r.id)).toEqual(["ada", "loc"]);
});

test("model picker lists installed models for a capability, never LoRAs or components", () => {
  const models = [
    model({ id: "a" }),
    model({ id: "b", status: "available" }),
    model({ id: "c", kind: "lora" }),
    model({ id: "d", kind: "vae" }),
    model({ id: "e", capabilities: ["image_edit"] }),
  ];
  expect(installedFor(models, "t2i").map((m) => m.id)).toEqual(["a"]);
  expect(installedFor(models, "image_edit").map((m) => m.id)).toEqual(["e"]);
});

test("playground image bodies carry the chosen model entry and LoRAs instead of an adapter", () => {
  const built = buildBody(
    "t2i",
    {
      ...DEFAULT_FORM,
      prompt: "a fox",
      adapterKey: "sdxl",
      modelId: "m1",
      loras: [{ id: "l1", weight: 0.8 }],
    },
    PID,
  );
  expect("body" in built && built.body).toMatchObject({
    model_entry_id: "m1",
    loras: [{ model_entry_id: "l1", weight: 0.8 }],
  });
  expect("body" in built && built.body.adapter_key).toBeUndefined();
  // Video kinds pick a model too (M4), but take no LoRAs; audio kinds still pick an adapter.
  const video = buildBody(
    "t2v",
    { ...DEFAULT_FORM, prompt: "x", modelId: "m1", adapterKey: "fake", loras: [{ id: "l1", weight: 1 }] },
    PID,
  );
  expect("body" in video && video.body).toMatchObject({ model_entry_id: "m1" });
  expect("body" in video && video.body.adapter_key).toBeUndefined();
  expect("body" in video && video.body.loras).toBeUndefined();
  const speech = buildBody("tts", { ...DEFAULT_FORM, prompt: "hi", modelId: "m1", adapterKey: "fake" }, PID);
  expect("body" in speech && speech.body).toMatchObject({ adapter_key: "fake" });
  expect("body" in speech && speech.body.model_entry_id).toBeUndefined();
});

// --- ingredients ----------------------------------------------------------------------------------
test("ingredient card generates candidates, shows progress, and stars a primary", async () => {
  const ada = ingredient({ primary_asset_id: "a1" });
  const job = makeJob({
    id: "job-1",
    type: "t2i",
    status: "running",
    progress: 0.5,
    progress_message: "Step 10/20",
  });
  const calls = mockApi({
    [`GET /api/projects/${PID}/ingredients`]: () => jsonResponse([ada]),
    "GET /api/models": () => jsonResponse([model({ license_acknowledged: true })]),
    [`GET /api/projects/${PID}/assets`]: () =>
      jsonResponse({ items: [asset("a1"), asset("a2")], next_cursor: null }),
    "POST /api/ingredients/ing-ada/generate": () => jsonResponse(job),
    "GET /api/jobs/job-1": () => jsonResponse(job),
    "PATCH /api/ingredients/ing-ada/primary": () => jsonResponse({ ...ada, primary_asset_id: "a2" }),
  });
  renderWithProviders(<IngredientsTab projectId={PID} />);

  const card = await screen.findByTestId("ingredient-ada");
  expect(within(card).getByAltText("Ada reference")).toHaveAttribute(
    "src",
    expect.stringContaining(`/api/thumbs/${PID}/a1`),
  );
  await waitFor(() =>
    expect(within(card).getAllByRole("button", { name: /primary image/i })).toHaveLength(2),
  );
  const assetQuery = calls.find((c) => c.path.endsWith("/assets"));
  expect(assetQuery?.search).toContain("ingredient_id=ing-ada");

  await userEvent.selectOptions(screen.getByRole("combobox", { name: "Model" }), "m-flux");
  await userEvent.click(within(card).getByRole("button", { name: /4 more/i }));
  await waitFor(() => expect(calls.some((c) => c.path === "/api/ingredients/ing-ada/generate")).toBe(true));
  expect(calls.find((c) => c.path === "/api/ingredients/ing-ada/generate")?.body).toEqual({
    count: 4,
    model_entry_id: "m-flux",
  });
  expect(await within(card).findByText("Step 10/20")).toBeInTheDocument();

  await userEvent.click(within(card).getByRole("button", { name: "Use as primary image" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path === "/api/ingredients/ing-ada/primary")?.body).toEqual({
      asset_id: "a2",
    }),
  );
});

test("unsaved prompt edits are saved before generating", async () => {
  const ada = ingredient({});
  const calls = mockApi({
    [`GET /api/projects/${PID}/ingredients`]: () => jsonResponse([ada]),
    "GET /api/models": () => jsonResponse([]),
    [`GET /api/projects/${PID}/assets`]: () => jsonResponse({ items: [], next_cursor: null }),
    "PATCH /api/ingredients/ing-ada": () => jsonResponse({ ...ada, visual_prompt: "a short man" }),
    "POST /api/ingredients/ing-ada/generate": () => jsonResponse(makeJob({ id: "job-2", type: "t2i" })),
    "GET /api/jobs/job-2": () => jsonResponse(makeJob({ id: "job-2", type: "t2i" })),
  });
  renderWithProviders(<IngredientsTab projectId={PID} />);
  const card = await screen.findByTestId("ingredient-ada");
  const prompt = within(card).getByRole("textbox", { name: "Visual prompt" });
  await userEvent.clear(prompt);
  await userEvent.type(prompt, "a short man");
  await userEvent.click(within(card).getByRole("button", { name: /generate 4/i }));
  await waitFor(() => expect(calls.some((c) => c.path.endsWith("/generate"))).toBe(true));
  const order = calls.filter((c) => c.method !== "GET").map((c) => `${c.method} ${c.path}`);
  expect(order).toEqual(["PATCH /api/ingredients/ing-ada", "POST /api/ingredients/ing-ada/generate"]);
  expect(calls[calls.findIndex((c) => c.method === "PATCH")]?.body).toMatchObject({
    visual_prompt: "a short man",
  });
});

test("a license refusal opens the license gate and retries after acknowledgement", async () => {
  const ada = ingredient({});
  let attempts = 0;
  const calls = mockApi({
    [`GET /api/projects/${PID}/ingredients`]: () => jsonResponse([ada]),
    "GET /api/models": () => jsonResponse([model({})]),
    [`GET /api/projects/${PID}/assets`]: () => jsonResponse({ items: [], next_cursor: null }),
    "POST /api/ingredients/ing-ada/generate": () =>
      ++attempts === 1
        ? jsonResponse(
            {
              error: {
                code: "license_not_acknowledged",
                message: "Acknowledge the license first.",
                details: { model_entry_id: "m-flux" },
              },
            },
            409,
          )
        : jsonResponse(makeJob({ id: "job-3", type: "t2i" })),
    "POST /api/models/m-flux/acknowledge-license": () => jsonResponse(model({ license_acknowledged: true })),
    "GET /api/jobs/job-3": () => jsonResponse(makeJob({ id: "job-3", type: "t2i" })),
  });
  renderWithProviders(
    <>
      <IngredientsTab projectId={PID} />
      <LicenseGate />
    </>,
  );
  const card = await screen.findByTestId("ingredient-ada");
  await userEvent.click(within(card).getByRole("button", { name: /generate 4/i }));

  const dialog = await screen.findByRole("dialog", { name: /license: flux/i });
  expect(within(card).queryByText(/acknowledge the license first/i)).not.toBeInTheDocument();
  await userEvent.click(within(dialog).getByRole("checkbox", { name: /reviewed this license/i }));
  await userEvent.click(within(dialog).getByRole("button", { name: "Acknowledge" }));

  await waitFor(() => expect(attempts).toBe(2));
  expect(calls.map((c) => c.path)).toContain("/api/models/m-flux/acknowledge-license");
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
});

// --- keyframes ------------------------------------------------------------------------------------
function shot(o: Partial<Shot> = {}): Shot {
  return {
    id: "shot-1",
    scene_id: "scene-1",
    index: 0,
    status: "planned",
    generation_mode: "i2v",
    keyframe_prompt: "Ada lifts the lantern on the pier",
    ingredient_ids: ["prop", "ada"],
    selected_keyframe_asset_id: null,
    selected_end_keyframe_asset_id: null,
    updated_at: now,
    ...o,
  } as Shot;
}

const cast = [
  ingredient({ id: "ada", name: "Ada", primary_asset_id: "p-ada" }),
  ingredient({ id: "prop", kind: "prop", name: "Lantern", primary_asset_id: "p-lantern" }),
  ingredient({ id: "loc", kind: "location", name: "Harbor" }),
];

const ref = (asset_id: string, ingredient_id: string, name: string, kind: string, view = "primary") => ({
  asset_id,
  ingredient_id,
  name,
  kind,
  view,
  label: name,
});
const REFS = {
  automatic: [ref("p-ada", "ada", "Ada", "character"), ref("p-lantern", "prop", "Lantern", "prop")],
  selected: [ref("p-ada", "ada", "Ada", "character"), ref("p-lantern", "prop", "Lantern", "prop")],
  overridden: false,
  max: 3,
  options: [
    {
      ingredient_id: "ada",
      name: "Ada",
      kind: "character",
      images: [{ asset_id: "p-ada", view: "primary" }],
    },
    {
      ingredient_id: "prop",
      name: "Lantern",
      kind: "prop",
      images: [{ asset_id: "p-lantern", view: "primary" }],
    },
  ],
};

test("keyframe section composes from references and selects a candidate", async () => {
  const out = {
    job: makeJob({ id: "job-k", type: "image_edit" }),
    mode: "edit",
    prompt: "…",
    reference_asset_ids: ["p-ada", "p-lantern"],
    reference_names: ["Ada", "Lantern"],
    note: null,
  };
  const calls = mockApi({
    "GET /api/models": () =>
      jsonResponse([model({ id: "m-kontext", name: "Kontext", capabilities: ["image_edit"] })]),
    [`GET /api/projects/${PID}/assets`]: () =>
      jsonResponse({ items: [asset("k1"), asset("k2")], next_cursor: null }),
    "POST /api/shots/shot-1/generate-keyframe": () => jsonResponse(out),
    "GET /api/jobs/job-k": () => jsonResponse(out.job),
    "POST /api/shots/shot-1/select": () => jsonResponse(shot({ selected_keyframe_asset_id: "k2" })),
    "GET /api/shots/shot-1/references": () => jsonResponse(REFS),
  });
  const onRun = vi.fn();
  renderWithProviders(
    <KeyframeSection
      shot={shot()}
      projectId={PID}
      ingredients={cast}
      ingredientIds={["prop", "ada"]}
      locationId="loc"
      runs={{}}
      onRun={onRun}
    />,
  );

  const refs = screen.getByTestId("keyframe-references");
  expect((await within(refs).findAllByText(/Ada|Lantern|Harbor/)).map((e) => e.textContent)).toEqual([
    "Ada",
    "Lantern",
    "Harbor",
  ]);
  expect(within(refs).getByTitle(/Harbor: no primary image/)).toBeInTheDocument();
  const refQuery = calls.find((c) => c.path === "/api/shots/shot-1/references")?.search ?? "";
  expect(refQuery).toContain("ingredient_ids=prop&ingredient_ids=ada");

  await userEvent.click(screen.getByRole("button", { name: /compose keyframe/i }));
  await waitFor(() => expect(onRun).toHaveBeenCalled());
  expect(calls.find((c) => c.path.endsWith("/generate-keyframe"))?.body).toEqual({
    which: "start",
    mode: "auto",
    count: 2,
    model_entry_id: null,
  });
  expect(onRun.mock.calls[0]![0]).toMatchObject({ which: "start", jobId: "job-k" });

  const grid = await screen.findByTestId("keyframe-candidates");
  const options = within(grid).getAllByRole("button", { name: "Use as keyframe" });
  expect(options).toHaveLength(2);
  expect(calls.find((c) => c.path.endsWith("/assets"))?.search).toMatch(
    /shot_id=shot-1.*slot=keyframe|slot=keyframe.*shot_id=shot-1/,
  );
  await userEvent.click(options[1]!);
  await waitFor(() =>
    expect(calls.find((c) => c.path === "/api/shots/shot-1/select")?.body).toEqual({
      slot: "keyframe",
      asset_id: "k2",
    }),
  );
});

test("first/last-frame shots pick the end keyframe separately and show fallback notes", async () => {
  const calls = mockApi({
    "GET /api/models": () => jsonResponse([]),
    [`GET /api/projects/${PID}/assets`]: () => jsonResponse({ items: [], next_cursor: null }),
    "GET /api/shots/shot-1/references": () =>
      jsonResponse({ ...REFS, automatic: [], selected: [], options: [] }),
  });
  renderWithProviders(
    <KeyframeSection
      shot={shot({ generation_mode: "flf2v", selected_keyframe_asset_id: "k1" })}
      projectId={PID}
      ingredients={cast}
      ingredientIds={[]}
      locationId={null}
      runs={{
        end: {
          which: "end",
          jobId: "job-x",
          out: {
            job: makeJob({ id: "job-x" }),
            mode: "t2i",
            prompt: "",
            reference_asset_ids: [],
            reference_names: [],
            note: "No image-edit model is installed, so the keyframe uses text-to-image without references.",
          },
        },
      }}
      onRun={vi.fn()}
    />,
  );
  expect(screen.getByAltText("Selected keyframe")).toBeInTheDocument();
  expect(screen.queryByText(/no image-edit model is installed/i)).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Last frame" }));
  expect(screen.getByText(/no image-edit model is installed/i)).toBeInTheDocument();
  expect(screen.getByText("No keyframe selected")).toBeInTheDocument();
  await waitFor(() => expect(calls.some((c) => c.search.includes("slot=end_keyframe"))).toBe(true));
});
