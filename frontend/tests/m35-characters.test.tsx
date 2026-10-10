import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { vi } from "vitest";
import type { Ingredient, Shot } from "@/api/client";
import { handleServerEvent } from "@/api/events";
import { MentionTextarea } from "@/components/MentionTextarea";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { CastLibraryPage } from "@/features/cast-library/CastLibraryPage";
import { CharacterWizard } from "@/features/characters/CharacterWizard";
import { IngredientsTab } from "@/features/ingredients/IngredientsTab";
import { KeyframeSection } from "@/features/shots/KeyframeSection";
import {
  insertMention,
  matchMentions,
  mentionQuery,
  parseMentions,
  unknownMentions,
  type MentionOption,
} from "@/lib/mentions";
import { useUi } from "@/stores/ui";
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

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  useUi.setState({ toasts: [] });
});

const PID = "01PROJ00000000000000000001";
const PLAYGROUND = "01PLAY00000000000000000001";
const now = new Date().toISOString();

function ingredient(o: Partial<Ingredient>): Ingredient {
  return {
    id: "ing-mara",
    project_id: PID,
    kind: "character",
    key: "mara",
    name: "Mara",
    description: "",
    visual_prompt: "a woman with silver hair",
    negative_prompt: "",
    character_info: null,
    voice_id: null,
    primary_asset_id: null,
    lora_ids: [],
    origin: "user",
    locked: true,
    views: {},
    voice_hints: {},
    draft: false,
    created_at: now,
    updated_at: now,
    ...o,
  } as Ingredient;
}

const OPTIONS: MentionOption[] = [
  { key: "market", name: "Fish market", kind: "location" },
  { key: "mara", name: "Mara", kind: "character" },
  { key: "lantern", name: "Brass lantern", kind: "prop" },
  { key: "old-tom", name: "Old Tom", kind: "character" },
];

// --- mention units ---------------------------------------------------------------------------------
test("mentions parse like the backend: @@ is literal and emails are not mentions", () => {
  expect(parseMentions("@Mara meets @old-tom (@@dawn) mail a@b.com @x-")).toEqual([
    { key: "mara", start: 0, end: 5 },
    { key: "old-tom", start: 12, end: 20 },
    { key: "x", start: 43, end: 45 }, // a trailing hyphen isn't part of a key
  ]);
  expect(unknownMentions("@mara and @ghost and @Ghost", ["mara"])).toEqual(["ghost"]);
  expect(unknownMentions("@mara and @gh", ["mara"], 13)).toEqual([]); // still being typed
});

test("autocomplete matches characters first, then locations and props", () => {
  expect(mentionQuery("walk with @ma", 13)).toEqual({ start: 10, query: "ma" });
  expect(mentionQuery("mail a@ma", 9)).toBeNull();
  expect(mentionQuery("@@ma", 4)).toBeNull();
  expect(matchMentions(OPTIONS, "ma").map((o) => o.key)).toEqual(["mara", "market"]);
  expect(matchMentions(OPTIONS, "").map((o) => o.kind)).toEqual([
    "character",
    "character",
    "location",
    "prop",
  ]);
  expect(matchMentions(OPTIONS, "tom").map((o) => o.key)).toEqual(["old-tom"]);
  expect(insertMention("hi @ma there", 3, 6, "mara")).toEqual({ text: "hi @mara there", caret: 8 });
  expect(insertMention("hi @ma", 3, 6, "mara")).toEqual({ text: "hi @mara ", caret: 9 });
});

function MentionHarness({ initial = "" }: { initial?: string }) {
  const [v, setV] = useState(initial);
  return (
    <>
      <MentionTextarea aria-label="Prompt" value={v} onChange={setV} options={OPTIONS} />
      <output data-testid="value">{v}</output>
    </>
  );
}

test("typing @ opens suggestions; Enter inserts the key; unknown mentions warn", async () => {
  renderWithProviders(<MentionHarness />);
  const box = screen.getByRole("textbox", { name: "Prompt" });
  await userEvent.type(box, "Walk with @ma");
  const list = screen.getByRole("listbox", { name: "Mention suggestions" });
  const opts = within(list).getAllByRole("option");
  expect(opts.map((o) => o.textContent)).toEqual([
    expect.stringContaining("@mara"),
    expect.stringContaining("@market"),
  ]);
  expect(opts[0]).toHaveAttribute("aria-selected", "true");
  await userEvent.keyboard("{ArrowDown}{Enter}");
  expect(screen.getByTestId("value")).toHaveTextContent("Walk with @market");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();

  await userEvent.type(box, "and @ghost ");
  expect(screen.getByRole("alert")).toHaveTextContent("Unknown mention @ghost");
  await userEvent.type(box, "@o");
  await userEvent.click(within(screen.getByRole("listbox")).getByText("Old Tom"));
  expect(screen.getByTestId("value")).toHaveTextContent("Walk with @market and @ghost @old-tom");
  await userEvent.type(box, " @l");
  await userEvent.keyboard("{Escape}");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
});

test("Escape closes the suggestions without closing the surrounding dialog", async () => {
  function InDialog() {
    const [open, setOpen] = useState(true);
    return (
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent aria-describedby={undefined}>
          <DialogTitle>Shot</DialogTitle>
          <MentionHarness />
        </DialogContent>
      </Dialog>
    );
  }
  renderWithProviders(<InDialog />);
  await userEvent.type(screen.getByRole("textbox", { name: "Prompt" }), "@ma");
  expect(screen.getByRole("listbox")).toBeInTheDocument();
  await userEvent.keyboard("{Escape}");
  expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  await userEvent.keyboard("{Escape}");
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
});

// --- wizard ----------------------------------------------------------------------------------------
const FISHER = {
  id: null,
  key: "weathered-fisherman",
  name: "Weathered fisherman",
  category: "realistic",
  builtin: true,
  visual_prompt: "an elderly fisherman with a white beard, yellow oilskin coat",
  negative_prompt: "",
  character_info: "Taciturn; reads the weather.",
  body_outfit_prompt: "yellow oilskin coat, rubber boots",
  voice_hints: { gender: "male", age: "elderly", tone: "low, gravelly and slow" },
  thumbnail_url: "/api/character-templates/weathered-fisherman/thumbnail",
};
const CAPS = { expand: true, describe: true, edit_model: false, voices: false };

test("template wizard: portraits, skip views, name, voice, save to the project and library", async () => {
  let draft = ingredient({ id: "d1", key: "draft-abc", name: "Weathered fisherman", draft: true });
  const job = makeJob({ id: "job-p", type: "t2i", status: "running", progress: 0.3 });
  const calls = mockApi({
    "GET /api/character-templates": () => jsonResponse([FISHER]),
    "GET /api/characters/capabilities": () => jsonResponse(CAPS),
    "GET /api/models": () => jsonResponse([]),
    "GET /api/playground": () => jsonResponse({ project_id: PLAYGROUND }),
    [`POST /api/projects/${PID}/characters`]: () => jsonResponse(draft, 201),
    "PATCH /api/ingredients/d1": async (req) => {
      draft = { ...draft, ...((await req.json()) as Partial<Ingredient>) };
      return jsonResponse(draft);
    },
    "GET /api/ingredients/d1": () => jsonResponse(draft),
    "POST /api/ingredients/d1/generate": () => jsonResponse(job, 201),
    "GET /api/jobs/job-p": () => jsonResponse(job),
    [`GET /api/projects/${PID}/assets`]: () =>
      jsonResponse({
        items: draft.primary_asset_id
          ? [{ id: "a1", project_id: PID, type: "image", thumb_url: `/api/thumbs/${PID}/a1` }]
          : [],
        next_cursor: null,
      }),
    "POST /api/ingredients/d1/save-to-library": () => jsonResponse({ id: "lib-1", key: "old-tom" }, 201),
  });
  const onOpenChange = vi.fn();
  const onSaved = vi.fn();
  const qc = makeQueryClient();
  renderWithProviders(
    <CharacterWizard projectId={PID} open onOpenChange={onOpenChange} onSaved={onSaved} />,
    qc,
  );

  const wizard = await screen.findByTestId("character-wizard");
  await userEvent.click(await within(wizard).findByRole("option", { name: /Weathered fisherman/ }));
  expect(within(wizard).getByLabelText(/Visual prompt/)).toHaveValue(FISHER.visual_prompt);
  const next = within(wizard).getByRole("button", { name: "Next" });
  expect(next).toBeDisabled();

  await userEvent.click(within(wizard).getByRole("button", { name: "Generate 4 portraits" }));
  await waitFor(() => expect(calls.some((c) => c.path === "/api/ingredients/d1/generate")).toBe(true));
  expect(calls.find((c) => c.path === `/api/projects/${PID}/characters`)?.body).toMatchObject({
    source: "template",
    template_key: "weathered-fisherman",
    fields: { visual_prompt: FISHER.visual_prompt, voice_hints: FISHER.voice_hints },
  });
  expect(calls.find((c) => c.path.endsWith("/generate"))?.body).toEqual({ count: 4, model_entry_id: null });
  expect(await within(wizard).findByText("30%", { exact: false })).toBeInTheDocument();

  // The job finishes: the backend auto-selects the first candidate and announces the change.
  draft = { ...draft, primary_asset_id: "a1", views: { primary: "a1" } };
  handleServerEvent(qc, { type: "job.completed", job: { ...job, status: "completed", project_id: PID } });
  handleServerEvent(qc, { type: "ingredients.updated", project_id: PID });
  await waitFor(() => expect(next).toBeEnabled());
  expect(await within(wizard).findByRole("button", { name: "Chosen portrait image" })).toBeInTheDocument();

  await userEvent.click(next);
  expect(
    await within(wizard).findByText(/views use text-to-image \(lower consistency\)/),
  ).toBeInTheDocument();
  expect(within(wizard).getByLabelText(/Body and outfit/)).toHaveValue("");
  await userEvent.click(within(wizard).getByRole("button", { name: "Skip" }));

  const name = within(wizard).getByLabelText("Name");
  expect(name).toHaveValue("Weathered fisherman");
  expect(within(wizard).getByLabelText(/^Key/)).toHaveValue("weathered-fisherman");
  await userEvent.clear(name);
  await userEvent.type(name, "Old Tom");
  expect(within(wizard).getByLabelText(/^Key/)).toHaveValue("old-tom");
  await userEvent.click(within(wizard).getByRole("button", { name: "Next" }));

  expect(within(wizard).getByText(/suggests a matching voice/)).toBeInTheDocument();
  expect(await within(wizard).findByTestId("no-voices")).toBeInTheDocument(); // no TTS model here
  expect(within(wizard).getByLabelText("gender")).toHaveValue("male");
  await userEvent.click(within(wizard).getByRole("button", { name: "Next" }));

  await userEvent.click(within(wizard).getByRole("checkbox", { name: /Also save to the Cast library/ }));
  await userEvent.click(within(wizard).getByRole("button", { name: "Save character" }));
  await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
  const patches = calls.filter((c) => c.method === "PATCH").map((c) => c.body);
  expect(patches.at(-1)).toMatchObject({
    name: "Old Tom",
    key: "old-tom",
    draft: false,
    character_info: "Taciturn; reads the weather.",
    voice_hints: FISHER.voice_hints,
    template_key: "weathered-fisherman",
  });
  expect(calls.some((c) => c.path === "/api/ingredients/d1/save-to-library")).toBe(true);
  expect(calls.some((c) => c.method === "DELETE")).toBe(false);
  expect(onSaved).toHaveBeenCalledWith(expect.objectContaining({ library: { id: "lib-1", key: "old-tom" } }));
});

test("expand with AI fills the sheet; cancelling deletes the draft and its images", async () => {
  const draft = ingredient({ id: "d2", key: "draft-xyz", draft: true, primary_asset_id: "a1" });
  const calls = mockApi({
    "GET /api/character-templates": () => jsonResponse([]),
    "GET /api/characters/capabilities": () => jsonResponse(CAPS),
    "GET /api/models": () => jsonResponse([]),
    "GET /api/playground": () => jsonResponse({ project_id: PLAYGROUND }),
    "POST /api/characters/expand": () =>
      jsonResponse({
        name: "Mara",
        visual_prompt: "a woman in her sixties, short silver hair",
        body_outfit_prompt: "navy peacoat",
        negative_prompt: "young",
        character_info: "Dry humor.",
        voice_hints: { gender: "female", age: "60s", tone: "warm" },
        usage: {},
      }),
    [`POST /api/projects/${PID}/characters`]: () => jsonResponse(draft, 201),
    "POST /api/ingredients/d2/generate": () => jsonResponse(makeJob({ id: "job-2", type: "t2i" }), 201),
    "GET /api/jobs/job-2": () => jsonResponse(makeJob({ id: "job-2", type: "t2i" })),
    "GET /api/ingredients/d2": () => jsonResponse(draft),
    [`GET /api/projects/${PID}/assets`]: () => jsonResponse({ items: [], next_cursor: null }),
    "DELETE /api/ingredients/d2": () => new Response(null, { status: 204 }),
  });
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  const onOpenChange = vi.fn();
  renderWithProviders(<CharacterWizard projectId={PID} open onOpenChange={onOpenChange} />);
  const wizard = await screen.findByTestId("character-wizard");
  await userEvent.click(within(wizard).getByRole("tab", { name: "Describe" }));
  await userEvent.type(within(wizard).getByLabelText(/Describe the character/), "an old harbor keeper");
  await userEvent.click(within(wizard).getByRole("button", { name: "Expand with AI" }));
  await waitFor(() =>
    expect(within(wizard).getByLabelText(/Visual prompt/)).toHaveValue(
      "a woman in her sixties, short silver hair",
    ),
  );
  expect(calls.find((c) => c.path === "/api/characters/expand")).toMatchObject({
    search: `?project_id=${PID}`,
    body: { description: "an old harbor keeper", template_key: null },
  });

  await userEvent.click(within(wizard).getByRole("button", { name: /Generate 4 portraits/ }));
  await waitFor(() => expect(calls.some((c) => c.path === "/api/ingredients/d2/generate")).toBe(true));
  expect(calls.find((c) => c.path === `/api/projects/${PID}/characters`)?.body).toMatchObject({
    source: "prompt",
    fields: { name: "Mara", negative_prompt: "young", voice_hints: { tone: "warm" } },
  });
  await userEvent.click(within(wizard).getByRole("button", { name: "Cancel" }));
  expect(confirm).toHaveBeenCalled();
  await waitFor(() =>
    expect(calls.some((c) => c.method === "DELETE" && c.path === "/api/ingredients/d2")).toBe(true),
  );
  expect(onOpenChange).toHaveBeenCalledWith(false);
});

test("library mode works on a Playground draft, saves to the library, then removes the draft", async () => {
  const draft = ingredient({
    id: "d3",
    project_id: PLAYGROUND,
    key: "draft-q",
    draft: true,
    primary_asset_id: "a1",
  });
  const calls = mockApi({
    "GET /api/character-templates": () => jsonResponse([FISHER]),
    "GET /api/characters/capabilities": () => jsonResponse(CAPS),
    "GET /api/models": () => jsonResponse([]),
    "GET /api/playground": () => jsonResponse({ project_id: PLAYGROUND }),
    [`POST /api/projects/${PLAYGROUND}/characters`]: () => jsonResponse(draft, 201),
    "POST /api/ingredients/d3/generate": () => jsonResponse(makeJob({ id: "job-3", type: "t2i" }), 201),
    "GET /api/jobs/job-3": () => jsonResponse(makeJob({ id: "job-3", type: "t2i" })),
    "GET /api/ingredients/d3": () => jsonResponse(draft),
    "PATCH /api/ingredients/d3": () => jsonResponse(draft),
    [`GET /api/projects/${PLAYGROUND}/assets`]: () => jsonResponse({ items: [], next_cursor: null }),
    "POST /api/ingredients/d3/save-to-library": () =>
      jsonResponse({ id: "lib-9", key: "weathered-fisherman" }, 201),
    "DELETE /api/ingredients/d3": () => new Response(null, { status: 204 }),
  });
  const onOpenChange = vi.fn();
  renderWithProviders(<CharacterWizard open onOpenChange={onOpenChange} />);
  const wizard = await screen.findByTestId("character-wizard");
  expect(within(wizard).getByText("New Cast library character")).toBeInTheDocument();
  expect(within(wizard).queryByRole("tab", { name: "Import" })).not.toBeInTheDocument();
  await userEvent.click(await within(wizard).findByRole("option", { name: /Weathered fisherman/ }));
  await waitFor(() =>
    expect(within(wizard).getByRole("button", { name: /Generate 4 portraits/ })).toBeEnabled(),
  );
  await userEvent.click(within(wizard).getByRole("button", { name: /Generate 4 portraits/ }));
  await waitFor(() => expect(within(wizard).getByRole("button", { name: "Next" })).toBeEnabled());
  for (let i = 0; i < 4; i++) await userEvent.click(within(wizard).getByRole("button", { name: "Next" }));
  expect(within(wizard).queryByRole("checkbox")).not.toBeInTheDocument();
  await userEvent.click(within(wizard).getByRole("button", { name: "Save to Cast library" }));
  await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
  const order = calls.filter((c) => c.method !== "GET").map((c) => `${c.method} ${c.path}`);
  expect(order.slice(-3)).toEqual([
    "PATCH /api/ingredients/d3",
    "POST /api/ingredients/d3/save-to-library",
    "DELETE /api/ingredients/d3",
  ]);
  expect(calls.filter((c) => c.method === "PATCH").at(-1)?.body).not.toHaveProperty("draft");
});

// --- Cast library ----------------------------------------------------------------------------------
const LIB = {
  id: "lib-1",
  key: "mara",
  name: "Mara",
  description: "",
  visual_prompt: "a woman with silver hair",
  negative_prompt: "",
  character_info: "Dry humor.",
  body_outfit_prompt: "navy peacoat",
  voice_id: null,
  voice_hints: { gender: "female", age: "60s", tone: "warm" },
  lora_ids: [],
  template_key: null,
  source_project_id: PID,
  images: [
    {
      n: 0,
      view: "primary",
      is_primary: true,
      url: "/api/library/characters/lib-1/images/0",
      thumb_url: "/api/library/characters/lib-1/images/0?thumb=1",
    },
    {
      n: 1,
      view: "full_body",
      is_primary: false,
      url: "/api/library/characters/lib-1/images/1",
      thumb_url: "/api/library/characters/lib-1/images/1?thumb=1",
    },
  ],
  created_at: now,
  updated_at: now,
};

test("Cast library page: search, edit, cast into a project", async () => {
  const calls = mockApi({
    "GET /api/library/characters": () => jsonResponse([LIB]),
    "GET /api/projects": () => jsonResponse([{ id: PID, name: "Harbor" }]),
    "PATCH /api/library/characters/lib-1": () => jsonResponse({ ...LIB, character_info: "Grumpier." }),
    [`POST /api/projects/${PID}/ingredients/import`]: () => jsonResponse(ingredient({ key: "mara-2" }), 201),
  });
  renderWithProviders(<CastLibraryPage />);
  const card = await screen.findByTestId("library-mara");
  expect(within(card).getByText("@mara")).toBeInTheDocument();
  expect(within(card).getByText("1 view")).toBeInTheDocument();
  expect(within(card).getByAltText("Mara")).toHaveAttribute(
    "src",
    expect.stringMatching(/images\/0\?thumb=1&token=/),
  );

  await userEvent.type(screen.getByRole("textbox", { name: "Search the Cast library" }), "silver");
  await waitFor(() => expect(calls.some((c) => c.search === "?q=silver")).toBe(true));

  await userEvent.click(within(card).getByRole("button", { name: "Open Mara" }));
  const dialog = await screen.findByRole("dialog", { name: "Mara" });
  const info = within(dialog).getByLabelText("Character info");
  await userEvent.clear(info);
  await userEvent.type(info, "Grumpier.");
  await userEvent.click(within(dialog).getByRole("button", { name: "Save changes" }));
  await waitFor(() =>
    expect(calls.find((c) => c.method === "PATCH")?.body).toMatchObject({
      character_info: "Grumpier.",
      voice_hints: { tone: "warm" },
    }),
  );
  await userEvent.selectOptions(within(dialog).getByRole("combobox", { name: "Project to cast into" }), PID);
  await userEvent.click(within(dialog).getByRole("button", { name: "Cast" }));
  await waitFor(() =>
    expect(calls.find((c) => c.path.endsWith("/ingredients/import"))?.body).toEqual({
      library_character_id: "lib-1",
    }),
  );
  await waitFor(() => expect(useUi.getState().toasts.at(-1)?.title).toBe("Mara cast into Harbor as @mara-2"));
});

test("character cards show views and offer Update from library when the library changed", async () => {
  const mara = ingredient({
    primary_asset_id: "p1",
    views: { primary: "p1", full_body: "fb1" },
    library_character_id: "lib-1",
    library_status: "update_available",
  });
  const calls = mockApi({
    [`GET /api/projects/${PID}/ingredients`]: () => jsonResponse([mara]),
    "GET /api/models": () => jsonResponse([]),
    [`GET /api/projects/${PID}/assets`]: () => jsonResponse({ items: [], next_cursor: null }),
    "POST /api/ingredients/ing-mara/update-from-library": () =>
      jsonResponse({ ...mara, library_status: "synced" }),
  });
  vi.spyOn(window, "confirm").mockReturnValue(true);
  renderWithProviders(<IngredientsTab projectId={PID} />);
  const extras = await screen.findByTestId("character-extras-mara");
  expect(within(extras).getByAltText("Mara full body")).toHaveAttribute(
    "src",
    expect.stringContaining(`/api/thumbs/${PID}/fb1`),
  );
  expect(within(extras).getByText("Library has updates")).toBeInTheDocument();
  expect(within(extras).queryByRole("button", { name: /Save to library/ })).not.toBeInTheDocument();
  await userEvent.click(within(extras).getByRole("button", { name: "Update from library" }));
  await waitFor(() => expect(calls.some((c) => c.path.endsWith("/update-from-library"))).toBe(true));
  expect(calls.find((c) => c.path.endsWith("/assets"))?.search).toContain("view=primary");
  expect(screen.getByRole("button", { name: "New character" })).toBeInTheDocument();
});

test("library.updated refreshes the library and every project's ingredients", () => {
  const qc = makeQueryClient();
  const spy = vi.spyOn(qc, "invalidateQueries");
  handleServerEvent(qc, { type: "library.updated", library_character_id: "lib-1" });
  expect(spy).toHaveBeenCalledWith({ queryKey: ["library"] });
  expect(spy).toHaveBeenCalledWith({ queryKey: ["ingredients"] });
});

// --- reference override ----------------------------------------------------------------------------
test("the shot's references can be chosen by hand and reset to automatic", async () => {
  const shot = {
    id: "shot-1",
    scene_id: "s1",
    index: 0,
    status: "planned",
    generation_mode: "i2v",
    shot_type: "wide",
    keyframe_prompt: "@mara on the pier",
    ingredient_ids: ["ing-mara"],
    selected_keyframe_asset_id: null,
    selected_end_keyframe_asset_id: null,
    updated_at: now,
  } as unknown as Shot;
  const refs = {
    automatic: [
      {
        asset_id: "fb1",
        ingredient_id: "ing-mara",
        name: "Mara",
        kind: "character",
        view: "full_body",
        label: "Mara (full body)",
      },
    ],
    selected: [
      {
        asset_id: "fb1",
        ingredient_id: "ing-mara",
        name: "Mara",
        kind: "character",
        view: "full_body",
        label: "Mara (full body)",
      },
    ],
    overridden: false,
    max: 3,
    options: [
      {
        ingredient_id: "ing-mara",
        name: "Mara",
        kind: "character",
        images: [
          { asset_id: "p1", view: "primary" },
          { asset_id: "fb1", view: "full_body" },
        ],
      },
    ],
  };
  const calls = mockApi({
    "GET /api/models": () => jsonResponse([]),
    [`GET /api/projects/${PID}/assets`]: () => jsonResponse({ items: [], next_cursor: null }),
    "GET /api/shots/shot-1/references": () => jsonResponse(refs),
  });
  const onSave = vi.fn().mockResolvedValue(undefined);
  renderWithProviders(
    <KeyframeSection
      shot={shot}
      projectId={PID}
      ingredients={[ingredient({ id: "ing-mara", primary_asset_id: "p1" })]}
      ingredientIds={["ing-mara"]}
      locationId={null}
      runs={{}}
      onRun={vi.fn()}
      shotType="close"
      onSaveReferences={onSave}
    />,
  );
  expect(await screen.findByText("Mara (full body)")).toBeInTheDocument();
  expect(calls.find((c) => c.path.endsWith("/references"))?.search).toContain("shot_type=close");

  await userEvent.click(screen.getByRole("button", { name: "Choose" }));
  const picker = screen.getByTestId("reference-picker");
  expect(within(picker).getByRole("button", { name: "Mara (full body)" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await userEvent.click(within(picker).getByRole("button", { name: "Mara (full body)" }));
  await userEvent.click(within(picker).getByRole("button", { name: "Mara (portrait)" }));
  await userEvent.click(within(picker).getByRole("button", { name: "Use these" }));
  expect(onSave).toHaveBeenCalledWith(["p1"]);
  await waitFor(() => expect(screen.queryByTestId("reference-picker")).not.toBeInTheDocument());

  await userEvent.click(screen.getByRole("button", { name: "Choose" }));
  await userEvent.click(screen.getByRole("button", { name: "Automatic" }));
  expect(onSave).toHaveBeenLastCalledWith(null);
});
