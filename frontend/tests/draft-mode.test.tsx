import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { Schemas } from "@/api/client";
import { mk, type ModelEntry } from "@/api/models";
import { qk } from "@/api/queries";
import { DraftModeToggle } from "@/features/projects/DraftModeToggle";
import { ModelCard } from "@/features/models/ModelCard";
import { jsonResponse, makeQueryClient, renderWithProviders } from "./helpers";

type Project = Schemas["ProjectOut"];

function makeProject(o: Partial<Project> = {}): Project {
  const now = new Date().toISOString();
  return {
    id: "p1",
    name: "The Last Rose",
    description: "",
    aspect_ratio: "16:9",
    fps: 24,
    default_video_model_id: null,
    default_image_model_id: null,
    draft_mode: false,
    style_notes: "",
    archived: false,
    created_at: now,
    updated_at: now,
    ...o,
  };
}

function makeModel(o: Partial<ModelEntry> = {}): ModelEntry {
  const now = new Date().toISOString();
  return {
    id: "m1",
    catalog_key: "wan22-ti2v-5b-turbo",
    source: "hf",
    source_ref: "hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF",
    name: "Wan 2.2 TI2V-5B Turbo · GGUF Q8_0 (draft, 4 steps)",
    description: "",
    purpose: "Fast video drafts",
    recommended: false,
    tier: "draft",
    kind: "checkpoint",
    capabilities: ["t2v", "i2v", "extend"],
    adapter_key: "wan",
    adapter_name: "Wan",
    adapter_available: true,
    base_model: "wan2.2-5b",
    license: "Apache 2.0",
    license_url: "https://example.com",
    license_acknowledged: true,
    license_acknowledged_at: now,
    nsfw: false,
    parts: [],
    local_path: null,
    size_bytes: 5_400_000_000,
    approx_size_bytes: 19_700_000_000,
    status: "installed",
    partial: false,
    error: null,
    default_params: {},
    vram_profiles: {},
    gated: false,
    access: null,
    access_message: null,
    access_checked_at: null,
    trigger_words: [],
    preview_url: null,
    requires_pickle: false,
    pickle_blocked: false,
    experimental: false,
    notes: "",
    model_url: null,
    job_id: null,
    civitai: null,
    installed_at: now,
    created_at: now,
    updated_at: now,
    ...o,
  };
}

afterEach(() => vi.unstubAllGlobals());

function mockPatch() {
  const bodies: unknown[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (req: Request) => {
      if (req.method === "PATCH") {
        const body = (await req.clone().json()) as Record<string, unknown>;
        bodies.push(body);
        return jsonResponse(makeProject({ draft_mode: Boolean(body.draft_mode) }));
      }
      return jsonResponse({ error: { code: "not_mocked", message: req.url } }, 404);
    }),
  );
  return bodies;
}

test("switching to Draft patches the project and updates the cached project", async () => {
  const bodies = mockPatch();
  const qc = makeQueryClient();
  qc.setQueryData(mk.models, [makeModel()]);
  renderWithProviders(<DraftModeToggle project={makeProject()} />, qc);

  expect(screen.getByRole("button", { name: "Final" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "Draft" })).toHaveAttribute("aria-pressed", "false");
  await userEvent.click(screen.getByRole("button", { name: "Draft" }));

  await waitFor(() => expect(bodies).toEqual([{ draft_mode: true }]));
  await waitFor(() => expect(qc.getQueryData<Project>(qk.project("p1"))?.draft_mode).toBe(true));
});

test("clicking the mode that's already on does nothing", async () => {
  const bodies = mockPatch();
  renderWithProviders(<DraftModeToggle project={makeProject()} />);
  await userEvent.click(screen.getByRole("button", { name: "Final" }));
  expect(bodies).toEqual([]);
});

test("Draft mode lists the kinds with no draft model installed, and links to the catalog", () => {
  const qc = makeQueryClient();
  qc.setQueryData(mk.models, [makeModel()]); // only the video draft model is installed
  renderWithProviders(<DraftModeToggle project={makeProject({ draft_mode: true })} />, qc);

  expect(screen.getByRole("button", { name: "Draft" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText(/Not installed yet: images, keyframes\./)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Models" })).toHaveAttribute("href", "/models?tab=catalog");
});

test("no warning once every kind has a draft model", () => {
  const qc = makeQueryClient();
  qc.setQueryData(mk.models, [
    makeModel(),
    makeModel({ id: "m2", capabilities: ["t2i"], name: "FLUX turbo" }),
    makeModel({ id: "m3", capabilities: ["image_edit"], name: "Kontext turbo" }),
  ]);
  renderWithProviders(<DraftModeToggle project={makeProject({ draft_mode: true })} />, qc);
  expect(screen.getByText(/New images, keyframes and videos use the fast models/)).toBeInTheDocument();
  expect(screen.queryByText(/Not installed yet/)).not.toBeInTheDocument();
});

test("the model card marks draft models", () => {
  renderWithProviders(<ModelCard model={makeModel()} />);
  expect(screen.getByText(/draft · fast/)).toBeInTheDocument();
  renderWithProviders(<ModelCard model={makeModel({ id: "m9", tier: "final", name: "Wan final" })} />);
  expect(screen.getAllByText(/draft · fast/)).toHaveLength(1);
});
