import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { SecretStatus } from "@/api/client";
import { handleServerEvent } from "@/api/events";
import { formatBytes, isInstalledTab, mk, type HfInspect, type ModelEntry } from "@/api/models";
import { HfPanel } from "@/features/models/HfPanel";
import { ModelCard } from "@/features/models/ModelCard";
import { ApiKeysCard } from "@/features/settings/ApiKeysCard";
import { useUi } from "@/stores/ui";
import { jsonResponse, makeQueryClient, renderWithProviders } from "./helpers";

type Handler = (req: Request) => Response | Promise<Response>;

function mockApi(routes: Record<string, Handler>) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (req: Request) => {
      const path = new URL(req.url).pathname;
      const key = `${req.method} ${path}`;
      const body =
        req.method === "GET"
          ? undefined
          : await req
              .clone()
              .json()
              .catch(() => undefined);
      calls.push({ method: req.method, path, body });
      const match = Object.entries(routes).find(([pattern]) => new RegExp(`^${pattern}$`).test(key));
      if (!match) return jsonResponse({ error: { code: "not_mocked", message: key } }, 404);
      return match[1](req);
    }),
  );
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
  useUi.setState({ downloads: {} });
});

function makeModel(o: Partial<ModelEntry> = {}): ModelEntry {
  const now = new Date().toISOString();
  return {
    id: "m1",
    catalog_key: "flux1-dev-q8",
    source: "hf",
    source_ref: "black-forest-labs/FLUX.1-dev",
    name: "FLUX.1 [dev] · GGUF Q8_0",
    description: "Text-to-image",
    purpose: "Text-to-image (quality)",
    recommended: true,
    kind: "checkpoint",
    capabilities: ["t2i"],
    adapter_key: "flux",
    adapter_name: "FLUX.1",
    adapter_available: false,
    base_model: "flux1",
    license: "FLUX.1 [dev] Non-Commercial License",
    license_url: "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
    license_acknowledged: false,
    license_acknowledged_at: null,
    nsfw: false,
    parts: [],
    local_path: null,
    size_bytes: 22_650_000_000,
    approx_size_bytes: 22_700_000_000,
    status: "available",
    partial: false,
    error: null,
    default_params: {},
    vram_profiles: {},
    gated: true,
    access: null,
    access_message: null,
    access_checked_at: null,
    trigger_words: [],
    preview_url: null,
    requires_pickle: false,
    pickle_blocked: false,
    experimental: false,
    notes: "",
    model_url: "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    job_id: null,
    civitai: null,
    installed_at: null,
    created_at: now,
    updated_at: now,
    ...o,
  };
}

// --- units ----------------------------------------------------------------------------------------
test("formatBytes and tab membership", () => {
  expect(formatBytes(0)).toBe("0 B");
  expect(formatBytes(1_500)).toBe("1.5 KB");
  expect(formatBytes(22_700_000_000)).toBe("22.7 GB");
  expect(formatBytes(340_000_000)).toBe("340 MB");
  expect(formatBytes(null)).toBe("–");
  expect(isInstalledTab(makeModel())).toBe(false);
  expect(isInstalledTab(makeModel({ status: "downloading" }))).toBe(true);
  expect(isInstalledTab(makeModel({ catalog_key: null }))).toBe(true);
});

test("model.status and download.progress events update the cache and store", () => {
  const qc = makeQueryClient();
  qc.setQueryData(mk.models, [makeModel(), makeModel({ id: "m2", name: "Other" })]);
  handleServerEvent(qc, {
    type: "download.progress",
    job_id: "j1",
    model_entry_id: "m1",
    downloaded_bytes: 5,
    total_bytes: 10,
    bytes_per_s: 2,
  });
  expect(useUi.getState().downloads.m1).toEqual({ jobId: "j1", downloaded: 5, total: 10, bytesPerS: 2 });

  handleServerEvent(qc, { type: "model.status", model: makeModel({ status: "installed" }) });
  expect(qc.getQueryData<ModelEntry[]>(mk.models)?.[0]?.status).toBe("installed");
  expect(useUi.getState().downloads.m1).toBeUndefined(); // cleared once no longer downloading

  handleServerEvent(qc, { type: "model.status", model: makeModel({ id: "m3", name: "New" }) });
  expect(qc.getQueryData<ModelEntry[]>(mk.models)).toHaveLength(3);
  handleServerEvent(qc, { type: "model.status", model_entry_id: "m2", deleted: true });
  expect(qc.getQueryData<ModelEntry[]>(mk.models)?.map((m) => m.id)).toEqual(["m1", "m3"]);
});

// --- ModelCard ------------------------------------------------------------------------------------
test("gated model without accepted license links to the model page and can check again", async () => {
  const calls = mockApi({
    "POST /api/models/m1/check-access": () => jsonResponse(makeModel({ access: "ok" })),
    "GET /api/models(/options)?": () => jsonResponse([]),
  });
  renderWithProviders(
    <ModelCard
      model={makeModel({
        access: "license_not_accepted",
        access_message:
          "black-forest-labs/FLUX.1-dev is gated. You're signed in as lrama1, but haven't accepted its license.",
      })}
    />,
  );
  const alert = screen.getByRole("alert");
  expect(within(alert).getByText("License not accepted")).toBeInTheDocument();
  expect(within(alert).getByText(/signed in as lrama1/)).toBeInTheDocument();
  expect(within(alert).getByRole("link", { name: /open model page/i })).toHaveAttribute(
    "href",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev",
  );
  await userEvent.click(within(alert).getByRole("button", { name: /check again/i }));
  await waitFor(() => expect(calls.some((c) => c.path === "/api/models/m1/check-access")).toBe(true));
});

test("needs_token points to Settings", () => {
  renderWithProviders(
    <ModelCard model={makeModel({ access: "needs_token", access_message: "Add a token" })} />,
  );
  expect(screen.getByRole("link", { name: /add a hugging face token/i })).toHaveAttribute(
    "href",
    "/settings",
  );
});

test("downloading card shows live progress and cancels the job", async () => {
  const calls = mockApi({ "POST /api/jobs/j9/cancel": () => jsonResponse({}) });
  useUi.getState().setDownload("m1", {
    jobId: "j9",
    downloaded: 2_000_000_000,
    total: 8_000_000_000,
    bytesPerS: 50_000_000,
  });
  renderWithProviders(<ModelCard model={makeModel({ status: "downloading", job_id: "j9" })} />);
  const bar = screen.getByTestId("download-progress");
  expect(within(bar).getByText("2.0 GB of 8.0 GB")).toBeInTheDocument();
  expect(within(bar).getByText("50.0 MB/s")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
  await waitFor(() => expect(calls.map((c) => c.path)).toContain("/api/jobs/j9/cancel"));
});

test("pickle-only models can't be installed by default", () => {
  renderWithProviders(
    <ModelCard model={makeModel({ requires_pickle: true, pickle_blocked: true, gated: false })} />,
  );
  expect(screen.getByRole("button", { name: /install/i })).toBeDisabled();
  expect(screen.getByText(/allow_pickle_weights/)).toBeInTheDocument();
});

test("installed model asks for a one-time license acknowledgement", async () => {
  const calls = mockApi({
    "POST /api/models/m1/acknowledge-license": () => jsonResponse(makeModel({ license_acknowledged: true })),
  });
  renderWithProviders(<ModelCard model={makeModel({ status: "installed", access: "ok" })} />);
  await userEvent.click(screen.getByRole("button", { name: /review license/i }));
  const dialog = await screen.findByRole("dialog");
  await userEvent.click(within(dialog).getByRole("button", { name: "Acknowledge" }));
  expect(calls).toHaveLength(0); // checkbox not ticked yet
  await userEvent.click(within(dialog).getByRole("checkbox", { name: /reviewed this license/i }));
  await userEvent.click(within(dialog).getByRole("button", { name: "Acknowledge" }));
  await waitFor(() => expect(calls.map((c) => c.path)).toContain("/api/models/m1/acknowledge-license"));
});

// --- Hugging Face panel ------------------------------------------------------------------------------
const inspect = (o: Partial<HfInspect> = {}): HfInspect => ({
  repo_id: "someone/tiny-lora",
  revision: null,
  sha: "abc",
  gated: false,
  gated_mode: null,
  private: false,
  access: "ok",
  access_message: null,
  model_url: "https://huggingface.co/someone/tiny-lora",
  license: "apache-2.0",
  license_url: "https://huggingface.co/someone/tiny-lora",
  pipeline_tag: null,
  library_name: null,
  tags: ["lora"],
  files: [
    {
      path: "lora.safetensors",
      size: 64_000_000,
      file_class: "weight",
      blocked_reason: null,
      default_selected: true,
    },
    {
      path: "train.py",
      size: 100,
      file_class: "code",
      blocked_reason: "Code files are never downloaded.",
      default_selected: false,
    },
    {
      path: "old.bin",
      size: 64_000_000,
      file_class: "pickle",
      blocked_reason: "Pickle weights are blocked",
      default_selected: false,
    },
  ],
  total_size: 128_000_100,
  default_selection: ["lora.safetensors"],
  selected_size: 64_000_000,
  suggestion: {
    kind: "lora",
    base_model: "sdxl",
    adapter_key: "sdxl",
    capabilities: ["t2i"],
    supported: true,
    reason: "x",
  },
  existing_entry_id: null,
  account: null,
  ...o,
});

const options = {
  adapters: [
    {
      key: "sdxl",
      name: "SDXL / Civitai checkpoints",
      capabilities: ["t2i"],
      base_models: ["sdxl"],
      supports_lora: true,
      milestone: "M3",
      available: false,
    },
  ],
  kinds: ["checkpoint", "lora"],
  allow_pickle_weights: false,
  allow_nsfw_models: false,
  models_dir: "C:/LocalFlow/models",
  free_bytes: 1e12,
  hf: { configured: false },
  civitai: { configured: false },
};

test("HF inspect → choose adapter → install sends the selected files", async () => {
  const calls = mockApi({
    "POST /api/models/hf/inspect": () => jsonResponse(inspect()),
    "GET /api/models/options": () => jsonResponse(options),
    "GET /api/models": () => jsonResponse([]),
    "POST /api/models/hf/install": () =>
      jsonResponse(
        { entry: makeModel({ id: "u1", name: "tiny-lora", catalog_key: null }), job_id: "j1" },
        201,
      ),
  });
  renderWithProviders(<HfPanel />);
  await userEvent.type(screen.getByLabelText("Hugging Face repo id"), "someone/tiny-lora");
  await userEvent.click(screen.getByRole("button", { name: /inspect/i }));
  const card = await screen.findByTestId("hf-inspect");
  expect(within(card).getByLabelText("old.bin")).toBeDisabled();
  expect(within(card).queryByLabelText("train.py")).toBeNull(); // configs/docs/code hidden until "Show all"
  await userEvent.click(within(card).getByRole("button", { name: /show all 3 files/i }));
  expect(within(card).getByLabelText("train.py")).toBeDisabled();
  expect(within(card).getByLabelText("lora.safetensors")).toBeChecked();
  await waitFor(() => expect(within(card).getByLabelText(/adapter/i)).toHaveValue("sdxl"));
  await userEvent.click(within(card).getByRole("button", { name: /install 64\.0 MB/i }));
  await screen.findByText(/download started for tiny-lora/i);
  const body = calls.find((c) => c.path === "/api/models/hf/install")?.body as Record<string, unknown>;
  expect(body).toEqual({
    repo_id: "someone/tiny-lora",
    allow_patterns: ["lora.safetensors"],
    adapter_key: "sdxl",
    kind: "lora",
    name: "tiny-lora",
  });
});

test("HF inspect of a gated repo without a token disables install", async () => {
  mockApi({
    "POST /api/models/hf/inspect": () =>
      jsonResponse(
        inspect({
          repo_id: "black-forest-labs/FLUX.1-dev",
          gated: true,
          gated_mode: "auto",
          access: "needs_token",
          access_message: "Add a token",
        }),
      ),
    "GET /api/models/options": () => jsonResponse(options),
  });
  renderWithProviders(<HfPanel />);
  await userEvent.type(screen.getByLabelText("Hugging Face repo id"), "black-forest-labs/FLUX.1-dev");
  await userEvent.click(screen.getByRole("button", { name: /inspect/i }));
  const card = await screen.findByTestId("hf-inspect");
  expect(within(card).getByText("Token needed")).toBeInTheDocument();
  expect(within(card).getByRole("button", { name: /^install/i })).toBeDisabled();
});

// --- Settings ------------------------------------------------------------------------------------
test("Settings can import an existing Hugging Face CLI login", async () => {
  const hf: SecretStatus = {
    name: "hf_token",
    label: "Hugging Face",
    kind: "hub",
    configured: false,
    source: null,
    env_var: null,
    env_vars: ["HF_TOKEN"],
    stored: false,
    overridden: false,
    last4: null,
    validated_at: null,
    ok: null,
    account: null,
    token_type: null,
    warnings: [],
    error: null,
    token_url: "https://huggingface.co/settings/tokens",
  };
  const calls = mockApi({
    "GET /api/settings/secrets": () => jsonResponse([hf]),
    "POST /api/settings/secrets/hf_token/import-cli": () =>
      jsonResponse({ ...hf, configured: true, source: "keyring", ok: true, account: { username: "lrama1" } }),
  });
  renderWithProviders(<ApiKeysCard />);
  await userEvent.click(await screen.findByRole("button", { name: /import existing hugging face login/i }));
  await screen.findByText(/^Imported/);
  expect(calls.map((c) => `${c.method} ${c.path}`)).toContain(
    "POST /api/settings/secrets/hf_token/import-cli",
  );
});
