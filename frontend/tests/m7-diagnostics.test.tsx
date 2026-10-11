import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { Diagnostics } from "@/api/diagnostics";
import { DiagnosticsCard } from "@/features/settings/DiagnosticsCard";
import { jsonResponse, renderWithProviders } from "./helpers";

type Handler = (req: Request) => Response | Promise<Response>;

function mockApi(routes: Record<string, Handler>) {
  const calls: { method: string; path: string; search: string }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (req: Request) => {
      const url = new URL(req.url);
      calls.push({ method: req.method, path: url.pathname, search: url.search });
      const key = `${req.method} ${url.pathname}`;
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
});

const DIAG: Diagnostics = {
  generated_at: new Date().toISOString(),
  checks: [
    { key: "data_dir", label: "Data folder", status: "ok", detail: "D:\\LocalFlow is writable" },
    { key: "ffmpeg", label: "ffmpeg", status: "ok", detail: "7.1 at C:\\ffmpeg\\bin\\ffmpeg.exe" },
    { key: "cuda", label: "PyTorch + CUDA", status: "ok", detail: "torch 2.9 · CUDA 13.0 · RTX 5080" },
    { key: "keyring", label: "Secret storage", status: "ok", detail: "Windows WinVaultKeyring" },
    {
      key: "llm",
      label: "LLM providers",
      status: "warn",
      detail: "No API key: the story tools are unavailable.",
    },
  ],
  versions: { localflow: "0.1.0", python: "3.12.10", torch: "2.9.0+cu130", cuda: "13.0" },
  storage: [
    { key: "models", path: "D:\\LocalFlow\\models", size_mb: 81234.5, files: 120 },
    { key: "database", path: "D:\\LocalFlow\\localflow.db", size_mb: 2.1, files: 1 },
  ],
  disk_free_mb: 400_000,
  disk_total_mb: 1_000_000,
  logs: [
    { name: "localflow.log", size_kb: 120.5, modified: new Date().toISOString() },
    { name: "worker.log", size_kb: 40, modified: new Date().toISOString() },
  ],
};

test("diagnostics: checks, storage, a log tail and the support bundle", async () => {
  const calls = mockApi({
    "GET /api/system/diagnostics": () => jsonResponse(DIAG),
    "GET /api/system/status": () => jsonResponse({}, 404),
    "GET /api/system/logs/worker.log": () =>
      jsonResponse({ name: "worker.log", lines: ["2026-10-10 12:00:00 INFO    w: ready"], total_lines: 9 }),
    "GET /api/system/diagnostics/bundle": () =>
      new Response(new Blob(["PK"]), {
        status: 200,
        headers: {
          "Content-Type": "application/zip",
          "Content-Disposition": 'attachment; filename="localflow-diagnostics-1.zip"',
        },
      }),
  });
  const createObjectURL = vi.fn(() => "blob:bundle");
  vi.stubGlobal("URL", Object.assign(URL, { createObjectURL, revokeObjectURL: vi.fn() }));
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  renderWithProviders(<DiagnosticsCard />);

  const list = await screen.findByRole("list", { name: "Checks" });
  expect(within(list).getAllByRole("listitem")).toHaveLength(5);
  expect(within(list).getByText("Windows WinVaultKeyring")).toBeInTheDocument();
  expect(screen.getByText("1 to look at")).toBeInTheDocument();
  expect(screen.getByText(/120 files/)).toBeInTheDocument();
  expect(screen.getByText("2.9.0+cu130")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: /worker\.log/ }));
  const viewer = await screen.findByTestId("log-viewer");
  expect(await within(viewer).findByText(/w: ready/)).toBeInTheDocument();
  expect(within(viewer).getByText("last 1 of 9 lines")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "Run checks" }));
  await waitFor(() => expect(calls.some((c) => c.search.includes("refresh=true"))).toBe(true));

  await userEvent.click(screen.getByRole("button", { name: /Support bundle/ }));
  await waitFor(() => expect(click).toHaveBeenCalled());
  expect(createObjectURL).toHaveBeenCalled();
});
