import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { TOKEN_HEADER, type SystemStatus } from "@/api/client";
import { qk } from "@/api/queries";
import { GpuStatusChip } from "@/components/GpuStatusChip";
import { SchemaForm } from "@/components/SchemaForm";
import { JobCard } from "@/features/jobs/JobCard";
import { useUi } from "@/stores/ui";
import { jsonResponse, makeJob, makeQueryClient, renderWithProviders } from "./helpers";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("running job shows progress, message, and step preview", () => {
  const job = makeJob({ status: "running", progress: 0.42, progress_message: "Step 5/12" });
  useUi.setState({ previews: { [job.id]: "/api/jobs/x/preview?n=1" } });
  renderWithProviders(<JobCard job={job} />);
  expect(screen.getByTestId("job-status")).toHaveTextContent("running");
  expect(screen.getByText("42%")).toBeInTheDocument();
  expect(screen.getByText("Step 5/12")).toBeInTheDocument();
  expect(screen.getByAltText("Step preview").getAttribute("src")).toContain("token=");
});

test("cancel button posts to the cancel endpoint with the token header", async () => {
  const meta = document.createElement("meta");
  meta.name = "localflow-token";
  meta.content = "tok123";
  document.head.appendChild(meta);
  const job = makeJob({ status: "running" });
  const fetchMock = vi.fn(async (_req: Request) => jsonResponse({ ...job, progress_message: "Cancelling…" }));
  vi.stubGlobal("fetch", fetchMock);

  renderWithProviders(<JobCard job={job} />);
  await userEvent.click(screen.getByRole("button", { name: "Cancel job" }));

  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  const req = fetchMock.mock.calls[0]![0];
  expect(req.method).toBe("POST");
  expect(new URL(req.url).pathname).toBe(`/api/jobs/${job.id}/cancel`);
  expect(req.headers.get(TOKEN_HEADER)).toBe("tok123");
  meta.remove();
});

test("failed job shows the error and a retry button", () => {
  renderWithProviders(<JobCard job={makeJob({ status: "failed", error: "Worker crashed" })} />);
  expect(screen.getByText("Worker crashed")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Retry job" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Cancel job" })).not.toBeInTheDocument();
});

test("SchemaForm renders adapter params and reports changes", async () => {
  const onChange = vi.fn();
  const schema = {
    properties: {
      steps: { type: "integer", title: "Steps", default: 12, minimum: 1, maximum: 200 },
      step_delay_s: { anyOf: [{ type: "number" }, { type: "null" }], title: "Step Delay S" },
      crash: { type: "boolean", title: "Crash", default: false },
    },
  };
  renderWithProviders(<SchemaForm schema={schema} value={{}} onChange={onChange} />);
  await userEvent.type(screen.getByLabelText("Steps"), "5");
  expect(onChange).toHaveBeenLastCalledWith({ steps: 5 });
  await userEvent.click(screen.getByLabelText("Crash"));
  expect(onChange).toHaveBeenLastCalledWith({ crash: true });
  expect(screen.getByLabelText("Step Delay S")).toHaveAttribute("type", "number");
});

test("GPU chip shows the GPU and VRAM usage", async () => {
  const status = {
    gpu: { name: "NVIDIA GeForce RTX 5080", vram_used_mb: 4000, vram_total_mb: 16000, driver_version: "610" },
    worker: {
      state: "idle",
      pid: 1,
      current_job_id: null,
      loaded_model: { adapter_key: "fake" },
      restarts: 0,
      last_exit_code: null,
      engine: "fake",
    },
  } as unknown as SystemStatus;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse(status)),
  );
  const qc = makeQueryClient();
  qc.setQueryData(qk.system, status);
  renderWithProviders(<GpuStatusChip />, qc);
  expect(await screen.findByText("RTX 5080")).toBeInTheDocument();
  expect(screen.getByTitle("25% VRAM")).toBeInTheDocument();
});
