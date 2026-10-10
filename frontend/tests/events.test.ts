import { handleServerEvent } from "@/api/events";
import type { Schemas, SystemStatus } from "@/api/client";
import { qk } from "@/api/queries";
import { useUi } from "@/stores/ui";
import { makeJob, makeQueryClient } from "./helpers";

type JobPage = Schemas["Page_JobOut_"];

beforeEach(() => {
  useUi.setState({ previews: {}, toasts: [] });
});

test("job events upsert into the jobs list and per-job cache", () => {
  const qc = makeQueryClient();
  qc.setQueryData<JobPage>(qk.jobs, { items: [], total: 0, offset: 0, limit: 50 });

  const job = makeJob();
  handleServerEvent(qc, { type: "job.created", job });
  expect(qc.getQueryData<JobPage>(qk.jobs)?.items.map((j) => j.id)).toEqual([job.id]);
  expect(qc.getQueryData(qk.job(job.id))).toEqual(job);

  handleServerEvent(qc, { type: "job.progress", job_id: job.id, progress: 0.4, message: "Step 4/10" });
  const updated = qc.getQueryData<JobPage>(qk.jobs)!.items[0]!;
  expect(updated.progress).toBe(0.4);
  expect(updated.status).toBe("running");
  expect(updated.progress_message).toBe("Step 4/10");

  handleServerEvent(qc, {
    type: "job.completed",
    job: { ...job, status: "completed", progress: 1, result: { asset_ids: ["a1"] } },
  });
  const page = qc.getQueryData<JobPage>(qk.jobs)!;
  expect(page.items).toHaveLength(1);
  expect(page.items[0]!.status).toBe("completed");
  expect(useUi.getState().toasts.at(-1)).toMatchObject({
    title: "Text → video finished",
    variant: "success",
  });
});

test("completed jobs invalidate the project's assets", () => {
  const qc = makeQueryClient();
  const job = makeJob({ status: "completed" });
  qc.setQueryData(qk.assets(job.project_id!, {}), { items: [], total: 0, offset: 0, limit: 100 });
  handleServerEvent(qc, { type: "job.completed", job });
  expect(qc.getQueryState(qk.assets(job.project_id!, {}))?.isInvalidated).toBe(true);
});

test("failed jobs raise an error toast with the message", () => {
  const qc = makeQueryClient();
  handleServerEvent(qc, {
    type: "job.failed",
    job: makeJob({ status: "failed", error: "Out of GPU memory" }),
  });
  expect(useUi.getState().toasts.at(-1)).toMatchObject({
    variant: "error",
    description: "Out of GPU memory",
  });
});

test("previews and worker status are applied", () => {
  const qc = makeQueryClient();
  qc.setQueryData<Partial<SystemStatus>>(qk.system, { version: "0.1.0" });
  handleServerEvent(qc, { type: "job.preview", job_id: "j1", url: "/api/jobs/j1/preview?n=3" });
  expect(useUi.getState().previews.j1).toBe("/api/jobs/j1/preview?n=3");

  const worker = {
    state: "busy" as const,
    pid: 42,
    current_job_id: "j1",
    loaded_model: { adapter_key: "fake" },
    restarts: 0,
    last_exit_code: null,
    engine: "fake",
  };
  handleServerEvent(qc, { type: "worker.status", worker });
  expect(qc.getQueryData<SystemStatus>(qk.system)?.worker).toEqual(worker);
});
