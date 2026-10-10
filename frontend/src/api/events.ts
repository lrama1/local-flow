import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { useUi } from "@/stores/ui";
import { JOB_TYPE_LABELS } from "@/lib/utils";
import { ck } from "./characters";
import { getToken, type Asset, type Job, type Schemas, type SystemStatus, type WorkerInfo } from "./client";
import { mk, type ModelEntry } from "./models";
import { JOBS_PAGE_SIZE, qk } from "./queries";
import { sk } from "./story";
import { tk } from "./timeline";

type JobPage = Schemas["Page_JobOut_"];

export type ServerEvent =
  | { type: "job.created" | "job.updated" | "job.completed" | "job.failed"; job: Job }
  | { type: "job.progress"; job_id: string; progress: number; message: string | null }
  | { type: "job.preview"; job_id: string; url: string }
  | { type: "worker.status"; worker: WorkerInfo }
  | { type: "asset.created"; asset: Asset }
  | { type: "asset.deleted"; asset_id: string }
  | { type: "project.deleted"; project_id: string }
  | { type: "story.updated"; project_id: string; revision: number }
  | { type: "board.updated"; project_id: string }
  | { type: "timeline.updated"; project_id: string }
  | { type: "ingredients.updated"; project_id: string }
  | { type: "library.updated"; library_character_id: string }
  | { type: "model.status"; model: ModelEntry }
  | { type: "model.status"; model_entry_id: string; deleted: true }
  | {
      type: "download.progress";
      job_id: string;
      model_entry_id: string;
      downloaded_bytes: number;
      total_bytes: number;
      bytes_per_s: number;
    }
  | { type: "heartbeat" };

function upsertJob(page: JobPage | undefined, job: Job): JobPage | undefined {
  if (!page) return page;
  const idx = page.items.findIndex((j) => j.id === job.id);
  if (idx >= 0) {
    const items = page.items.slice();
    items[idx] = job;
    return { ...page, items };
  }
  return { ...page, items: [job, ...page.items].slice(0, JOBS_PAGE_SIZE), total: page.total + 1 };
}

function patchJob(page: JobPage | undefined, id: string, patch: Partial<Job>): JobPage | undefined {
  if (!page || !page.items.some((j) => j.id === id)) return page;
  return { ...page, items: page.items.map((j) => (j.id === id ? { ...j, ...patch } : j)) };
}

/** Applies one server event to the query cache / UI store. Exported for tests. */
export function handleServerEvent(qc: QueryClient, ev: ServerEvent): void {
  const ui = useUi.getState();
  switch (ev.type) {
    case "job.created":
    case "job.updated":
    case "job.completed":
    case "job.failed": {
      qc.setQueryData<JobPage>(qk.jobs, (p) => upsertJob(p, ev.job));
      qc.setQueryData<Job>(qk.job(ev.job.id), ev.job);
      if (ev.type === "job.completed" && ev.job.project_id) {
        void qc.invalidateQueries({ queryKey: qk.assetsRoot(ev.job.project_id) });
        void qc.invalidateQueries({ queryKey: qk.projects });
      }
      if (ev.type === "job.completed" || ev.type === "job.failed") {
        const label = JOB_TYPE_LABELS[ev.job.type] ?? ev.job.type;
        ui.pushToast({
          title: ev.type === "job.completed" ? `${label} finished` : `${label} failed`,
          description: ev.type === "job.failed" ? (ev.job.error ?? undefined) : undefined,
          variant: ev.type === "job.completed" ? "success" : "error",
          action: { label: "View", onClick: () => useUi.getState().setJobsDrawerOpen(true) },
        });
      }
      break;
    }
    case "job.progress": {
      const patch = { progress: ev.progress, progress_message: ev.message, status: "running" as const };
      qc.setQueryData<JobPage>(qk.jobs, (p) => patchJob(p, ev.job_id, patch));
      qc.setQueryData<Job>(qk.job(ev.job_id), (j) => (j ? { ...j, ...patch } : j));
      break;
    }
    case "job.preview":
      ui.setPreview(ev.job_id, ev.url);
      break;
    case "worker.status":
      qc.setQueryData<SystemStatus>(qk.system, (s) => (s ? { ...s, worker: ev.worker } : s));
      if (ev.worker.state === "idle") void qc.invalidateQueries({ queryKey: qk.adapters });
      break;
    case "asset.created":
      void qc.invalidateQueries({ queryKey: qk.assetsRoot(ev.asset.project_id) });
      break;
    case "asset.deleted":
      void qc.invalidateQueries({ queryKey: ["assets"] });
      break;
    case "project.deleted":
      void qc.invalidateQueries({ queryKey: qk.projects });
      break;
    case "story.updated":
      void qc.invalidateQueries({ queryKey: sk.story(ev.project_id) });
      void qc.invalidateQueries({ queryKey: sk.revisions(ev.project_id) });
      break;
    case "board.updated":
      void qc.invalidateQueries({ queryKey: sk.board(ev.project_id) });
      void qc.invalidateQueries({ queryKey: ck.referencesRoot });
      break;
    case "timeline.updated":
      void qc.invalidateQueries({ queryKey: tk.timeline(ev.project_id) });
      break;
    case "ingredients.updated":
      void qc.invalidateQueries({ queryKey: sk.ingredients(ev.project_id) });
      void qc.invalidateQueries({ queryKey: ck.referencesRoot });
      break;
    case "library.updated":
      // Library edits change the "update available" badge on linked characters in every project.
      void qc.invalidateQueries({ queryKey: ck.libraryRoot });
      void qc.invalidateQueries({ queryKey: ["ingredients"] });
      break;
    case "model.status": {
      if ("deleted" in ev) {
        qc.setQueryData<ModelEntry[]>(mk.models, (list) => list?.filter((m) => m.id !== ev.model_entry_id));
        ui.clearDownload(ev.model_entry_id);
      } else {
        const m = ev.model;
        qc.setQueryData<ModelEntry[]>(mk.models, (list) => {
          if (!list) return list;
          return list.some((x) => x.id === m.id) ? list.map((x) => (x.id === m.id ? m : x)) : [...list, m];
        });
        if (m.status !== "downloading") ui.clearDownload(m.id);
      }
      void qc.invalidateQueries({ queryKey: mk.options });
      break;
    }
    case "download.progress":
      ui.setDownload(ev.model_entry_id, {
        jobId: ev.job_id,
        downloaded: ev.downloaded_bytes,
        total: ev.total_bytes,
        bytesPerS: ev.bytes_per_s,
      });
      break;
    case "heartbeat":
      break;
  }
}

/** Keeps a WebSocket open to `/ws/events`, reconnecting with exponential backoff. */
export function useEventStream(): void {
  const qc = useQueryClient();
  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let disposed = false;
    const { setConnection } = useUi.getState();

    const connect = () => {
      setConnection("connecting");
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(
        `${proto}://${window.location.host}/ws/events?token=${encodeURIComponent(getToken())}`,
      );
      ws.onopen = () => {
        if (attempt > 0) void qc.invalidateQueries(); // resync anything missed while disconnected
        attempt = 0;
        setConnection("open");
      };
      ws.onmessage = (msg) => {
        try {
          handleServerEvent(qc, JSON.parse(msg.data as string) as ServerEvent);
        } catch (e) {
          console.error("bad event", e);
        }
      };
      ws.onclose = () => {
        if (disposed) return; // a stale socket from an earlier mount must not touch shared state
        setConnection("closed");
        timer = setTimeout(connect, Math.min(30_000, 500 * 2 ** attempt++));
      };
    };

    // Deferred so React StrictMode's mount/unmount/mount doesn't open and immediately drop a socket.
    timer = setTimeout(connect, 0);
    return () => {
      disposed = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, [qc]);
}
