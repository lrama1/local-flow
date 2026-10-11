import { create } from "zustand";

export type ConnectionState = "connecting" | "open" | "closed";

export interface Toast {
  id: number;
  title: string;
  description?: string;
  variant: "default" | "success" | "error";
  action?: { label: string; onClick: () => void };
}

interface UiState {
  jobsDrawerOpen: boolean;
  setJobsDrawerOpen: (open: boolean) => void;
  toggleJobsDrawer: () => void;

  connection: ConnectionState;
  setConnection: (c: ConnectionState) => void;

  /** Latest step preview per running job (from `job.preview` events). */
  previews: Record<string, string>;
  setPreview: (jobId: string, url: string) => void;

  toasts: Toast[];
  pushToast: (t: Omit<Toast, "id">) => void;
  dismissToast: (id: number) => void;

  /** Live download progress per model entry (from `download.progress` events). */
  downloads: Record<string, DownloadProgress>;
  setDownload: (entryId: string, p: DownloadProgress) => void;
  clearDownload: (entryId: string) => void;

  /** A generation was refused until a model's license is acknowledged; retried after (Section 9.5). */
  licensePrompt: LicensePrompt | null;
  setLicensePrompt: (p: LicensePrompt | null) => void;

  /** "Reuse settings" from the asset library: the Playground opens with these (Section 13). */
  playgroundPrefill: PlaygroundPrefill | null;
  setPlaygroundPrefill: (p: PlaygroundPrefill | null) => void;
}

export interface PlaygroundPrefill {
  kind: string;
  form: Record<string, unknown>;
}

export interface LicensePrompt {
  modelEntryId: string;
  retry: () => void;
}

export interface DownloadProgress {
  jobId: string;
  downloaded: number;
  total: number;
  bytesPerS: number;
}

let toastId = 0;

export const useUi = create<UiState>((set) => ({
  jobsDrawerOpen: false,
  setJobsDrawerOpen: (open) => set({ jobsDrawerOpen: open }),
  toggleJobsDrawer: () => set((s) => ({ jobsDrawerOpen: !s.jobsDrawerOpen })),

  connection: "connecting",
  setConnection: (connection) => set({ connection }),

  previews: {},
  setPreview: (jobId, url) => set((s) => ({ previews: { ...s.previews, [jobId]: url } })),

  toasts: [],
  pushToast: (t) => set((s) => ({ toasts: [...s.toasts.slice(-4), { ...t, id: ++toastId }] })),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  downloads: {},
  setDownload: (entryId, p) => set((s) => ({ downloads: { ...s.downloads, [entryId]: p } })),
  clearDownload: (entryId) =>
    set((s) => {
      if (!(entryId in s.downloads)) return s;
      const { [entryId]: _gone, ...rest } = s.downloads;
      return { downloads: rest };
    }),

  licensePrompt: null,
  setLicensePrompt: (licensePrompt) => set({ licensePrompt }),

  playgroundPrefill: null,
  setPlaygroundPrefill: (playgroundPrefill) => set({ playgroundPrefill }),
}));
