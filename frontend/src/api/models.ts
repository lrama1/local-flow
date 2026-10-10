import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";
import { sk } from "./story";

export type ModelEntry = Schemas["ModelEntryOut"];
export type ModelOptions = Schemas["ModelOptions"];
export type AdapterOption = Schemas["AdapterOption"];
export type AccessState = NonNullable<ModelEntry["access"]>;
export type HfInspect = Schemas["HfInspectOut"];
export type HfInstallBody = Schemas["HfInstallBody"];
export type CivitaiModel = Schemas["CivitaiModelOut"];
export type CivitaiVersion = Schemas["CivitaiVersionOut"];
export type CivitaiFile = Schemas["CivitaiFileOut"];
export type CivitaiSearchPage = Schemas["CivitaiSearchOut"];

export const mk = {
  models: ["models"] as const,
  options: ["model-options"] as const,
  civitaiSearch: (p: CivitaiSearchParams) => ["civitai-search", p] as const,
  civitaiModel: (id: number) => ["civitai-model", id] as const,
};

export interface CivitaiSearchParams {
  query?: string;
  types?: ("Checkpoint" | "LORA" | "LoCon" | "DoRA" | "VAE")[];
  base_models?: string[];
  sort?: "Highest Rated" | "Most Downloaded" | "Newest";
}

export function useModels() {
  return useQuery({ queryKey: mk.models, queryFn: () => unwrap(api.GET("/api/models")) });
}

/** Installed models that can run `capability`, excluding LoRAs and components. */
export function installedFor(models: ModelEntry[] | undefined, capability: string): ModelEntry[] {
  return (models ?? []).filter(
    (m) =>
      m.status === "installed" &&
      m.capabilities.includes(capability) &&
      !["lora", "vae", "text_encoder"].includes(m.kind),
  );
}

export function useModelOptions() {
  return useQuery({ queryKey: mk.options, queryFn: () => unwrap(api.GET("/api/models/options")) });
}

function useModelMutation<V, R>(fn: (v: V) => Promise<R>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: mk.models });
      void qc.invalidateQueries({ queryKey: mk.options });
    },
  });
}

const entryPath = (id: string) => ({ params: { path: { entry_id: id } } });

export function useInstallModel() {
  return useModelMutation((id: string) => unwrap(api.POST("/api/models/{entry_id}/install", entryPath(id))));
}

export function useDeleteModel() {
  return useModelMutation((id: string) => unwrap(api.DELETE("/api/models/{entry_id}", entryPath(id))));
}

export function useCheckAccess() {
  return useModelMutation((id: string) =>
    unwrap(api.POST("/api/models/{entry_id}/check-access", entryPath(id))),
  );
}

export function useAcknowledgeLicense() {
  return useModelMutation((id: string) =>
    unwrap(api.POST("/api/models/{entry_id}/acknowledge-license", entryPath(id))),
  );
}

export function useCancelDownload() {
  return useModelMutation((jobId: string) =>
    unwrap(api.POST("/api/jobs/{job_id}/cancel", { params: { path: { job_id: jobId } } })),
  );
}

// --- Hugging Face ---------------------------------------------------------------------------------
export function useHfInspect() {
  return useMutation({
    mutationFn: (body: Schemas["HfInspectBody"]) => unwrap(api.POST("/api/models/hf/inspect", { body })),
  });
}

export function useHfInstall() {
  return useModelMutation((body: HfInstallBody) => unwrap(api.POST("/api/models/hf/install", { body })));
}

// --- Civitai ----------------------------------------------------------------------------------------
export function useCivitaiSearch(params: CivitaiSearchParams, enabled: boolean) {
  return useInfiniteQuery({
    queryKey: mk.civitaiSearch(params),
    enabled,
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last: CivitaiSearchPage) => last.next_cursor ?? undefined,
    queryFn: ({ pageParam, signal }) =>
      unwrap(
        api.GET("/api/models/civitai/search", {
          params: { query: { ...params, cursor: pageParam, limit: 24 } },
          signal,
        }),
      ),
    staleTime: 60_000,
  });
}

export function useCivitaiInstall() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Schemas["CivitaiInstallBody"]) =>
      unwrap(api.POST("/api/models/civitai/install", { body })),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: mk.models });
      void qc.invalidateQueries({ queryKey: ["civitai-search"] });
    },
  });
}

// --- hub accounts --------------------------------------------------------------------------------
export function useImportHfCli() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => unwrap(api.POST("/api/settings/secrets/hf_token/import-cli")),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: sk.secrets });
      void qc.invalidateQueries({ queryKey: mk.options });
    },
  });
}

// --- display helpers -------------------------------------------------------------------------------
export function formatBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return i === 0 ? `${v} B` : `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

export const ACCESS_LABELS: Record<AccessState, string> = {
  ok: "Access OK",
  needs_token: "Token needed",
  license_not_accepted: "License not accepted",
  pending_approval: "Awaiting approval",
  token_no_access: "Token lacks access",
  not_found: "Not found",
  invalid_token: "Token rejected",
  forbidden: "Forbidden",
};

/** Which entries belong on the Installed tab: anything with files on disk, in flight, or user-added. */
export function isInstalledTab(m: ModelEntry): boolean {
  return m.status !== "available" || m.partial || m.catalog_key == null;
}
