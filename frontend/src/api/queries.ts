import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Job, type Schemas } from "./client";

export const qk = {
  projects: ["projects"] as const,
  project: (id: string) => ["projects", id] as const,
  assetsRoot: (projectId: string) => ["assets", projectId] as const,
  assets: (projectId: string, filters: AssetFilters) => ["assets", projectId, filters] as const,
  jobs: ["jobs"] as const,
  job: (id: string) => ["job", id] as const,
  system: ["system-status"] as const,
  adapters: ["adapters"] as const,
  playground: ["playground"] as const,
};

export interface AssetFilters {
  type?: "image" | "video" | "audio";
  favorite?: boolean;
  limit?: number;
}

export const JOBS_PAGE_SIZE = 50;

// --- system ---------------------------------------------------------------------------------------
export function useSystemStatus() {
  return useQuery({
    queryKey: qk.system,
    queryFn: () => unwrap(api.GET("/api/system/status")),
    refetchInterval: 5000,
  });
}

export function useAdapters() {
  return useQuery({
    queryKey: qk.adapters,
    queryFn: () => unwrap(api.GET("/api/models/adapters")),
    refetchInterval: (q) => (q.state.data?.length ? false : 2000), // the worker may still be starting
  });
}

export function usePlayground() {
  return useQuery({
    queryKey: qk.playground,
    queryFn: () => unwrap(api.GET("/api/playground")),
    staleTime: Infinity,
  });
}

// --- projects -------------------------------------------------------------------------------------
export function useProjects() {
  return useQuery({ queryKey: qk.projects, queryFn: () => unwrap(api.GET("/api/projects")) });
}

export function useProject(id: string | undefined) {
  return useQuery({
    queryKey: qk.project(id ?? ""),
    enabled: !!id,
    queryFn: () => unwrap(api.GET("/api/projects/{project_id}", { params: { path: { project_id: id! } } })),
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Schemas["ProjectCreate"]) => unwrap(api.POST("/api/projects", { body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects }),
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      unwrap(api.DELETE("/api/projects/{project_id}", { params: { path: { project_id: id } } })),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects }),
  });
}

// --- assets ---------------------------------------------------------------------------------------
export function useAssets(projectId: string | undefined, filters: AssetFilters = {}) {
  return useQuery({
    queryKey: qk.assets(projectId ?? "", filters),
    enabled: !!projectId,
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/assets", {
          params: { path: { project_id: projectId! }, query: { limit: 100, ...filters } },
        }),
      ),
  });
}

export function useToggleFavorite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, favorite }: { id: string; favorite: boolean }) =>
      unwrap(api.PATCH("/api/assets/{asset_id}", { params: { path: { asset_id: id } }, body: { favorite } })),
    onSuccess: (a) => qc.invalidateQueries({ queryKey: qk.assetsRoot(a.project_id) }),
  });
}

// --- jobs -----------------------------------------------------------------------------------------
export function useJobs() {
  return useQuery({
    queryKey: qk.jobs,
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { limit: JOBS_PAGE_SIZE } } })),
  });
}

export function useJob(id: string | undefined) {
  return useQuery({
    queryKey: qk.job(id ?? ""),
    enabled: !!id,
    queryFn: () => unwrap(api.GET("/api/jobs/{job_id}", { params: { path: { job_id: id! } } })),
  });
}

export function useCancelJob() {
  return useMutation({
    mutationFn: (id: string) =>
      unwrap(api.POST("/api/jobs/{job_id}/cancel", { params: { path: { job_id: id } } })),
  });
}

export function useRetryJob() {
  return useMutation({
    mutationFn: (id: string) =>
      unwrap(api.POST("/api/jobs/{job_id}/retry", { params: { path: { job_id: id } } })),
  });
}

// --- generation -----------------------------------------------------------------------------------
export interface GenerateBodies {
  t2i: Schemas["T2IBody"];
  image_edit: Schemas["ImageEditBody"];
  t2v: Schemas["T2VBody"];
  i2v: Schemas["I2VBody"];
  flf2v: Schemas["FLF2VBody"];
  ref2v: Schemas["Ref2VBody"];
  extend: Schemas["ExtendBody"];
  tts: Schemas["TTSBody"];
  v2a: Schemas["V2ABody"];
  music: Schemas["MusicBody"];
}
export type GenerateKind = keyof GenerateBodies;

export function generate<K extends GenerateKind>(kind: K, body: GenerateBodies[K]): Promise<Job> {
  // One typed route per kind exists server-side; the cast keeps this helper generic.
  return unwrap(
    api.POST(`/api/generate/${kind}` as "/api/generate/t2i", { body: body as Schemas["T2IBody"] }),
  );
}

export interface GenerateArgs {
  kind: GenerateKind;
  body: GenerateBodies[GenerateKind];
}

export function useGenerate() {
  const qc = useQueryClient();
  return useMutation<Job, Error, GenerateArgs>({
    mutationFn: (args) => generate(args.kind, args.body),
    onSuccess: (job) => qc.setQueryData(qk.job(job.id), job),
  });
}
