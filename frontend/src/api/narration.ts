import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";
import { sk } from "./story";
import { tk } from "./timeline";

export type NarrationPlan = Schemas["NarrationPlanOut"];
export type NarrationShot = Schemas["NarrationShotOut"];
export type NarrationWriteBody = Schemas["NarrationWriteBody"];
export type NarrationGenerateBody = Schemas["NarrationGenerateBody"];

export const nk = {
  root: ["narration"] as const,
  plan: (projectId: string, speed: number) => ["narration", projectId, speed] as const,
};

const projectPath = (projectId: string) => ({ params: { path: { project_id: projectId } } });

/** One row per shot: its line, voiced status, where it starts, and whether it overruns the next line. */
export function useNarrationPlan(projectId: string, speed: number) {
  return useQuery({
    queryKey: nk.plan(projectId, speed),
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/narration", {
          params: { path: { project_id: projectId }, query: { speed } },
        }),
      ),
    placeholderData: keepPreviousData,
  });
}

export function useWriteNarration(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: NarrationWriteBody) =>
      unwrap(api.POST("/api/projects/{project_id}/narration/write", { ...projectPath(projectId), body })),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: sk.board(projectId) });
      void qc.invalidateQueries({ queryKey: nk.root });
    },
  });
}

export function useGenerateNarration(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: NarrationGenerateBody) =>
      unwrap(api.POST("/api/projects/{project_id}/narration/generate", { ...projectPath(projectId), body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: nk.root }),
  });
}

export function useResyncNarration(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => unwrap(api.POST("/api/projects/{project_id}/narration/resync", projectPath(projectId))),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: tk.timeline(projectId) });
      void qc.invalidateQueries({ queryKey: nk.root });
    },
  });
}
