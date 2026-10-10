import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";
import { sk } from "./story";

export type VideoBody = Schemas["VideoBody"];
export type VideoOut = Schemas["VideoOut"];
export type VideoPlan = Schemas["VideoPlanOut"];
export type ExtendShotBody = Schemas["ExtendShotBody"];
export type GenerateAllBody = Schemas["GenerateAllBody"];
export type GenerateAllOut = Schemas["GenerateAllOut"];
export type VideoMode = NonNullable<VideoBody["mode"]>;
export type CameraIntensity = NonNullable<VideoBody["camera_intensity"]>;

export const vk = {
  plan: (shotId: string, body: VideoBody, version: string) => ["video-plan", shotId, body, version] as const,
  takes: (projectId: string, shotId: string) => ["assets", projectId, "takes", shotId] as const,
  cameraMoves: ["camera-moves"] as const,
};

const shotPath = (shotId: string) => ({ params: { path: { shot_id: shotId } } });

/** What "Generate video" would do for the saved shot. `version` (the shot's updated_at) refreshes it. */
export function useVideoPlan(shotId: string, body: VideoBody, version: string) {
  return useQuery({
    queryKey: vk.plan(shotId, body, version),
    queryFn: () => unwrap(api.POST("/api/shots/{shot_id}/video-plan", { ...shotPath(shotId), body })),
    placeholderData: keepPreviousData,
    retry: false,
  });
}

/** The shot's video takes, newest first. */
export function useTakes(projectId: string, shotId: string, enabled = true) {
  return useQuery({
    queryKey: vk.takes(projectId, shotId),
    enabled: enabled && !!shotId,
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/assets", {
          params: {
            path: { project_id: projectId },
            query: { type: "video", shot_id: shotId, slot: "video", limit: 24 },
          },
        }),
      ),
  });
}

export function useGenerateVideo(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ shotId, body }: { shotId: string; body: VideoBody }) =>
      unwrap(api.POST("/api/shots/{shot_id}/generate-video", { ...shotPath(shotId), body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: sk.board(projectId) }),
  });
}

export function useExtendShot(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ shotId, body }: { shotId: string; body: ExtendShotBody }) =>
      unwrap(api.POST("/api/shots/{shot_id}/extend", { ...shotPath(shotId), body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: sk.board(projectId) }),
  });
}

export function useGenerateAll(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: GenerateAllBody) =>
      unwrap(
        api.POST("/api/projects/{project_id}/generate-all", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: sk.board(projectId) }),
  });
}

/** Camera move → intensity → the prompt fragment it adds (for tooltips). */
export function useCameraMoves() {
  return useQuery({
    queryKey: vk.cameraMoves,
    staleTime: Infinity,
    queryFn: () => unwrap(api.GET("/api/camera-moves")),
  });
}
