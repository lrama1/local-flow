import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Timeline } from "@/features/timeline/math";
import { api, unwrap, type Schemas } from "./client";

export type TimelineOut = Schemas["TimelineOut"];
export type AutoAssembleBody = Schemas["AutoAssembleBody"];
export type AutoAssembleOut = Schemas["AutoAssembleOut"];
export type ExportBody = Schemas["ExportBody"];
export type ExportPreset = NonNullable<ExportBody["preset"]>;

export const tk = {
  timeline: (projectId: string) => ["timeline", projectId] as const,
  exports: (projectId: string) => ["assets", projectId, "exports"] as const,
  media: (projectId: string) => ["assets", projectId, "timeline-media"] as const,
};

const projectPath = (projectId: string) => ({ params: { path: { project_id: projectId } } });

export function useTimeline(projectId: string) {
  return useQuery({
    queryKey: tk.timeline(projectId),
    queryFn: () => unwrap(api.GET("/api/projects/{project_id}/timeline", projectPath(projectId))),
  });
}

export function useSaveTimeline(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (timeline: Timeline) =>
      unwrap(api.PUT("/api/projects/{project_id}/timeline", { ...projectPath(projectId), body: timeline })),
    onSuccess: (out) => qc.setQueryData(tk.timeline(projectId), out),
  });
}

export function useAutoAssemble(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: AutoAssembleBody) =>
      unwrap(
        api.POST("/api/projects/{project_id}/timeline/auto-assemble", { ...projectPath(projectId), body }),
      ),
    onSuccess: (out) => qc.setQueryData(tk.timeline(projectId), out),
  });
}

export function useExportTimeline(projectId: string) {
  return useMutation({
    mutationFn: (body: ExportBody) =>
      unwrap(api.POST("/api/projects/{project_id}/export", { ...projectPath(projectId), body })),
  });
}

/** Finished exports, newest first. */
export function useExports(projectId: string) {
  return useQuery({
    queryKey: tk.exports(projectId),
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/assets", {
          params: { path: { project_id: projectId }, query: { type: "video", source: "export", limit: 20 } },
        }),
      ),
  });
}

/** Uploads a file (image, video or audio; checked by content) into the project. */
export function useUploadAsset(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return unwrap(
        api.POST("/api/projects/{project_id}/assets/upload", {
          ...projectPath(projectId),
          // FormData is sent as-is (multipart); the typed body is only a schema placeholder.
          body: form as unknown as Schemas["Body_upload_asset_api_projects__project_id__assets_upload_post"],
          bodySerializer: (b) => b as unknown as FormData,
        }),
      );
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["assets", projectId] }),
  });
}

/** Every video and audio asset in the project, by id: durations, sizes and thumbnails for the editor. */
export function useTimelineMedia(projectId: string) {
  return useQuery({
    queryKey: tk.media(projectId),
    queryFn: async () => {
      const [videos, audio] = await Promise.all(
        (["video", "audio"] as const).map((type) =>
          unwrap(
            api.GET("/api/projects/{project_id}/assets", {
              params: { path: { project_id: projectId }, query: { type, limit: 500 } },
            }),
          ),
        ),
      );
      return new Map([...videos!.items, ...audio!.items].map((a) => [a.id, a]));
    },
  });
}
