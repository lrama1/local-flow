import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";
import { sk } from "./story";
import { tk } from "./timeline";

export type AudioPlan = Schemas["AudioPlanOut"];
export type ShotAudioParams = Schemas["ShotAudioParams"];
export type ShotAudioBody = Schemas["ShotAudioBody"];
export type AudioSlot = "dialogue" | "sfx";
export type MusicGenBody = Schemas["MusicGenBody"];

export const ak = {
  plan: (shotId: string, params: ShotAudioParams, version: string) =>
    ["audio-plan", shotId, params, version] as const,
  candidates: (projectId: string, shotId: string, slot: AudioSlot) =>
    ["assets", projectId, "audio", shotId, slot] as const,
};

const shotPath = (shotId: string) => ({ params: { path: { shot_id: shotId } } });

/** Who says what with which voice, whether it fits the clip, and the SFX prompt. `version` refreshes it
 * (the shot's updated_at plus anything else that changes voices). */
export function useAudioPlan(shotId: string, params: ShotAudioParams, version: string) {
  return useQuery({
    queryKey: ak.plan(shotId, params, version),
    queryFn: () => unwrap(api.POST("/api/shots/{shot_id}/audio-plan", { ...shotPath(shotId), body: params })),
    placeholderData: keepPreviousData,
    retry: false,
  });
}

/** The shot's dialogue or SFX takes, newest first. */
export function useAudioCandidates(projectId: string, shotId: string, slot: AudioSlot) {
  return useQuery({
    queryKey: ak.candidates(projectId, shotId, slot),
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/assets", {
          params: {
            path: { project_id: projectId },
            query: { type: "audio", shot_id: shotId, slot, limit: 12 },
          },
        }),
      ),
  });
}

export function useGenerateShotAudio(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ shotId, body }: { shotId: string; body: ShotAudioBody }) =>
      unwrap(api.POST("/api/shots/{shot_id}/generate-audio", { ...shotPath(shotId), body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: sk.board(projectId) }),
  });
}

export function useSelectAudio(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ shotId, assetId, slot }: { shotId: string; assetId: string; slot?: AudioSlot }) =>
      unwrap(
        api.POST("/api/shots/{shot_id}/select-audio", {
          ...shotPath(shotId),
          body: { asset_id: assetId, slot: slot ?? null },
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: sk.board(projectId) });
      void qc.invalidateQueries({ queryKey: tk.timeline(projectId) });
      void qc.invalidateQueries({ queryKey: ["audio-plan"] });
    },
  });
}

export function useGenerateMusic(projectId: string) {
  return useMutation({
    mutationFn: (body: MusicGenBody) =>
      unwrap(
        api.POST("/api/projects/{project_id}/generate-music", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
  });
}
