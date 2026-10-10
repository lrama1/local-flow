import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, withToken, type Schemas } from "./client";
import { sk } from "./story";

export type KeyframeOut = Schemas["KeyframeOut"];
export type KeyframeBody = Schemas["KeyframeBody"];
export type IngredientGenerateBody = Schemas["IngredientGenerateBody"];

type View = "primary" | "full_body" | "three_view" | "side" | "back" | "face_closeup";

/** `view` narrows a character's candidates to one view ("primary" = images made without a view). */
export type CandidateTarget =
  { ingredient_id: string; view?: View } | { shot_id: string; slot: "keyframe" | "end_keyframe" };

export const gk = {
  candidates: (projectId: string, target: CandidateTarget) =>
    ["assets", projectId, "candidates", target] as const,
  asset: (id: string) => ["asset", id] as const,
};

/** Thumbnail URL for an asset id (the backend's URL scheme), with the media token. */
export function thumbUrl(projectId: string, assetId: string | null | undefined): string | undefined {
  return assetId ? withToken(`/api/thumbs/${projectId}/${assetId}`) : undefined;
}

export function fileUrl(projectId: string, assetId: string | null | undefined): string | undefined {
  return assetId ? withToken(`/api/files/${projectId}/${assetId}`) : undefined;
}

export function useCandidates(projectId: string | undefined, target: CandidateTarget, enabled = true) {
  return useQuery({
    queryKey: gk.candidates(projectId ?? "", target),
    enabled: !!projectId && enabled,
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/assets", {
          params: { path: { project_id: projectId! }, query: { type: "image", limit: 48, ...target } },
        }),
      ),
  });
}

export function useAsset(id: string | null | undefined) {
  return useQuery({
    queryKey: gk.asset(id ?? ""),
    enabled: !!id,
    staleTime: Infinity,
    queryFn: () => unwrap(api.GET("/api/assets/{asset_id}", { params: { path: { asset_id: id! } } })),
  });
}

// --- ingredients ------------------------------------------------------------------------------------
export function useGenerateIngredient() {
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: IngredientGenerateBody }) =>
      unwrap(
        api.POST("/api/ingredients/{ingredient_id}/generate", {
          params: { path: { ingredient_id: id } },
          body,
        }),
      ),
  });
}

export function useSetPrimary(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, assetId }: { id: string; assetId: string | null }) =>
      unwrap(
        api.PATCH("/api/ingredients/{ingredient_id}/primary", {
          params: { path: { ingredient_id: id } },
          body: { asset_id: assetId },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: sk.ingredients(projectId) }),
  });
}

export function useUploadIngredientImage(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, file, view }: { id: string; file: File; view?: View }) => {
      const form = new FormData();
      form.append("file", file);
      form.append("set_primary", "true");
      if (view) form.append("view", view);
      return unwrap(
        api.POST("/api/ingredients/{ingredient_id}/assets", {
          params: { path: { ingredient_id: id } },
          // FormData is sent as-is (multipart); the typed body is only a schema placeholder.
          body: form as unknown as Schemas["Body_upload_ingredient_image_api_ingredients__ingredient_id__assets_post"],
          bodySerializer: (b) => b as unknown as FormData,
        }),
      );
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: sk.ingredients(projectId) });
      void qc.invalidateQueries({ queryKey: ["assets", projectId] });
    },
  });
}

// --- shots -------------------------------------------------------------------------------------------
export function useGenerateKeyframe() {
  return useMutation({
    mutationFn: ({ shotId, body }: { shotId: string; body: KeyframeBody }) =>
      unwrap(
        api.POST("/api/shots/{shot_id}/generate-keyframe", { params: { path: { shot_id: shotId } }, body }),
      ),
  });
}

export function useSelectForShot(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      shotId,
      slot,
      assetId,
    }: {
      shotId: string;
      slot: "keyframe" | "end_keyframe" | "video";
      assetId: string | null;
    }) =>
      unwrap(
        api.POST("/api/shots/{shot_id}/select", {
          params: { path: { shot_id: shotId } },
          body: { slot, asset_id: assetId },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: sk.board(projectId) }),
  });
}
