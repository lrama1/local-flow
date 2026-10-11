import {
  skipToken,
  useIsMutating,
  useMutation,
  useMutationState,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";

export type ShotReview = Schemas["ShotReviewOut"];
export type VisionCapability = Schemas["VisionCapability"];

export const rk = {
  vision: (purpose: string) => ["llm-vision", purpose] as const,
  review: (shotId: string) => ["shot-review", shotId] as const,
  result: (shotId: string, assetId: string) => ["shot-review", shotId, assetId] as const,
};

/** Whether the model a purpose resolves to can read images (Section 8.3 step 5). */
export function useVisionCapability(purpose = "vision_review") {
  return useQuery({
    queryKey: rk.vision(purpose),
    queryFn: () => unwrap(api.GET("/api/llm/vision", { params: { query: { purpose } } })),
    staleTime: 60_000,
  });
}

/** Runs a consistency check. The result, pending state and error live in the query/mutation caches rather
 * than component state: saving the shot panel's draft first remounts the panel, and the check (which costs
 * an LLM call) must still show up in the new one. */
export function useReviewShot(shotId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationKey: rk.review(shotId),
    mutationFn: (assetId: string) =>
      unwrap(
        api.POST("/api/shots/{shot_id}/review", {
          params: { path: { shot_id: shotId } },
          body: { asset_id: assetId },
        }),
      ),
    onSuccess: (r) => qc.setQueryData(rk.result(shotId, r.asset_id), r),
  });
}

/** The latest check of `assetId` in this shot (null until one ran, or after it's cleared). */
export function useShotReview(shotId: string, assetId: string) {
  const qc = useQueryClient();
  const { data } = useQuery<ShotReview | null>({
    queryKey: rk.result(shotId, assetId),
    queryFn: skipToken,
    staleTime: Infinity,
  });
  const pending = useIsMutating({ mutationKey: rk.review(shotId) }) > 0;
  const states = useMutationState({ filters: { mutationKey: rk.review(shotId) }, select: (m) => m.state });
  const last = states.at(-1);
  return {
    result: data ?? null,
    pending,
    error: last?.status === "error" ? (last.error as Error) : null,
    clear: () => qc.setQueryData<ShotReview | null>(rk.result(shotId, assetId), null),
  };
}
