import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";
import { qk } from "./queries";

export type Collection = Schemas["CollectionOut"];
export type AssetFacets = Schemas["AssetFacets"];
export type AssetSource = "generated" | "uploaded" | "extracted" | "export" | "imported";

/** Asset library filters (Section 13); empty values are left out of the request. */
export interface LibraryFilters {
  type?: "image" | "video" | "audio";
  source?: AssetSource;
  favorite?: boolean;
  tag?: string;
  collection?: string;
  model?: string;
  scene_id?: string;
  featuring?: string;
  q?: string;
  sort?: "newest" | "oldest";
}

export const PAGE_SIZE = 60;

export const lk = {
  library: (projectId: string, f: LibraryFilters) => ["assets", projectId, "library", f] as const,
  facets: (projectId: string) => ["assets", projectId, "facets"] as const,
  collections: (projectId: string) => ["collections", projectId] as const,
  assetCollections: (assetId: string) => ["asset-collections", assetId] as const,
};

export function cleanFilters(f: LibraryFilters): LibraryFilters {
  return Object.fromEntries(
    Object.entries(f).filter(([, v]) => v !== undefined && v !== "" && v !== null),
  ) as LibraryFilters;
}

export function useLibraryAssets(projectId: string, filters: LibraryFilters) {
  const f = cleanFilters(filters);
  return useInfiniteQuery({
    queryKey: lk.library(projectId, f),
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      unwrap(
        api.GET("/api/projects/{project_id}/assets", {
          params: { path: { project_id: projectId }, query: { ...f, offset: pageParam, limit: PAGE_SIZE } },
        }),
      ),
    getNextPageParam: (last) =>
      last.offset + last.items.length < last.total ? last.offset + PAGE_SIZE : undefined,
  });
}

export function useFacets(projectId: string) {
  return useQuery({
    queryKey: lk.facets(projectId),
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/assets/facets", { params: { path: { project_id: projectId } } }),
      ),
  });
}

export function useCollections(projectId: string) {
  return useQuery({
    queryKey: lk.collections(projectId),
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/collections", { params: { path: { project_id: projectId } } }),
      ),
  });
}

export function useAssetCollections(assetId: string | null | undefined) {
  return useQuery({
    queryKey: lk.assetCollections(assetId ?? ""),
    enabled: !!assetId,
    queryFn: () =>
      unwrap(api.GET("/api/assets/{asset_id}/collections", { params: { path: { asset_id: assetId! } } })),
  });
}

/** Refreshes everything the library shows after an edit. */
function useRefresh(projectId: string) {
  const qc = useQueryClient();
  return () => {
    void qc.invalidateQueries({ queryKey: qk.assetsRoot(projectId) });
    void qc.invalidateQueries({ queryKey: ["asset"] });
    void qc.invalidateQueries({ queryKey: lk.collections(projectId) });
    void qc.invalidateQueries({ queryKey: ["asset-collections"] });
  };
}

export function useCollectionMutations(projectId: string) {
  const refresh = useRefresh(projectId);
  const path = (id: string) => ({ params: { path: { collection_id: id } } });
  return {
    create: useMutation({
      mutationFn: (body: Schemas["CollectionCreate"]) =>
        unwrap(
          api.POST("/api/projects/{project_id}/collections", {
            params: { path: { project_id: projectId } },
            body,
          }),
        ),
      onSuccess: refresh,
    }),
    update: useMutation({
      mutationFn: ({ id, body }: { id: string; body: Schemas["CollectionUpdate"] }) =>
        unwrap(api.PATCH("/api/collections/{collection_id}", { ...path(id), body })),
      onSuccess: refresh,
    }),
    remove: useMutation({
      mutationFn: (id: string) => unwrap(api.DELETE("/api/collections/{collection_id}", path(id))),
      onSuccess: refresh,
    }),
    addAssets: useMutation({
      mutationFn: ({ id, assetIds }: { id: string; assetIds: string[] }) =>
        unwrap(
          api.POST("/api/collections/{collection_id}/assets", { ...path(id), body: { asset_ids: assetIds } }),
        ),
      onSuccess: refresh,
    }),
    removeAssets: useMutation({
      mutationFn: ({ id, assetIds }: { id: string; assetIds: string[] }) =>
        unwrap(
          api.DELETE("/api/collections/{collection_id}/assets", {
            ...path(id),
            body: { asset_ids: assetIds },
          }),
        ),
      onSuccess: refresh,
    }),
  };
}

export function useAssetMutations(projectId: string) {
  const refresh = useRefresh(projectId);
  return {
    bulk: useMutation({
      mutationFn: (body: Schemas["BulkAssetsBody"]) =>
        unwrap(
          api.POST("/api/projects/{project_id}/assets/bulk", {
            params: { path: { project_id: projectId } },
            body,
          }),
        ),
      onSuccess: refresh,
    }),
    tags: useMutation({
      mutationFn: ({ id, tags }: { id: string; tags: string[] }) =>
        unwrap(api.PATCH("/api/assets/{asset_id}", { params: { path: { asset_id: id } }, body: { tags } })),
      onSuccess: refresh,
    }),
    extractFrame: useMutation({
      mutationFn: ({ id, timeS }: { id: string; timeS: number }) =>
        unwrap(
          api.POST("/api/assets/{asset_id}/extract-frame", {
            params: { path: { asset_id: id } },
            body: { time_s: timeS },
          }),
        ),
      onSuccess: refresh,
    }),
    remove: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.DELETE("/api/assets/{asset_id}", { params: { path: { asset_id: id } } })),
      onSuccess: refresh,
    }),
  };
}
