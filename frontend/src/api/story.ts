import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Board, type LlmProvider, type Schemas, type Shot, type Story } from "./client";

export const sk = {
  secrets: ["secrets"] as const,
  appSettings: ["app-settings"] as const,
  providers: ["llm-providers"] as const,
  models: (provider: string) => ["llm-models", provider] as const,
  usage: ["llm-usage"] as const,
  story: (pid: string) => ["story", pid] as const,
  revisions: (pid: string) => ["story-revisions", pid] as const,
  revision: (pid: string, rev: number) => ["story-revision", pid, rev] as const,
  ingredients: (pid: string) => ["ingredients", pid] as const,
  board: (pid: string) => ["board", pid] as const,
};

// --- settings & keys ------------------------------------------------------------------------------
export function useSecrets() {
  return useQuery({ queryKey: sk.secrets, queryFn: () => unwrap(api.GET("/api/settings/secrets")) });
}

function useSecretMutation<V>(fn: (v: V) => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: sk.secrets });
      void qc.invalidateQueries({ queryKey: sk.providers });
    },
  });
}

export function useSetSecret() {
  return useSecretMutation(({ name, value }: { name: string; value: string }) =>
    unwrap(api.PUT("/api/settings/secrets/{name}", { params: { path: { name } }, body: { value } })),
  );
}

export function useDeleteSecret() {
  return useSecretMutation((name: string) =>
    unwrap(api.DELETE("/api/settings/secrets/{name}", { params: { path: { name } } })),
  );
}

export function useTestSecret() {
  return useSecretMutation((name: string) =>
    unwrap(api.POST("/api/settings/secrets/{name}/test", { params: { path: { name } } })),
  );
}

export function useAppSettings() {
  return useQuery({ queryKey: sk.appSettings, queryFn: () => unwrap(api.GET("/api/settings")) });
}

export function usePatchSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Schemas["AppSettingsPatch"]) => unwrap(api.PATCH("/api/settings", { body })),
    onSuccess: (data) => qc.setQueryData(sk.appSettings, data),
  });
}

export function useLlmProviders() {
  return useQuery({
    queryKey: sk.providers,
    queryFn: () => unwrap(api.GET("/api/llm/providers")),
    staleTime: 60_000,
  });
}

export function useLlmModels(provider: LlmProvider | undefined, enabled = true) {
  return useQuery({
    queryKey: sk.models(provider ?? ""),
    enabled: !!provider && enabled,
    staleTime: 10 * 60_000,
    queryFn: () => unwrap(api.GET("/api/llm/models", { params: { query: { provider: provider! } } })),
  });
}

export function useLlmUsage() {
  return useQuery({
    queryKey: sk.usage,
    queryFn: () => unwrap(api.GET("/api/llm/usage", { params: { query: { limit: 20 } } })),
  });
}

// --- story ----------------------------------------------------------------------------------------
export function useStory(pid: string) {
  return useQuery({
    queryKey: sk.story(pid),
    queryFn: () =>
      unwrap(api.GET("/api/projects/{project_id}/story", { params: { path: { project_id: pid } } })),
  });
}

export function useRevisions(pid: string, enabled = true) {
  return useQuery({
    queryKey: sk.revisions(pid),
    enabled,
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/story/revisions", { params: { path: { project_id: pid } } }),
      ),
  });
}

export function useRevision(pid: string, revision: number | undefined) {
  return useQuery({
    queryKey: sk.revision(pid, revision ?? 0),
    enabled: !!revision,
    staleTime: Infinity,
    queryFn: () =>
      unwrap(
        api.GET("/api/projects/{project_id}/story/revisions/{revision}", {
          params: { path: { project_id: pid, revision: revision! } },
        }),
      ),
  });
}

/** Invalidates everything a story step can change. */
export function useStoryInvalidate(pid: string) {
  const qc = useQueryClient();
  return () => {
    for (const key of [sk.story(pid), sk.revisions(pid), sk.ingredients(pid), sk.board(pid), sk.usage]) {
      void qc.invalidateQueries({ queryKey: key });
    }
  };
}

type Signal = { signal?: AbortSignal };
const path = (pid: string) => ({ params: { path: { project_id: pid } } });

export const storySteps = {
  refine: (pid: string, body: Schemas["RefineBody"], { signal }: Signal = {}) =>
    unwrap(api.POST("/api/projects/{project_id}/story/refine", { ...path(pid), body, signal })),
  bible: (pid: string, body: Schemas["BibleBody"], { signal }: Signal = {}) =>
    unwrap(api.POST("/api/projects/{project_id}/story/bible", { ...path(pid), body, signal })),
  breakdown: (pid: string, body: Schemas["BreakdownBody"], { signal }: Signal = {}) =>
    unwrap(api.POST("/api/projects/{project_id}/story/breakdown", { ...path(pid), body, signal })),
  patch: (pid: string, body: Schemas["StoryPatch"]) =>
    unwrap(api.PATCH("/api/projects/{project_id}/story", { ...path(pid), body })),
  restore: (pid: string, revision: number) =>
    unwrap(api.POST("/api/projects/{project_id}/story/restore", { ...path(pid), body: { revision } })),
};

export function hasStory(s: Story | undefined): boolean {
  return !!s && s.revision != null;
}

// --- ingredients ----------------------------------------------------------------------------------
export function useIngredients(pid: string) {
  return useQuery({
    queryKey: sk.ingredients(pid),
    queryFn: () => unwrap(api.GET("/api/projects/{project_id}/ingredients", path(pid))),
  });
}

export function useIngredientMutations(pid: string) {
  const qc = useQueryClient();
  const done = () => {
    void qc.invalidateQueries({ queryKey: sk.ingredients(pid) });
    void qc.invalidateQueries({ queryKey: sk.board(pid) });
  };
  return {
    create: useMutation({
      mutationFn: (body: Schemas["IngredientCreate"]) =>
        unwrap(api.POST("/api/projects/{project_id}/ingredients", { ...path(pid), body })),
      onSuccess: done,
    }),
    update: useMutation({
      mutationFn: ({ id, body }: { id: string; body: Schemas["IngredientUpdate"] }) =>
        unwrap(
          api.PATCH("/api/ingredients/{ingredient_id}", { params: { path: { ingredient_id: id } }, body }),
        ),
      onSuccess: done,
    }),
    remove: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.DELETE("/api/ingredients/{ingredient_id}", { params: { path: { ingredient_id: id } } })),
      onSuccess: done,
    }),
  };
}

// --- board ----------------------------------------------------------------------------------------
export function useBoard(pid: string) {
  return useQuery({
    queryKey: sk.board(pid),
    queryFn: () => unwrap(api.GET("/api/projects/{project_id}/scenes", path(pid))),
  });
}

/** Board mutations with optimistic reordering (Section 17.3). */
export function useBoardMutations(pid: string) {
  const qc = useQueryClient();
  const invalidate = () => void qc.invalidateQueries({ queryKey: sk.board(pid) });
  const setBoard = (fn: (b: Board) => Board) => qc.setQueryData<Board>(sk.board(pid), (b) => (b ? fn(b) : b));

  return {
    createScene: useMutation({
      mutationFn: (body: Schemas["SceneCreate"]) =>
        unwrap(api.POST("/api/projects/{project_id}/scenes", { ...path(pid), body })),
      onSuccess: invalidate,
    }),
    updateScene: useMutation({
      mutationFn: ({ id, body }: { id: string; body: Schemas["SceneUpdate"] }) =>
        unwrap(api.PATCH("/api/scenes/{scene_id}", { params: { path: { scene_id: id } }, body })),
      onSuccess: invalidate,
    }),
    deleteScene: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.DELETE("/api/scenes/{scene_id}", { params: { path: { scene_id: id } } })),
      onSuccess: invalidate,
    }),
    reorderScenes: useMutation({
      mutationFn: (ids: string[]) =>
        unwrap(api.POST("/api/projects/{project_id}/scenes/reorder", { ...path(pid), body: { ids } })),
      onMutate: (ids) =>
        setBoard((b) => ({
          ...b,
          scenes: ids.map((id, index) => ({ ...b.scenes.find((s) => s.id === id)!, index })),
        })),
      onSettled: invalidate,
    }),
    createShot: useMutation({
      mutationFn: ({ sceneId, body }: { sceneId: string; body: Schemas["ShotCreate"] }) =>
        unwrap(api.POST("/api/scenes/{scene_id}/shots", { params: { path: { scene_id: sceneId } }, body })),
      onSuccess: invalidate,
    }),
    updateShot: useMutation({
      mutationFn: ({ id, body }: { id: string; body: Schemas["ShotUpdate"] }) =>
        unwrap(api.PATCH("/api/shots/{shot_id}", { params: { path: { shot_id: id } }, body })),
      onSuccess: (shot: Shot) =>
        setBoard((b) => ({
          ...b,
          scenes: b.scenes.map((s) => ({ ...s, shots: s.shots.map((x) => (x.id === shot.id ? shot : x)) })),
        })),
      onSettled: invalidate,
    }),
    deleteShot: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.DELETE("/api/shots/{shot_id}", { params: { path: { shot_id: id } } })),
      onSuccess: invalidate,
    }),
    reorderShots: useMutation({
      mutationFn: ({ sceneId, ids }: { sceneId: string; ids: string[] }) =>
        unwrap(
          api.POST("/api/scenes/{scene_id}/shots/reorder", {
            params: { path: { scene_id: sceneId } },
            body: { ids },
          }),
        ),
      onMutate: ({ sceneId, ids }) => setBoard((b) => moveShots(b, sceneId, ids)),
      onSettled: invalidate,
    }),
  };
}

/** Pure helper: the board after `sceneId` gets exactly `ids` (which may come from other scenes). */
export function moveShots(board: Board, sceneId: string, ids: string[]): Board {
  const all = new Map(board.scenes.flatMap((s) => s.shots.map((sh) => [sh.id, sh] as const)));
  const moving = new Set(ids);
  return {
    ...board,
    scenes: board.scenes.map((s) => {
      if (s.id === sceneId) {
        return {
          ...s,
          shots: ids.map((id, index) => ({ ...all.get(id)!, scene_id: sceneId, index })),
        };
      }
      return {
        ...s,
        shots: s.shots.filter((sh) => !moving.has(sh.id)).map((sh, index) => ({ ...sh, index })),
      };
    }),
  };
}

export function rewriteShot(id: string, instruction: string, { signal }: Signal = {}) {
  return unwrap(
    api.POST("/api/shots/{shot_id}/rewrite-prompt", {
      params: { path: { shot_id: id } },
      body: { instruction },
      signal,
    }),
  );
}
