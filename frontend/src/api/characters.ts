import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, withToken, type Ingredient, type Schemas } from "./client";
import { sk } from "./story";

export type CharacterTemplate = Schemas["TemplateOut"];
export type TemplateCategory = NonNullable<Schemas["TemplateCreate"]["category"]>;
export type LibraryCharacter = Schemas["LibraryCharacterOut"];
export type LibraryImage = Schemas["LibraryImageOut"];
export type CharacterCapabilities = Schemas["CharacterCapabilities"];
export type CharacterFields = Schemas["CharacterFields"];
export type VoiceHints = Schemas["VoiceHintsIn"];
export type ExtraView = Schemas["ViewsBody"]["views"] extends (infer V)[] | undefined ? V : never;
export type CharacterView = "primary" | ExtraView;
export type ShotReferences = Schemas["ReferencesOut"];
export type RefOut = Schemas["RefOut"];

export const EXTRA_VIEWS: { view: ExtraView; label: string; hint: string }[] = [
  { view: "full_body", label: "Full body", hint: "Head to toe; used for wide shots" },
  { view: "three_view", label: "Turnaround", hint: "Front, side and back on one sheet" },
  { view: "side", label: "Side", hint: "Profile" },
  { view: "back", label: "Back", hint: "From behind" },
  { view: "face_closeup", label: "Face close-up", hint: "Used for close-ups" },
];

export const VIEW_LABELS: Record<string, string> = {
  primary: "Portrait",
  ...Object.fromEntries(EXTRA_VIEWS.map((v) => [v.view, v.label])),
};

export const TEMPLATE_CATEGORIES: { value: TemplateCategory; label: string }[] = [
  { value: "realistic", label: "Realistic" },
  { value: "fantastical", label: "Fantastical" },
  { value: "sci-fi", label: "Sci-fi" },
  { value: "historical", label: "Historical" },
  { value: "anime", label: "Anime" },
  { value: "cartoon", label: "Cartoon" },
  { value: "creature", label: "Creature" },
  { value: "animal", label: "Animal" },
  { value: "custom", label: "My templates" },
];

export const ck = {
  templates: ["character-templates"] as const,
  capabilities: (pid: string) => ["character-capabilities", pid] as const,
  libraryRoot: ["library"] as const,
  library: (q: string) => ["library", "list", q] as const,
  references: (shotId: string, shotType: string, ids: string[]) =>
    ["references", shotId, shotType, ids] as const,
  referencesRoot: ["references"] as const,
};

/** Library and template image URLs come from the backend; <img> needs the media token appended. */
export const mediaUrl = (url: string | null | undefined) => withToken(url);

export function primaryImage(lib: LibraryCharacter): LibraryImage | undefined {
  return lib.images.find((i) => i.is_primary) ?? lib.images[0];
}

/** Lowercase slug, exactly like the backend's `slugify` (keys are what `@mentions` use). */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48)
    .replace(/-+$/, "");
}

// --- templates ----------------------------------------------------------------------------------
export function useTemplates() {
  return useQuery({
    queryKey: ck.templates,
    staleTime: 5 * 60_000,
    queryFn: () => unwrap(api.GET("/api/character-templates")),
  });
}

export function useTemplateMutations() {
  const qc = useQueryClient();
  const done = () => void qc.invalidateQueries({ queryKey: ck.templates });
  return {
    create: useMutation({
      mutationFn: (body: Schemas["TemplateCreate"]) => unwrap(api.POST("/api/character-templates", { body })),
      onSuccess: done,
    }),
    remove: useMutation({
      mutationFn: (id: string) =>
        unwrap(
          api.DELETE("/api/character-templates/{template_id}", { params: { path: { template_id: id } } }),
        ),
      onSuccess: done,
    }),
  };
}

// --- wizard -------------------------------------------------------------------------------------
export function useCapabilities(projectId: string | undefined) {
  return useQuery({
    queryKey: ck.capabilities(projectId ?? ""),
    enabled: !!projectId,
    staleTime: 30_000,
    queryFn: () =>
      unwrap(api.GET("/api/characters/capabilities", { params: { query: { project_id: projectId } } })),
  });
}

export function createDraft(projectId: string, body: Schemas["DraftCreate"]) {
  return unwrap(
    api.POST("/api/projects/{project_id}/characters", { params: { path: { project_id: projectId } }, body }),
  );
}

export function patchIngredient(id: string, body: Schemas["IngredientUpdate"]) {
  return unwrap(
    api.PATCH("/api/ingredients/{ingredient_id}", { params: { path: { ingredient_id: id } }, body }),
  );
}

/** One ingredient (wizard drafts aren't in the project list). Keyed under the project's ingredients so
 * `ingredients.updated` events refresh it too. */
export function useIngredient(projectId: string | undefined, id: string | null) {
  return useQuery({
    queryKey: [...sk.ingredients(projectId ?? ""), "one", id ?? ""] as const,
    enabled: !!projectId && !!id,
    queryFn: () =>
      unwrap(api.GET("/api/ingredients/{ingredient_id}", { params: { path: { ingredient_id: id! } } })),
  });
}

export function deleteIngredient(id: string) {
  return unwrap(api.DELETE("/api/ingredients/{ingredient_id}", { params: { path: { ingredient_id: id } } }));
}

export function useExpand() {
  return useMutation({
    mutationFn: ({ projectId, body }: { projectId?: string; body: Schemas["ExpandBody"] }) =>
      unwrap(api.POST("/api/characters/expand", { params: { query: { project_id: projectId } }, body })),
  });
}

export function useDescribe() {
  return useMutation({
    mutationFn: (body: Schemas["DescribeBody"]) => unwrap(api.POST("/api/characters/describe", { body })),
  });
}

export function useGenerateViews() {
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: Schemas["ViewsBody"] }) =>
      unwrap(
        api.POST("/api/ingredients/{ingredient_id}/generate-views", {
          params: { path: { ingredient_id: id } },
          body,
        }),
      ),
  });
}

export function useCleanReference() {
  return useMutation({
    mutationFn: ({ id, assetId }: { id: string; assetId: string }) =>
      unwrap(
        api.POST("/api/ingredients/{ingredient_id}/clean-reference", {
          params: { path: { ingredient_id: id } },
          body: { asset_id: assetId },
        }),
      ),
  });
}

export function useSetView(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, view, assetId }: { id: string; view: CharacterView; assetId: string | null }) =>
      unwrap(
        api.PUT("/api/ingredients/{ingredient_id}/views/{view}", {
          params: { path: { ingredient_id: id, view } },
          body: { asset_id: assetId },
        }),
      ),
    onSuccess: (ing) => {
      qc.setQueryData<Ingredient[]>(sk.ingredients(projectId), (list) =>
        list?.map((i) => (i.id === ing.id ? ing : i)),
      );
      void qc.invalidateQueries({ queryKey: sk.ingredients(projectId) });
    },
  });
}

// --- Cast library -------------------------------------------------------------------------------
export function useLibrary(q = "", enabled = true) {
  return useQuery({
    queryKey: ck.library(q),
    enabled,
    placeholderData: keepPreviousData,
    queryFn: () => unwrap(api.GET("/api/library/characters", { params: { query: { q: q || undefined } } })),
  });
}

/** Mutations that change the library or a project's link to it; both sides are refreshed. */
export function useLibraryMutations(projectId?: string) {
  const qc = useQueryClient();
  const done = () => {
    void qc.invalidateQueries({ queryKey: ck.libraryRoot });
    void qc.invalidateQueries({ queryKey: projectId ? sk.ingredients(projectId) : ["ingredients"] });
  };
  const ingPath = (id: string) => ({ params: { path: { ingredient_id: id } } });
  return {
    update: useMutation({
      mutationFn: ({ id, body }: { id: string; body: Schemas["LibraryCharacterUpdate"] }) =>
        unwrap(
          api.PATCH("/api/library/characters/{library_id}", { params: { path: { library_id: id } }, body }),
        ),
      onSuccess: done,
    }),
    remove: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.DELETE("/api/library/characters/{library_id}", { params: { path: { library_id: id } } })),
      onSuccess: done,
    }),
    save: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.POST("/api/ingredients/{ingredient_id}/save-to-library", { ...ingPath(id), body: {} })),
      onSuccess: done,
    }),
    push: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.POST("/api/ingredients/{ingredient_id}/push-to-library", ingPath(id))),
      onSuccess: done,
    }),
    pull: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.POST("/api/ingredients/{ingredient_id}/update-from-library", ingPath(id))),
      onSuccess: done,
    }),
  };
}

export function useImportCharacter() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ projectId, body }: { projectId: string; body: Schemas["ImportBody"] }) =>
      unwrap(
        api.POST("/api/projects/{project_id}/ingredients/import", {
          params: { path: { project_id: projectId } },
          body,
        }),
      ),
    onSuccess: (_ing, { projectId }) => {
      void qc.invalidateQueries({ queryKey: sk.ingredients(projectId) });
      void qc.invalidateQueries({ queryKey: ["assets", projectId] });
    },
  });
}

// --- references ---------------------------------------------------------------------------------
export function useShotReferences(shotId: string, shotType: string, ingredientIds: string[]) {
  return useQuery({
    queryKey: ck.references(shotId, shotType, ingredientIds),
    placeholderData: keepPreviousData,
    queryFn: () =>
      unwrap(
        api.GET("/api/shots/{shot_id}/references", {
          params: {
            path: { shot_id: shotId },
            query: { shot_type: shotType, ingredient_ids: ingredientIds.length ? ingredientIds : undefined },
          },
        }),
      ),
  });
}
