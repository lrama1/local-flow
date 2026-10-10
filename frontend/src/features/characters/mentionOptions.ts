import { mediaUrl, primaryImage, useLibrary } from "@/api/characters";
import type { Ingredient } from "@/api/client";
import { thumbUrl } from "@/api/imagegen";
import { useIngredients } from "@/api/story";
import type { MentionOption } from "@/lib/mentions";

export function ingredientOptions(ingredients: Ingredient[], excludeId?: string): MentionOption[] {
  return ingredients
    .filter((i) => i.kind !== "style" && !i.draft && i.id !== excludeId)
    .map((i) => ({
      key: i.key,
      name: i.name,
      kind: i.kind,
      thumb: thumbUrl(i.project_id, i.primary_asset_id),
    }));
}

/** What a project prompt can `@mention`: its saved characters, locations and props. */
export function useProjectMentionOptions(projectId: string, excludeId?: string): MentionOption[] | undefined {
  const { data } = useIngredients(projectId);
  return data ? ingredientOptions(data, excludeId) : undefined;
}

/** The Playground has no cast of its own, so its prompts mention Cast library characters. */
export function useLibraryMentionOptions(enabled = true): MentionOption[] | undefined {
  const { data } = useLibrary("", enabled);
  return data?.map((c) => ({
    key: c.key,
    name: c.name,
    kind: "character",
    thumb: mediaUrl(primaryImage(c)?.thumb_url),
  }));
}
