import type { Ingredient } from "@/api/client";
import type { KeyframeOut } from "@/api/imagegen";

export type KeyframeWhich = "start" | "end";

/** The last compose request per shot+frame; kept by the board because the panel remounts on shot updates. */
export interface KeyframeRun {
  which: KeyframeWhich;
  jobId: string;
  out: KeyframeOut;
}

/** Mirrors the backend (Section 11.2): at most this many references go on the reference sheet. */
export const MAX_REFERENCES = 3;
const REF_ORDER: Record<string, number> = { character: 0, prop: 1, location: 2 };

/** Ingredients the backend will use as references, in its order (characters, props, then the location). */
export function referenceIngredients(
  ingredientIds: string[],
  locationId: string | null | undefined,
  ingredients: Ingredient[],
): Ingredient[] {
  const byId = new Map(ingredients.map((i) => [i.id, i]));
  const ids = [...ingredientIds];
  if (locationId && !ids.includes(locationId)) ids.push(locationId);
  return ids
    .map((id) => byId.get(id))
    .filter((i): i is Ingredient => !!i && i.kind !== "style")
    .sort((a, b) => (REF_ORDER[a.kind] ?? 9) - (REF_ORDER[b.kind] ?? 9));
}
