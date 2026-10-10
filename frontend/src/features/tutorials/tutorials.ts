import { useQuery } from "@tanstack/react-query";

export interface TutorialChapter {
  time_s: number;
  title: string;
}

/** One entry of `public/tutorials/index.json`, written by `scripts/tutorials/make_tutorials.py`. */
export interface Tutorial {
  id: string;
  title: string;
  summary: string;
  duration_s: number;
  video: string;
  poster?: string;
  captions?: string;
  chapters: TutorialChapter[];
  voice?: string;
  generated_at?: string;
}

export interface TutorialManifest {
  version: number;
  tutorials: Tutorial[];
}

export const TUTORIALS_URL = "/tutorials/index.json";

/** Static files shipped with the frontend (no API token needed). A missing manifest means "none yet". */
export async function fetchTutorials(): Promise<TutorialManifest> {
  const res = await fetch(TUTORIALS_URL, { cache: "no-cache" });
  // The production SPA fallback answers unknown paths with index.html, so check the type too.
  if (!res.ok || !(res.headers.get("content-type") ?? "").includes("json")) {
    return { version: 1, tutorials: [] };
  }
  return (await res.json()) as TutorialManifest;
}

export function useTutorials() {
  return useQuery({ queryKey: ["tutorials"], queryFn: fetchTutorials, staleTime: 60_000 });
}

/** The chapter playing at `time` (the last one that started). */
export function currentChapter(chapters: TutorialChapter[], time: number): number {
  let idx = -1;
  chapters.forEach((c, i) => {
    if (c.time_s <= time + 0.25) idx = i;
  });
  return idx;
}
