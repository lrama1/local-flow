import type { Job, Shot } from "@/api/client";
import type { VideoPlan } from "@/api/video";

/** The shot's queued/running video jobs, and the newest one if it failed (its error stays visible). */
export function shotVideoJobs(jobs: Job[] | undefined, shotId: string) {
  const mine = (jobs ?? [])
    .filter((j) => j.target_ref?.["shot_id"] === shotId && j.target_ref?.["slot"] === "video")
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const active = mine.filter((j) => j.status === "queued" || j.status === "running");
  const failed = mine[0]?.status === "failed" ? mine[0] : undefined;
  return { active, failed };
}

/** The shot's queued/running dialogue or SFX jobs, and the newest one if it failed. */
export function shotAudioJobs(jobs: Job[] | undefined, shotId: string, slot: "dialogue" | "sfx") {
  const mine = (jobs ?? [])
    .filter((j) => j.target_ref?.["shot_id"] === shotId && j.target_ref?.["slot"] === slot)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  return {
    active: mine.filter((j) => j.status === "queued" || j.status === "running"),
    failed: mine[0]?.status === "failed" ? mine[0] : undefined,
  };
}

/** Extensions needed for the shot's duration with this plan's clip length (the backend caps at 4). */
export function segmentsFor(shot: Shot, plan: VideoPlan | undefined): number {
  if (!plan || shot.duration_s <= plan.duration_s + 0.05) return 0;
  const auto = Math.ceil((shot.duration_s - plan.duration_s) / Math.max(plan.max_clip_seconds, 0.5));
  return Math.min(4, Math.max(1, shot.extend_segments ?? auto));
}
