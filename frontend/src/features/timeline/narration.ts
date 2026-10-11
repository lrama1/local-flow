import type { NarrationShot } from "@/api/narration";

/** About 2.3 words per second of the shot, less a beat of silence (mirrors the backend's budget). */
export function budgetWords(durationS: number): number {
  return Math.max(0, Math.floor((durationS - 0.6) * 2.3));
}

export const wordCount = (text: string): number => text.trim().split(/\s+/).filter(Boolean).length;

const STATUS_LABELS: Record<NarrationShot["status"], string> = {
  empty: "silent",
  text: "not voiced",
  voiced: "voiced",
  stale: "out of date",
};

export const statusLabel = (s: NarrationShot["status"]): string => STATUS_LABELS[s];

/** Lines that still need a voice: written but not voiced, or edited since they were. */
export function needsVoice(shots: NarrationShot[] | undefined): NarrationShot[] {
  return (shots ?? []).filter((s) => s.status === "text" || s.status === "stale");
}
