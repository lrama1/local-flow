import {
  clipDuration,
  snapCandidates,
  snapTime,
  snapToFrame,
  type Clip,
  type Timeline,
  type Track,
} from "./math";

export type DragMode = "move" | "trim-start" | "trim-end";

export interface Drag {
  clipId: string;
  kind: Track["kind"];
  mode: DragMode;
  x0: number;
  orig: Clip;
  /** Pointer travel so far, in seconds. */
  dt: number;
  moved: boolean;
}

export interface DragPreview {
  start: number;
  duration: number;
  /** What dropping changes (video moves are reorders, handled by the caller). */
  patch: Partial<Clip>;
  snapTo: number | null;
}

const MIN_AUDIO_S = 0.05;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/**
 * Where the dragged clip is drawn and what dropping it changes. Trims snap to clip edges and the
 * playhead, then to the frame grid (video) or milliseconds (audio); audio moves snap either edge.
 */
export function dragPreview(
  d: Drag,
  timeline: Timeline,
  sourceDuration: number | null | undefined,
  pxPerSec: number,
  playhead: number,
): DragPreview {
  const fps = timeline.fps;
  const o = d.orig;
  const max = sourceDuration ?? Number.POSITIVE_INFINITY;
  const cands = snapCandidates(timeline, playhead, o.id);
  const minLen = d.kind === "video" ? 1 / fps : MIN_AUDIO_S;
  const grid = (s: number) => (d.kind === "video" ? snapToFrame(s, fps) : Math.round(s * 1000) / 1000);
  if (d.mode === "trim-start") {
    let inS = clamp(o.in_s + d.dt, 0, o.out_s - minLen);
    const snap = snapTime(o.start_s + (inS - o.in_s), cands, pxPerSec);
    inS = grid(clamp(o.in_s + (snap.t - o.start_s), 0, o.out_s - minLen));
    let start = o.start_s + (inS - o.in_s);
    if (d.kind === "audio" && start < 0) {
      inS -= start;
      start = 0;
    }
    const patch: Partial<Clip> = d.kind === "audio" ? { in_s: inS, start_s: start } : { in_s: inS };
    return { start, duration: o.out_s - inS, patch, snapTo: snap.to };
  }
  if (d.mode === "trim-end") {
    let outS = clamp(o.out_s + d.dt, o.in_s + minLen, max);
    const snap = snapTime(o.start_s + (outS - o.in_s), cands, pxPerSec);
    outS = grid(clamp(o.in_s + (snap.t - o.start_s), o.in_s + minLen, max));
    return { start: o.start_s, duration: outS - o.in_s, patch: { out_s: outS }, snapTo: snap.to };
  }
  const dur = clipDuration(o);
  const raw = Math.max(0, o.start_s + d.dt);
  if (d.kind === "video") return { start: raw, duration: dur, patch: {}, snapTo: null };
  const head = snapTime(raw, cands, pxPerSec);
  const tail = snapTime(raw + dur, cands, pxPerSec);
  let start = raw;
  let snapTo: number | null = null;
  if (head.to !== null && (tail.to === null || Math.abs(head.t - raw) <= Math.abs(tail.t - raw - dur))) {
    start = head.t;
    snapTo = head.to;
  } else if (tail.to !== null) {
    start = Math.max(0, tail.t - dur);
    snapTo = tail.to;
  }
  start = grid(start);
  return { start, duration: dur, patch: { start_s: start }, snapTo };
}
