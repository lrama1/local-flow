import type { Asset, Job, Schemas } from "@/api/client";

export type TransitionType = "cut" | "crossfade" | "fade_from_black";

export interface Transition {
  type: TransitionType;
  duration_s: number;
}

export interface Clip {
  id: string;
  asset_id: string;
  start_s: number;
  in_s: number;
  out_s: number;
  transition_in: Transition;
  gain_db: number;
  shot_id: string | null;
}

export interface Track {
  id: string;
  kind: "video" | "audio";
  name: string;
  muted: boolean;
  gain_db: number;
  clips: Clip[];
}

export interface Timeline {
  fps: number;
  width: number;
  height: number;
  tracks: Track[];
}

/** The server always returns every field; this fills defaults so the editor can rely on them. */
export function complete(t: Schemas["TimelineData"]): Timeline {
  return {
    fps: t.fps ?? 24,
    width: t.width ?? 1280,
    height: t.height ?? 720,
    tracks: t.tracks.map((tr) => ({
      id: tr.id,
      kind: tr.kind,
      name: tr.name ?? tr.id,
      muted: tr.muted ?? false,
      gain_db: tr.gain_db ?? 0,
      clips: (tr.clips ?? []).map((c) => ({
        id: c.id ?? newClipId(),
        asset_id: c.asset_id,
        start_s: c.start_s ?? 0,
        in_s: c.in_s ?? 0,
        out_s: c.out_s,
        transition_in: {
          type: c.transition_in?.type ?? "cut",
          duration_s: c.transition_in?.duration_s ?? 0,
        },
        gain_db: c.gain_db ?? 0,
        shot_id: c.shot_id ?? null,
      })),
    })),
  };
}

export function newClipId(): string {
  return `c${Math.random().toString(16).slice(2, 12).padEnd(10, "0")}`;
}

export const videoTrack = (t: Timeline): Track => t.tracks.find((x) => x.kind === "video")!;
export const audioTracks = (t: Timeline): Track[] => t.tracks.filter((x) => x.kind === "audio");

// --- time ↔ pixels ---------------------------------------------------------------------------------
export const timeToPx = (t: number, pxPerSec: number): number => t * pxPerSec;
export const pxToTime = (px: number, pxPerSec: number): number => px / pxPerSec;

export const frames = (seconds: number, fps: number): number => Math.round(seconds * fps + 1e-9);
export const snapToFrame = (seconds: number, fps: number): number => frames(seconds, fps) / fps;
export const clipDuration = (c: Clip): number => c.out_s - c.in_s;
export const clipEnd = (c: Clip): number => c.start_s + clipDuration(c);

/**
 * Snaps `t` to the nearest candidate within `thresholdPx` (at this zoom); otherwise returns it unchanged.
 * `to` is the candidate it snapped to (for a guide line).
 */
export function snapTime(
  t: number,
  candidates: number[],
  pxPerSec: number,
  thresholdPx = 8,
): { t: number; to: number | null } {
  let best: number | null = null;
  let bestDist = thresholdPx / pxPerSec;
  for (const c of candidates) {
    const d = Math.abs(c - t);
    if (d <= bestDist) {
      best = c;
      bestDist = d;
    }
  }
  return best === null ? { t, to: null } : { t: best, to: best };
}

/** Clip edges (video and audio), zero and the playhead, except the clip being dragged. */
export function snapCandidates(t: Timeline, playhead: number, exceptClipId?: string): number[] {
  const out = [0, playhead];
  for (const track of t.tracks) {
    for (const c of track.clips) {
      if (c.id === exceptClipId) continue;
      out.push(c.start_s, clipEnd(c));
    }
  }
  return out;
}

/**
 * Mirrors the backend's normalization of the video track (frame grid, sequential starts, crossfade
 * limits), so the layout updates instantly while the save is in flight. The server's answer wins.
 */
export function sequence(t: Timeline): Timeline {
  const fps = t.fps;
  let cursor = 0;
  let prev: number | null = null;
  const tracks = t.tracks.map((track) => {
    if (track.kind !== "video") return track;
    const clips = track.clips.map((c) => {
      const a = frames(c.in_s, fps);
      const n = Math.max(1, frames(c.out_s, fps) - a);
      let type = c.transition_in.type;
      let k = frames(c.transition_in.duration_s, fps);
      if (type === "crossfade" && prev === null) type = "fade_from_black";
      if (type === "cut" || k === 0) {
        type = "cut";
        k = 0;
      } else if (type === "crossfade") {
        k = Math.min(k, Math.floor(Math.min(prev!, n) / 2));
        if (k === 0) type = "cut";
      } else {
        k = Math.min(k, n);
      }
      const start = cursor - (type === "crossfade" ? k : 0);
      cursor = start + n;
      prev = n;
      return {
        ...c,
        in_s: a / fps,
        out_s: (a + n) / fps,
        start_s: start / fps,
        transition_in: { type, duration_s: k / fps },
      };
    });
    return { ...track, clips };
  });
  return { ...t, tracks };
}

export function totalDuration(t: Timeline): number {
  const clips = videoTrack(t).clips;
  return clips.length ? clipEnd(clips[clips.length - 1]!) : 0;
}

/** The video clip showing at `time`; during a crossfade, the incoming one. */
export function videoClipAt(t: Timeline, time: number): Clip | undefined {
  const clips = videoTrack(t).clips;
  for (let i = clips.length - 1; i >= 0; i--) {
    const c = clips[i]!;
    if (time >= c.start_s - 1e-6 && time < clipEnd(c) - 1e-6) return c;
  }
  return undefined;
}

/** Where a dragged video clip lands: the number of other clips whose centre is left of its centre. */
export function dropIndex(clips: Clip[], draggedId: string, centre: number): number {
  return clips.filter((c) => c.id !== draggedId && c.start_s + clipDuration(c) / 2 < centre).length;
}

export function moveClip(t: Timeline, clipId: string, to: number): Timeline {
  return {
    ...t,
    tracks: t.tracks.map((track) => {
      const from = track.clips.findIndex((c) => c.id === clipId);
      if (from < 0 || track.kind !== "video") return track;
      const clips = track.clips.slice();
      const [c] = clips.splice(from, 1);
      clips.splice(Math.max(0, Math.min(to, clips.length)), 0, c!);
      return { ...track, clips };
    }),
  };
}

export function updateClip(t: Timeline, clipId: string, patch: Partial<Clip>): Timeline {
  return {
    ...t,
    tracks: t.tracks.map((track) => ({
      ...track,
      clips: track.clips.map((c) => (c.id === clipId ? { ...c, ...patch } : c)),
    })),
  };
}

export function updateTrack(t: Timeline, trackId: string, patch: Partial<Track>): Timeline {
  return { ...t, tracks: t.tracks.map((tr) => (tr.id === trackId ? { ...tr, ...patch } : tr)) };
}

export function removeClip(t: Timeline, clipId: string): Timeline {
  return {
    ...t,
    tracks: t.tracks.map((track) => ({ ...track, clips: track.clips.filter((c) => c.id !== clipId) })),
  };
}

export function findClip(t: Timeline, clipId: string | null): { clip: Clip; track: Track } | undefined {
  if (!clipId) return undefined;
  for (const track of t.tracks) {
    const clip = track.clips.find((c) => c.id === clipId);
    if (clip) return { clip, track };
  }
  return undefined;
}

/** `m:ss.ff` (the last part is frames) for the transport display. */
export function timecode(seconds: number, fps: number): string {
  const total = Math.max(0, frames(seconds, fps));
  const f = total % fps;
  const s = Math.floor(total / fps);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}.${String(f).padStart(2, "0")}`;
}

/** Ruler tick spacing (seconds) that keeps labels at least ~60 px apart. */
export function tickStep(pxPerSec: number): number {
  for (const step of [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]) {
    if (step * pxPerSec >= 60) return step;
  }
  return 600;
}

/** "Shot 1.2" for video clips; the upload name, spoken text or prompt for audio clips. */
export function clipLabel(
  clip: Clip,
  track: Track,
  shotNumbers: Map<string, string>,
  media: Map<string, Asset> | undefined,
): string {
  if (track.kind === "video") {
    const n = clip.shot_id ? shotNumbers.get(clip.shot_id) : undefined;
    return n ? `Shot ${n}` : "Clip";
  }
  return assetLabel(media?.get(clip.asset_id));
}

export function assetLabel(asset: Asset | undefined): string {
  const gen = (asset?.generation ?? {}) as Record<string, unknown>;
  const params = (gen.params ?? {}) as Record<string, unknown>;
  return String(gen.uploaded_name ?? params.text ?? params.prompt ?? "Audio");
}

/** Mirrors the export's fit: within 3% of the frame's aspect ratio it's cropped to fill, else padded. */
export function fillsFrame(asset: Asset | undefined, width: number, height: number): boolean {
  if (!asset?.width || !asset.height) return false;
  return Math.abs(asset.width / asset.height / (width / height) - 1) <= 0.03;
}

/** Mirrors the backend: the even size with this short side in the timeline's aspect ratio. */
export function sizeFor(width: number, height: number, shortSide: number): [number, number] {
  const even = (x: number) => Math.max(2, 2 * Math.round(x / 2));
  return width >= height
    ? [even((shortSide * width) / height), even(shortSide)]
    : [even(shortSide), even((shortSide * height) / width)];
}

/** The project's export jobs: the queued/running one, and the newest one if it failed. */
export function exportJobs(jobs: Job[] | undefined, projectId: string) {
  const mine = (jobs ?? [])
    .filter((j) => j.type === "export" && j.project_id === projectId)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  return {
    active: mine.find((j) => j.status === "queued" || j.status === "running"),
    failed: mine[0]?.status === "failed" ? mine[0] : undefined,
  };
}

/** HTML media volume can't go above 1, so the preview caps boosts; the export applies them. */
export const dbToVolume = (db: number): number => Math.min(1, Math.max(0, 10 ** (db / 20)));
