import { useEffect, useRef, type RefObject } from "react";
import type { Asset } from "@/api/client";
import { fileUrl } from "@/api/imagegen";
import { cn } from "@/lib/utils";
import { audioTracks, clipEnd, dbToVolume, fillsFrame, videoTrack, type Clip, type Timeline } from "./math";

// A playing element that drifts from the clock is nudged with playbackRate; only big jumps re-seek,
// because seeking a playing video stalls it (and the stall then looks like more drift).
const NUDGE_S = 0.04;
const SEEK_S = 0.5;

/** Keeps one media element in step with the timeline clock while `time` is inside its clip. */
function useFollow(
  ref: RefObject<HTMLMediaElement | null>,
  clip: Clip,
  time: number,
  playing: boolean,
  fps: number,
) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const inside = time >= clip.start_s - 1e-3 && time < clipEnd(clip);
    const local = clip.in_s + Math.max(0, time - clip.start_s);
    if (!inside) {
      if (!el.paused) el.pause();
      // Parked at its in point, so it starts without a seek when its turn comes.
      if (time < clip.start_s && Math.abs(el.currentTime - clip.in_s) > 0.05 && !el.seeking) {
        el.currentTime = clip.in_s;
      }
      return;
    }
    if (playing) {
      const drift = el.currentTime - local; // > 0: ahead of the clock
      if (Math.abs(drift) > SEEK_S && !el.seeking) el.currentTime = local;
      el.playbackRate = Math.abs(drift) > NUDGE_S ? (drift > 0 ? 0.9 : 1.1) : 1;
      if (el.paused) void el.play().catch(() => {});
    } else {
      if (!el.paused) el.pause();
      el.playbackRate = 1;
      if (Math.abs(el.currentTime - local) > 0.5 / fps && !el.seeking) el.currentTime = local;
    }
  }, [ref, clip, time, playing, fps]);
}

function ClipVideo({
  clip,
  src,
  time,
  playing,
  fps,
  opacity,
  volume,
  muted,
  fill,
}: {
  clip: Clip;
  src: string | undefined;
  time: number;
  playing: boolean;
  fps: number;
  opacity: number;
  volume: number;
  muted: boolean;
  /** Crop to fill (near-matching aspect ratio) like the export does, else letterbox. */
  fill: boolean;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  useFollow(ref, clip, time, playing, fps);
  useEffect(() => {
    if (ref.current) ref.current.volume = volume;
  }, [volume]);
  return (
    <video
      ref={ref}
      src={src}
      muted={muted}
      playsInline
      preload="auto"
      className={cn("absolute inset-0 size-full", fill ? "object-cover" : "object-contain")}
      style={{ opacity }}
      data-testid="preview-video"
      data-clip={clip.id}
    />
  );
}

function ClipAudio({
  clip,
  src,
  time,
  playing,
  fps,
  volume,
}: {
  clip: Clip;
  src: string | undefined;
  time: number;
  playing: boolean;
  fps: number;
  volume: number;
}) {
  const ref = useRef<HTMLAudioElement>(null);
  useFollow(ref, clip, time, playing, fps);
  useEffect(() => {
    if (ref.current) ref.current.volume = volume;
  }, [volume]);
  return <audio ref={ref} src={src} preload="auto" data-testid="preview-audio" />;
}

const ramp = (time: number, from: number, length: number) =>
  length > 0 ? Math.min(1, Math.max(0, (time - from) / length)) : 1;

export function PreviewPlayer({
  timeline,
  media,
  projectId,
  time,
  playing,
}: {
  timeline: Timeline;
  media: Map<string, Asset> | undefined;
  projectId: string;
  time: number;
  playing: boolean;
}) {
  const video = videoTrack(timeline);
  const clips = video.clips;
  let index = -1;
  for (let i = clips.length - 1; i >= 0; i--) {
    if (time >= clips[i]!.start_s - 1e-6 && time < clipEnd(clips[i]!)) {
      index = i;
      break;
    }
  }
  if (index < 0 && clips.length && time >= clipEnd(clips[clips.length - 1]!) - 1e-6) index = -2; // past end
  const active = index >= 0 ? clips[index] : undefined;
  const fading = active && active.transition_in.duration_s > 0 && active.transition_in.type !== "cut";
  const previous =
    active &&
    index > 0 &&
    active.transition_in.type === "crossfade" &&
    time < active.start_s + active.transition_in.duration_s
      ? clips[index - 1]
      : undefined;
  // Mount the previous (crossfade), active and next clips; the next one preloads at its in point.
  const upcoming =
    index >= 0 ? clips[index + 1] : index === -1 ? clips.find((c) => c.start_s > time) : undefined;
  const mounted = [previous, active, upcoming].filter((c): c is Clip => !!c);

  const sounding = audioTracks(timeline).flatMap((track) =>
    track.muted
      ? []
      : track.clips
          .filter((c) => c.start_s <= time + 1.5 && clipEnd(c) > time - 0.1)
          .map((c) => ({ clip: c, volume: dbToVolume(c.gain_db + track.gain_db) })),
  );

  return (
    <div
      className="relative mx-auto overflow-hidden rounded-lg bg-black"
      style={{
        aspectRatio: `${timeline.width} / ${timeline.height}`,
        // At most ~45% of the window's height, so the tracks stay in view.
        width: `min(100%, calc(45vh * ${timeline.width} / ${timeline.height}))`,
      }}
      data-testid="preview"
    >
      {mounted.map((c) => (
        <ClipVideo
          key={c.id}
          clip={c}
          src={fileUrl(projectId, c.asset_id)}
          time={time}
          playing={playing}
          fps={timeline.fps}
          opacity={
            c === active
              ? fading
                ? ramp(time, c.start_s, c.transition_in.duration_s)
                : 1
              : c === previous
                ? 1
                : 0
          }
          volume={dbToVolume(c.gain_db + video.gain_db)}
          muted={video.muted || !media?.get(c.asset_id)?.has_audio}
          fill={fillsFrame(media?.get(c.asset_id), timeline.width, timeline.height)}
        />
      ))}
      {sounding.map(({ clip, volume }) => (
        <ClipAudio
          key={clip.id}
          clip={clip}
          src={fileUrl(projectId, clip.asset_id)}
          time={time}
          playing={playing}
          fps={timeline.fps}
          volume={volume}
        />
      ))}
      {!clips.length && (
        <div className="text-muted-foreground absolute inset-0 flex items-center justify-center text-sm">
          Nothing on the timeline yet
        </div>
      )}
    </div>
  );
}
