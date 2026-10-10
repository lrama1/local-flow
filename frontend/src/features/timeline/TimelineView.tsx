import { PlusIcon, Volume2Icon, VolumeXIcon } from "lucide-react";
import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import type { Asset } from "@/api/client";
import { thumbUrl } from "@/api/imagegen";
import { Button } from "@/components/ui/button";
import { cn, formatDuration } from "@/lib/utils";
import { dragPreview, type Drag, type DragMode, type DragPreview } from "./drag";
import {
  clipDuration,
  clipLabel,
  dropIndex,
  moveClip,
  tickStep,
  timeToPx,
  totalDuration,
  updateClip,
  type Clip,
  type Timeline,
  type Track,
} from "./math";

const RULER_H = 24;
const ROW_H = { video: 64, audio: 40 } as const;
const HEADER_W = 148;
const AUDIO_COLORS: Record<string, string> = {
  a_dialogue: "bg-sky-500/30 border-sky-400/60",
  a_sfx: "bg-amber-500/30 border-amber-400/60",
  a_music: "bg-emerald-500/30 border-emerald-400/60",
};

type Down = (e: ReactPointerEvent, clip: Clip, track: Track, mode: DragMode) => void;
type Handler = (e: ReactPointerEvent) => void;

const ClipBlock = memo(function ClipBlock({
  clip,
  track,
  index,
  label,
  thumb,
  pxPerSec,
  selected,
  preview,
  onDown,
  onMove,
  onUp,
}: {
  clip: Clip;
  track: Track;
  index: number;
  label: string;
  thumb: string | undefined;
  pxPerSec: number;
  selected: boolean;
  preview: DragPreview | null;
  onDown: Down;
  onMove: Handler;
  onUp: Handler;
}) {
  const start = preview?.start ?? clip.start_s;
  const duration = preview?.duration ?? clipDuration(clip);
  const video = track.kind === "video";
  const t = clip.transition_in;
  const handle = (mode: DragMode, side: string) => (
    <span
      aria-hidden
      data-handle={mode}
      className={cn("absolute inset-y-0 z-10 w-1.5 cursor-ew-resize hover:bg-white/40", side)}
      onPointerDown={(e) => onDown(e, clip, track, mode)}
    />
  );
  return (
    <div
      role="button"
      tabIndex={-1}
      aria-label={`${video ? `Clip ${index + 1}` : track.name}: ${label}, ${formatDuration(duration)}`}
      aria-pressed={selected}
      data-testid={`clip-${clip.id}`}
      className={cn(
        "absolute top-1 bottom-1 cursor-grab touch-none overflow-hidden rounded-md border text-[11px] select-none",
        video
          ? "bg-muted border-border"
          : (AUDIO_COLORS[track.id] ?? "border-violet-400/60 bg-violet-500/30"),
        selected && "ring-primary ring-2",
        preview && "z-20 cursor-grabbing opacity-90 shadow-lg",
      )}
      style={{
        left: timeToPx(start, pxPerSec),
        width: Math.max(4, timeToPx(duration, pxPerSec)),
        backgroundImage: video && thumb ? `url("${thumb}")` : undefined,
        backgroundSize: video ? "auto 100%" : undefined,
        backgroundRepeat: "repeat-x",
      }}
      onPointerDown={(e) => onDown(e, clip, track, "move")}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
    >
      {video && t.type !== "cut" && t.duration_s > 0 && (
        <span
          aria-hidden
          title={t.type === "crossfade" ? "Crossfade" : "Fade from black"}
          className={cn(
            "absolute inset-y-0 left-0",
            t.type === "crossfade"
              ? "bg-gradient-to-r from-white/50 to-transparent"
              : "bg-gradient-to-r from-black to-transparent",
          )}
          style={{ width: timeToPx(t.duration_s, pxPerSec) }}
        />
      )}
      <span
        className={cn(
          "relative block truncate px-1.5 py-0.5 font-medium",
          video && "w-fit max-w-full rounded-br bg-black/60 text-white",
        )}
      >
        {label}
      </span>
      {handle("trim-start", "left-0")}
      {handle("trim-end", "right-0")}
    </div>
  );
});

export function TimelineView({
  timeline,
  media,
  projectId,
  pxPerSec,
  time,
  playing,
  selectedId,
  shotNumbers,
  onSelect,
  onSeek,
  onChange,
  onToggleMute,
  onAddAudio,
}: {
  timeline: Timeline;
  media: Map<string, Asset> | undefined;
  projectId: string;
  pxPerSec: number;
  time: number;
  playing: boolean;
  selectedId: string | null;
  /** shot id → board number ("1.2") */
  shotNumbers: Map<string, string>;
  onSelect: (clipId: string | null) => void;
  onSeek: (t: number) => void;
  onChange: (next: Timeline) => void;
  onToggleMute: (track: Track) => void;
  onAddAudio: (track: Track) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const scrubbing = useRef(false);
  // Handlers read the latest props through a ref, so they stay stable and memoized clips don't re-render
  // on every playback frame.
  const latest = useRef({ drag, timeline, media, pxPerSec, time, onSelect, onChange });
  useLayoutEffect(() => {
    latest.current = { drag, timeline, media, pxPerSec, time, onSelect, onChange };
  });

  const total = totalDuration(timeline);
  const audioEnd = Math.max(
    0,
    ...timeline.tracks.flatMap((t) => t.clips.map((c) => c.start_s + clipDuration(c))),
  );
  const width = timeToPx(Math.max(total, audioEnd) + 10, pxPerSec);
  const step = tickStep(pxPerSec);
  const ticks = Array.from({ length: Math.ceil(width / pxPerSec / step) + 1 }, (_, i) => i * step);
  const preview =
    drag && drag.moved
      ? dragPreview(drag, timeline, media?.get(drag.orig.asset_id)?.duration_s, pxPerSec, time)
      : null;

  useEffect(() => {
    const el = scroller.current;
    if (!el || !playing) return;
    const x = timeToPx(time, pxPerSec);
    if (x < el.scrollLeft || x > el.scrollLeft + el.clientWidth - 40) el.scrollLeft = Math.max(0, x - 80);
  }, [time, playing, pxPerSec]);

  const xToTime = (clientX: number) => {
    const el = scroller.current!;
    return Math.max(0, (clientX - el.getBoundingClientRect().left + el.scrollLeft) / latest.current.pxPerSec);
  };

  const onDown = useCallback<Down>((e, clip, track, mode) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    (e.currentTarget.closest("[data-testid^='clip-']") as HTMLElement | null)?.setPointerCapture?.(
      e.pointerId,
    );
    setDrag({ clipId: clip.id, kind: track.kind, mode, x0: e.clientX, orig: clip, dt: 0, moved: false });
  }, []);
  const onMove = useCallback<Handler>((e) => {
    const { drag: d, pxPerSec: pps } = latest.current;
    if (!d) return;
    const dx = e.clientX - d.x0;
    setDrag({ ...d, dt: dx / pps, moved: d.moved || Math.abs(dx) > 3 });
  }, []);
  const onUp = useCallback<Handler>(() => {
    const {
      drag: d,
      timeline: tl,
      media: m,
      pxPerSec: pps,
      time: t,
      onSelect: select,
      onChange: change,
    } = latest.current;
    if (!d) return;
    setDrag(null);
    select(d.clipId);
    if (!d.moved) return;
    const p = dragPreview(d, tl, m?.get(d.orig.asset_id)?.duration_s, pps, t);
    if (d.kind === "video" && d.mode === "move") {
      const clips = tl.tracks.find((x) => x.kind === "video")!.clips;
      const to = dropIndex(clips, d.clipId, p.start + p.duration / 2);
      if (to !== clips.findIndex((c) => c.id === d.clipId)) change(moveClip(tl, d.clipId, to));
    } else {
      change(updateClip(tl, d.clipId, p.patch));
    }
  }, []);

  return (
    <div className="flex overflow-hidden rounded-lg border" data-testid="timeline-view">
      <div className="bg-card shrink-0 border-r" style={{ width: HEADER_W }}>
        <div style={{ height: RULER_H }} className="border-b" />
        {timeline.tracks.map((track) => (
          <div
            key={track.id}
            className="flex items-center gap-1 border-b px-2 text-xs"
            style={{ height: ROW_H[track.kind] }}
          >
            <span className="min-w-0 flex-1 truncate font-medium">{track.name}</span>
            <Button
              size="icon-sm"
              variant="ghost"
              aria-label={`${track.muted ? "Unmute" : "Mute"} ${track.name}`}
              aria-pressed={track.muted}
              title={track.kind === "video" ? "The clips' own sound" : undefined}
              onClick={() => onToggleMute(track)}
            >
              {track.muted ? <VolumeXIcon /> : <Volume2Icon />}
            </Button>
            {track.kind === "audio" && (
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label={`Add audio to ${track.name}`}
                onClick={() => onAddAudio(track)}
              >
                <PlusIcon />
              </Button>
            )}
          </div>
        ))}
      </div>

      <div ref={scroller} className="relative min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
        <div className="relative" style={{ width }}>
          <div
            className="bg-card text-muted-foreground relative cursor-pointer border-b text-[10px]"
            style={{ height: RULER_H }}
            data-testid="ruler"
            onPointerDown={(e) => {
              scrubbing.current = true;
              e.currentTarget.setPointerCapture?.(e.pointerId);
              onSeek(xToTime(e.clientX));
            }}
            onPointerMove={(e) => scrubbing.current && onSeek(xToTime(e.clientX))}
            onPointerUp={() => (scrubbing.current = false)}
          >
            {ticks.map((t) => (
              <span
                key={t}
                className="absolute top-0 h-full border-l pl-1 leading-6"
                style={{ left: timeToPx(t, pxPerSec) }}
              >
                {formatDuration(t)}
              </span>
            ))}
          </div>

          {timeline.tracks.map((track) => (
            <div
              key={track.id}
              className={cn("relative border-b", track.muted && "opacity-60")}
              style={{ height: ROW_H[track.kind] }}
              data-testid={`track-${track.id}`}
              onPointerDown={(e) => {
                if (e.target !== e.currentTarget) return;
                onSelect(null);
                onSeek(xToTime(e.clientX));
              }}
            >
              {track.clips.map((clip, i) => (
                <ClipBlock
                  key={clip.id}
                  clip={clip}
                  track={track}
                  index={i}
                  label={clipLabel(clip, track, shotNumbers, media)}
                  thumb={track.kind === "video" ? thumbUrl(projectId, clip.asset_id) : undefined}
                  pxPerSec={pxPerSec}
                  selected={clip.id === selectedId}
                  preview={drag?.clipId === clip.id ? preview : null}
                  onDown={onDown}
                  onMove={onMove}
                  onUp={onUp}
                />
              ))}
            </div>
          ))}

          {preview?.snapTo != null && (
            <div
              aria-hidden
              className="pointer-events-none absolute top-0 bottom-0 z-30 w-px bg-amber-400"
              style={{ left: timeToPx(preview.snapTo, pxPerSec) }}
            />
          )}
          <div
            aria-hidden
            className="pointer-events-none absolute top-0 bottom-0 z-30 w-px bg-red-500"
            style={{ left: timeToPx(time, pxPerSec) }}
            data-testid="playhead"
          >
            <span className="absolute -top-px -left-1.5 size-3 rotate-45 bg-red-500" />
          </div>
        </div>
      </div>
    </div>
  );
}
