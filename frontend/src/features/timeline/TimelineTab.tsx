import {
  FilmIcon,
  Loader2Icon,
  PauseIcon,
  PlayIcon,
  ScanIcon,
  SkipBackIcon,
  WandSparklesIcon,
} from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Asset, Shot } from "@/api/client";
import { useBoard } from "@/api/story";
import { useAutoAssemble, useSaveTimeline, useTimeline, useTimelineMedia } from "@/api/timeline";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/primitives";
import { plural } from "@/lib/utils";
import { AddAudioDialog } from "./AddAudioDialog";
import { ClipInspector } from "./ClipInspector";
import { ExportPanel } from "./ExportPanel";
import {
  complete,
  findClip,
  newClipId,
  removeClip,
  sequence,
  timecode,
  totalDuration,
  updateTrack,
  type Timeline,
  type Track,
} from "./math";
import { PreviewPlayer } from "./PreviewPlayer";
import { TimelineView } from "./TimelineView";
import { usePlayback } from "./usePlayback";

const SAVE_DELAY_MS = 500;
const ZOOM = { min: 10, max: 300, initial: 60 };

/** Section 15.2: preview, tracks, inspector and export. Edits apply locally at once and are saved
 * (debounced, one request at a time); the server's normalized timeline then replaces the local one. */
export function TimelineTab({ projectId }: { projectId: string }) {
  const { data, isLoading, error } = useTimeline(projectId);
  const { data: media } = useTimelineMedia(projectId);
  const { data: board } = useBoard(projectId);
  const saveTimeline = useSaveTimeline(projectId);
  const saveMutate = saveTimeline.mutateAsync; // stable across renders (unlike the mutation object)
  const assemble = useAutoAssemble(projectId);

  const [draft, setDraft] = useState<Timeline | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [pxPerSec, setPxPerSec] = useState(ZOOM.initial);
  const [notes, setNotes] = useState<string[]>([]);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [adding, setAdding] = useState<Track | null>(null);
  const [assembled, setAssembled] = useState<string | null>(null);

  const edits = useRef({ version: 0, saved: 0, timer: 0, latest: null as Timeline | null });
  const queue = useRef<Promise<void>>(Promise.resolve());

  // Adopt the server's timeline whenever there are no unsaved local edits.
  useEffect(() => {
    if (data && edits.current.version === edits.current.saved) {
      const tl = complete(data.timeline);
      edits.current.latest = tl;
      setDraft(tl);
    }
  }, [data]);

  const flush = useCallback((): Promise<void> => {
    window.clearTimeout(edits.current.timer);
    queue.current = queue.current.then(async () => {
      const e = edits.current;
      if (e.version === e.saved || !e.latest) return;
      const version = e.version;
      try {
        const out = await saveMutate(e.latest);
        setSaveError(null);
        setNotes(out.notes ?? []);
        e.saved = Math.max(e.saved, version);
        if (e.version === version) {
          const tl = complete(out.timeline);
          e.latest = tl;
          setDraft(tl);
        }
      } catch (err) {
        e.saved = Math.max(e.saved, version); // don't retry a rejected edit forever
        setSaveError((err as Error).message);
      }
    });
    return queue.current;
  }, [saveMutate]);

  const change = useCallback(
    (next: Timeline) => {
      const tl = sequence(next);
      const e = edits.current;
      e.latest = tl;
      e.version += 1;
      setDraft(tl);
      setNotes([]);
      window.clearTimeout(e.timer);
      e.timer = window.setTimeout(() => void flush(), SAVE_DELAY_MS);
    },
    [flush],
  );

  // Save what's pending when leaving the tab.
  useEffect(() => () => void flush(), [flush]);

  const timeline = draft;
  const duration = timeline ? totalDuration(timeline) : 0;
  const { time, playing, toggle, pause, seek } = usePlayback(duration);

  const { shots, shotNumbers, takesReady } = useMemo(() => {
    const shots = new Map<string, Shot>();
    const shotNumbers = new Map<string, string>();
    board?.scenes.forEach((scene, si) =>
      scene.shots.forEach((sh, i) => {
        shots.set(sh.id, sh);
        shotNumbers.set(sh.id, `${si + 1}.${i + 1}`);
      }),
    );
    const takesReady = [...shots.values()].filter((s) => s.selected_video_asset_id).length;
    return { shots, shotNumbers, takesReady };
  }, [board]);

  // Keyboard (Section 17.3): space plays, ←/→ step a frame (Shift: a second), Delete removes the clip.
  // One listener reads the latest state through a ref instead of re-subscribing every frame.
  const keys = useRef({ timeline, time, duration, selected, toggle, pause, seek, change });
  useLayoutEffect(() => {
    keys.current = { timeline, time, duration, selected, toggle, pause, seek, change };
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = keys.current;
      if (!k.timeline || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      // The target can be the window itself, which has no `closest`.
      const target = e.target as Partial<Element> | null;
      const el = typeof target?.closest === "function" ? (target as Element) : null;
      if (el?.closest("input, textarea, select, [contenteditable='true'], [role='dialog']")) return;
      const step = e.shiftKey ? 1 : 1 / k.timeline.fps;
      if (e.key === " " && !el?.closest("button, a")) {
        e.preventDefault();
        k.toggle();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        k.pause();
        k.seek(k.time + (e.key === "ArrowLeft" ? -step : step));
      } else if (e.key === "Home") {
        k.seek(0);
      } else if (e.key === "End") {
        k.seek(k.duration);
      } else if ((e.key === "Delete" || e.key === "Backspace") && k.selected) {
        e.preventDefault();
        k.change(removeClip(k.timeline, k.selected));
        setSelected(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const runAssemble = async () => {
    const hasClips = timeline?.tracks.some((t) => t.kind === "video" && t.clips.length);
    if (
      hasClips &&
      !window.confirm(
        "Rebuild the timeline from each shot's selected take? Trims, order and transitions are replaced; the music track is kept.",
      )
    ) {
      return;
    }
    await flush();
    const out = await assemble.mutateAsync({});
    edits.current.saved = edits.current.version; // the server's timeline replaces local edits
    const tl = complete(out.timeline);
    edits.current.latest = tl;
    setDraft(tl);
    setSelected(null);
    seek(0);
    const placed = out.timeline.tracks.find((t) => t.kind === "video")?.clips?.length ?? 0;
    setAssembled(
      `${plural(placed, "clip")} placed${out.skipped?.length ? `; ${plural(out.skipped.length, "shot")} without a take skipped` : ""}.`,
    );
  };

  const addAudio = (trackId: string, asset: Asset) => {
    const track = timeline?.tracks.find((t) => t.id === trackId);
    if (!timeline || !track || !asset.duration_s) return;
    const clip = {
      id: newClipId(),
      asset_id: asset.id,
      start_s: Math.round(time * 1000) / 1000,
      in_s: 0,
      out_s: asset.duration_s,
      transition_in: { type: "cut" as const, duration_s: 0 },
      gain_db: 0,
      shot_id: null,
    };
    change(updateTrack(timeline, track.id, { clips: [...track.clips, clip] }));
    setSelected(clip.id);
    setAdding(null);
  };

  const fit = () => {
    const width = document.querySelector("[data-testid='timeline-view']")?.clientWidth ?? 900;
    const usable = Math.max(200, width - 170);
    setPxPerSec(Math.max(ZOOM.min, Math.min(ZOOM.max, usable / Math.max(1, duration + 1))));
  };

  if (isLoading || (!timeline && !error)) return <Skeleton className="h-80" />;
  if (error || !timeline) {
    return (
      <EmptyState icon={FilmIcon} title="Couldn't load the timeline">
        {error?.message}
      </EmptyState>
    );
  }

  const hasVideo = timeline.tracks.some((t) => t.kind === "video" && t.clips.length);
  const selection = findClip(timeline, selected);
  const assembleButton = (
    <Button
      variant={hasVideo ? "outline" : "default"}
      size="sm"
      disabled={assemble.isPending || takesReady === 0}
      title={takesReady === 0 ? "No shot has a selected take yet (Shots board → Video)" : undefined}
      onClick={() => void runAssemble()}
    >
      {assemble.isPending ? <Loader2Icon className="animate-spin" /> : <WandSparklesIcon />}
      Auto-assemble
    </Button>
  );

  return (
    <div className="space-y-4" data-testid="timeline-tab">
      {!hasVideo && (
        <EmptyState icon={FilmIcon} title="The timeline is empty" action={assembleButton}>
          {takesReady
            ? `Auto-assemble places the selected takes of ${plural(takesReady, "shot")} in board order.`
            : "Generate videos on the Shots board first; each shot's selected take goes on the timeline."}
        </EmptyState>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-2">
          <PreviewPlayer
            timeline={timeline}
            media={media}
            projectId={projectId}
            time={time}
            playing={playing}
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button size="icon-sm" variant="ghost" aria-label="Back to start" onClick={() => seek(0)}>
              <SkipBackIcon />
            </Button>
            <Button
              size="icon-sm"
              aria-label={playing ? "Pause" : "Play"}
              disabled={!hasVideo}
              onClick={toggle}
            >
              {playing ? <PauseIcon /> : <PlayIcon />}
            </Button>
            <span className="font-mono text-xs tabular-nums" data-testid="timecode">
              {timecode(time, timeline.fps)} / {timecode(duration, timeline.fps)}
            </span>
            <span className="text-muted-foreground text-xs">
              {timeline.width}×{timeline.height} · {timeline.fps} fps
            </span>
            <span className="text-muted-foreground ml-auto text-xs" aria-live="polite">
              {saveError ? (
                <span className="text-destructive">Not saved: {saveError}</span>
              ) : saveTimeline.isPending ? (
                "Saving…"
              ) : (
                ""
              )}
            </span>
            {hasVideo && assembleButton}
          </div>
          {assembled && <p className="text-muted-foreground text-xs">{assembled}</p>}
          {notes.length > 0 && <p className="text-warning text-xs">{notes.join(" ")}</p>}
        </div>

        <div className="space-y-4">
          <section className="rounded-lg border p-3">
            {selection ? (
              <ClipInspector
                clip={selection.clip}
                track={selection.track}
                timeline={timeline}
                media={media}
                shots={shots}
                shotNumbers={shotNumbers}
                projectId={projectId}
                onChange={change}
              />
            ) : (
              <p className="text-muted-foreground text-xs">
                Click a clip to trim it, change its transition or swap the take. Drag clips to reorder them
                and their edges to trim. Space plays, ←/→ step a frame, Delete removes the selected clip.
              </p>
            )}
          </section>
          <ExportPanel projectId={projectId} timeline={timeline} beforeExport={flush} />
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs">
        <span className="text-muted-foreground">Zoom</span>
        <input
          type="range"
          aria-label="Zoom"
          min={ZOOM.min}
          max={ZOOM.max}
          value={pxPerSec}
          onChange={(e) => setPxPerSec(Number(e.target.value))}
          className="w-40"
        />
        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={fit}>
          <ScanIcon />
          Fit
        </Button>
      </div>
      <TimelineView
        timeline={timeline}
        media={media}
        projectId={projectId}
        pxPerSec={pxPerSec}
        time={time}
        playing={playing}
        selectedId={selected}
        shotNumbers={shotNumbers}
        onSelect={setSelected}
        onSeek={(t) => {
          pause();
          seek(t);
        }}
        onChange={change}
        onToggleMute={(track) => change(updateTrack(timeline, track.id, { muted: !track.muted }))}
        onAddAudio={setAdding}
      />
      {adding && (
        <AddAudioDialog
          track={adding}
          media={media}
          projectId={projectId}
          playhead={time}
          fps={timeline.fps}
          onAdd={(a) => addAudio(adding.id, a)}
          onClose={() => setAdding(null)}
        />
      )}
    </div>
  );
}
