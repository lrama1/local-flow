import { ArrowLeftIcon, ArrowRightIcon, Trash2Icon } from "lucide-react";
import { useState } from "react";
import type { Asset, Shot } from "@/api/client";
import { fileUrl } from "@/api/imagegen";
import { useTakes } from "@/api/video";
import { AssetThumb } from "@/components/MediaPreview";
import { Button } from "@/components/ui/button";
import { Input, Label, NativeSelect } from "@/components/ui/primitives";
import { VideoSection } from "@/features/shots/VideoSection";
import { cn, formatDuration } from "@/lib/utils";
import {
  assetLabel,
  clipDuration,
  moveClip,
  removeClip,
  updateClip,
  videoTrack,
  type Clip,
  type Timeline,
  type Track,
  type TransitionType,
} from "./math";

/** A number input that commits on blur or Enter, so typing "1." doesn't re-layout the timeline. */
function NumberField({
  id,
  label,
  value,
  step,
  min,
  max,
  onCommit,
}: {
  id: string;
  label: string;
  value: number;
  step: number;
  min?: number;
  max?: number;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    if (text === null) return;
    const v = Number(text);
    setText(null);
    if (!Number.isFinite(v)) return;
    onCommit(Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v)));
  };
  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="text-xs">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        inputMode="decimal"
        step={step}
        min={min}
        max={max}
        className="h-8 text-xs"
        value={text ?? String(Math.round(value * 1000) / 1000)}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && commit()}
      />
    </div>
  );
}

function VideoClipInspector({
  clip,
  index,
  timeline,
  shot,
  shotNumber,
  source,
  projectId,
  onChange,
}: {
  clip: Clip;
  index: number;
  timeline: Timeline;
  shot: Shot | undefined;
  shotNumber: string | undefined;
  source: Asset | undefined;
  projectId: string;
  onChange: (next: Timeline) => void;
}) {
  const fps = timeline.fps;
  const count = videoTrack(timeline).clips.length;
  const [shotOpen, setShotOpen] = useState(false);
  const { data: takes } = useTakes(projectId, clip.shot_id ?? "", !!clip.shot_id);
  const t = clip.transition_in;
  const patch = (p: Partial<Clip>) => onChange(updateClip(timeline, clip.id, p));
  const setTransition = (type: TransitionType, duration_s = t.duration_s || 0.5) =>
    patch({ transition_in: { type, duration_s: type === "cut" ? 0 : duration_s } });

  return (
    <div className="space-y-3" data-testid="inspector">
      <div className="flex items-baseline gap-2">
        <h3 className="text-sm font-semibold">{shotNumber ? `Shot ${shotNumber}` : "Clip"}</h3>
        <span className="text-muted-foreground text-xs">
          Clip {index + 1} of {count} · {formatDuration(clipDuration(clip))}
          {source?.duration_s != null && ` of a ${formatDuration(source.duration_s)} take`}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <NumberField
          id="clip-in"
          label="In (s)"
          value={clip.in_s}
          step={1 / fps}
          min={0}
          max={clip.out_s - 1 / fps}
          onCommit={(v) => patch({ in_s: v })}
        />
        <NumberField
          id="clip-out"
          label="Out (s)"
          value={clip.out_s}
          step={1 / fps}
          min={clip.in_s + 1 / fps}
          max={source?.duration_s ?? undefined}
          onCommit={(v) => patch({ out_s: v })}
        />
      </div>
      <div className="flex items-end gap-2">
        <div className="flex-1 space-y-1">
          <Label htmlFor="clip-transition" className="text-xs">
            Transition in
          </Label>
          <NativeSelect
            id="clip-transition"
            className="h-8 text-xs"
            value={t.type}
            onChange={(e) => setTransition(e.target.value as TransitionType)}
          >
            <option value="cut">Cut</option>
            <option value="crossfade" disabled={index === 0}>
              Crossfade{index === 0 ? " (not on the first clip)" : ""}
            </option>
            <option value="fade_from_black">Fade from black</option>
          </NativeSelect>
        </div>
        {t.type !== "cut" && (
          <div className="w-24">
            <NumberField
              id="clip-transition-s"
              label="Length (s)"
              value={t.duration_s}
              step={1 / fps}
              min={1 / fps}
              max={3}
              onCommit={(v) => setTransition(t.type, v)}
            />
          </div>
        )}
      </div>

      {clip.shot_id && (takes?.items.length ?? 0) > 1 && (
        <div className="space-y-1">
          <Label className="text-xs">Take</Label>
          <div className="grid grid-cols-4 gap-1.5" data-testid="inspector-takes">
            {takes!.items.map((a) => {
              const on = a.id === clip.asset_id;
              return (
                <button
                  key={a.id}
                  type="button"
                  aria-pressed={on}
                  aria-label={on ? "Take on the timeline" : `Use this take (${formatDuration(a.duration_s)})`}
                  title={formatDuration(a.duration_s)}
                  className={cn(
                    "bg-muted aspect-video overflow-hidden rounded border-2",
                    on ? "border-primary" : "hover:border-muted-foreground/40 border-transparent",
                  )}
                  onClick={() =>
                    !on && a.duration_s && patch({ asset_id: a.id, in_s: 0, out_s: a.duration_s })
                  }
                >
                  <AssetThumb asset={a} />
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="flex flex-wrap gap-1.5">
        <Button
          size="sm"
          variant="outline"
          disabled={index === 0}
          onClick={() => onChange(moveClip(timeline, clip.id, index - 1))}
        >
          <ArrowLeftIcon />
          Earlier
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={index === count - 1}
          onClick={() => onChange(moveClip(timeline, clip.id, index + 1))}
        >
          Later
          <ArrowRightIcon />
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="text-muted-foreground ml-auto"
          onClick={() => onChange(removeClip(timeline, clip.id))}
        >
          <Trash2Icon />
          Remove
        </Button>
      </div>

      {shot && (
        <details className="rounded-lg border p-2" onToggle={(e) => setShotOpen(e.currentTarget.open)}>
          <summary className="cursor-pointer text-xs font-medium">Shot prompt, new takes and extend</summary>
          <p className="text-muted-foreground mt-2 text-xs whitespace-pre-wrap">
            {shot.video_prompt || shot.keyframe_prompt || "No prompt."}
          </p>
          <p className="text-muted-foreground mt-1 text-xs">
            New takes appear under Take above; click one to use it here.
          </p>
          {/* Mounted only when open: it plans a generation as soon as it renders. */}
          {shotOpen && (
            <div className="mt-2">
              <VideoSection shot={shot} projectId={projectId} />
            </div>
          )}
        </details>
      )}
    </div>
  );
}

function AudioClipInspector({
  clip,
  track,
  timeline,
  source,
  projectId,
  onChange,
}: {
  clip: Clip;
  track: Track;
  timeline: Timeline;
  source: Asset | undefined;
  projectId: string;
  onChange: (next: Timeline) => void;
}) {
  const patch = (p: Partial<Clip>) => onChange(updateClip(timeline, clip.id, p));
  const moveTo = (trackId: string) =>
    onChange({
      ...timeline,
      tracks: timeline.tracks.map((tr) =>
        tr.id === track.id
          ? { ...tr, clips: tr.clips.filter((c) => c.id !== clip.id) }
          : tr.id === trackId
            ? { ...tr, clips: [...tr.clips, clip] }
            : tr,
      ),
    });
  return (
    <div className="space-y-3" data-testid="inspector">
      <div className="flex items-baseline gap-2">
        <h3 className="min-w-0 truncate text-sm font-semibold">{assetLabel(source)}</h3>
        <span className="text-muted-foreground shrink-0 text-xs">{formatDuration(clipDuration(clip))}</span>
      </div>
      <div className="space-y-1">
        <Label htmlFor="clip-track" className="text-xs">
          Track
        </Label>
        <NativeSelect
          id="clip-track"
          className="h-8 text-xs"
          value={track.id}
          onChange={(e) => moveTo(e.target.value)}
        >
          {timeline.tracks
            .filter((tr) => tr.kind === "audio")
            .map((tr) => (
              <option key={tr.id} value={tr.id}>
                {tr.name}
              </option>
            ))}
        </NativeSelect>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <NumberField
          id="clip-start"
          label="Start (s)"
          value={clip.start_s}
          step={0.01}
          min={0}
          onCommit={(v) => patch({ start_s: v })}
        />
        <NumberField
          id="clip-in"
          label="In (s)"
          value={clip.in_s}
          step={0.01}
          min={0}
          max={clip.out_s - 0.05}
          onCommit={(v) => patch({ in_s: v })}
        />
        <NumberField
          id="clip-out"
          label="Out (s)"
          value={clip.out_s}
          step={0.01}
          min={clip.in_s + 0.05}
          max={source?.duration_s ?? undefined}
          onCommit={(v) => patch({ out_s: v })}
        />
      </div>
      <div className="space-y-1">
        <Label htmlFor="clip-gain" className="text-xs">
          Gain: {clip.gain_db > 0 ? "+" : ""}
          {clip.gain_db.toFixed(1)} dB
        </Label>
        <input
          id="clip-gain"
          type="range"
          min={-24}
          max={12}
          step={0.5}
          value={clip.gain_db}
          className="w-full"
          onChange={(e) => patch({ gain_db: Number(e.target.value) })}
        />
      </div>
      <audio src={fileUrl(projectId, clip.asset_id)} controls className="h-8 w-full" />
      <Button
        size="sm"
        variant="ghost"
        className="text-muted-foreground"
        onClick={() => onChange(removeClip(timeline, clip.id))}
      >
        <Trash2Icon />
        Remove
      </Button>
    </div>
  );
}

export function ClipInspector({
  clip,
  track,
  timeline,
  media,
  shots,
  shotNumbers,
  projectId,
  onChange,
}: {
  clip: Clip;
  track: Track;
  timeline: Timeline;
  media: Map<string, Asset> | undefined;
  shots: Map<string, Shot>;
  shotNumbers: Map<string, string>;
  projectId: string;
  onChange: (next: Timeline) => void;
}) {
  const source = media?.get(clip.asset_id);
  if (track.kind === "video") {
    return (
      <VideoClipInspector
        key={clip.id}
        clip={clip}
        index={track.clips.findIndex((c) => c.id === clip.id)}
        timeline={timeline}
        shot={clip.shot_id ? shots.get(clip.shot_id) : undefined}
        shotNumber={clip.shot_id ? shotNumbers.get(clip.shot_id) : undefined}
        source={source}
        projectId={projectId}
        onChange={onChange}
      />
    );
  }
  return (
    <AudioClipInspector
      key={clip.id}
      clip={clip}
      track={track}
      timeline={timeline}
      source={source}
      projectId={projectId}
      onChange={onChange}
    />
  );
}
