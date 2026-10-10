import {
  AudioLinesIcon,
  BadgeCheckIcon,
  Loader2Icon,
  MessageSquareTextIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useState } from "react";
import {
  useAudioCandidates,
  useAudioPlan,
  useGenerateShotAudio,
  useSelectAudio,
  type AudioSlot,
} from "@/api/audio";
import type { ApiError, Asset, Shot } from "@/api/client";
import { fileUrl } from "@/api/imagegen";
import { useJobs } from "@/api/queries";
import { JobProgress } from "@/components/JobProgress";
import { Button } from "@/components/ui/button";
import { Badge, Label, NativeSelect } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";
import { cn, formatDuration } from "@/lib/utils";
import { shotAudioJobs } from "./video";

function overflow(a: Asset): number {
  const out = (a.generation as Record<string, unknown> | null)?.["output"] as
    { overflow_s?: number } | undefined;
  return out?.overflow_s ?? 0;
}

function Candidates({
  title,
  items,
  selected,
  projectId,
  onUse,
  busy,
}: {
  title: string;
  items: Asset[];
  selected: string[];
  projectId: string;
  onUse: (a: Asset) => void;
  busy: boolean;
}) {
  if (!items.length) return null;
  return (
    <div className="space-y-1.5">
      <p className="text-muted-foreground text-xs font-medium">{title}</p>
      {items.map((a) => {
        const on = selected.includes(a.id);
        const over = overflow(a);
        return (
          <div
            key={a.id}
            className={cn("flex items-center gap-2 rounded-md border p-1.5", on && "border-primary")}
            data-testid={`audio-candidate-${a.id}`}
          >
            <audio controls preload="none" src={fileUrl(projectId, a.id)} className="h-8 min-w-0 flex-1" />
            <span className="text-muted-foreground shrink-0 text-xs tabular-nums">
              {formatDuration(a.duration_s ?? 0)}
            </span>
            {over > 0 && (
              <Badge variant="warning" title="The dialogue runs past the end of the clip">
                +{over.toFixed(1)} s
              </Badge>
            )}
            {on ? (
              <Badge variant="success">
                <BadgeCheckIcon className="size-3" />
                In use
              </Badge>
            ) : (
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs"
                disabled={busy}
                onClick={() => onUse(a)}
              >
                Use
              </Button>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function AudioSection({
  shot,
  projectId,
  beforeGenerate,
}: {
  shot: Shot;
  projectId: string;
  /** Saves unsaved panel edits first (dialogue lines, SFX prompt). */
  beforeGenerate?: () => Promise<unknown>;
}) {
  const [fit, setFit] = useState<"speed" | "none">("speed");
  const plan = useAudioPlan(shot.id, { fit }, shot.updated_at);
  const dialogue = useAudioCandidates(projectId, shot.id, "dialogue").data?.items ?? [];
  const sfx = useAudioCandidates(projectId, shot.id, "sfx").data?.items ?? [];
  const { data: jobsPage } = useJobs();
  const gen = useGenerateShotAudio(projectId);
  const select = useSelectAudio(projectId);
  const p = plan.data;
  const jobs = {
    dialogue: shotAudioJobs(jobsPage?.items, shot.id, "dialogue"),
    sfx: shotAudioJobs(jobsPage?.items, shot.id, "sfx"),
  };
  const lines = p?.lines ?? [];
  const missingVoice = lines.some((l) => !l.voice_id);
  const planError = plan.error as ApiError | null;
  const genError =
    gen.error && (gen.error as ApiError).code !== "license_not_acknowledged" ? gen.error : null;

  const run = async (parts: AudioSlot[]) => {
    try {
      await beforeGenerate?.();
      await gen.mutateAsync({ shotId: shot.id, body: { parts, params: { fit } } });
    } catch (e) {
      gateLicense(e, () => void run(parts));
    }
  };

  return (
    <section className="space-y-3 rounded-lg border p-3" data-testid="audio-section">
      <div className="flex items-center gap-2">
        <Label>Audio</Label>
        {p && (
          <span className="text-muted-foreground text-xs">
            for the {p.has_take ? "take" : "planned shot"} ({formatDuration(p.clip_s)})
          </span>
        )}
      </div>

      {planError && <p className="text-destructive text-xs">{planError.message}</p>}
      {lines.length > 0 ? (
        <ul className="space-y-1 text-xs" aria-label="Dialogue plan">
          {lines.map((l) => (
            <li key={l.index} className="flex items-start gap-1.5">
              <MessageSquareTextIcon className="text-muted-foreground mt-0.5 size-3.5 shrink-0" />
              <span className="min-w-0 flex-1">
                <span className="font-medium">{l.speaker}</span>{" "}
                <span className="text-muted-foreground">
                  ({l.voice_name ?? "no voice"}
                  {l.voice_source === "suggested" && ", suggested"})
                </span>
                : “{l.line}”
              </span>
              <span className="text-muted-foreground shrink-0 tabular-nums">
                ~{l.estimated_s.toFixed(1)} s
              </span>
            </li>
          ))}
        </ul>
      ) : (
        p && (
          <p className="text-muted-foreground text-xs">
            No dialogue lines: add them below to generate speech.
          </p>
        )
      )}
      {p && lines.length > 0 && (
        <p
          className={cn("text-xs", p.fits ? "text-muted-foreground" : "text-warning")}
          data-testid="audio-fit"
        >
          {!p.fits && <TriangleAlertIcon className="mr-1 inline size-3.5" />}
          Dialogue ~{p.speech_s.toFixed(1)} s of {p.clip_s.toFixed(1)} s
          {p.fit_speed > 1 && p.fits && `, sped up ${p.fit_speed.toFixed(2)}×`}
          {!p.fits && ": too long for the clip. Extend the shot or shorten the lines."}
        </p>
      )}
      {p?.notes
        .filter((n) => !n.startsWith("The dialogue"))
        .map((n) => (
          <p key={n} className="text-muted-foreground text-xs">
            {n}
          </p>
        ))}

      <div className="flex flex-wrap items-center gap-2">
        <NativeSelect
          aria-label="Fit to the clip"
          className="h-8 w-auto text-xs"
          value={fit}
          onChange={(e) => setFit(e.target.value as "speed" | "none")}
        >
          <option value="speed">Speed up to fit (≤ 1.15×)</option>
          <option value="none">Keep the natural pace</option>
        </NativeSelect>
        <Button
          size="sm"
          disabled={!lines.length || missingVoice || gen.isPending}
          title={missingVoice ? "Install a text-to-speech model first" : undefined}
          onClick={() => void run(["dialogue"])}
        >
          {gen.isPending ? <Loader2Icon className="animate-spin" /> : <MessageSquareTextIcon />}
          Generate dialogue
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!p?.has_take || gen.isPending}
          title={p?.has_take ? (p.sfx_prompt ? p.sfx_prompt : "Follows the picture") : "Select a take first"}
          onClick={() => void run(["sfx"])}
        >
          <AudioLinesIcon />
          Generate sound effects
        </Button>
      </div>
      {p && !p.has_take && (
        <p className="text-muted-foreground text-xs">
          Sound effects are made from the take: select one first.
        </p>
      )}
      {genError && <p className="text-destructive text-xs">{genError.message}</p>}
      {(["dialogue", "sfx"] as const).map((slot) =>
        jobs[slot].active.map((j) => <JobProgress key={j.id} jobId={j.id} showPreview={false} />),
      )}
      {(["dialogue", "sfx"] as const).map(
        (slot) =>
          !jobs[slot].active.length &&
          jobs[slot].failed && (
            <p key={slot} className="text-destructive text-xs">
              {slot === "sfx" ? "Sound effects" : "Dialogue"} failed: {jobs[slot].failed?.error}
            </p>
          ),
      )}

      <Candidates
        title="Dialogue takes"
        items={dialogue}
        selected={shot.selected_audio_asset_ids}
        projectId={projectId}
        busy={select.isPending}
        onUse={(a) => select.mutate({ shotId: shot.id, assetId: a.id })}
      />
      <Candidates
        title="Sound effects takes"
        items={sfx}
        selected={shot.selected_audio_asset_ids}
        projectId={projectId}
        busy={select.isPending}
        onUse={(a) => select.mutate({ shotId: shot.id, assetId: a.id })}
      />
      {select.error && <p className="text-destructive text-xs">{select.error.message}</p>}
    </section>
  );
}
