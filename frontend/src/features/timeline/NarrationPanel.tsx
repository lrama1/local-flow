import { AudioLinesIcon, Loader2Icon, MicIcon, RefreshCwIcon, WandSparklesIcon } from "lucide-react";
import { useState } from "react";
import type { ApiError } from "@/api/client";
import { fileUrl } from "@/api/imagegen";
import { useJobs, useProject, useUpdateProject } from "@/api/queries";
import {
  useGenerateNarration,
  useNarrationPlan,
  useResyncNarration,
  useWriteNarration,
  type NarrationShot,
} from "@/api/narration";
import { useBoardMutations } from "@/api/story";
import { VoicePicker } from "@/components/VoicePicker";
import { Button } from "@/components/ui/button";
import { Badge, Input, Label, NativeSelect, Textarea } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";
import { cn, plural } from "@/lib/utils";
import { needsVoice, statusLabel, wordCount } from "./narration";

const SPEEDS = [0.8, 0.9, 1, 1.1, 1.2];
const STATUS_VARIANT = { empty: "outline", text: "secondary", voiced: "success", stale: "warning" } as const;

function Row({
  shot,
  projectId,
  busy,
  onSave,
  onVoice,
}: {
  shot: NarrationShot;
  projectId: string;
  busy: boolean;
  onSave: (text: string) => void;
  onVoice: () => void;
}) {
  const [text, setText] = useState(shot.text);
  const words = wordCount(text);
  const over = words > shot.budget_words + Math.max(2, Math.floor(shot.budget_words / 3));
  return (
    <li className="space-y-1 rounded-md border p-2" data-testid={`narration-row-${shot.shot_id}`}>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-medium">Shot {shot.number}</span>
        <span className="text-muted-foreground min-w-0 flex-1 truncate">{shot.scene_title}</span>
        {shot.overflow_s > 0.05 && (
          <Badge variant="warning" title="The line runs into the next narrated line">
            +{shot.overflow_s.toFixed(1)} s over
          </Badge>
        )}
        <Badge variant={STATUS_VARIANT[shot.status]}>{statusLabel(shot.status)}</Badge>
      </div>
      <Textarea
        aria-label={`Narration for shot ${shot.number}`}
        rows={2}
        value={text}
        placeholder="Silent: nothing is spoken over this shot"
        onChange={(e) => setText(e.target.value)}
        onBlur={() => text.trim() !== shot.text && onSave(text)}
      />
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className={cn("tabular-nums", over ? "text-warning" : "text-muted-foreground")}>
          {words} / {shot.budget_words} words
        </span>
        <span className="text-muted-foreground">
          · {shot.duration_s} s shot
          {shot.status !== "empty" && ` · ${shot.asset_duration_s ?? shot.estimated_s} s of speech`}
        </span>
        {shot.selected_asset_id && (
          <audio
            controls
            preload="none"
            src={fileUrl(projectId, shot.selected_asset_id)}
            className="ml-auto h-7 w-44"
          />
        )}
        <Button
          size="sm"
          variant="outline"
          className={cn("h-7 text-xs", !shot.selected_asset_id && "ml-auto")}
          disabled={!shot.text || busy}
          onClick={onVoice}
        >
          <AudioLinesIcon />
          {shot.selected_asset_id ? "Re-voice" : "Voice"}
        </Button>
      </div>
    </li>
  );
}

/** Section 12.3: the narrator, the LLM writing step, and one editable line per shot, voiced onto the
 * Narration track. */
export function NarrationPanel({ projectId }: { projectId: string }) {
  const { data: project } = useProject(projectId);
  const updateProject = useUpdateProject();
  const [speed, setSpeed] = useState(1);
  const [style, setStyle] = useState("");
  const [rewrite, setRewrite] = useState(false);
  const plan = useNarrationPlan(projectId, speed);
  const write = useWriteNarration(projectId);
  const generate = useGenerateNarration(projectId);
  const resync = useResyncNarration(projectId);
  const board = useBoardMutations(projectId);
  const { data: jobsPage } = useJobs();
  const p = plan.data;
  const shots = p?.shots ?? [];
  const todo = needsVoice(shots);
  const voicing = (jobsPage?.items ?? []).filter(
    (j) =>
      j.type === "tts" &&
      j.project_id === projectId &&
      j.target_ref?.["slot"] === "narration" &&
      (j.status === "queued" || j.status === "running"),
  );
  const busy = voicing.length > 0 || generate.isPending;
  const hasClips = shots.some((s) => s.selected_asset_id);
  const writeError = write.error as ApiError | null;
  const genError =
    generate.error && (generate.error as ApiError).code !== "license_not_acknowledged"
      ? generate.error
      : null;

  const voice = async (shotIds?: string[]) => {
    try {
      await generate.mutateAsync({
        speed,
        shot_ids: shotIds ?? null,
        only_missing: !shotIds,
      });
    } catch (e) {
      gateLicense(e, () => void voice(shotIds));
    }
  };

  return (
    <section className="space-y-3 rounded-lg border p-3" data-testid="narration-panel">
      <h3 className="flex items-center gap-1.5 text-sm font-semibold">
        <MicIcon className="size-4" />
        Narration
      </h3>

      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto]">
        <div className="space-y-1">
          <Label htmlFor="narrator-voice">Narrator voice</Label>
          <VoicePicker
            id="narrator-voice"
            value={project?.narrator_voice_id}
            noneLabel="Suggested voice"
            onChange={(id) => updateProject.mutate({ id: projectId, body: { narrator_voice_id: id } })}
          />
          {p?.narrator.source === "suggested" && p.narrator.voice_name && (
            <p className="text-muted-foreground text-xs">Using {p.narrator.voice_name} (suggested).</p>
          )}
        </div>
        <div className="space-y-1">
          <Label htmlFor="narration-pace">Pace</Label>
          <NativeSelect
            id="narration-pace"
            className="h-8 w-24 text-xs"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s === 1 ? "Normal" : `${s}×`}
              </option>
            ))}
          </NativeSelect>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Input
          aria-label="Narration style"
          className="h-8 min-w-48 flex-1 text-xs"
          placeholder="Style (optional): a hushed storybook, first person…"
          value={style}
          onChange={(e) => setStyle(e.target.value)}
        />
        <Label className="flex items-center gap-1.5 text-xs font-normal">
          <input type="checkbox" checked={rewrite} onChange={(e) => setRewrite(e.target.checked)} />
          Rewrite existing lines
        </Label>
        <Button
          size="sm"
          variant="outline"
          disabled={write.isPending || shots.length === 0}
          onClick={() => write.mutate({ overwrite: rewrite, style: style.trim() || null })}
        >
          {write.isPending ? <Loader2Icon className="animate-spin" /> : <WandSparklesIcon />}
          Write narration
        </Button>
        <Button size="sm" disabled={busy || todo.length === 0} onClick={() => void voice()}>
          {busy ? <Loader2Icon className="animate-spin" /> : <AudioLinesIcon />}
          {busy && voicing.length > 0 ? `Voicing ${plural(voicing.length, "line")}…` : "Voice narration"}
          {!busy && todo.length > 0 && ` (${todo.length})`}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          disabled={!hasClips || resync.isPending}
          title="Move every narration clip back to its shot"
          onClick={() => resync.mutate()}
        >
          <RefreshCwIcon />
          Re-sync
        </Button>
      </div>

      {writeError && <p className="text-destructive text-xs">{writeError.message}</p>}
      {genError && <p className="text-destructive text-xs">{genError.message}</p>}
      {write.data && (
        <p className="text-muted-foreground text-xs" data-testid="narration-written">
          Wrote {plural(write.data.updated.length, "line")}. {write.data.warnings.join(" ")}
        </p>
      )}
      {(p?.notes.length ?? 0) > 0 && <p className="text-warning text-xs">{p!.notes.join(" ")}</p>}

      {shots.length === 0 ? (
        <p className="text-muted-foreground text-xs">No shots yet. Break the story into shots first.</p>
      ) : (
        <ul className="space-y-2">
          {shots.map((s) => (
            <Row
              key={`${s.shot_id}|${s.text}`}
              shot={s}
              projectId={projectId}
              busy={busy}
              onSave={(text) => board.updateShot.mutate({ id: s.shot_id, body: { narration: text } })}
              onVoice={() => void voice([s.shot_id])}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
