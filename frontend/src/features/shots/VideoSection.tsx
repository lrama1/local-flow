import {
  BadgeCheckIcon,
  ExpandIcon,
  FastForwardIcon,
  FilmIcon,
  Loader2Icon,
  SparklesIcon,
  SquareIcon,
  XIcon,
} from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { useShotReferences } from "@/api/characters";
import type { ApiError, Asset, Shot } from "@/api/client";
import { fileUrl, thumbUrl, useSelectForShot } from "@/api/imagegen";
import { useCancelJob, useJobs } from "@/api/queries";
import { useBoardMutations } from "@/api/story";
import {
  useExtendShot,
  useGenerateVideo,
  useTakes,
  useVideoPlan,
  type CameraIntensity,
  type VideoMode,
  type VideoPlan,
} from "@/api/video";
import { AssetViewer } from "@/components/AssetGrid";
import { JobProgress } from "@/components/JobProgress";
import { AssetThumb } from "@/components/MediaPreview";
import { ModelPicker } from "@/components/ModelPicker";
import { Button } from "@/components/ui/button";
import { Badge, Label, NativeSelect } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";
import { cn, formatDuration } from "@/lib/utils";
import { ReviewCheck, type PromptField } from "./ReviewCheck";
import { segmentsFor, shotVideoJobs } from "./video";

const MODES: { value: VideoMode; label: string }[] = [
  { value: "auto", label: "Auto (the shot's mode)" },
  { value: "i2v", label: "Keyframe → video" },
  { value: "flf2v", label: "First + last frame" },
  { value: "t2v", label: "Text only" },
];
const INTENSITIES: CameraIntensity[] = ["subtle", "normal", "strong"];

function planLine(p: VideoPlan): string {
  return `${p.mode} · ${p.model_name ?? "no model"} · ${p.width}×${p.height} · ${p.num_frames} frames @ ${p.fps} fps (${formatDuration(p.duration_s)})`;
}

export function VideoSection({
  shot,
  projectId,
  beforeGenerate,
  onApplyFix,
}: {
  shot: Shot;
  projectId: string;
  /** Saves unsaved panel edits first, so the video uses the prompts on screen. */
  beforeGenerate?: () => Promise<unknown>;
  /** Puts a vision-review prompt fix into the panel's draft. */
  onApplyFix?: (field: PromptField, text: string) => void;
}) {
  const [mode, setMode] = useState<VideoMode>("auto");
  const [modelId, setModelId] = useState<string | null>(null);
  const [variations, setVariations] = useState(1);
  const [intensity, setIntensity] = useState<CameraIntensity>("normal");
  const [extendAfter, setExtendAfter] = useState<number | null>(null); // null = what the shot needs
  const [segments, setSegments] = useState(1);
  const [steps, setSteps] = useState<number | null>(null); // null = the model's default
  const [viewing, setViewing] = useState<Asset | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  const planBody = { mode, model_entry_id: modelId, camera_intensity: intensity };
  const plan = useVideoPlan(shot.id, planBody, shot.updated_at);
  const { data: takesPage } = useTakes(projectId, shot.id);
  const { data: jobsPage } = useJobs();
  const gen = useGenerateVideo(projectId);
  const ext = useExtendShot(projectId);
  const select = useSelectForShot(projectId);
  const cancel = useCancelJob();
  const board = useBoardMutations(projectId);

  const takes = takesPage?.items ?? [];
  const selectedId = shot.selected_video_asset_id;
  // The saved shot's references: what a consistency check compares the take with.
  const { data: refs } = useShotReferences(shot.id, shot.shot_type, shot.ingredient_ids);
  const { active, failed } = shotVideoJobs(jobsPage?.items, shot.id);
  const needed = segmentsFor(shot, plan.data);
  const extendCount = extendAfter ?? needed;
  const capability = mode === "auto" ? (plan.data?.mode ?? "i2v") : mode;
  const planError = plan.error as ApiError | null;
  const approved = shot.status === "approved";

  const generate = async () => {
    try {
      await beforeGenerate?.();
      await gen.mutateAsync({
        shotId: shot.id,
        body: { ...planBody, variations, extend_segments: extendCount, params: { steps } },
      });
    } catch (e) {
      gateLicense(e, () => void generate());
    }
  };
  const extend = async () => {
    try {
      await ext.mutateAsync({
        shotId: shot.id,
        body: { segments, camera_intensity: intensity, params: { steps } },
      });
    } catch (e) {
      gateLicense(e, () => void extend());
    }
  };
  const mutationError = [gen.error, ext.error].find(
    (e) => e && (e as ApiError).code !== "license_not_acknowledged",
  );

  return (
    <section className="space-y-3 rounded-lg border p-3" data-testid="video-section">
      <div className="flex items-center gap-2">
        <Label>Video</Label>
        {selectedId && (
          <Button
            size="sm"
            variant={approved ? "default" : "outline"}
            className="ml-auto h-7 text-xs"
            aria-pressed={approved}
            title={approved ? "Approved for the final cut" : "Mark this take as final"}
            onClick={() =>
              board.updateShot.mutate({
                id: shot.id,
                body: { status: approved ? "video_ready" : "approved" },
              })
            }
          >
            <BadgeCheckIcon />
            {approved ? "Approved" : "Approve take"}
          </Button>
        )}
      </div>

      <div className="relative overflow-hidden rounded-md bg-black">
        {selectedId ? (
          <>
            <video
              key={selectedId}
              src={fileUrl(projectId, selectedId)}
              poster={thumbUrl(projectId, selectedId)}
              controls
              loop
              playsInline
              preload="metadata"
              className="aspect-video w-full"
              data-testid="selected-take"
            />
            <Button
              size="icon-sm"
              variant="secondary"
              aria-label="Clear selected take"
              title="Clear the selection"
              className="absolute top-1.5 right-1.5"
              onClick={() => select.mutate({ shotId: shot.id, slot: "video", assetId: null })}
            >
              <XIcon />
            </Button>
          </>
        ) : (
          <div className="bg-muted text-muted-foreground flex aspect-video items-center justify-center gap-1.5 text-xs">
            <FilmIcon className="size-4" />
            No take yet
          </div>
        )}
      </div>
      {selectedId && (refs?.selected.length ?? 0) > 0 && (
        <ReviewCheck
          shotId={shot.id}
          assetId={selectedId}
          what="take"
          beforeReview={beforeGenerate}
          onApplyFix={onApplyFix}
        />
      )}

      {plan.data && (
        <div className="space-y-1">
          <p className="text-muted-foreground text-xs" data-testid="video-plan">
            {planLine(plan.data)}{" "}
            <button
              type="button"
              className="text-primary hover:underline"
              aria-expanded={showPrompt}
              onClick={() => setShowPrompt((s) => !s)}
            >
              {showPrompt ? "hide prompt" : "prompt"}
            </button>
          </p>
          {showPrompt && (
            <p className="bg-muted rounded-md p-2 font-mono text-[11px] whitespace-pre-wrap">
              {plan.data.prompt}
            </p>
          )}
          {plan.data.note && <p className="text-warning text-xs">{plan.data.note}</p>}
          {!plan.data.model_entry_id && (
            <p className="text-warning text-xs">
              No {plan.data.mode} video model is installed.{" "}
              <Link to="/models?tab=catalog" className="text-primary hover:underline">
                Install one in Models
              </Link>
            </p>
          )}
        </div>
      )}
      {planError && <p className="text-destructive text-xs">{planError.message}</p>}

      <div className="flex flex-wrap items-center gap-2">
        <NativeSelect
          aria-label="Video mode"
          className="h-8 w-auto text-xs"
          value={mode}
          onChange={(e) => {
            setMode(e.target.value as VideoMode);
            setModelId(null);
          }}
        >
          {MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </NativeSelect>
        <ModelPicker capability={capability} value={modelId} onChange={setModelId} label="Video model" />
        <NativeSelect
          aria-label="Camera intensity"
          title="How strongly the shot's camera move is described"
          className="h-8 w-auto text-xs"
          value={intensity}
          onChange={(e) => setIntensity(e.target.value as CameraIntensity)}
        >
          {INTENSITIES.map((i) => (
            <option key={i} value={i}>
              Camera: {i}
            </option>
          ))}
        </NativeSelect>
        <NativeSelect
          aria-label="Takes"
          className="h-8 w-auto text-xs"
          value={variations}
          onChange={(e) => setVariations(Number(e.target.value))}
        >
          {[1, 2, 3, 4].map((n) => (
            <option key={n} value={n}>
              {n} {n === 1 ? "take" : "takes"}
            </option>
          ))}
        </NativeSelect>
        <NativeSelect
          aria-label="Steps"
          title="Fewer steps are faster but rougher; good for drafts"
          className="h-8 w-auto text-xs"
          value={steps ?? ""}
          onChange={(e) => setSteps(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">Default steps</option>
          {[10, 15, 20, 30, 40, 50].map((n) => (
            <option key={n} value={n}>
              {n} steps
            </option>
          ))}
        </NativeSelect>
        {(needed > 0 || extendAfter) && (
          <NativeSelect
            aria-label="Extend after generating"
            title={`The shot is ${formatDuration(shot.duration_s)}; one clip is ${formatDuration(plan.data?.duration_s ?? 0)}`}
            className={cn("h-8 w-auto text-xs", needed > 0 && extendCount === 0 && "border-warning")}
            value={extendCount}
            onChange={(e) => setExtendAfter(Number(e.target.value))}
          >
            {[0, 1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>
                {n === 0 ? "No extend" : `+ extend ×${n}`}
              </option>
            ))}
          </NativeSelect>
        )}
        <Button size="sm" disabled={gen.isPending || !plan.data} onClick={() => void generate()}>
          {gen.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
          {extendCount > 0 ? `Generate + extend ×${extendCount}` : "Generate video"}
        </Button>
      </div>

      {selectedId && (
        <div className="flex flex-wrap items-center gap-2">
          <NativeSelect
            aria-label="Extend segments"
            className="h-8 w-auto text-xs"
            value={segments}
            onChange={(e) => setSegments(Number(e.target.value))}
          >
            {[1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>
                {n} {n === 1 ? "segment" : "segments"}
              </option>
            ))}
          </NativeSelect>
          <Button size="sm" variant="outline" disabled={ext.isPending} onClick={() => void extend()}>
            {ext.isPending ? <Loader2Icon className="animate-spin" /> : <FastForwardIcon />}
            Extend take
          </Button>
          <span className="text-muted-foreground text-xs">Continues from its last frame</span>
        </div>
      )}

      {mutationError && <p className="text-destructive text-xs">{mutationError.message}</p>}
      {active.map((j) => (
        <div key={j.id} className="flex items-center gap-2" data-testid="video-job">
          <JobProgress jobId={j.id} showPreview={false} className="min-w-0 flex-1" />
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Cancel video job"
            title="Cancel"
            disabled={cancel.isPending}
            onClick={() => cancel.mutate(j.id)}
          >
            <SquareIcon />
          </Button>
        </div>
      ))}
      {active.length === 0 && failed && (
        <p className="text-destructive text-xs">Last video job failed: {failed.error ?? "unknown error"}</p>
      )}

      {takes.length > 0 && (
        <div className="grid grid-cols-3 gap-2" data-testid="video-takes">
          {takes.map((a) => {
            const on = a.id === selectedId;
            return (
              <div key={a.id} className="group relative">
                <button
                  type="button"
                  aria-label={on ? "Selected take" : "Use this take"}
                  aria-pressed={on}
                  onClick={() => !on && select.mutate({ shotId: shot.id, slot: "video", assetId: a.id })}
                  className={cn(
                    "bg-muted block aspect-video w-full overflow-hidden rounded-md border-2",
                    on ? "border-primary" : "hover:border-muted-foreground/40 border-transparent",
                  )}
                >
                  <AssetThumb asset={a} />
                </button>
                {a.duration_s != null && (
                  <Badge
                    variant="secondary"
                    className="pointer-events-none absolute top-1 right-1 px-1 text-[10px]"
                  >
                    {formatDuration(a.duration_s)}
                  </Badge>
                )}
                <button
                  type="button"
                  aria-label="Play take"
                  onClick={() => setViewing(a)}
                  className="absolute bottom-1 left-1 hidden rounded bg-black/60 p-0.5 text-white group-hover:block"
                >
                  <ExpandIcon className="size-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}
      <AssetViewer asset={viewing} onClose={() => setViewing(null)} />
    </section>
  );
}
