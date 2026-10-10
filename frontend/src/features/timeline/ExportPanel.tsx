import { DownloadIcon, FilmIcon, Loader2Icon, PlayIcon, SquareIcon } from "lucide-react";
import { useState } from "react";
import type { ApiError, Asset } from "@/api/client";
import { fileUrl, thumbUrl } from "@/api/imagegen";
import { useCancelJob, useJobs } from "@/api/queries";
import { useExportTimeline, useExports, type ExportPreset } from "@/api/timeline";
import { AssetViewer } from "@/components/AssetGrid";
import { JobProgress } from "@/components/JobProgress";
import { Button } from "@/components/ui/button";
import { Label, NativeSelect } from "@/components/ui/primitives";
import { formatDuration } from "@/lib/utils";
import { exportJobs, sizeFor, type Timeline } from "./math";

const PRESETS: { value: ExportPreset; short: number }[] = [
  { value: "720p", short: 720 },
  { value: "1080p", short: 1080 },
];

export function ExportPanel({
  projectId,
  timeline,
  beforeExport,
}: {
  projectId: string;
  timeline: Timeline;
  /** Saves pending timeline edits, so the export matches the screen. */
  beforeExport: () => Promise<void>;
}) {
  const [preset, setPreset] = useState<ExportPreset>("720p");
  const [loudnorm, setLoudnorm] = useState(false);
  const [viewing, setViewing] = useState<Asset | null>(null);
  const run = useExportTimeline(projectId);
  const cancel = useCancelJob();
  const { data: jobs } = useJobs();
  const { data: exports } = useExports(projectId);
  const { active, failed } = exportJobs(jobs?.items, projectId);
  const empty = !timeline.tracks.some((t) => t.kind === "video" && t.clips.length);

  const start = async () => {
    await beforeExport();
    run.mutate({ preset, loudnorm });
  };
  const error = run.error && (run.error as ApiError).code !== "license_not_acknowledged" ? run.error : null;

  return (
    <section className="space-y-3 rounded-lg border p-3" data-testid="export-panel">
      <h3 className="flex items-center gap-1.5 text-sm font-semibold">
        <FilmIcon className="size-4" />
        Export
      </h3>
      <div className="flex flex-wrap items-center gap-2">
        <NativeSelect
          aria-label="Export size"
          className="h-8 w-auto text-xs"
          value={preset}
          onChange={(e) => setPreset(e.target.value as ExportPreset)}
        >
          {PRESETS.map((p) => {
            const [w, h] = sizeFor(timeline.width, timeline.height, p.short);
            return (
              <option key={p.value} value={p.value}>
                {p.value} · {w}×{h}
              </option>
            );
          })}
        </NativeSelect>
        <Label className="flex items-center gap-1.5 text-xs font-normal">
          <input type="checkbox" checked={loudnorm} onChange={(e) => setLoudnorm(e.target.checked)} />
          Normalize loudness
        </Label>
        <Button
          size="sm"
          className="ml-auto"
          disabled={empty || !!active || run.isPending}
          onClick={() => void start()}
        >
          {run.isPending ? <Loader2Icon className="animate-spin" /> : <DownloadIcon />}
          Export MP4
        </Button>
      </div>
      {error && <p className="text-destructive text-xs">{error.message}</p>}
      {active && (
        <div className="flex items-center gap-2" data-testid="export-job">
          <JobProgress jobId={active.id} showPreview={false} className="min-w-0 flex-1" />
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Cancel export"
            title="Cancel"
            onClick={() => cancel.mutate(active.id)}
          >
            <SquareIcon />
          </Button>
        </div>
      )}
      {!active && failed && <p className="text-destructive text-xs">Last export failed: {failed.error}</p>}

      {(exports?.items.length ?? 0) > 0 && (
        <ul className="space-y-1.5" data-testid="export-history">
          {exports!.items.map((a) => (
            <li key={a.id} className="flex items-center gap-2 text-xs">
              <button
                type="button"
                aria-label="Play export"
                className="group relative h-9 w-16 shrink-0 overflow-hidden rounded bg-black"
                onClick={() => setViewing(a)}
              >
                <img src={thumbUrl(projectId, a.id)} alt="" className="size-full object-cover" />
                <PlayIcon className="absolute inset-0 m-auto size-4 text-white opacity-80 group-hover:opacity-100" />
              </button>
              <span className="min-w-0 flex-1">
                <span className="block">
                  {a.width}×{a.height} · {formatDuration(a.duration_s)}
                </span>
                <span className="text-muted-foreground block">{new Date(a.created_at).toLocaleString()}</span>
              </span>
              <Button asChild size="icon-sm" variant="ghost" aria-label="Download export" title="Download">
                <a href={fileUrl(projectId, a.id)} download={`export-${a.id}.mp4`}>
                  <DownloadIcon />
                </a>
              </Button>
            </li>
          ))}
        </ul>
      )}
      <AssetViewer asset={viewing} onClose={() => setViewing(null)} />
    </section>
  );
}
