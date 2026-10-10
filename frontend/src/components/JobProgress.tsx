import { Loader2Icon } from "lucide-react";
import { withToken } from "@/api/client";
import { useJob } from "@/api/queries";
import { Progress } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";
import { useUi } from "@/stores/ui";

/** Inline status for one job: progress, message, latest step preview, and errors. Hides when done. */
export function JobProgress({
  jobId,
  className,
  showPreview = true,
}: {
  jobId: string | null | undefined;
  className?: string;
  showPreview?: boolean;
}) {
  const { data: job } = useJob(jobId ?? undefined);
  const preview = useUi((s) => (jobId ? s.previews[jobId] : undefined));
  if (!jobId || !job) return null;
  if (job.status === "completed" || job.status === "cancelled") return null;
  if (job.status === "failed") {
    return <p className={cn("text-destructive text-xs", className)}>{job.error ?? "The job failed."}</p>;
  }
  return (
    <div className={cn("space-y-1.5", className)} data-testid="job-progress">
      <div className="text-muted-foreground flex items-center gap-2 text-xs">
        <Loader2Icon className="size-3.5 animate-spin" />
        <span className="truncate">
          {job.status === "queued" ? "Queued…" : (job.progress_message ?? "Running…")}
        </span>
        <span className="ml-auto tabular-nums">{Math.round(job.progress * 100)}%</span>
      </div>
      <Progress value={job.progress * 100} />
      {showPreview && preview && job.status === "running" && (
        <img src={withToken(preview)} alt="Step preview" className="max-h-40 rounded-md opacity-90" />
      )}
    </div>
  );
}
