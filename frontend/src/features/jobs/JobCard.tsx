import { RotateCcwIcon, XIcon } from "lucide-react";
import { withToken, type Job } from "@/api/client";
import { useCancelJob, useRetryJob } from "@/api/queries";
import { Button } from "@/components/ui/button";
import { Badge, Progress } from "@/components/ui/primitives";
import { JOB_TYPE_LABELS, cn, relativeTime } from "@/lib/utils";
import { useUi } from "@/stores/ui";

const STATUS_VARIANT = {
  queued: "outline",
  running: "default",
  completed: "success",
  failed: "destructive",
  cancelled: "secondary",
} as const;

export function JobCard({ job, compact = false }: { job: Job; compact?: boolean }) {
  const preview = useUi((s) => s.previews[job.id]);
  const cancel = useCancelJob();
  const retry = useRetryJob();
  const active = job.status === "queued" || job.status === "running";
  const prompt = (job.params.request as { prompt?: string; text?: string } | undefined) ?? {};
  const title = prompt.prompt || prompt.text || "";

  return (
    <div className="bg-card flex gap-3 rounded-lg border p-3" data-testid={`job-${job.id}`}>
      {!compact && job.status === "running" && preview && (
        <img
          src={withToken(preview)}
          alt="Step preview"
          className="size-14 shrink-0 rounded-md object-cover"
        />
      )}
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">{JOB_TYPE_LABELS[job.type] ?? job.type}</span>
          <Badge variant={STATUS_VARIANT[job.status]} data-testid="job-status">
            {job.status}
          </Badge>
          <span className="text-muted-foreground ml-auto shrink-0 text-[11px]">
            {relativeTime(job.created_at)}
          </span>
        </div>
        {title && <p className="text-muted-foreground line-clamp-2 text-xs">{title}</p>}
        {active && (
          <div className="space-y-1">
            <Progress
              value={job.progress * 100}
              aria-label="Job progress"
              indicatorClassName={cn(job.status === "queued" && "bg-muted-foreground")}
            />
            <div className="text-muted-foreground flex justify-between text-[11px]">
              <span className="truncate">
                {job.status === "queued" ? "Waiting in queue" : job.progress_message}
              </span>
              {job.status === "running" && <span>{Math.round(job.progress * 100)}%</span>}
            </div>
          </div>
        )}
        {job.status === "failed" && job.error && (
          <p className="text-destructive line-clamp-3 text-xs" title={job.error}>
            {job.error}
          </p>
        )}
      </div>
      <div className="flex shrink-0 flex-col gap-1">
        {active && (
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Cancel job"
            disabled={cancel.isPending}
            onClick={() => cancel.mutate(job.id)}
          >
            <XIcon />
          </Button>
        )}
        {(job.status === "failed" || job.status === "cancelled") && (
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Retry job"
            disabled={retry.isPending}
            onClick={() => retry.mutate(job.id)}
          >
            <RotateCcwIcon />
          </Button>
        )}
      </div>
    </div>
  );
}
