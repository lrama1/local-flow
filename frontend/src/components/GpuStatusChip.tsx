import { CpuIcon } from "lucide-react";
import { useSystemStatus } from "@/api/queries";
import { Tooltip } from "@/components/ui/tooltip";
import { cn, formatMb } from "@/lib/utils";

const STATE_COLOR: Record<string, string> = {
  idle: "bg-success",
  busy: "bg-primary animate-pulse",
  starting: "bg-warning animate-pulse",
  restarting: "bg-warning animate-pulse",
  stopped: "bg-destructive",
  disabled: "bg-muted-foreground",
};

export function GpuStatusChip() {
  const { data } = useSystemStatus();
  const gpu = data?.gpu;
  const worker = data?.worker;
  const pct = gpu ? Math.round((gpu.vram_used_mb / Math.max(1, gpu.vram_total_mb)) * 100) : 0;
  const lm = worker?.loaded_model as
    { adapter_key?: string; model_name?: string; vram_profile?: string; load_s?: number } | null | undefined;
  const loaded = lm?.model_name ?? lm?.adapter_key;
  const vram = worker?.vram as
    { allocated_mb?: number; baseline?: { device_used_mb?: number } | null } | null | undefined;

  return (
    <Tooltip
      content={
        <div className="space-y-1">
          <div className="font-medium">{!data ? "Loading…" : (gpu?.name ?? "No NVIDIA GPU detected")}</div>
          {gpu && (
            <div>
              VRAM {formatMb(gpu.vram_used_mb)} / {formatMb(gpu.vram_total_mb)}
            </div>
          )}
          <div>
            Worker: {worker?.state ?? "unknown"}
            {worker?.engine === "fake" && " (FakeEngine)"}
          </div>
          {worker?.engine_note && <div className="text-warning">{worker.engine_note}</div>}
          <div>
            Loaded model: {loaded ?? "none"}
            {lm?.vram_profile &&
              ` (${lm.vram_profile}${lm.load_s != null ? `, loaded in ${lm.load_s} s` : ""})`}
          </div>
          {vram?.allocated_mb != null && (
            <div>
              Worker allocation {formatMb(vram.allocated_mb)}
              {vram.baseline?.device_used_mb != null &&
                ` · idle baseline ${formatMb(vram.baseline.device_used_mb)}`}
            </div>
          )}
          {worker?.restarts ? <div>Restarts: {worker.restarts}</div> : null}
        </div>
      }
    >
      <div
        className="flex h-8 items-center gap-2 rounded-md border px-2.5 text-xs"
        data-testid="gpu-chip"
        aria-label="GPU status"
      >
        <span
          className={cn("size-2 rounded-full", STATE_COLOR[worker?.state ?? ""] ?? "bg-muted-foreground")}
        />
        <CpuIcon className="text-muted-foreground size-3.5" />
        <span className="hidden max-w-36 truncate md:inline">
          {!data ? "…" : gpu ? gpu.name.replace(/^NVIDIA (GeForce )?/, "") : "No GPU"}
        </span>
        {gpu && (
          <div className="bg-secondary h-1.5 w-16 overflow-hidden rounded-full" title={`${pct}% VRAM`}>
            <div
              className={cn("h-full", pct > 90 ? "bg-destructive" : "bg-primary")}
              style={{ width: `${pct}%` }}
            />
          </div>
        )}
        {loaded && (
          <span className="text-muted-foreground hidden max-w-44 truncate lg:inline">· {loaded}</span>
        )}
      </div>
    </Tooltip>
  );
}
