import { CheckCircle2Icon, XCircleIcon } from "lucide-react";
import type { ReactNode } from "react";
import { useSystemStatus } from "@/api/queries";
import { ApiKeysCard } from "./ApiKeysCard";
import { LlmSettingsCard } from "./LlmSettingsCard";
import { Card, CardContent, CardHeader, CardTitle, Skeleton } from "@/components/ui/primitives";
import { formatMb } from "@/lib/utils";

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b py-2 text-sm last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="truncate text-right font-mono text-xs">{children}</span>
    </div>
  );
}

function Check({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {ok ? (
        <CheckCircle2Icon className="text-success size-3.5" />
      ) : (
        <XCircleIcon className="text-destructive size-3.5" />
      )}
      {children}
    </span>
  );
}

export function SettingsPage() {
  const { data: s } = useSystemStatus();
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      <div>
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-muted-foreground text-sm">API keys, default models, and diagnostics.</p>
      </div>

      <ApiKeysCard />
      <LlmSettingsCard />

      <Card>
        <CardHeader>
          <CardTitle>Diagnostics</CardTitle>
        </CardHeader>
        <CardContent>
          {!s ? (
            <Skeleton className="h-48" />
          ) : (
            <div data-testid="diagnostics">
              <Row label="LocalFlow">{s.version}</Row>
              <Row label="Data directory">{s.data_dir}</Row>
              <Row label="GPU">
                {s.gpu
                  ? `${s.gpu.name} · ${formatMb(s.gpu.vram_total_mb)} · driver ${s.gpu.driver_version}`
                  : "none"}
              </Row>
              <Row label="System RAM">
                {formatMb(s.ram_used_mb)} / {formatMb(s.ram_total_mb)}
              </Row>
              <Row label="Disk free">{formatMb(s.disk_free_mb)}</Row>
              <Row label="ffmpeg">
                <Check ok={s.ffmpeg_found && s.ffprobe_found}>
                  {s.ffmpeg_found ? (s.ffmpeg_version ?? "found") : "not found"}
                </Check>
              </Row>
              <Row label="Worker">
                {s.worker.state} · engine {s.worker.engine} · pid {s.worker.pid ?? "–"} · restarts{" "}
                {s.worker.restarts}
              </Row>
              <Row label="VRAM profile">{s.vram_profile}</Row>
              <Row label="Python">{s.python_version}</Row>
              <Row label="torch / diffusers">
                {s.torch_version ?? "not installed"} / {s.diffusers_version ?? "not installed"}
              </Row>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
