import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  DownloadIcon,
  FileTextIcon,
  Loader2Icon,
  MinusCircleIcon,
  RefreshCwIcon,
  XCircleIcon,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import {
  downloadBundle,
  useDiagnostics,
  useLogTail,
  useRerunDiagnostics,
  type DiagCheck,
} from "@/api/diagnostics";
import { useSystemStatus } from "@/api/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, NativeSelect, Skeleton } from "@/components/ui/primitives";
import { cn, formatMb } from "@/lib/utils";
import { useUi } from "@/stores/ui";

const STATUS_ICON = {
  ok: <CheckCircle2Icon className="text-success size-4" />,
  warn: <AlertTriangleIcon className="text-warning size-4" />,
  error: <XCircleIcon className="text-destructive size-4" />,
  skip: <MinusCircleIcon className="text-muted-foreground size-4" />,
} as const;

const FOLDER_LABELS: Record<string, string> = {
  models: "Models",
  projects: "Projects",
  library: "Cast library",
  tmp: "Temporary files",
  logs: "Logs",
  database: "Database",
};

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b py-1.5 text-sm last:border-0">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span className="truncate text-right font-mono text-xs">{children}</span>
    </div>
  );
}

function CheckRow({ check }: { check: DiagCheck }) {
  return (
    <li className="flex items-start gap-2.5 py-1.5" data-status={check.status}>
      <span className="mt-0.5" aria-label={check.status}>
        {STATUS_ICON[check.status]}
      </span>
      <div className="min-w-0">
        <div className="text-sm font-medium">{check.label}</div>
        <div className="text-muted-foreground text-xs break-words">{check.detail}</div>
      </div>
    </li>
  );
}

function LogViewer({ name }: { name: string }) {
  const [lines, setLines] = useState(200);
  const { data, isFetching, refetch } = useLogTail(name, lines);
  return (
    <div className="space-y-2" data-testid="log-viewer">
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs">{name}</span>
        {data && (
          <span className="text-muted-foreground text-xs">
            last {data.lines.length} of {data.total_lines.toLocaleString()} lines
          </span>
        )}
        <NativeSelect
          aria-label="Lines"
          className="ml-auto h-7 w-auto text-xs"
          value={lines}
          onChange={(e) => setLines(Number(e.target.value))}
        >
          {[200, 1000, 5000].map((n) => (
            <option key={n} value={n}>
              {n} lines
            </option>
          ))}
        </NativeSelect>
        <Button size="icon-sm" variant="ghost" aria-label="Reload log" onClick={() => void refetch()}>
          <RefreshCwIcon className={cn(isFetching && "animate-spin")} />
        </Button>
      </div>
      <pre className="bg-muted max-h-80 overflow-auto rounded-md p-3 text-[11px] leading-relaxed whitespace-pre-wrap">
        {data ? data.lines.join("\n") || "(empty)" : "Loading…"}
      </pre>
    </div>
  );
}

/** Settings → Diagnostics (Section 19): health checks, versions, storage, logs and a support bundle. */
export function DiagnosticsCard() {
  const { data: d, error } = useDiagnostics();
  const { data: s } = useSystemStatus();
  const rerun = useRerunDiagnostics();
  const toast = useUi((st) => st.pushToast);
  const [log, setLog] = useState<string | null>(null);
  const [bundling, setBundling] = useState(false);
  const problems = d?.checks.filter((c) => c.status === "error" || c.status === "warn").length ?? 0;

  const bundle = async () => {
    setBundling(true);
    try {
      await downloadBundle();
    } catch (e) {
      toast({ title: "Couldn't download the bundle", description: (e as Error).message, variant: "error" });
    } finally {
      setBundling(false);
    }
  };

  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2">
        <CardTitle>Diagnostics</CardTitle>
        {d && (
          <span className={cn("text-xs", problems ? "text-warning" : "text-success")}>
            {problems ? `${problems} to look at` : "All good"}
          </span>
        )}
        <div className="ml-auto flex gap-2">
          <Button size="sm" variant="outline" onClick={() => rerun.mutate()} disabled={rerun.isPending}>
            <RefreshCwIcon className={cn(rerun.isPending && "animate-spin")} />
            Run checks
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void bundle()}
            disabled={bundling}
            title="A zip with checks, versions, config and recent logs (keys and tokens redacted) for bug reports"
          >
            {bundling ? <Loader2Icon className="animate-spin" /> : <DownloadIcon />}
            Support bundle
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-5" data-testid="diagnostics">
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {!d ? (
          <Skeleton className="h-48" />
        ) : (
          <>
            <ul className="grid gap-x-6 sm:grid-cols-2" aria-label="Checks">
              {d.checks.map((c) => (
                <CheckRow key={c.key} check={c} />
              ))}
            </ul>

            <div className="grid gap-x-8 gap-y-4 md:grid-cols-2">
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase">System</h4>
                <Row label="Data folder">{s?.data_dir ?? "–"}</Row>
                {s?.gpu && (
                  <Row label="VRAM">
                    {formatMb(s.gpu.vram_used_mb)} / {formatMb(s.gpu.vram_total_mb)}
                  </Row>
                )}
                {s && (
                  <Row label="System RAM">
                    {formatMb(s.ram_used_mb)} / {formatMb(s.ram_total_mb)}
                  </Row>
                )}
                {s && (
                  <Row label="Worker">
                    {s.worker.state} · pid {s.worker.pid ?? "–"} · restarts {s.worker.restarts}
                  </Row>
                )}
                {Object.entries(d.versions).map(([k, v]) => (
                  <Row key={k} label={k}>
                    {v ?? "–"}
                  </Row>
                ))}
              </div>
              <div>
                <h4 className="mb-1 text-xs font-semibold uppercase">Storage</h4>
                <Row label="Disk free">
                  {formatMb(d.disk_free_mb)} of {formatMb(d.disk_total_mb)}
                </Row>
                {d.storage.map((f) => (
                  <Row key={f.key} label={FOLDER_LABELS[f.key] ?? f.key}>
                    <span title={f.path}>
                      {formatMb(f.size_mb)} · {f.files.toLocaleString()} files
                    </span>
                  </Row>
                ))}
                <h4 className="mt-4 mb-1 text-xs font-semibold uppercase">Logs</h4>
                <div className="flex flex-wrap gap-1.5">
                  {d.logs.map((l) => (
                    <Button
                      key={l.name}
                      size="sm"
                      variant={log === l.name ? "secondary" : "ghost"}
                      className="h-7 text-xs"
                      aria-pressed={log === l.name}
                      onClick={() => setLog(log === l.name ? null : l.name)}
                    >
                      <FileTextIcon />
                      {l.name} <span className="text-muted-foreground">{l.size_kb} KB</span>
                    </Button>
                  ))}
                </div>
              </div>
            </div>
            {log && <LogViewer name={log} />}
          </>
        )}
      </CardContent>
    </Card>
  );
}
