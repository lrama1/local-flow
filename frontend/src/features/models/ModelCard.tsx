import {
  AlertTriangleIcon,
  CheckIcon,
  DownloadIcon,
  ExternalLinkIcon,
  FlaskConicalIcon,
  Loader2Icon,
  LockIcon,
  RefreshCwIcon,
  ScaleIcon,
  StarIcon,
  Trash2Icon,
  XIcon,
  ZapIcon,
} from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import type { ApiError } from "@/api/client";
import {
  ACCESS_LABELS,
  formatBytes,
  useAcknowledgeLicense,
  useCancelDownload,
  useCheckAccess,
  useDeleteModel,
  useInstallModel,
  type AccessState,
  type ModelEntry,
} from "@/api/models";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardHeader, CardTitle, Progress } from "@/components/ui/primitives";
import { JOB_TYPE_LABELS } from "@/lib/utils";
import { useUi } from "@/stores/ui";

const STATUS_VARIANT = {
  available: "outline",
  downloading: "default",
  installed: "success",
  error: "destructive",
} as const;

const TOKEN_STATES: AccessState[] = ["needs_token", "invalid_token"];
const PAGE_STATES: AccessState[] = [
  "license_not_accepted",
  "pending_approval",
  "token_no_access",
  "forbidden",
];

export function AccessNotice({
  access,
  message,
  modelUrl,
  source,
  onCheckAgain,
  checking,
}: {
  access: AccessState | null | undefined;
  message?: string | null;
  modelUrl?: string | null;
  source: string;
  onCheckAgain?: () => void;
  checking?: boolean;
}) {
  if (!access || access === "ok") return null;
  const hub = source === "civitai" ? "Civitai" : "Hugging Face";
  return (
    <div className="border-warning/40 bg-warning/5 space-y-2 rounded-md border p-2.5 text-xs" role="alert">
      <div className="text-warning flex items-center gap-1.5 font-medium">
        <LockIcon className="size-3.5" />
        {ACCESS_LABELS[access]}
      </div>
      {message && <p className="text-muted-foreground">{message}</p>}
      <div className="flex flex-wrap gap-2">
        {TOKEN_STATES.includes(access) && (
          <Link to="/settings" className="text-primary inline-flex items-center gap-1 hover:underline">
            Add a {hub} token in Settings
          </Link>
        )}
        {(PAGE_STATES.includes(access) || access === "not_found") && modelUrl && (
          <a
            href={modelUrl}
            target="_blank"
            rel="noreferrer"
            className="text-primary inline-flex items-center gap-1 hover:underline"
          >
            Open model page <ExternalLinkIcon className="size-3" />
          </a>
        )}
        {access === "token_no_access" && source === "hf" && (
          <a
            href="https://huggingface.co/settings/tokens"
            target="_blank"
            rel="noreferrer"
            className="text-primary inline-flex items-center gap-1 hover:underline"
          >
            Update token permissions <ExternalLinkIcon className="size-3" />
          </a>
        )}
        {onCheckAgain && (
          <Button
            size="sm"
            variant="outline"
            className="h-6 px-2 text-xs"
            disabled={checking}
            onClick={onCheckAgain}
          >
            {checking ? <Loader2Icon className="animate-spin" /> : <RefreshCwIcon />}
            Check again
          </Button>
        )}
      </div>
    </div>
  );
}

export function LicenseDialog({
  model,
  open,
  onClose,
  onAcknowledged,
}: {
  model: ModelEntry;
  open: boolean;
  onClose: () => void;
  onAcknowledged?: () => void;
}) {
  const ack = useAcknowledgeLicense();
  const [checked, setChecked] = useState(false);
  return (
    <ConfirmDialog
      open={open}
      title={`License: ${model.name}`}
      description="Review the model's license before first use. The acknowledgement is saved for this model."
      confirmLabel={ack.isPending ? "Saving…" : "Acknowledge"}
      onCancel={onClose}
      onConfirm={() => {
        if (!checked) return;
        ack.mutate(model.id, { onSuccess: onAcknowledged ?? onClose });
      }}
    >
      <div className="space-y-3 text-sm">
        <div className="flex items-center gap-2">
          <ScaleIcon className="text-muted-foreground size-4" />
          <span className="font-medium">{model.license ?? "Unknown license"}</span>
        </div>
        {model.license_url && (
          <a
            href={model.license_url}
            target="_blank"
            rel="noreferrer"
            className="text-primary inline-flex items-center gap-1 text-xs hover:underline"
          >
            Read the license <ExternalLinkIcon className="size-3" />
          </a>
        )}
        {model.civitai?.permissions && Object.keys(model.civitai.permissions).length > 0 && (
          <PermissionList permissions={model.civitai.permissions} />
        )}
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
            aria-label="I have reviewed this license"
          />
          I have reviewed this license
        </label>
      </div>
    </ConfirmDialog>
  );
}

export function PermissionList({ permissions }: { permissions: Record<string, unknown> }) {
  const labels: Record<string, string> = {
    allowCommercialUse: "Commercial use",
    allowDerivatives: "Derivatives allowed",
    allowNoCredit: "Use without credit",
    allowDifferentLicense: "Relicensing allowed",
  };
  return (
    <ul className="text-muted-foreground space-y-0.5 text-xs">
      {Object.entries(labels).map(([k, label]) => {
        const v = permissions[k];
        const text = Array.isArray(v) ? (v.length ? v.join(", ") : "none") : v ? "yes" : "no";
        return (
          <li key={k}>
            {label}: <span className="text-foreground">{text}</span>
          </li>
        );
      })}
    </ul>
  );
}

function errorText(e: unknown): string | null {
  if (!e) return null;
  const err = e as ApiError;
  if (err.code && err.code in ACCESS_LABELS) return null; // shown by <AccessNotice> from the entry's state
  return err.message ?? String(e);
}

export function ModelCard({ model }: { model: ModelEntry }) {
  const install = useInstallModel();
  const del = useDeleteModel();
  const check = useCheckAccess();
  const cancel = useCancelDownload();
  const progress = useUi((s) => s.downloads[model.id]);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [licenseOpen, setLicenseOpen] = useState(false);

  const downloading = model.status === "downloading";
  const size = model.size_bytes ?? model.approx_size_bytes;
  const pct = progress && progress.total ? (progress.downloaded / progress.total) * 100 : 0;
  const partial = model.partial && !downloading;
  const canDelete = model.status === "installed" || model.status === "error" || partial || !model.catalog_key;
  const jobId = model.job_id ?? progress?.jobId;
  const err = errorText(install.error) ?? errorText(del.error);

  return (
    <Card className="flex flex-col" data-testid={`model-${model.catalog_key ?? model.id}`}>
      <CardHeader className="gap-1.5">
        <div className="flex items-start gap-2">
          <CardTitle className="leading-snug">{model.name}</CardTitle>
          {model.recommended && (
            <StarIcon className="text-warning size-3.5 shrink-0" aria-label="Recommended" />
          )}
          <Badge variant={STATUS_VARIANT[model.status]} className="ml-auto shrink-0">
            {model.status}
          </Badge>
        </div>
        {model.purpose && <p className="text-muted-foreground text-xs">{model.purpose}</p>}
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3">
        {model.description && (
          <p className="text-muted-foreground line-clamp-3 text-xs">{model.description}</p>
        )}
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="outline">{model.source === "hf" ? "Hugging Face" : model.source}</Badge>
          <Badge variant="outline">{model.kind}</Badge>
          {model.capabilities.map((c) => (
            <Badge key={c} variant="secondary">
              {JOB_TYPE_LABELS[c] ?? c}
            </Badge>
          ))}
          {model.gated && (
            <Badge variant={model.access === "ok" ? "outline" : "warning"}>
              <LockIcon className="size-3" /> gated
            </Badge>
          )}
          {model.experimental && (
            <Badge variant="warning">
              <FlaskConicalIcon className="size-3" /> experimental
            </Badge>
          )}
          {model.tier === "draft" && (
            <Badge variant="secondary">
              <ZapIcon className="size-3" /> draft · fast
            </Badge>
          )}
          {model.nsfw && <Badge variant="destructive">NSFW</Badge>}
        </div>
        <dl className="text-muted-foreground grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
          <dt>Size</dt>
          <dd className="text-foreground">
            {model.status === "installed" ? formatBytes(size) : `≈ ${formatBytes(size)}`}
          </dd>
          <dt>Adapter</dt>
          <dd className="text-foreground">
            {model.adapter_name ?? "none"}
            {model.adapter_key && !model.adapter_available && (
              <span className="text-muted-foreground"> · not loaded yet</span>
            )}
          </dd>
          {model.base_model && (
            <>
              <dt>Base</dt>
              <dd className="text-foreground">{model.base_model}</dd>
            </>
          )}
          <dt>License</dt>
          <dd className="flex min-w-0 items-center gap-1">
            {model.license_url ? (
              <a
                href={model.license_url}
                target="_blank"
                rel="noreferrer"
                className="text-foreground truncate hover:underline"
                title={model.license ?? undefined}
              >
                {model.license ?? "license"}
              </a>
            ) : (
              <span className="text-foreground">{model.license ?? "unknown"}</span>
            )}
            {model.license_acknowledged && (
              <CheckIcon className="text-success size-3" aria-label="Acknowledged" />
            )}
          </dd>
        </dl>
        {model.trigger_words.length > 0 && (
          <p className="text-xs">
            <span className="text-muted-foreground">Trigger words: </span>
            {model.trigger_words.join(", ")}
          </p>
        )}
        <AccessNotice
          access={model.access}
          message={model.access_message}
          modelUrl={model.model_url}
          source={model.source}
          checking={check.isPending}
          onCheckAgain={() => check.mutate(model.id)}
        />
        {model.pickle_blocked && (
          <p className="text-warning flex gap-1.5 text-xs">
            <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
            Only published as PyTorch pickle weights. Set <code>allow_pickle_weights = true</code> in
            config.toml to install it.
          </p>
        )}
        {model.notes && !model.pickle_blocked && (
          <p className="text-muted-foreground text-xs">{model.notes}</p>
        )}
        {model.error && model.status !== "downloading" && !model.access_message && (
          <p className="text-destructive text-xs">{model.error}</p>
        )}
        {err && <p className="text-destructive text-xs">{err}</p>}
        {downloading && (
          <div className="space-y-1" data-testid="download-progress">
            <Progress value={pct} />
            <div className="text-muted-foreground flex justify-between text-[11px]">
              <span>
                {progress
                  ? `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)}`
                  : "Waiting to start…"}
              </span>
              {progress && progress.bytesPerS > 0 && <span>{formatBytes(progress.bytesPerS)}/s</span>}
            </div>
          </div>
        )}
        <div className="mt-auto flex flex-wrap items-center gap-2 pt-1">
          {(model.status === "available" || model.status === "error") && (
            <Button
              size="sm"
              disabled={install.isPending || model.pickle_blocked}
              onClick={() => install.mutate(model.id)}
            >
              {install.isPending ? <Loader2Icon className="animate-spin" /> : <DownloadIcon />}
              {partial || model.status === "error" ? "Resume install" : "Install"}
            </Button>
          )}
          {downloading && jobId && (
            <Button
              size="sm"
              variant="outline"
              disabled={cancel.isPending}
              onClick={() => cancel.mutate(jobId)}
            >
              <XIcon /> Cancel
            </Button>
          )}
          {model.status === "installed" && !model.license_acknowledged && (
            <Button size="sm" variant="outline" onClick={() => setLicenseOpen(true)}>
              <ScaleIcon /> Review license
            </Button>
          )}
          {model.model_url && (
            <a
              href={model.model_url}
              target="_blank"
              rel="noreferrer"
              className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs"
            >
              Model page <ExternalLinkIcon className="size-3" />
            </a>
          )}
          {canDelete && !downloading && (
            <Button
              size="sm"
              variant="ghost"
              className="text-destructive ml-auto"
              aria-label={`Delete ${model.name}`}
              disabled={del.isPending}
              onClick={() => setConfirmDelete(true)}
            >
              <Trash2Icon />
            </Button>
          )}
        </div>
      </CardContent>
      <ConfirmDialog
        open={confirmDelete}
        title={`Delete ${model.name}?`}
        description={
          model.catalog_key
            ? "Deletes the downloaded files. The model stays in the catalog so you can install it again."
            : "Deletes the downloaded files and removes the model from the registry."
        }
        confirmLabel="Delete"
        destructive
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => {
          setConfirmDelete(false);
          del.mutate(model.id);
        }}
      />
      <LicenseDialog model={model} open={licenseOpen} onClose={() => setLicenseOpen(false)} />
    </Card>
  );
}
