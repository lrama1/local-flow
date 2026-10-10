import {
  DownloadIcon,
  ExternalLinkIcon,
  ImageOffIcon,
  Loader2Icon,
  LockIcon,
  SearchIcon,
} from "lucide-react";
import { useState } from "react";
import type { ApiError } from "@/api/client";
import {
  ACCESS_LABELS,
  formatBytes,
  useCivitaiInstall,
  useCivitaiSearch,
  type AccessState,
  type CivitaiModel,
  type CivitaiSearchParams,
} from "@/api/models";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, Input, NativeSelect, Skeleton } from "@/components/ui/primitives";
import { AccessNotice, PermissionList } from "./ModelCard";

const TYPE_FILTERS: Record<string, CivitaiSearchParams["types"]> = {
  all: ["Checkpoint", "LORA", "LoCon", "DoRA", "VAE"],
  checkpoint: ["Checkpoint"],
  lora: ["LORA", "LoCon", "DoRA"],
  vae: ["VAE"],
};
const BASE_MODELS = [
  "SDXL 1.0",
  "Pony",
  "Illustrious",
  "NoobAI",
  "Flux.1 D",
  "Flux.1 S",
  "Wan Video 2.2 TI2V-5B",
  "Wan Video 2.2 I2V-A14B",
  "Wan Video 2.2 T2V-A14B",
  "Wan Video 14B i2v 720p",
];
const SORTS = ["Most Downloaded", "Highest Rated", "Newest"] as const;
const ACCESS_CODES = Object.keys(ACCESS_LABELS);

export function CivitaiPanel() {
  const [form, setForm] = useState({
    query: "",
    type: "lora",
    base: "",
    sort: "Most Downloaded" as (typeof SORTS)[number],
  });
  const [params, setParams] = useState<CivitaiSearchParams | null>(null);
  const search = useCivitaiSearch(params ?? {}, params !== null);
  const items = search.data?.pages.flatMap((p) => p.items) ?? [];
  const hidden = search.data?.pages.reduce((n, p) => n + (p.hidden_nsfw ?? 0), 0) ?? 0;

  return (
    <div className="space-y-4">
      <form
        className="flex flex-wrap gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setParams({
            query: form.query.trim() || undefined,
            types: TYPE_FILTERS[form.type],
            base_models: form.base ? [form.base] : undefined,
            sort: form.sort,
          });
        }}
      >
        <Input
          aria-label="Search Civitai"
          placeholder="Search Civitai (e.g. detail, film grain, camera motion)"
          value={form.query}
          onChange={(e) => setForm({ ...form, query: e.target.value })}
          className="max-w-sm"
        />
        <NativeSelect
          aria-label="Type"
          value={form.type}
          onChange={(e) => setForm({ ...form, type: e.target.value })}
          className="w-36"
        >
          <option value="all">All types</option>
          <option value="checkpoint">Checkpoints</option>
          <option value="lora">LoRAs</option>
          <option value="vae">VAEs</option>
        </NativeSelect>
        <NativeSelect
          aria-label="Base model"
          value={form.base}
          onChange={(e) => setForm({ ...form, base: e.target.value })}
          className="w-52"
        >
          <option value="">Any base model</option>
          {BASE_MODELS.map((b) => (
            <option key={b}>{b}</option>
          ))}
        </NativeSelect>
        <NativeSelect
          aria-label="Sort"
          value={form.sort}
          onChange={(e) => setForm({ ...form, sort: e.target.value as (typeof SORTS)[number] })}
          className="w-44"
        >
          {SORTS.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </NativeSelect>
        <Button type="submit" disabled={search.isFetching && !search.isFetchingNextPage}>
          {search.isFetching && !search.isFetchingNextPage ? (
            <Loader2Icon className="animate-spin" />
          ) : (
            <SearchIcon />
          )}
          Search
        </Button>
      </form>
      {search.error && <p className="text-destructive text-sm">{(search.error as ApiError).message}</p>}
      {params === null && (
        <EmptyState icon={SearchIcon} title="Browse Civitai">
          Search checkpoints and LoRAs. Public files download without an account; login-required and
          early-access versions need a Civitai API key (Settings).
        </EmptyState>
      )}
      {search.isLoading && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-72" />
          ))}
        </div>
      )}
      {params !== null && !search.isLoading && items.length === 0 && !search.error && (
        <p className="text-muted-foreground text-sm">No results.</p>
      )}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {items.map((m) => (
          <CivitaiCard key={m.id} model={m} />
        ))}
      </div>
      {hidden > 0 && (
        <p className="text-muted-foreground text-xs">
          {hidden} NSFW result{hidden === 1 ? "" : "s"} hidden (allow_nsfw_models is off).
        </p>
      )}
      {search.hasNextPage && (
        <Button
          variant="outline"
          disabled={search.isFetchingNextPage}
          onClick={() => void search.fetchNextPage()}
        >
          {search.isFetchingNextPage && <Loader2Icon className="animate-spin" />}
          Load more
        </Button>
      )}
    </div>
  );
}

function CivitaiCard({ model }: { model: CivitaiModel }) {
  const [versionId, setVersionId] = useState(model.versions[0]?.id);
  const install = useCivitaiInstall();
  const v = model.versions.find((x) => x.id === versionId) ?? model.versions[0];
  const file = v?.files.find((f) => f.primary) ?? v?.files.find((f) => !f.blocked_reason) ?? v?.files[0];
  const err = install.error as ApiError | null;
  const errAccess = err && ACCESS_CODES.includes(err.code) ? (err.code as AccessState) : null;
  const preview = v?.preview_url ?? model.preview_url;

  return (
    <Card className="flex flex-col overflow-hidden" data-testid={`civitai-${model.id}`}>
      <div className="bg-muted aspect-[4/3] w-full overflow-hidden">
        {preview ? (
          <img
            src={preview}
            alt=""
            loading="lazy"
            className="size-full object-cover"
            referrerPolicy="no-referrer"
          />
        ) : (
          <div className="text-muted-foreground flex size-full items-center justify-center">
            <ImageOffIcon className="size-6" />
          </div>
        )}
      </div>
      <CardContent className="flex flex-1 flex-col gap-2 pt-3">
        <div className="flex items-start gap-2">
          <a
            href={model.model_url}
            target="_blank"
            rel="noreferrer"
            className="line-clamp-2 text-sm leading-snug font-semibold hover:underline"
          >
            {model.name}
          </a>
          <Badge variant="outline" className="ml-auto shrink-0">
            {model.type}
          </Badge>
        </div>
        <p className="text-muted-foreground text-xs">
          {model.creator ? `by ${model.creator}` : ""}
          {model.downloads != null ? ` · ${model.downloads.toLocaleString()} downloads` : ""}
        </p>
        {model.versions.length > 1 && (
          <NativeSelect
            aria-label="Version"
            value={versionId}
            onChange={(e) => setVersionId(Number(e.target.value))}
            className="h-8 text-xs"
          >
            {model.versions.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} · {x.base_model_raw}
              </option>
            ))}
          </NativeSelect>
        )}
        {v && (
          <>
            <div className="flex flex-wrap gap-1.5">
              <Badge variant={v.supported ? "secondary" : "warning"}>
                {v.base_model_raw ?? "unknown base"}
                {v.supported ? ` → ${v.adapter_key}` : " · unsupported"}
              </Badge>
              {v.lock_reason && (
                <Badge variant="warning">
                  <LockIcon className="size-3" /> {v.lock_reason}
                </Badge>
              )}
            </div>
            {v.trained_words.length > 0 && (
              <p className="line-clamp-2 text-xs">
                <span className="text-muted-foreground">Trigger words: </span>
                {v.trained_words.join(", ")}
              </p>
            )}
            {file && (
              <p className="text-muted-foreground text-xs">
                {file.name} · {formatBytes(file.size_bytes)}
                {file.format ? ` · ${file.format}` : ""}
                {file.fp ? ` · ${file.fp}` : ""}
              </p>
            )}
            {file?.blocked_reason && <p className="text-warning text-xs">{file.blocked_reason}</p>}
          </>
        )}
        <details className="text-xs">
          <summary className="text-muted-foreground cursor-pointer">Permissions</summary>
          <div className="pt-1">
            <PermissionList permissions={model.permissions} />
          </div>
        </details>
        {errAccess ? (
          <AccessNotice
            access={errAccess}
            message={err?.message}
            modelUrl={model.model_url}
            source="civitai"
          />
        ) : (
          err && <p className="text-destructive text-xs">{err.message}</p>
        )}
        <div className="mt-auto flex items-center gap-2 pt-1">
          {v?.installed_entry_id || install.isSuccess ? (
            <Badge variant="success">In registry</Badge>
          ) : (
            <Button
              size="sm"
              disabled={!v || !file || !!file.blocked_reason || !v.downloadable || install.isPending}
              onClick={() => v && install.mutate({ version_id: v.id, file_id: file?.id ?? null })}
            >
              {install.isPending ? <Loader2Icon className="animate-spin" /> : <DownloadIcon />}
              Install
            </Button>
          )}
          <a
            href={model.model_url}
            target="_blank"
            rel="noreferrer"
            className="text-muted-foreground hover:text-foreground ml-auto inline-flex items-center gap-1 text-xs"
          >
            civitai.com <ExternalLinkIcon className="size-3" />
          </a>
        </div>
      </CardContent>
    </Card>
  );
}
