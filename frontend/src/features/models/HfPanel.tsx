import { DownloadIcon, ExternalLinkIcon, Loader2Icon, LockIcon, SearchIcon } from "lucide-react";
import { useMemo, useState } from "react";
import type { ApiError } from "@/api/client";
import {
  formatBytes,
  useHfInspect,
  useHfInstall,
  useModelOptions,
  type HfInspect,
  type HfInstallBody,
} from "@/api/models";
import { Button } from "@/components/ui/button";
import {
  Badge,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Label,
  NativeSelect,
} from "@/components/ui/primitives";
import { AccessNotice } from "./ModelCard";

const KINDS = ["checkpoint", "lora", "vae", "text_encoder", "audio"] as const;
type Kind = (typeof KINDS)[number];

export function HfPanel() {
  const [repo, setRepo] = useState("");
  const inspect = useHfInspect();
  const data = inspect.data;

  const run = (repoId = repo) => {
    if (repoId.trim()) inspect.mutate({ repo_id: repoId.trim() });
  };

  return (
    <div className="space-y-4">
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <Input
          aria-label="Hugging Face repo id"
          placeholder="org/name or https://huggingface.co/org/name"
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          className="max-w-xl"
        />
        <Button type="submit" disabled={!repo.trim() || inspect.isPending}>
          {inspect.isPending ? <Loader2Icon className="animate-spin" /> : <SearchIcon />}
          Inspect
        </Button>
      </form>
      {inspect.error && <p className="text-destructive text-sm">{(inspect.error as ApiError).message}</p>}
      {data && (
        <InspectResult
          key={`${data.repo_id}:${data.sha}`}
          data={data}
          onRecheck={() => run(data.repo_id)}
          rechecking={inspect.isPending}
        />
      )}
    </div>
  );
}

function InspectResult({
  data,
  onRecheck,
  rechecking,
}: {
  data: HfInspect;
  onRecheck: () => void;
  rechecking: boolean;
}) {
  const { data: options } = useModelOptions();
  const install = useHfInstall();
  const [selected, setSelected] = useState(() => new Set(data.default_selection));
  const [adapter, setAdapter] = useState<string>(data.suggestion.adapter_key ?? "");
  const [kind, setKind] = useState<Kind>(
    (KINDS as readonly string[]).includes(data.suggestion.kind)
      ? (data.suggestion.kind as Kind)
      : "checkpoint",
  );
  const [name, setName] = useState(data.repo_id.split("/")[1] ?? data.repo_id);
  const [showAll, setShowAll] = useState(false);

  const sizeOf = useMemo(() => new Map(data.files.map((f) => [f.path, f.size ?? 0])), [data.files]);
  const selectedSize = [...selected].reduce((n, p) => n + (sizeOf.get(p) ?? 0), 0);
  const files = showAll
    ? data.files
    : data.files.filter(
        (f) =>
          f.default_selected ||
          selected.has(f.path) ||
          f.file_class === "weight" ||
          f.file_class === "pickle",
      );
  const accessOk = data.access === "ok";
  const done = install.data;

  const toggle = (path: string) =>
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const submit = () => {
    const body: HfInstallBody = {
      repo_id: data.repo_id,
      allow_patterns: [...selected].sort(),
      adapter_key: adapter || null,
      kind,
      name: name.trim() || null,
    };
    install.mutate(body);
  };

  return (
    <Card data-testid="hf-inspect">
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="font-mono">{data.repo_id}</CardTitle>
          {data.gated && (
            <Badge variant={accessOk ? "outline" : "warning"}>
              <LockIcon className="size-3" /> gated{data.gated_mode ? ` (${data.gated_mode})` : ""}
            </Badge>
          )}
          {data.private && <Badge variant="outline">private</Badge>}
          {accessOk && <Badge variant="success">Access OK</Badge>}
          <a
            href={data.model_url}
            target="_blank"
            rel="noreferrer"
            className="text-muted-foreground hover:text-foreground ml-auto inline-flex items-center gap-1 text-xs"
          >
            Model page <ExternalLinkIcon className="size-3" />
          </a>
        </div>
        <div className="text-muted-foreground flex flex-wrap gap-x-4 gap-y-1 text-xs">
          <span>
            License:{" "}
            <a
              href={data.license_url}
              target="_blank"
              rel="noreferrer"
              className="text-foreground hover:underline"
            >
              {data.license ?? "unknown"}
            </a>
          </span>
          {data.pipeline_tag && <span>Pipeline: {data.pipeline_tag}</span>}
          {data.library_name && <span>Library: {data.library_name}</span>}
          <span>Repo size: {formatBytes(data.total_size)}</span>
          {data.account && <span>Signed in as {data.account}</span>}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <AccessNotice
          access={data.access}
          message={data.access_message}
          modelUrl={data.model_url}
          source="hf"
          onCheckAgain={onRecheck}
          checking={rechecking}
        />
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label>Files to download</Label>
            <span className="text-muted-foreground text-xs">
              {selected.size} selected · {formatBytes(selectedSize)}
            </span>
            <Button
              size="sm"
              variant="ghost"
              className="ml-auto h-6 text-xs"
              onClick={() => setShowAll((v) => !v)}
            >
              {showAll ? "Hide configs/docs" : `Show all ${data.files.length} files`}
            </Button>
          </div>
          <ul className="max-h-72 overflow-y-auto rounded-md border text-xs">
            {files.map((f) => (
              <li key={f.path} className="flex items-center gap-2 border-b px-2 py-1 last:border-0">
                <input
                  type="checkbox"
                  aria-label={f.path}
                  checked={selected.has(f.path)}
                  disabled={!!f.blocked_reason}
                  onChange={() => toggle(f.path)}
                />
                <span className={f.blocked_reason ? "text-muted-foreground line-through" : "font-mono"}>
                  {f.path}
                </span>
                {f.blocked_reason && (
                  <span className="text-muted-foreground truncate">— {f.blocked_reason}</span>
                )}
                <span className="text-muted-foreground ml-auto shrink-0">{formatBytes(f.size)}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-1">
            <Label htmlFor="hf-name">Name</Label>
            <Input id="hf-name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="space-y-1">
            <Label htmlFor="hf-kind">Kind</Label>
            <NativeSelect id="hf-kind" value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </NativeSelect>
          </div>
          <div className="space-y-1">
            <Label htmlFor="hf-adapter">Adapter (confirm)</Label>
            <NativeSelect id="hf-adapter" value={adapter} onChange={(e) => setAdapter(e.target.value)}>
              <option value="">None (component / unsupported)</option>
              {options?.adapters.map((a) => (
                <option key={a.key} value={a.key}>
                  {a.name}
                  {a.available ? "" : ` (arrives in ${a.milestone})`}
                </option>
              ))}
            </NativeSelect>
          </div>
        </div>
        <p className="text-muted-foreground text-xs">
          Suggested: {data.suggestion.kind}
          {data.suggestion.base_model ? ` · ${data.suggestion.base_model}` : ""}
          {data.suggestion.adapter_key ? ` · adapter ${data.suggestion.adapter_key}` : ""} (
          {data.suggestion.reason})
        </p>
        {install.error && <p className="text-destructive text-sm">{(install.error as ApiError).message}</p>}
        {done ? (
          <p className="text-success text-sm">
            Download started for {done.entry.name}. Follow progress on the Installed tab or in the Jobs
            drawer.
          </p>
        ) : (
          <Button
            disabled={!accessOk || selected.size === 0 || install.isPending || !!data.existing_entry_id}
            onClick={submit}
          >
            {install.isPending ? <Loader2Icon className="animate-spin" /> : <DownloadIcon />}
            Install {formatBytes(selectedSize)}
          </Button>
        )}
        {data.existing_entry_id && !done && (
          <p className="text-muted-foreground text-xs">
            This repo is already in your registry (Installed tab). Delete it there to change the selected
            files.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
