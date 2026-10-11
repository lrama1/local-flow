import { BoxesIcon, CheckIcon, FlaskConicalIcon, Loader2Icon, SparklesIcon, XIcon } from "lucide-react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router";
import { withToken, type Asset, type Job } from "@/api/client";
import { useAsset } from "@/api/imagegen";
import { useModels } from "@/api/models";
import { useAdapters, useAssets, useGenerate, useJob, usePlayground, type GenerateKind } from "@/api/queries";
import { AssetGrid } from "@/components/AssetGrid";
import { EmptyState } from "@/components/EmptyState";
import { MediaPlayer, AssetThumb } from "@/components/MediaPreview";
import { ModelPicker } from "@/components/ModelPicker";
import { VoicePicker } from "@/components/VoicePicker";
import { SchemaForm } from "@/components/SchemaForm";
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
  Progress,
} from "@/components/ui/primitives";
import { MentionTextarea } from "@/components/MentionTextarea";
import { useLibraryMentionOptions } from "@/features/characters/mentionOptions";
import { gateLicense } from "@/features/models/gateLicense";
import { cn } from "@/lib/utils";
import { useUi } from "@/stores/ui";
import { LoraPicker } from "./LoraPicker";

// Generation kinds whose prompts resolve @mentions on the backend (speech, audio and music don't).
const MENTION_KINDS = new Set<GenerateKind>(["t2i", "image_edit", "t2v", "ref2v", "i2v", "flf2v", "extend"]);
import {
  DEFAULT_FORM,
  KIND_ORDER,
  KINDS,
  SIZE_PRESETS,
  buildBody,
  type AssetInput,
  type PlaygroundForm,
  type SizePreset,
} from "./kinds";

function AssetPicker({
  input,
  assets,
  selected,
  onChange,
}: {
  input: AssetInput;
  assets: Asset[];
  selected: string[];
  onChange: (ids: string[]) => void;
}) {
  const options = assets.filter((a) => a.type === input.type);
  // Inputs reused from another project's asset ("Reuse settings") aren't in the Playground's list.
  const foreign = selected.filter((id) => !options.some((a) => a.id === id));
  return (
    <div className="space-y-1.5">
      <Label>
        {input.label}
        {input.multiple && " (pick up to 8)"}
      </Label>
      {foreign.length > 0 && (
        <div className="flex flex-wrap gap-1.5" data-testid={`foreign-${input.field}`}>
          {foreign.map((id) => (
            <ForeignPick key={id} id={id} onRemove={() => onChange(selected.filter((x) => x !== id))} />
          ))}
        </div>
      )}
      {options.length === 0 ? (
        !foreign.length && (
          <p className="text-muted-foreground rounded-md border border-dashed p-3 text-xs">
            No {input.type}s yet. Generate one first with{" "}
            {input.type === "image" ? "Text → image" : "Text → video"}.
          </p>
        )
      ) : (
        <div className="grid grid-cols-5 gap-1.5">
          {options.slice(0, 20).map((a) => {
            const on = selected.includes(a.id);
            return (
              <button
                type="button"
                key={a.id}
                aria-pressed={on}
                aria-label={`Select ${input.type}`}
                onClick={() =>
                  onChange(
                    input.multiple
                      ? on
                        ? selected.filter((x) => x !== a.id)
                        : [...selected, a.id].slice(0, 8)
                      : on
                        ? []
                        : [a.id],
                  )
                }
                className={cn(
                  "relative aspect-square overflow-hidden rounded-md border-2",
                  on ? "border-primary" : "border-transparent opacity-80 hover:opacity-100",
                )}
              >
                <AssetThumb asset={a} />
                {on && <CheckIcon className="bg-primary absolute top-1 right-1 size-4 rounded-full p-0.5" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ForeignPick({ id, onRemove }: { id: string; onRemove: () => void }) {
  const { data: asset } = useAsset(id);
  return (
    <span className="bg-muted border-primary relative inline-flex size-16 overflow-hidden rounded-md border-2">
      {asset && <AssetThumb asset={asset} />}
      <button
        type="button"
        aria-label="Remove input"
        title="From another project"
        onClick={onRemove}
        className="bg-background/80 absolute top-0.5 right-0.5 rounded-full p-0.5"
      >
        <XIcon className="size-3" />
      </button>
    </span>
  );
}

function CurrentJob({ job, assets }: { job: Job; assets: Asset[] }) {
  const preview = useUi((s) => s.previews[job.id]);
  const ids = (job.result?.asset_ids as string[] | undefined) ?? [];
  const outputs = ids.map((id) => assets.find((a) => a.id === id)).filter((a): a is Asset => !!a);
  const active = job.status === "queued" || job.status === "running";

  return (
    <Card data-testid="current-job">
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>{KINDS[job.type as GenerateKind]?.label ?? job.type}</CardTitle>
        <Badge
          variant={
            job.status === "completed" ? "success" : job.status === "failed" ? "destructive" : "default"
          }
          data-testid="current-job-status"
        >
          {job.status}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        {active && (
          <>
            <div className="bg-muted flex aspect-video items-center justify-center overflow-hidden rounded-lg">
              {preview ? (
                <img src={withToken(preview)} alt="Step preview" className="size-full object-contain" />
              ) : (
                <Loader2Icon className="text-muted-foreground size-8 animate-spin" />
              )}
            </div>
            <Progress value={job.progress * 100} aria-label="Current job progress" />
            <div className="text-muted-foreground flex justify-between text-xs">
              <span>{job.status === "queued" ? "Waiting for the GPU worker…" : job.progress_message}</span>
              <span>{Math.round(job.progress * 100)}%</span>
            </div>
          </>
        )}
        {job.status === "completed" &&
          (outputs.length ? (
            <div className={cn("grid gap-2", outputs.length > 1 && "grid-cols-2")}>
              {outputs.map((a) => (
                <MediaPlayer key={a.id} asset={a} />
              ))}
            </div>
          ) : (
            <Loader2Icon className="text-muted-foreground mx-auto size-6 animate-spin" />
          ))}
        {job.status === "failed" && <p className="text-destructive text-sm">{job.error}</p>}
        {job.status === "cancelled" && <p className="text-muted-foreground text-sm">Cancelled.</p>}
      </CardContent>
    </Card>
  );
}

export function PlaygroundPage() {
  // "Reuse settings" from the asset library opens the Playground prefilled (read once, then cleared).
  const [kind, setKind] = useState<GenerateKind>(
    () => (useUi.getState().playgroundPrefill?.kind as GenerateKind | undefined) ?? "t2v",
  );
  const [form, setForm] = useState<PlaygroundForm>(() => ({
    ...DEFAULT_FORM,
    ...(useUi.getState().playgroundPrefill?.form as Partial<PlaygroundForm> | undefined),
  }));
  useEffect(() => useUi.getState().setPlaygroundPrefill(null), []);
  const [jobId, setJobId] = useState<string>();
  const [formError, setFormError] = useState<string>();
  const cfg = KINDS[kind];

  const { data: playground } = usePlayground();
  const { data: adapters, isLoading: adaptersLoading } = useAdapters();
  const { data: models } = useModels();
  const { data: assetPage } = useAssets(playground?.project_id);
  const { data: job } = useJob(jobId);
  const generate = useGenerate();
  const mentionOptions = useLibraryMentionOptions();
  const assets = useMemo(() => assetPage?.items ?? [], [assetPage]);

  const usable = (adapters ?? []).filter((a) => a.available && a.capabilities.includes(kind));
  const imageKind = cfg.output === "image";
  const modelKind = cfg.output !== "audio"; // image and video kinds pick a registry model
  const model = modelKind ? models?.find((m) => m.id === form.modelId) : undefined;
  const adapter =
    (model && usable.find((a) => a.adapter_key === model.adapter_key)) ??
    usable.find((a) => a.adapter_key === form.adapterKey) ??
    usable[0];
  const loraTarget = model?.adapter_key ?? adapter?.adapter_key;
  const loraChoices = (models ?? []).filter(
    (m) =>
      m.kind === "lora" &&
      m.status === "installed" &&
      (!m.adapter_key || loraTarget === "fake" || m.adapter_key === loraTarget),
  );
  const update = (patch: Partial<PlaygroundForm>) => setForm((f) => ({ ...f, ...patch }));

  const run = () => {
    const built = buildBody(kind, form, playground?.project_id);
    if ("error" in built) {
      setFormError(built.error);
      return;
    }
    setFormError(undefined);
    generate.mutate(
      { kind, body: built.body as never },
      {
        onSuccess: (j) => setJobId(j.id),
        onError: (err) => {
          if (!gateLicense(err, run)) setFormError(err.message);
        },
      },
    );
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    run();
  };

  return (
    <div className="mx-auto grid max-w-7xl gap-6 p-6 lg:grid-cols-[420px_1fr]">
      <div className="space-y-4">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold">
            <FlaskConicalIcon className="size-5" /> Playground
          </h1>
          <p className="text-muted-foreground text-sm">
            Try any capability directly, outside the story flow.
          </p>
        </div>

        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Generation type">
          {KIND_ORDER.map((k) => (
            <Button
              key={k}
              type="button"
              role="tab"
              aria-selected={k === kind}
              size="sm"
              variant={k === kind ? "default" : "outline"}
              onClick={() => {
                setKind(k);
                setFormError(undefined);
              }}
            >
              {KINDS[k].label}
            </Button>
          ))}
        </div>

        {!adaptersLoading && adapters && adapters.length > 0 && usable.length === 0 && (
          <EmptyState
            icon={BoxesIcon}
            title={`No model installed for ${cfg.label}`}
            action={
              <Button asChild size="sm" variant="outline">
                <Link to="/models">Go to Models</Link>
              </Button>
            }
          />
        )}

        <form onSubmit={submit} className="space-y-4" aria-label="Generation form">
          <div className="space-y-1.5">
            <Label htmlFor="pg-prompt">{cfg.promptField === "text" ? "Text to speak" : "Prompt"}</Label>
            <MentionTextarea
              id="pg-prompt"
              value={form.prompt}
              options={MENTION_KINDS.has(kind) ? mentionOptions : undefined}
              onChange={(v) => update({ prompt: v })}
              placeholder={
                cfg.promptField === "text"
                  ? "Hello! This is my narrator voice."
                  : "A lighthouse on a cliff at dusk, waves crashing, cinematic"
              }
              rows={4}
            />
            {MENTION_KINDS.has(kind) && !!mentionOptions?.length && (
              <p className="text-muted-foreground text-xs">Type @ to use a Cast library character.</p>
            )}
          </div>

          {cfg.negative && (
            <div className="space-y-1.5">
              <Label htmlFor="pg-negative">Negative prompt</Label>
              <Input
                id="pg-negative"
                value={form.negative}
                onChange={(e) => update({ negative: e.target.value })}
                placeholder="blurry, text, watermark"
              />
            </div>
          )}

          {modelKind && (
            <div className="space-y-1.5">
              <Label>Model</Label>
              <ModelPicker
                capability={kind}
                value={form.modelId || null}
                onChange={(id) => update({ modelId: id ?? "", loras: [], extra: {} })}
                className="w-full"
              />
            </div>
          )}

          {cfg.voice && (
            <div className="space-y-1.5">
              <Label htmlFor="pg-voice">Voice</Label>
              <VoicePicker
                id="pg-voice"
                value={form.voiceId || null}
                onChange={(id) => update({ voiceId: id ?? "" })}
                noneLabel="Default voice"
              />
            </div>
          )}

          {imageKind && loraChoices.length > 0 && (
            <LoraPicker choices={loraChoices} value={form.loras} onChange={(loras) => update({ loras })} />
          )}

          {(cfg.assets ?? []).map((input) => (
            <AssetPicker
              key={input.field}
              input={input}
              assets={assets}
              selected={form.assets[input.field] ?? []}
              onChange={(ids) => update({ assets: { ...form.assets, [input.field]: ids } })}
            />
          ))}

          <div className="grid grid-cols-2 gap-3">
            {cfg.size && (
              <div className="space-y-1.5">
                <Label htmlFor="pg-aspect">Aspect</Label>
                <NativeSelect
                  id="pg-aspect"
                  value={form.preset}
                  onChange={(e) => update({ preset: e.target.value as SizePreset })}
                >
                  {Object.entries(SIZE_PRESETS).map(([k, v]) => {
                    const [w, h] = v[cfg.output === "image" ? "image" : "video"];
                    return (
                      <option key={k} value={k}>
                        {k} · {w}×{h}
                      </option>
                    );
                  })}
                </NativeSelect>
              </div>
            )}
            {cfg.numImages && (
              <div className="space-y-1.5">
                <Label htmlFor="pg-count">Images</Label>
                <NativeSelect
                  id="pg-count"
                  value={form.numImages}
                  onChange={(e) => update({ numImages: Number(e.target.value) })}
                >
                  {[1, 2, 4].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </NativeSelect>
              </div>
            )}
            {cfg.duration && (
              <div className="space-y-1.5">
                <Label htmlFor="pg-duration">Duration (s)</Label>
                <Input
                  id="pg-duration"
                  type="number"
                  min={0.5}
                  max={cfg.output === "audio" ? 600 : 120}
                  step={0.5}
                  value={form.duration}
                  onChange={(e) => update({ duration: Number(e.target.value) })}
                />
              </div>
            )}
            {cfg.duration && cfg.output === "video" && (
              <div className="space-y-1.5">
                <Label htmlFor="pg-fps">FPS</Label>
                <NativeSelect
                  id="pg-fps"
                  value={form.fps}
                  onChange={(e) => update({ fps: Number(e.target.value) })}
                >
                  {[16, 24, 30].map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </NativeSelect>
              </div>
            )}
            <div className="space-y-1.5">
              <Label htmlFor="pg-seed">Seed</Label>
              <Input
                id="pg-seed"
                inputMode="numeric"
                value={form.seed}
                onChange={(e) => update({ seed: e.target.value })}
                placeholder="random"
              />
            </div>
            {!modelKind && usable.length > 1 && (
              <div className="space-y-1.5">
                <Label htmlFor="pg-adapter">Model</Label>
                <NativeSelect
                  id="pg-adapter"
                  value={adapter?.adapter_key ?? ""}
                  onChange={(e) => update({ adapterKey: e.target.value, extra: {} })}
                >
                  {usable.map((a) => (
                    <option key={a.adapter_key} value={a.adapter_key}>
                      {a.name}
                    </option>
                  ))}
                </NativeSelect>
              </div>
            )}
          </div>

          {cfg.withAudio && (
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.withAudio}
                onChange={(e) => update({ withAudio: e.target.checked })}
              />
              Generate native audio (if the model supports it)
            </label>
          )}

          <details className="rounded-lg border p-3">
            <summary className="cursor-pointer text-sm font-medium">
              Advanced{adapter ? ` · ${adapter.name}` : ""}
            </summary>
            <div className="pt-3">
              <SchemaForm
                schema={adapter?.param_schema as never}
                value={form.extra}
                onChange={(extra) => update({ extra })}
              />
            </div>
          </details>

          {formError && (
            <p className="text-destructive text-sm" role="alert">
              {formError}
            </p>
          )}
          <Button type="submit" className="w-full" disabled={generate.isPending || !adapter}>
            {generate.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
            Generate
          </Button>
        </form>
      </div>

      <div className="min-w-0 space-y-6">
        {job ? (
          <CurrentJob job={job} assets={assets} />
        ) : (
          <EmptyState icon={SparklesIcon} title="Nothing generated yet">
            Pick a capability, write a prompt, and hit Generate. Progress streams in live.
          </EmptyState>
        )}
        {assets.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-muted-foreground text-sm font-medium">Playground outputs</h2>
            <AssetGrid assets={assets} />
          </section>
        )}
      </div>
    </div>
  );
}
