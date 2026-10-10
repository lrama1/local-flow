import { useQueryClient } from "@tanstack/react-query";
import {
  CheckIcon,
  ImageIcon,
  ImportIcon,
  Loader2Icon,
  SparklesIcon,
  UploadIcon,
  WandSparklesIcon,
} from "lucide-react";
import { useRef, useState } from "react";
import {
  createDraft,
  deleteIngredient,
  EXTRA_VIEWS,
  mediaUrl,
  patchIngredient,
  slugify,
  TEMPLATE_CATEGORIES,
  useCapabilities,
  useCleanReference,
  useDescribe,
  useExpand,
  useIngredient,
  useLibraryMutations,
  useTemplates,
  type CharacterTemplate,
  type LibraryCharacter,
  type VoiceHints,
} from "@/api/characters";
import type { ApiError, Ingredient, Schemas } from "@/api/client";
import { thumbUrl, useGenerateIngredient, useUploadIngredientImage } from "@/api/imagegen";
import { usePlayground } from "@/api/queries";
import { sk } from "@/api/story";
import { ModelPicker } from "@/components/ModelPicker";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input, Label, NativeSelect, Textarea } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";
import { cn } from "@/lib/utils";
import { ViewCandidates } from "./ViewCandidates";
import { ViewsEditor } from "./ViewsEditor";

type Source = "template" | "prompt" | "upload" | "import";
const SOURCES: { id: Source; label: string }[] = [
  { id: "template", label: "Template" },
  { id: "prompt", label: "Describe" },
  { id: "upload", label: "Upload" },
  { id: "import", label: "Import" },
];
const STEPS = [
  { id: "look", label: "Look" },
  { id: "views", label: "Views" },
  { id: "identity", label: "Identity" },
  { id: "voice", label: "Voice" },
  { id: "save", label: "Save" },
] as const;
type Step = (typeof STEPS)[number]["id"];

interface Fields {
  name: string;
  visual_prompt: string;
  negative_prompt: string;
  character_info: string;
  body_outfit_prompt: string;
  voice_hints: VoiceHints;
  template_key: string | null;
}

const EMPTY: Fields = {
  name: "",
  visual_prompt: "",
  negative_prompt: "",
  character_info: "",
  body_outfit_prompt: "",
  voice_hints: { gender: "", age: "", tone: "" },
  template_key: null,
};

function fromTemplate(t: CharacterTemplate): Fields {
  return {
    name: t.name,
    visual_prompt: t.visual_prompt,
    negative_prompt: t.negative_prompt,
    character_info: t.character_info,
    body_outfit_prompt: t.body_outfit_prompt,
    voice_hints: { gender: "", age: "", tone: "", ...t.voice_hints },
    template_key: t.key,
  };
}

function updateBody(f: Fields): Schemas["IngredientUpdate"] {
  return {
    name: f.name.trim() || undefined,
    visual_prompt: f.visual_prompt,
    negative_prompt: f.negative_prompt,
    character_info: f.character_info || null,
    body_outfit_prompt: f.body_outfit_prompt || null,
    voice_hints: f.voice_hints,
    template_key: f.template_key,
  };
}

function TemplatePicker({ value, onPick }: { value: string | null; onPick: (t: CharacterTemplate) => void }) {
  const { data = [], isLoading } = useTemplates();
  const [category, setCategory] = useState<string>("all");
  const shown = category === "all" ? data : data.filter((t) => t.category === category);
  const used = new Set(data.map((t) => t.category));
  if (isLoading) return <Loader2Icon className="size-4 animate-spin" />;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {[{ value: "all", label: "All" }, ...TEMPLATE_CATEGORIES.filter((c) => used.has(c.value))].map(
          (c) => (
            <Button
              key={c.value}
              size="sm"
              variant={category === c.value ? "secondary" : "ghost"}
              className="h-7 text-xs"
              aria-pressed={category === c.value}
              onClick={() => setCategory(c.value)}
            >
              {c.label}
            </Button>
          ),
        )}
      </div>
      <div
        className="grid max-h-64 grid-cols-3 gap-2 overflow-y-auto sm:grid-cols-4"
        role="listbox"
        aria-label="Templates"
      >
        {shown.map((t) => (
          <button
            key={t.key}
            type="button"
            role="option"
            aria-selected={value === t.key}
            onClick={() => onPick(t)}
            title={t.visual_prompt}
            className={cn(
              "overflow-hidden rounded-lg border-2 text-left text-xs",
              value === t.key ? "border-primary" : "hover:border-muted-foreground/40 border-transparent",
            )}
          >
            <span className="bg-muted flex aspect-square items-center justify-center">
              {t.thumbnail_url ? (
                <img src={mediaUrl(t.thumbnail_url)} alt="" className="size-full object-cover" />
              ) : (
                <ImageIcon className="text-muted-foreground size-5" />
              )}
            </span>
            <span className="block truncate px-1.5 py-1 font-medium">{t.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function WizardBody({
  projectId,
  onClose,
  onSaved,
  onImport,
}: {
  projectId?: string;
  onClose: () => void;
  onSaved?: (r: { ingredient?: Ingredient; library?: LibraryCharacter }) => void;
  onImport?: () => void;
}) {
  const qc = useQueryClient();
  const libraryMode = !projectId;
  const playground = usePlayground().data?.project_id;
  const workPid = projectId ?? playground;
  const caps = useCapabilities(workPid).data;

  const [step, setStep] = useState<Step>("look");
  const [reached, setReached] = useState(0);
  const [source, setSource] = useState<Source>("template");
  const [fields, setFields] = useState<Fields>(EMPTY);
  const [description, setDescription] = useState("");
  const [draftId, setDraftId] = useState<string | null>(null);
  const [primaryJob, setPrimaryJob] = useState<string>();
  const [count, setCount] = useState(4);
  const [modelId, setModelId] = useState<string | null>(null);
  const [key, setKey] = useState("");
  const [keyTouched, setKeyTouched] = useState(false);
  const [alsoLibrary, setAlsoLibrary] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const savedRef = useRef(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const draft = useIngredient(workPid, draftId).data;
  const gen = useGenerateIngredient();
  const upload = useUploadIngredientImage(workPid ?? "");
  const expand = useExpand();
  const describe = useDescribe();
  const clean = useCleanReference();
  const library = useLibraryMutations(projectId);
  const set = (patch: Partial<Fields>) => setFields((f) => ({ ...f, ...patch }));
  const effectiveKey = keyTouched ? key : slugify(fields.name);
  const stepIndex = STEPS.findIndex((s) => s.id === step);
  const hasPrimary = !!draft?.primary_asset_id;

  const go = (s: Step) => {
    const i = STEPS.findIndex((x) => x.id === s);
    setReached((r) => Math.max(r, i));
    setError(null);
    setStep(s);
  };

  /** Creates the draft on first use, otherwise saves the wizard's fields onto it. */
  const ensureDraft = async (): Promise<string> => {
    if (!workPid) throw new Error("Still loading…");
    if (draftId) {
      const ing = await patchIngredient(draftId, updateBody(fields));
      qc.setQueryData([...sk.ingredients(workPid), "one", draftId], ing);
      return draftId;
    }
    const body = updateBody(fields);
    const d = await createDraft(workPid, {
      source: source === "import" ? "prompt" : source,
      template_key: fields.template_key,
      fields: { ...body, voice_hints: fields.voice_hints },
    });
    qc.setQueryData([...sk.ingredients(workPid), "one", d.id], d);
    setDraftId(d.id);
    return d.id;
  };

  const generate = async () => {
    setError(null);
    try {
      const id = await ensureDraft();
      const job = await gen.mutateAsync({ id, body: { count, model_entry_id: modelId } });
      setPrimaryJob(job.id);
    } catch (e) {
      if ((e as ApiError).code === "license_not_acknowledged") gateLicense(e, () => void generate());
      else setError((e as Error).message);
    }
  };

  const onUpload = async (file: File) => {
    setError(null);
    try {
      const id = await ensureDraft();
      await upload.mutateAsync({ id, file });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const runExpand = () =>
    expand.mutate(
      { projectId, body: { description, template_key: fields.template_key } },
      {
        onSuccess: (x) =>
          set({
            name: fields.name || x.name,
            visual_prompt: x.visual_prompt,
            body_outfit_prompt: x.body_outfit_prompt,
            negative_prompt: x.negative_prompt,
            character_info: x.character_info,
            voice_hints: { gender: "", age: "", tone: "", ...x.voice_hints },
          }),
      },
    );

  const runDescribe = () =>
    draft?.primary_asset_id &&
    describe.mutate(
      { asset_ids: [draft.primary_asset_id], name: fields.name || null },
      {
        onSuccess: (x) =>
          set({
            visual_prompt: x.visual_prompt,
            body_outfit_prompt: x.body_outfit_prompt || fields.body_outfit_prompt,
          }),
      },
    );

  const runClean = () =>
    draftId &&
    draft?.primary_asset_id &&
    clean.mutate(
      { id: draftId, assetId: draft.primary_asset_id },
      { onSuccess: (job) => setPrimaryJob(job.id) },
    );

  const save = async () => {
    if (!draftId) return;
    setBusy(true);
    setError(null);
    try {
      if (libraryMode) {
        await patchIngredient(draftId, updateBody(fields));
        const lib = await library.save.mutateAsync(draftId);
        savedRef.current = true;
        await deleteIngredient(draftId);
        onSaved?.({ library: lib });
      } else {
        const ing = await patchIngredient(draftId, {
          ...updateBody(fields),
          key: effectiveKey,
          draft: false,
        });
        savedRef.current = true;
        let lib: LibraryCharacter | undefined;
        if (alsoLibrary) lib = await library.save.mutateAsync(draftId);
        void qc.invalidateQueries({ queryKey: sk.ingredients(workPid!) });
        onSaved?.({ ingredient: ing, library: lib });
      }
      onClose();
    } catch (e) {
      const err = e as ApiError;
      setError(err.message);
      if (err.code === "key_taken" || err.code === "invalid_key") setStep("identity");
    } finally {
      setBusy(false);
    }
  };

  const cancel = () => {
    if (draftId && !savedRef.current) {
      if (hasPrimary && !window.confirm("Discard this character? Its generated images are deleted too."))
        return;
      void deleteIngredient(draftId).finally(() => qc.invalidateQueries({ queryKey: ["assets", workPid] }));
    }
    onClose();
  };

  const canNext =
    step === "look" ? hasPrimary : step === "identity" ? !!fields.name.trim() && !!effectiveKey : true;

  return (
    <Dialog open onOpenChange={(o) => !o && cancel()}>
      <DialogContent className="max-w-3xl" data-testid="character-wizard">
        <DialogHeader>
          <DialogTitle>{libraryMode ? "New Cast library character" : "New character"}</DialogTitle>
          <DialogDescription>
            Pick a look, add extra views for consistency, then name it. Use its @key in prompts.
          </DialogDescription>
        </DialogHeader>

        <ol className="flex flex-wrap gap-1 text-xs" aria-label="Steps">
          {STEPS.map((s, i) => (
            <li key={s.id}>
              <button
                type="button"
                disabled={i > reached || (i > 0 && !hasPrimary)}
                aria-current={s.id === step ? "step" : undefined}
                onClick={() => go(s.id)}
                className={cn(
                  "flex items-center gap-1 rounded-full border px-2.5 py-1 disabled:opacity-40",
                  s.id === step ? "border-primary text-primary" : "text-muted-foreground",
                )}
              >
                {i < stepIndex ? <CheckIcon className="size-3" /> : <span>{i + 1}</span>}
                {s.label}
              </button>
            </li>
          ))}
        </ol>

        {step === "look" && (
          <div className="space-y-4">
            <div
              className="bg-muted flex w-fit rounded-md p-0.5 text-sm"
              role="tablist"
              aria-label="Start from"
            >
              {SOURCES.filter((s) => s.id !== "import" || !libraryMode).map((s) => (
                <button
                  key={s.id}
                  type="button"
                  role="tab"
                  aria-selected={source === s.id}
                  onClick={() => setSource(s.id)}
                  className={cn("rounded px-3 py-1", source === s.id && "bg-background shadow-sm")}
                >
                  {s.label}
                </button>
              ))}
            </div>

            {source === "import" ? (
              <div className="space-y-2 rounded-lg border p-4 text-sm">
                <p>Bring in a character you already made: from the Cast library or another project.</p>
                <Button size="sm" onClick={() => onImport?.()}>
                  <ImportIcon />
                  Browse characters
                </Button>
              </div>
            ) : (
              <>
                {source === "template" && (
                  <TemplatePicker value={fields.template_key} onPick={(t) => setFields(fromTemplate(t))} />
                )}
                {source === "prompt" && (
                  <div className="space-y-1">
                    <Label htmlFor="cw-desc">Describe the character in your own words</Label>
                    <Textarea
                      id="cw-desc"
                      rows={2}
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="a gruff old lighthouse keeper who secretly loves birds"
                    />
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={!description.trim() || expand.isPending || caps?.expand === false}
                        title={caps?.expand === false ? (caps.expand_reason ?? undefined) : undefined}
                        onClick={runExpand}
                      >
                        {expand.isPending ? <Loader2Icon className="animate-spin" /> : <WandSparklesIcon />}
                        Expand with AI
                      </Button>
                      {caps?.expand === false && (
                        <span className="text-muted-foreground text-xs">{caps.expand_reason}</span>
                      )}
                    </div>
                    {expand.error && <p className="text-destructive text-xs">{expand.error.message}</p>}
                  </div>
                )}
                {source === "upload" && (
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={upload.isPending || !workPid}
                        onClick={() => fileRef.current?.click()}
                      >
                        {upload.isPending ? <Loader2Icon className="animate-spin" /> : <UploadIcon />}
                        Upload a reference image
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={!hasPrimary || describe.isPending || caps?.describe === false}
                        title={caps?.describe === false ? (caps.describe_reason ?? undefined) : undefined}
                        onClick={runDescribe}
                      >
                        {describe.isPending ? <Loader2Icon className="animate-spin" /> : <WandSparklesIcon />}
                        Describe with AI
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={!hasPrimary || clean.isPending || !caps?.edit_model}
                        title={
                          caps?.edit_model
                            ? "Re-render on a plain background, front-facing"
                            : "Needs an image-edit model (FLUX.1 Kontext)"
                        }
                        onClick={runClean}
                      >
                        {clean.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
                        Clean up reference
                      </Button>
                    </div>
                    <input
                      ref={fileRef}
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      className="hidden"
                      aria-label="Upload a reference image"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) void onUpload(file);
                        e.target.value = "";
                      }}
                    />
                    {(describe.error || clean.error) && (
                      <p className="text-destructive text-xs">{(describe.error ?? clean.error)?.message}</p>
                    )}
                  </div>
                )}

                <div className="space-y-1">
                  <Label htmlFor="cw-visual">Visual prompt (appearance only)</Label>
                  <Textarea
                    id="cw-visual"
                    rows={3}
                    value={fields.visual_prompt}
                    onChange={(e) => set({ visual_prompt: e.target.value })}
                    placeholder="a woman in her sixties with short silver hair, green eyes, navy wool peacoat"
                  />
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <ModelPicker capability="t2i" value={modelId} onChange={setModelId} />
                  <NativeSelect
                    aria-label="Portraits per run"
                    className="h-8 w-auto text-xs"
                    value={count}
                    onChange={(e) => setCount(Number(e.target.value))}
                  >
                    {[1, 2, 4, 6].map((n) => (
                      <option key={n} value={n}>
                        {n} per run
                      </option>
                    ))}
                  </NativeSelect>
                  <Button
                    size="sm"
                    disabled={!fields.visual_prompt.trim() || gen.isPending || !workPid}
                    onClick={() => void generate()}
                  >
                    {gen.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
                    {hasPrimary ? `${count} more portraits` : `Generate ${count} portraits`}
                  </Button>
                </div>
                {draft && (
                  <ViewCandidates
                    ing={draft}
                    projectId={workPid!}
                    view="primary"
                    jobId={primaryJob}
                    emptyText="Generate or upload a portrait, then pick the one to build on."
                  />
                )}
              </>
            )}
          </div>
        )}

        {step === "views" && draft && workPid && (
          <ViewsEditor ing={draft} projectId={workPid} beforeGenerate={ensureDraft} />
        )}

        {step === "identity" && (
          <div className="space-y-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <Label htmlFor="cw-name">Name</Label>
                <Input id="cw-name" value={fields.name} onChange={(e) => set({ name: e.target.value })} />
              </div>
              {!libraryMode && (
                <div className="space-y-1">
                  <Label htmlFor="cw-key">Key (used as @{effectiveKey || "key"} in prompts)</Label>
                  <Input
                    id="cw-key"
                    className="font-mono"
                    value={effectiveKey}
                    onChange={(e) => {
                      setKeyTouched(true);
                      setKey(slugify(e.target.value) || e.target.value.toLowerCase());
                    }}
                  />
                </div>
              )}
            </div>
            <p className="text-muted-foreground text-xs">
              {libraryMode
                ? `Its key comes from the name (@${slugify(fields.name) || "key"}); a number is added if it's taken.`
                : "The key can't change after saving, so prompts that mention it keep working."}
            </p>
            <div className="space-y-1">
              <Label htmlFor="cw-info">
                Character info (personality, quirks; only the story LLM sees this)
              </Label>
              <Textarea
                id="cw-info"
                rows={3}
                value={fields.character_info}
                onChange={(e) => set({ character_info: e.target.value })}
              />
            </div>
          </div>
        )}

        {step === "voice" && (
          <div className="space-y-3">
            {!caps?.voices && (
              <p className="text-muted-foreground rounded-md border p-3 text-xs">
                Choosing and previewing a voice arrives with text-to-speech (M6). Describe the voice now: the
                story LLM uses it for dialogue, and it pre-selects a voice later.
              </p>
            )}
            <div className="grid gap-3 sm:grid-cols-3">
              {(["gender", "age", "tone"] as const).map((k) => (
                <div key={k} className="space-y-1">
                  <Label htmlFor={`cw-voice-${k}`} className="capitalize">
                    {k}
                  </Label>
                  <Input
                    id={`cw-voice-${k}`}
                    value={fields.voice_hints[k] ?? ""}
                    placeholder={{ gender: "female", age: "60s", tone: "warm and low" }[k]}
                    onChange={(e) => set({ voice_hints: { ...fields.voice_hints, [k]: e.target.value } })}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {step === "save" && draft && workPid && (
          <div className="space-y-3">
            <div className="flex gap-3 rounded-lg border p-3">
              <div className="flex gap-1.5">
                {[draft.primary_asset_id, ...EXTRA_VIEWS.map((v) => draft.views?.[v.view])]
                  .filter((a): a is string => !!a)
                  .map((a) => (
                    <img
                      key={a}
                      src={thumbUrl(workPid, a)}
                      alt=""
                      className="bg-muted h-24 w-auto rounded-md object-cover"
                    />
                  ))}
              </div>
              <div className="min-w-0 space-y-1 text-sm">
                <p className="font-medium">{fields.name || "Unnamed"}</p>
                {!libraryMode && <p className="text-muted-foreground font-mono text-xs">@{effectiveKey}</p>}
                <p className="text-muted-foreground line-clamp-3 text-xs">{fields.visual_prompt}</p>
              </div>
            </div>
            {!libraryMode && (
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={alsoLibrary}
                  onChange={(e) => setAlsoLibrary(e.target.checked)}
                />
                Also save to the Cast library (reuse it in other projects)
              </label>
            )}
          </div>
        )}

        {error && <p className="text-destructive text-sm">{error}</p>}

        <DialogFooter className="items-center">
          <Button variant="ghost" className="mr-auto" onClick={cancel}>
            Cancel
          </Button>
          {stepIndex > 0 && (
            <Button variant="outline" onClick={() => go(STEPS[stepIndex - 1]!.id)}>
              Back
            </Button>
          )}
          {step === "views" && (
            <Button variant="ghost" onClick={() => go("identity")}>
              Skip
            </Button>
          )}
          {step !== "save" ? (
            source !== "import" && (
              <Button
                disabled={!canNext}
                onClick={() => {
                  if (step === "look") ensureDraft().catch((e: Error) => setError(e.message));
                  go(STEPS[stepIndex + 1]!.id);
                }}
              >
                Next
              </Button>
            )
          ) : (
            <Button disabled={busy || !fields.name.trim()} onClick={() => void save()}>
              {busy && <Loader2Icon className="animate-spin" />}
              {libraryMode ? "Save to Cast library" : "Save character"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** The character creator (Section 11.7.1). Without `projectId` it creates a Cast library character,
 * working on a draft in the Playground project. */
export function CharacterWizard({
  open,
  onOpenChange,
  projectId,
  onSaved,
  onImport,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId?: string;
  onSaved?: (r: { ingredient?: Ingredient; library?: LibraryCharacter }) => void;
  /** Called from the Import source; the parent swaps the wizard for the import dialog. */
  onImport?: () => void;
}) {
  if (!open) return null;
  return (
    <WizardBody
      projectId={projectId}
      onClose={() => onOpenChange(false)}
      onSaved={onSaved}
      onImport={() => {
        onOpenChange(false);
        onImport?.();
      }}
    />
  );
}
