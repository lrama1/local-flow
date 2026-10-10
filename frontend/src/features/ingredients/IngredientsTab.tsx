import {
  BookmarkPlusIcon,
  ExpandIcon,
  ImageIcon,
  ImportIcon,
  LayersIcon,
  Loader2Icon,
  LockIcon,
  PlusIcon,
  SparklesIcon,
  StarIcon,
  Trash2Icon,
  UnlockIcon,
  UploadIcon,
  UserPlusIcon,
} from "lucide-react";
import { useRef, useState, type FormEvent } from "react";
import { EXTRA_VIEWS, useLibraryMutations, useTemplateMutations } from "@/api/characters";
import type { ApiError, Asset, Ingredient, IngredientKind } from "@/api/client";
import {
  thumbUrl,
  useCandidates,
  useGenerateIngredient,
  useSetPrimary,
  useUploadIngredientImage,
} from "@/api/imagegen";
import { useIngredientMutations, useIngredients } from "@/api/story";
import { AssetViewer } from "@/components/AssetGrid";
import { EmptyState } from "@/components/EmptyState";
import { JobProgress } from "@/components/JobProgress";
import { AssetThumb } from "@/components/MediaPreview";
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
import { MentionTextarea } from "@/components/MentionTextarea";
import { Badge, Card, Input, Label, NativeSelect, Skeleton, Textarea } from "@/components/ui/primitives";
import { VoicePicker } from "@/components/VoicePicker";
import { CharacterWizard } from "@/features/characters/CharacterWizard";
import { ImportCharacterDialog } from "@/features/characters/ImportCharacterDialog";
import { useProjectMentionOptions } from "@/features/characters/mentionOptions";
import { ViewsEditor } from "@/features/characters/ViewsEditor";
import { gateLicense } from "@/features/models/gateLicense";
import { VoiceDesigner } from "@/features/voices/VoiceDesigner";
import { cn } from "@/lib/utils";
import { useUi } from "@/stores/ui";

const KINDS: { kind: IngredientKind; title: string }[] = [
  { kind: "character", title: "Characters" },
  { kind: "location", title: "Locations" },
  { kind: "prop", title: "Props" },
  { kind: "style", title: "Style" },
];

/** Tab-level generation settings and per-ingredient job ids (cards remount when an ingredient changes). */
interface GenContext {
  modelId: string | null;
  count: number;
  jobs: Record<string, string>;
  onJob: (ingredientId: string, jobId: string) => void;
}

function IngredientImages({
  ing,
  projectId,
  gen: ctx,
  beforeGenerate,
}: {
  ing: Ingredient;
  projectId: string;
  gen: GenContext;
  beforeGenerate?: () => Promise<unknown>;
}) {
  const { data } = useCandidates(projectId, { ingredient_id: ing.id, view: "primary" });
  const gen = useGenerateIngredient();
  const setPrimary = useSetPrimary(projectId);
  const upload = useUploadIngredientImage(projectId);
  const [viewing, setViewing] = useState<Asset | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const items = data?.items ?? [];
  const jobId = ctx.jobs[ing.id];

  const run = async () => {
    try {
      await beforeGenerate?.();
      const job = await gen.mutateAsync({
        id: ing.id,
        body: { count: ctx.count, model_entry_id: ctx.modelId },
      });
      ctx.onJob(ing.id, job.id);
    } catch (e) {
      gateLicense(e, () => void run());
    }
  };
  const genError =
    gen.error && (gen.error as ApiError).code !== "license_not_acknowledged" ? gen.error.message : null;

  return (
    <div className="space-y-2" data-testid={`images-${ing.key}`}>
      {items.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {items.slice(0, 12).map((a) => {
            const primary = a.id === ing.primary_asset_id;
            return (
              <div key={a.id} className="group relative">
                <button
                  type="button"
                  aria-label={primary ? "Primary image" : "Use as primary image"}
                  title={primary ? "Primary image" : "Click to use as primary image"}
                  onClick={() => !primary && setPrimary.mutate({ id: ing.id, assetId: a.id })}
                  className={cn(
                    "bg-muted block size-14 overflow-hidden rounded-md border-2",
                    primary ? "border-primary" : "hover:border-muted-foreground/40 border-transparent",
                  )}
                >
                  <AssetThumb asset={a} />
                </button>
                {primary && (
                  <StarIcon className="fill-primary text-primary absolute top-0.5 right-0.5 size-3" />
                )}
                <button
                  type="button"
                  aria-label="View image"
                  onClick={() => setViewing(a)}
                  className="absolute bottom-0.5 left-0.5 hidden rounded bg-black/60 p-0.5 text-white group-hover:block"
                >
                  <ExpandIcon className="size-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" disabled={gen.isPending} onClick={() => void run()}>
          {gen.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
          {items.length ? `${ctx.count} more` : `Generate ${ctx.count}`}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          disabled={upload.isPending}
          onClick={() => fileRef.current?.click()}
        >
          {upload.isPending ? <Loader2Icon className="animate-spin" /> : <UploadIcon />}
          Upload
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="hidden"
          aria-label={`Upload image for ${ing.name}`}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) upload.mutate({ id: ing.id, file });
            e.target.value = "";
          }}
        />
      </div>
      <JobProgress jobId={jobId} />
      {(genError || upload.error) && (
        <p className="text-destructive text-xs">{genError ?? upload.error?.message}</p>
      )}
      <AssetViewer asset={viewing} onClose={() => setViewing(null)} />
    </div>
  );
}

const LIBRARY_BADGE = {
  synced: { label: "In Cast library", variant: "success" },
  update_available: { label: "Library has updates", variant: "warning" },
  missing: { label: "Removed from library", variant: "outline" },
} as const;

/** Views strip, Cast library link and template actions for a character card (Section 11.7). */
function CharacterExtras({ ing, projectId }: { ing: Ingredient; projectId: string }) {
  const [viewsOpen, setViewsOpen] = useState(false);
  const [designing, setDesigning] = useState(false);
  const { update } = useIngredientMutations(projectId);
  const lib = useLibraryMutations(projectId);
  const templates = useTemplateMutations();
  const toast = useUi((s) => s.pushToast);
  const views = EXTRA_VIEWS.filter((v) => ing.views?.[v.view]);
  const status = ing.library_character_id ? ing.library_status : null;
  const badge = status ? LIBRARY_BADGE[status] : null;
  const error = lib.save.error ?? lib.push.error ?? lib.pull.error ?? templates.create.error;
  const done = (title: string) => () => toast({ title, variant: "success" });

  return (
    <div className="space-y-2" data-testid={`character-extras-${ing.key}`}>
      <div className="flex flex-wrap items-center gap-1.5">
        {views.map((v) => (
          <img
            key={v.view}
            src={thumbUrl(projectId, ing.views![v.view])}
            alt={`${ing.name} ${v.label.toLowerCase()}`}
            title={v.label}
            className="bg-muted h-12 w-auto rounded object-cover"
          />
        ))}
        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => setViewsOpen(true)}>
          <LayersIcon />
          {views.length ? "Views" : "Add views"}
        </Button>
      </div>
      <div className="space-y-1" data-testid={`character-voice-${ing.key}`}>
        <Label htmlFor={`voice-${ing.id}`}>Voice</Label>
        <VoicePicker
          id={`voice-${ing.id}`}
          value={ing.voice_id}
          hints={ing.voice_hints}
          onChange={(v) => update.mutate({ id: ing.id, body: { voice_id: v } })}
          onDesign={() => setDesigning(true)}
        />
        {update.error && <p className="text-destructive text-xs">{update.error.message}</p>}
        {designing && (
          <VoiceDesigner
            open
            onOpenChange={setDesigning}
            initialName={ing.name}
            initialDescription={Object.values(ing.voice_hints ?? {})
              .filter(Boolean)
              .join(", ")}
            onSaved={(v) => update.mutate({ id: ing.id, body: { voice_id: v.id } })}
          />
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {badge && <Badge variant={badge.variant}>{badge.label}</Badge>}
        {status === "update_available" && (
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            disabled={lib.pull.isPending}
            onClick={() =>
              window.confirm(`Replace ${ing.name} here with the Cast library version?`) &&
              lib.pull.mutate(ing.id, { onSuccess: done(`${ing.name} updated from the library`) })
            }
          >
            Update from library
          </Button>
        )}
        {(status === "synced" || status === "update_available") && (
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            disabled={lib.push.isPending}
            onClick={() =>
              window.confirm(`Overwrite ${ing.name} in the Cast library with this version?`) &&
              lib.push.mutate(ing.id, { onSuccess: done(`${ing.name} pushed to the library`) })
            }
          >
            Push to library
          </Button>
        )}
        {(!status || status === "missing") && (
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            disabled={lib.save.isPending || !ing.primary_asset_id}
            title={ing.primary_asset_id ? undefined : "Choose a primary image first"}
            onClick={() =>
              lib.save.mutate(ing.id, { onSuccess: done(`${ing.name} saved to the Cast library`) })
            }
          >
            <BookmarkPlusIcon />
            Save to library
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          className="text-muted-foreground h-7 text-xs"
          disabled={templates.create.isPending}
          onClick={() =>
            templates.create.mutate(
              { from_ingredient_id: ing.id, name: ing.name, category: "custom" },
              { onSuccess: done(`Saved "${ing.name}" as a template`) },
            )
          }
        >
          Save as template
        </Button>
      </div>
      {error && <p className="text-destructive text-xs">{error.message}</p>}
      <Dialog open={viewsOpen} onOpenChange={setViewsOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{ing.name}: views</DialogTitle>
            <DialogDescription>
              Extra angles keep the character consistent: wide shots use the full-body view and close-ups the
              face close-up.
            </DialogDescription>
          </DialogHeader>
          <ViewsEditor ing={ing} projectId={projectId} editable />
        </DialogContent>
      </Dialog>
    </div>
  );
}

function IngredientCard({ ing, projectId, gen }: { ing: Ingredient; projectId: string; gen: GenContext }) {
  const { update, remove } = useIngredientMutations(projectId);
  const mentionOptions = useProjectMentionOptions(projectId, ing.id);
  const [draft, setDraft] = useState({
    name: ing.name,
    visual_prompt: ing.visual_prompt,
    description: ing.description,
    negative_prompt: ing.negative_prompt,
    character_info: ing.character_info ?? "",
  });
  const dirty =
    draft.name !== ing.name ||
    draft.visual_prompt !== ing.visual_prompt ||
    draft.description !== ing.description ||
    draft.negative_prompt !== ing.negative_prompt ||
    draft.character_info !== (ing.character_info ?? "");

  const saveBody = () => ({
    ...draft,
    character_info: ing.kind === "character" ? draft.character_info || null : undefined,
  });
  const save = () => update.mutate({ id: ing.id, body: saveBody() });

  return (
    <Card className="flex flex-col gap-3 p-4" data-testid={`ingredient-${ing.key}`}>
      <div className="flex items-start gap-3">
        <div className="bg-muted flex size-20 shrink-0 items-center justify-center overflow-hidden rounded-lg">
          {ing.primary_asset_id ? (
            <img
              src={thumbUrl(projectId, ing.primary_asset_id)}
              alt={`${ing.name} reference`}
              className="size-full object-cover"
            />
          ) : (
            <ImageIcon className="text-muted-foreground size-5" />
          )}
        </div>
        <div className="min-w-0 flex-1 space-y-1">
          <Input
            aria-label="Name"
            className="h-8 font-medium"
            value={draft.name}
            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
          />
          <div className="flex flex-wrap gap-1">
            <Badge variant="outline">{ing.key}</Badge>
            <Badge variant="secondary">{ing.origin === "bible" ? "from bible" : ing.origin}</Badge>
          </div>
        </div>
        <Button
          size="icon-sm"
          variant={ing.locked ? "secondary" : "ghost"}
          aria-label={ing.locked ? "Unlock" : "Lock"}
          title={
            ing.locked ? "Locked: bible rebuilds won't change it" : "Lock to protect from bible rebuilds"
          }
          onClick={() => update.mutate({ id: ing.id, body: { locked: !ing.locked } })}
        >
          {ing.locked ? <LockIcon className="text-warning" /> : <UnlockIcon />}
        </Button>
      </div>
      <div className="space-y-1">
        <Label>Visual prompt</Label>
        <MentionTextarea
          aria-label="Visual prompt"
          rows={3}
          value={draft.visual_prompt}
          options={ing.kind === "style" ? undefined : mentionOptions}
          onChange={(v) => setDraft({ ...draft, visual_prompt: v })}
        />
      </div>
      {ing.kind === "character" && (
        <div className="space-y-1">
          <Label>Character info (personality, quirks; only the story LLM sees this)</Label>
          <Textarea
            aria-label="Character info"
            rows={2}
            value={draft.character_info}
            onChange={(e) => setDraft({ ...draft, character_info: e.target.value })}
          />
        </div>
      )}
      {ing.kind === "style" && (
        <div className="space-y-1">
          <Label>Negative prompt</Label>
          <Input
            aria-label="Negative prompt"
            value={draft.negative_prompt}
            onChange={(e) => setDraft({ ...draft, negative_prompt: e.target.value })}
          />
        </div>
      )}
      <IngredientImages
        ing={ing}
        projectId={projectId}
        gen={gen}
        beforeGenerate={dirty ? () => update.mutateAsync({ id: ing.id, body: saveBody() }) : undefined}
      />
      {ing.kind === "character" && <CharacterExtras ing={ing} projectId={projectId} />}
      <div className="mt-auto flex items-center gap-2">
        {dirty && (
          <Button size="sm" disabled={update.isPending} onClick={save}>
            Save
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          className="text-muted-foreground ml-auto"
          onClick={() => window.confirm(`Delete "${ing.name}"?`) && remove.mutate(ing.id)}
        >
          <Trash2Icon />
          Delete
        </Button>
      </div>
    </Card>
  );
}

function AddIngredientDialog({
  projectId,
  open,
  onOpenChange,
}: {
  projectId: string;
  open: boolean;
  onOpenChange: (o: boolean) => void;
}) {
  const { create } = useIngredientMutations(projectId);
  const [kind, setKind] = useState<IngredientKind>("character");
  const [name, setName] = useState("");
  const [visual, setVisual] = useState("");
  const [info, setInfo] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    create.mutate(
      {
        kind,
        name: name.trim(),
        visual_prompt: visual,
        character_info: kind === "character" ? info || null : null,
      },
      {
        onSuccess: () => {
          onOpenChange(false);
          setName("");
          setVisual("");
          setInfo("");
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <form className="space-y-3" onSubmit={submit}>
          <DialogHeader>
            <DialogTitle>Add ingredient</DialogTitle>
            <DialogDescription>
              Ingredients you add are locked by default, so story bible rebuilds keep them as they are.
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-[140px_1fr] gap-3">
            <div className="space-y-1">
              <Label htmlFor="ing-kind">Kind</Label>
              <NativeSelect
                id="ing-kind"
                value={kind}
                onChange={(e) => setKind(e.target.value as IngredientKind)}
              >
                {KINDS.filter((k) => k.kind !== "style").map((k) => (
                  <option key={k.kind} value={k.kind}>
                    {k.title.replace(/s$/, "")}
                  </option>
                ))}
              </NativeSelect>
            </div>
            <div className="space-y-1">
              <Label htmlFor="ing-name">Name</Label>
              <Input id="ing-name" required value={name} onChange={(e) => setName(e.target.value)} />
            </div>
          </div>
          <div className="space-y-1">
            <Label htmlFor="ing-visual">Visual prompt</Label>
            <Textarea
              id="ing-visual"
              rows={3}
              value={visual}
              onChange={(e) => setVisual(e.target.value)}
              placeholder="Appearance only: age, build, hair, wardrobe and colors…"
            />
          </div>
          {kind === "character" && (
            <div className="space-y-1">
              <Label htmlFor="ing-info">Character info</Label>
              <Textarea id="ing-info" rows={2} value={info} onChange={(e) => setInfo(e.target.value)} />
            </div>
          )}
          {create.error && <p className="text-destructive text-sm">{create.error.message}</p>}
          <DialogFooter>
            <Button type="submit" disabled={!name.trim() || create.isPending}>
              Add
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function IngredientsTab({ projectId }: { projectId: string }) {
  const { data, isLoading } = useIngredients(projectId);
  const [open, setOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [modelId, setModelId] = useState<string | null>(null);
  const [count, setCount] = useState(4);
  const [jobs, setJobs] = useState<Record<string, string>>({});
  if (isLoading) return <Skeleton className="h-40" />;
  const items = data ?? [];
  const gen: GenContext = {
    modelId,
    count,
    jobs,
    onJob: (ingredientId, jobId) => setJobs((j) => ({ ...j, [ingredientId]: jobId })),
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-muted-foreground max-w-xl text-sm">
          Built from the story bible, or created here. Generate reference images and star the one shots should
          use; keyframes are composed from it. Characters can add extra views (wide shots use the full body).
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <ModelPicker capability="t2i" value={modelId} onChange={setModelId} />
          <NativeSelect
            aria-label="Images per run"
            className="h-8 w-auto text-xs"
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
          >
            {[1, 2, 4, 6, 8].map((n) => (
              <option key={n} value={n}>
                {n} per run
              </option>
            ))}
          </NativeSelect>
          <Button size="sm" onClick={() => setWizardOpen(true)}>
            <UserPlusIcon />
            New character
          </Button>
          <Button size="sm" variant="outline" onClick={() => setImportOpen(true)}>
            <ImportIcon />
            Import
          </Button>
          <Button size="sm" variant="outline" onClick={() => setOpen(true)}>
            <PlusIcon />
            Add ingredient
          </Button>
        </div>
      </div>
      {items.length === 0 && (
        <EmptyState icon={SparklesIcon} title="No ingredients yet">
          Build the story bible on the Story tab, create a character, or add locations and props yourself.
        </EmptyState>
      )}
      {KINDS.map(({ kind, title }) => {
        const group = items.filter((i) => i.kind === kind);
        if (!group.length) return null;
        return (
          <section key={kind} className="space-y-2">
            <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
              {title} · {group.length}
            </h3>
            <div className={cn("grid gap-3", kind === "style" ? "" : "md:grid-cols-2 xl:grid-cols-3")}>
              {group.map((ing) => (
                <IngredientCard
                  key={`${ing.id}:${ing.updated_at}`}
                  ing={ing}
                  projectId={projectId}
                  gen={gen}
                />
              ))}
            </div>
          </section>
        );
      })}
      <AddIngredientDialog projectId={projectId} open={open} onOpenChange={setOpen} />
      <CharacterWizard
        projectId={projectId}
        open={wizardOpen}
        onOpenChange={setWizardOpen}
        onImport={() => setImportOpen(true)}
      />
      <ImportCharacterDialog projectId={projectId} open={importOpen} onOpenChange={setImportOpen} />
    </div>
  );
}
