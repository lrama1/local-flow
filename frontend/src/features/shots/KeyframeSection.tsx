import { ExpandIcon, ImageIcon, Loader2Icon, SlidersHorizontalIcon, SparklesIcon, XIcon } from "lucide-react";
import { useState } from "react";
import { useShotReferences, VIEW_LABELS, type ShotReferences } from "@/api/characters";
import type { ApiError, Asset, Ingredient, Shot } from "@/api/client";
import {
  thumbUrl,
  useCandidates,
  useGenerateKeyframe,
  useSelectForShot,
  type KeyframeBody,
} from "@/api/imagegen";
import { AssetViewer } from "@/components/AssetGrid";
import { JobProgress } from "@/components/JobProgress";
import { AssetThumb } from "@/components/MediaPreview";
import { ModelPicker } from "@/components/ModelPicker";
import { Button } from "@/components/ui/button";
import { Badge, Label, NativeSelect } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";
import { cn } from "@/lib/utils";
import { MAX_REFERENCES, referenceIngredients, type KeyframeRun, type KeyframeWhich } from "./references";
import { ReviewCheck, type PromptField } from "./ReviewCheck";

type Mode = KeyframeBody["mode"];

/** Per-shot reference override (Section 11.7.3): pick up to `max` images across the shot's ingredients. */
function ReferencePicker({
  refs,
  projectId,
  onSave,
  onClose,
}: {
  refs: ShotReferences;
  projectId: string;
  onSave: (ids: string[] | null) => Promise<unknown>;
  onClose: () => void;
}) {
  const [sel, setSel] = useState<string[]>(refs.selected.map((r) => r.asset_id));
  const [saving, setSaving] = useState(false);
  const toggle = (id: string) =>
    setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length < refs.max ? [...s, id] : s));
  const save = async (ids: string[] | null) => {
    setSaving(true);
    try {
      await onSave(ids);
      onClose();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-2 rounded-md border p-2" data-testid="reference-picker">
      <p className="text-muted-foreground text-xs">
        Pick up to {refs.max} images, in order. Automatic picks the view that suits the shot type.
      </p>
      {refs.options.map((o) => (
        <div key={o.ingredient_id} className="space-y-1">
          <span className="text-xs font-medium">{o.name}</span>
          <div className="flex flex-wrap gap-1.5">
            {o.images.map((img) => {
              const n = sel.indexOf(img.asset_id);
              const label = VIEW_LABELS[img.view] ?? img.view;
              return (
                <button
                  key={img.asset_id}
                  type="button"
                  aria-pressed={n >= 0}
                  aria-label={`${o.name} (${label.toLowerCase()})`}
                  onClick={() => toggle(img.asset_id)}
                  className={cn(
                    "relative overflow-hidden rounded-md border-2 text-[10px]",
                    n >= 0 ? "border-primary" : "hover:border-muted-foreground/40 border-transparent",
                  )}
                >
                  <img
                    src={thumbUrl(projectId, img.asset_id)}
                    alt=""
                    className="bg-muted size-14 object-cover"
                  />
                  <span className="absolute inset-x-0 bottom-0 truncate bg-black/60 px-0.5 text-white">
                    {label}
                  </span>
                  {n >= 0 && (
                    <span className="bg-primary text-primary-foreground absolute top-0.5 left-0.5 rounded px-1">
                      {n + 1}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      ))}
      <div className="flex gap-2">
        <Button size="sm" disabled={saving || !sel.length} onClick={() => void save(sel)}>
          Use these
        </Button>
        <Button size="sm" variant="outline" disabled={saving} onClick={() => void save(null)}>
          Automatic
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

const MODES: { value: Mode; label: string }[] = [
  { value: "auto", label: "Auto: compose from references" },
  { value: "edit", label: "Compose from references only" },
  { value: "t2i", label: "Text only" },
];

export function KeyframeSection({
  shot,
  projectId,
  ingredients,
  ingredientIds,
  locationId,
  runs,
  onRun,
  beforeGenerate,
  shotType,
  onSaveReferences,
  onApplyFix,
}: {
  shot: Shot;
  projectId: string;
  ingredients: Ingredient[];
  /** The panel's (possibly unsaved) ingredient selection. */
  ingredientIds: string[];
  locationId: string | null | undefined;
  runs: Partial<Record<KeyframeWhich, KeyframeRun>>;
  onRun: (run: KeyframeRun) => void;
  beforeGenerate?: () => Promise<unknown>;
  /** The panel's (possibly unsaved) shot type; it decides which character view is used. */
  shotType?: string;
  /** Saves the per-shot reference override (null = automatic), together with any unsaved panel edits. */
  onSaveReferences?: (ids: string[] | null) => Promise<unknown>;
  /** Puts a vision-review prompt fix into the panel's draft. */
  onApplyFix?: (field: PromptField, text: string) => void;
}) {
  const [which, setWhich] = useState<KeyframeWhich>("start");
  const [mode, setMode] = useState<Mode>("auto");
  const [count, setCount] = useState(2);
  const [modelId, setModelId] = useState<string | null>(null);
  const [viewing, setViewing] = useState<Asset | null>(null);
  const slot = which === "start" ? "keyframe" : "end_keyframe";
  const { data } = useCandidates(projectId, { shot_id: shot.id, slot });
  const gen = useGenerateKeyframe();
  const select = useSelectForShot(projectId);
  const flf = shot.generation_mode === "flf2v";
  const selectedId =
    which === "start" ? shot.selected_keyframe_asset_id : shot.selected_end_keyframe_asset_id;
  const candidates = data?.items ?? [];
  const { data: refData } = useShotReferences(shot.id, shotType ?? shot.shot_type, ingredientIds);
  const [picking, setPicking] = useState(false);
  const selectedRefs = refData?.selected ?? [];
  const withImages = new Set(refData?.options.map((o) => o.ingredient_id));
  // Ingredients in the shot that have no image yet can't be references; they're shown greyed out.
  const missing = referenceIngredients(ingredientIds, locationId, ingredients).filter(
    (i) => refData && !withImages.has(i.id),
  );
  const unused = (refData?.options.length ?? 0) > selectedRefs.length && !refData?.overridden;
  const run = runs[which];

  const compose = async () => {
    try {
      await beforeGenerate?.();
      const out = await gen.mutateAsync({
        shotId: shot.id,
        body: { which, mode, count, model_entry_id: modelId },
      });
      onRun({ which, jobId: out.job.id, out });
    } catch (e) {
      gateLicense(e, () => void compose());
    }
  };
  const genError =
    gen.error && (gen.error as ApiError).code !== "license_not_acknowledged" ? gen.error.message : null;

  return (
    <section className="space-y-3 rounded-lg border p-3" data-testid="keyframe-section">
      <div className="flex items-center gap-2">
        <Label>Keyframe</Label>
        {flf && (
          <div className="bg-muted ml-auto flex rounded-md p-0.5 text-xs">
            {(["start", "end"] as const).map((w) => (
              <button
                key={w}
                type="button"
                aria-pressed={which === w}
                onClick={() => setWhich(w)}
                className={cn("rounded px-2 py-0.5", which === w && "bg-background shadow-sm")}
              >
                {w === "start" ? "First frame" : "Last frame"}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="bg-muted relative flex aspect-video items-center justify-center overflow-hidden rounded-md">
        {selectedId ? (
          <>
            <img
              src={thumbUrl(projectId, selectedId)}
              alt="Selected keyframe"
              className="size-full object-cover"
            />
            <Button
              size="icon-sm"
              variant="secondary"
              aria-label="Clear selected keyframe"
              title="Clear the selection"
              className="absolute top-1.5 right-1.5"
              onClick={() => select.mutate({ shotId: shot.id, slot, assetId: null })}
            >
              <XIcon />
            </Button>
          </>
        ) : (
          <span className="text-muted-foreground flex items-center gap-1.5 text-xs">
            <ImageIcon className="size-4" />
            No keyframe selected
          </span>
        )}
      </div>
      {selectedId && selectedRefs.length > 0 && (
        <ReviewCheck
          shotId={shot.id}
          assetId={selectedId}
          what="keyframe"
          beforeReview={beforeGenerate}
          onApplyFix={onApplyFix}
        />
      )}

      <div className="space-y-1">
        <div className="flex items-center gap-1.5">
          <span className="text-muted-foreground text-xs">
            References
            {unused ? ` (the best ${refData?.max ?? MAX_REFERENCES} for this shot type)` : ""}
          </span>
          {refData?.overridden && <Badge variant="secondary">Custom</Badge>}
          {onSaveReferences && (refData?.options.length ?? 0) > 0 && (
            <Button
              size="sm"
              variant="ghost"
              className="ml-auto h-6 px-1.5 text-xs"
              aria-expanded={picking}
              onClick={() => setPicking((p) => !p)}
            >
              <SlidersHorizontalIcon />
              Choose
            </Button>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5" data-testid="keyframe-references">
          {selectedRefs.map((r) => (
            <span
              key={r.asset_id}
              title={`${r.name}: ${VIEW_LABELS[r.view] ?? r.view}`}
              className="flex items-center gap-1.5 rounded-full border py-0.5 pr-2 pl-0.5 text-xs"
            >
              <img
                src={thumbUrl(projectId, r.asset_id)}
                alt=""
                className="size-5 rounded-full object-cover"
              />
              {r.label}
            </span>
          ))}
          {missing.map((i) => (
            <span
              key={i.id}
              title={`${i.name}: no primary image yet (Ingredients tab)`}
              className="text-muted-foreground flex items-center gap-1.5 rounded-full border border-dashed py-0.5 pr-2 pl-0.5 text-xs"
            >
              <span className="bg-muted size-5 rounded-full" />
              {i.name}
            </span>
          ))}
          {refData && selectedRefs.length === 0 && missing.length === 0 && (
            <span className="text-muted-foreground text-xs">
              None: add characters, props or a scene location (or @mention them) to compose from their images.
            </span>
          )}
        </div>
        {picking && refData && onSaveReferences && (
          <ReferencePicker
            refs={refData}
            projectId={projectId}
            onSave={onSaveReferences}
            onClose={() => setPicking(false)}
          />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <NativeSelect
          aria-label="Keyframe mode"
          className="h-8 w-auto text-xs"
          value={mode}
          onChange={(e) => {
            setMode(e.target.value as Mode);
            setModelId(null);
          }}
        >
          {MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </NativeSelect>
        <ModelPicker
          capability={mode === "t2i" ? "t2i" : "image_edit"}
          value={modelId}
          onChange={setModelId}
        />
        <NativeSelect
          aria-label="Keyframe candidates"
          className="h-8 w-auto text-xs"
          value={count}
          onChange={(e) => setCount(Number(e.target.value))}
        >
          {[1, 2, 3, 4].map((n) => (
            <option key={n} value={n}>
              {n} {n === 1 ? "image" : "images"}
            </option>
          ))}
        </NativeSelect>
        <Button size="sm" disabled={gen.isPending} onClick={() => void compose()}>
          {gen.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
          {mode === "t2i" || selectedRefs.length === 0 ? "Generate keyframe" : "Compose keyframe"}
        </Button>
      </div>

      {run?.out.note && <p className="text-warning text-xs">{run.out.note}</p>}
      {genError && <p className="text-destructive text-xs">{genError}</p>}
      <JobProgress jobId={run?.jobId} />

      {candidates.length > 0 && (
        <div className="grid grid-cols-3 gap-2" data-testid="keyframe-candidates">
          {candidates.map((a) => {
            const on = a.id === selectedId;
            return (
              <div key={a.id} className="group relative">
                <button
                  type="button"
                  aria-label={on ? "Selected keyframe" : "Use as keyframe"}
                  aria-pressed={on}
                  onClick={() => !on && select.mutate({ shotId: shot.id, slot, assetId: a.id })}
                  className={cn(
                    "bg-muted block aspect-video w-full overflow-hidden rounded-md border-2",
                    on ? "border-primary" : "hover:border-muted-foreground/40 border-transparent",
                  )}
                >
                  <AssetThumb asset={a} />
                </button>
                <button
                  type="button"
                  aria-label="View image"
                  onClick={() => setViewing(a)}
                  className="absolute bottom-1 left-1 hidden rounded bg-black/60 p-0.5 text-white group-hover:block"
                >
                  <ExpandIcon className="size-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}
      <AssetViewer asset={viewing} onClose={() => setViewing(null)} />
    </section>
  );
}
