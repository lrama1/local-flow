import { Loader2Icon, SparklesIcon } from "lucide-react";
import { useState } from "react";
import { EXTRA_VIEWS, useCapabilities, useGenerateViews, type ExtraView } from "@/api/characters";
import type { ApiError, Ingredient } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Label, NativeSelect, Textarea } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";
import { cn } from "@/lib/utils";
import { ViewCandidates } from "./ViewCandidates";

/** Extra views of a character (Section 11.7.1 step 2): outfit, which views, generate, and pick per view. */
export function ViewsEditor({
  ing,
  projectId,
  editable = false,
  beforeGenerate,
}: {
  ing: Ingredient;
  projectId: string;
  /** Per-view upload/clear (the character editor; the wizard keeps it simple). */
  editable?: boolean;
  beforeGenerate?: () => Promise<unknown>;
}) {
  const caps = useCapabilities(projectId).data;
  const gen = useGenerateViews();
  const existing = EXTRA_VIEWS.filter((v) => ing.views?.[v.view]).map((v) => v.view);
  const [views, setViews] = useState<ExtraView[]>(existing.length ? existing : ["full_body", "three_view"]);
  const [outfit, setOutfit] = useState(ing.body_outfit_prompt ?? "");
  const [count, setCount] = useState(2);
  const [jobs, setJobs] = useState<Partial<Record<ExtraView, string>>>({});
  const [note, setNote] = useState<string | null>(null);
  const hasPrimary = !!ing.primary_asset_id;

  const run = async () => {
    try {
      await beforeGenerate?.();
      const out = await gen.mutateAsync({
        id: ing.id,
        body: { views, body_outfit_prompt: outfit, count },
      });
      setJobs((j) => ({ ...j, ...Object.fromEntries(out.views.map((v, i) => [v, out.jobs[i]!.id])) }));
      setNote(out.note ?? null);
    } catch (e) {
      gateLicense(e, () => void run());
    }
  };
  const error =
    gen.error && (gen.error as ApiError).code !== "license_not_acknowledged" ? gen.error.message : null;
  const shown = EXTRA_VIEWS.filter((v) => views.includes(v.view) || ing.views?.[v.view] || jobs[v.view]);

  return (
    <div className="space-y-4" data-testid="views-editor">
      <p className={cn("text-xs", caps?.edit_model ? "text-muted-foreground" : "text-warning")}>
        {caps?.edit_model
          ? `Views are edits of the portrait with ${caps.edit_model_name ?? "the image-edit model"}, so the face and outfit stay consistent.`
          : "No image-edit model is installed, so views use text-to-image (lower consistency). Install FLUX.1 Kontext on the Models page for consistent views."}
      </p>
      <div className="space-y-1">
        <Label htmlFor={`outfit-${ing.id}`}>Body and outfit (what the portrait can&apos;t show)</Label>
        <Textarea
          id={`outfit-${ing.id}`}
          rows={2}
          value={outfit}
          onChange={(e) => setOutfit(e.target.value)}
          placeholder="navy wool peacoat with brass buttons, grey trousers, black rubber boots"
        />
      </div>
      <fieldset className="space-y-1.5">
        <legend className="text-muted-foreground mb-1 text-xs font-medium">Views</legend>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          {EXTRA_VIEWS.map((v) => (
            <label key={v.view} className="flex items-center gap-1.5 text-sm" title={v.hint}>
              <input
                type="checkbox"
                checked={views.includes(v.view)}
                onChange={(e) =>
                  setViews(e.target.checked ? [...views, v.view] : views.filter((x) => x !== v.view))
                }
              />
              {v.label}
            </label>
          ))}
        </div>
      </fieldset>
      <div className="flex flex-wrap items-center gap-2">
        <NativeSelect
          aria-label="Candidates per view"
          className="h-8 w-auto text-xs"
          value={count}
          onChange={(e) => setCount(Number(e.target.value))}
        >
          {[1, 2, 3, 4].map((n) => (
            <option key={n} value={n}>
              {n} per view
            </option>
          ))}
        </NativeSelect>
        <Button size="sm" disabled={!views.length || !hasPrimary || gen.isPending} onClick={() => void run()}>
          {gen.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
          Generate views
        </Button>
        {!hasPrimary && <span className="text-muted-foreground text-xs">Choose a portrait first.</span>}
      </div>
      {note && <p className="text-warning text-xs">{note}</p>}
      {error && <p className="text-destructive text-xs">{error}</p>}
      {shown.map((v) => (
        <div key={v.view} className="space-y-1">
          <h4 className="text-xs font-medium">
            {v.label} <span className="text-muted-foreground font-normal">· {v.hint}</span>
          </h4>
          <ViewCandidates
            ing={ing}
            projectId={projectId}
            view={v.view}
            jobId={jobs[v.view]}
            editable={editable}
            size="sm"
            emptyText="No images yet."
          />
        </div>
      ))}
    </div>
  );
}
