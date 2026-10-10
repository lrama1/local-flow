import { ExpandIcon, Loader2Icon, StarIcon, UploadIcon, XIcon } from "lucide-react";
import { useRef, useState } from "react";
import { useSetView, VIEW_LABELS, type CharacterView } from "@/api/characters";
import type { Asset, Ingredient } from "@/api/client";
import { useCandidates, useUploadIngredientImage } from "@/api/imagegen";
import { AssetViewer } from "@/components/AssetGrid";
import { JobProgress } from "@/components/JobProgress";
import { AssetThumb } from "@/components/MediaPreview";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

function chosenFor(ing: Ingredient, view: CharacterView): string | null {
  return view === "primary" ? ing.primary_asset_id : (ing.views?.[view] ?? null);
}

/** Candidate images for one view of a character; clicking one chooses it for that view. */
export function ViewCandidates({
  ing,
  projectId,
  view,
  jobId,
  editable = false,
  size = "md",
  emptyText,
}: {
  ing: Ingredient;
  projectId: string;
  view: CharacterView;
  jobId?: string;
  /** Show Upload and Clear for this view. */
  editable?: boolean;
  size?: "sm" | "md";
  emptyText?: string;
}) {
  const { data } = useCandidates(projectId, { ingredient_id: ing.id, view });
  const setView = useSetView(projectId);
  const upload = useUploadIngredientImage(projectId);
  const fileRef = useRef<HTMLInputElement>(null);
  const [viewing, setViewing] = useState<Asset | null>(null);
  const items = data?.items ?? [];
  const chosen = chosenFor(ing, view);
  const label = VIEW_LABELS[view] ?? view;

  return (
    <div className="space-y-1.5" data-testid={`view-${view}`}>
      {items.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {items.slice(0, 16).map((a) => {
            const on = a.id === chosen;
            return (
              <div key={a.id} className="group relative">
                <button
                  type="button"
                  aria-label={on ? `Chosen ${label.toLowerCase()} image` : `Use for ${label.toLowerCase()}`}
                  aria-pressed={on}
                  onClick={() => !on && setView.mutate({ id: ing.id, view, assetId: a.id })}
                  className={cn(
                    "bg-muted block overflow-hidden rounded-md border-2",
                    size === "sm" ? "size-14" : "size-20",
                    on ? "border-primary" : "hover:border-muted-foreground/40 border-transparent",
                  )}
                >
                  <AssetThumb asset={a} />
                </button>
                {on && <StarIcon className="fill-primary text-primary absolute top-0.5 right-0.5 size-3" />}
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
      ) : (
        emptyText && <p className="text-muted-foreground text-xs">{emptyText}</p>
      )}
      {editable && (
        <div className="flex flex-wrap items-center gap-1.5">
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            disabled={upload.isPending}
            onClick={() => fileRef.current?.click()}
          >
            {upload.isPending ? <Loader2Icon className="animate-spin" /> : <UploadIcon />}
            Upload
          </Button>
          {chosen && view !== "primary" && (
            <Button
              size="sm"
              variant="ghost"
              className="text-muted-foreground h-7 text-xs"
              onClick={() => setView.mutate({ id: ing.id, view, assetId: null })}
            >
              <XIcon />
              Clear
            </Button>
          )}
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            aria-label={`Upload ${label.toLowerCase()} image`}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) upload.mutate({ id: ing.id, file, view });
              e.target.value = "";
            }}
          />
        </div>
      )}
      <JobProgress jobId={jobId} />
      {(setView.error || upload.error) && (
        <p className="text-destructive text-xs">{(setView.error ?? upload.error)?.message}</p>
      )}
      <AssetViewer asset={viewing} onClose={() => setViewing(null)} />
    </div>
  );
}
