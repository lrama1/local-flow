import { ZapIcon } from "lucide-react";
import { Link } from "react-router";
import type { Schemas } from "@/api/client";
import { installedFor, useModels } from "@/api/models";
import { useUpdateProject } from "@/api/queries";
import { cn } from "@/lib/utils";

const KINDS = [
  { capability: "t2i", label: "images" },
  { capability: "image_edit", label: "keyframes" },
  { capability: "i2v", label: "video" },
];

/** Final/Draft switch: in Draft mode a model that isn't chosen by hand defaults to the fast Turbo models. */
export function DraftModeToggle({ project }: { project: Schemas["ProjectOut"] }) {
  const update = useUpdateProject();
  const { data: models } = useModels();
  const draft = project.draft_mode;
  const missing = models
    ? KINDS.filter((k) => !installedFor(models, k.capability).some((m) => m.tier === "draft")).map(
        (k) => k.label,
      )
    : [];

  return (
    <div className="flex flex-col items-end gap-1.5">
      <div
        role="group"
        aria-label="Generation quality"
        className="bg-muted inline-flex rounded-md p-0.5 text-xs"
      >
        {([false, true] as const).map((value) => (
          <button
            key={String(value)}
            type="button"
            aria-pressed={draft === value}
            disabled={update.isPending}
            title={
              value
                ? "Draft: fast, lower-quality takes for trying things out"
                : "Final: the full-quality models"
            }
            onClick={() => {
              if (value !== draft) update.mutate({ id: project.id, body: { draft_mode: value } });
            }}
            className={cn(
              "flex items-center gap-1 rounded px-2.5 py-1 font-medium transition-colors",
              draft === value ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {value && <ZapIcon className="size-3" />}
            {value ? "Draft" : "Final"}
          </button>
        ))}
      </div>
      {draft && (
        <p className="text-muted-foreground max-w-xs text-right text-xs">
          New images, keyframes and videos use the fast models unless you pick one.
          {missing.length > 0 && (
            <>
              {" "}
              Not installed yet: {missing.join(", ")}.{" "}
              <Link to="/models?tab=catalog" className="text-primary hover:underline">
                Models
              </Link>
            </>
          )}
        </p>
      )}
    </div>
  );
}
