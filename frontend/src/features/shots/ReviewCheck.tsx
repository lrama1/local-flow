import { AlertTriangleIcon, CheckCircle2Icon, Loader2Icon, ScanEyeIcon } from "lucide-react";
import { useReviewShot, useShotReview, useVisionCapability, type ShotReview } from "@/api/review";
import { DiffView } from "@/components/DiffView";
import { UsageLine } from "@/components/LlmPicker";
import { Button } from "@/components/ui/button";

export type PromptField = ShotReview["prompt_field"];

/** "Check consistency": a vision model compares the keyframe or take with the shot's reference images and
 * can propose a prompt fix (Section 8.3 step 5). The fix goes into the panel's unsaved draft. */
export function ReviewCheck({
  shotId,
  assetId,
  what,
  beforeReview,
  onApplyFix,
}: {
  shotId: string;
  assetId: string;
  what: "keyframe" | "take";
  beforeReview?: () => Promise<unknown>;
  onApplyFix?: (field: PromptField, text: string) => void;
}) {
  const { data: cap } = useVisionCapability();
  const review = useReviewShot(shotId);
  // Keyed by shot and asset: picking another take or keyframe shows that one's check (if any).
  const { result: shown, pending, error, clear } = useShotReview(shotId, assetId);

  const run = async () => {
    clear();
    await beforeReview?.(); // may remount this component (the panel is keyed on the shot's updated_at)
    await review.mutateAsync(assetId);
  };

  return (
    <div className="space-y-2" data-testid={`review-${what}`}>
      <Button
        size="sm"
        variant="outline"
        className="h-7 text-xs"
        disabled={!cap?.available || pending}
        title={
          cap?.available
            ? `Compare this ${what} with the shot's reference images (${cap.model})`
            : (cap?.reason ?? "Needs a vision-capable LLM (Settings)")
        }
        onClick={() => void run().catch(() => undefined)}
      >
        {pending ? <Loader2Icon className="animate-spin" /> : <ScanEyeIcon />}
        Check consistency
      </Button>
      {error && !pending && <p className="text-destructive text-xs">{error.message}</p>}
      {shown && (
        <div
          className="space-y-2 rounded-md border p-2.5 text-sm"
          role="region"
          aria-label="Consistency check"
        >
          {shown.consistent ? (
            <p className="text-success flex items-center gap-1.5 font-medium">
              <CheckCircle2Icon className="size-4" />
              Matches the references
            </p>
          ) : (
            <p className="text-warning flex items-center gap-1.5 font-medium">
              <AlertTriangleIcon className="size-4" />
              {shown.issues.length === 1 ? "1 issue" : `${shown.issues.length} issues`}
            </p>
          )}
          {shown.issues.length > 0 && (
            <ul className="text-muted-foreground list-disc space-y-0.5 pl-5 text-xs">
              {shown.issues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          )}
          <p className="text-muted-foreground text-xs">
            Compared with {shown.references.map((r) => r.label).join(", ")}
            {shown.frames_s.length > 0 && ` at ${shown.frames_s.map((t) => `${t}s`).join(" and ")}`}.
          </p>
          {shown.suggested_prompt_fix && (
            <>
              <DiffView
                before={shown.current_prompt}
                after={shown.suggested_prompt_fix}
                className="max-h-40"
              />
              <div className="flex gap-2">
                {onApplyFix && (
                  <Button
                    size="sm"
                    onClick={() => {
                      onApplyFix(shown.prompt_field, shown.suggested_prompt_fix);
                      clear();
                    }}
                  >
                    Apply fix
                  </Button>
                )}
                <Button size="sm" variant="ghost" onClick={clear}>
                  Dismiss
                </Button>
              </div>
            </>
          )}
          <UsageLine usage={shown.usage} />
        </div>
      )}
    </div>
  );
}
