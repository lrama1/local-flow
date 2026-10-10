import { Loader2Icon, SparklesIcon } from "lucide-react";
import { useState } from "react";
import type { ApiError, Shot } from "@/api/client";
import { useGenerateAll, type CameraIntensity, type GenerateAllOut } from "@/api/video";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { NativeSelect } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";
import { plural } from "@/lib/utils";

/** Section 11.6: keyframes for shots without one, then videos for every shot without a take. */
export function GenerateAllDialog({
  projectId,
  shots,
  numbers,
  onClose,
}: {
  projectId: string;
  shots: Shot[];
  /** Shot id → its board number ("1.2"), for the skipped list. */
  numbers: Map<string, string>;
  onClose: () => void;
}) {
  const [keyframes, setKeyframes] = useState(true);
  const [videos, setVideos] = useState(true);
  const [autoApprove, setAutoApprove] = useState(false);
  const [extend, setExtend] = useState(true);
  const [intensity, setIntensity] = useState<CameraIntensity>("normal");
  const [result, setResult] = useState<GenerateAllOut>();
  const run = useGenerateAll(projectId);

  const withoutTake = shots.filter((s) => !s.selected_video_asset_id);
  const withoutKeyframe = withoutTake.filter(
    (s) => s.generation_mode !== "t2v" && !s.selected_keyframe_asset_id,
  );
  const longShots = withoutTake.filter((s) => s.needs_extend).length;

  const submit = async () => {
    try {
      setResult(
        await run.mutateAsync({
          keyframes,
          videos,
          auto_approve_keyframes: autoApprove,
          extend,
          camera_intensity: intensity,
        }),
      );
    } catch (e) {
      gateLicense(e, () => void submit());
    }
  };
  const error = run.error && (run.error as ApiError).code !== "license_not_acknowledged" ? run.error : null;

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Generate all</DialogTitle>
          <DialogDescription>
            {plural(withoutTake.length, "shot")} without a take
            {withoutKeyframe.length > 0 && `, ${withoutKeyframe.length} of them without a keyframe`}. Shots
            that already have a take or are generating are skipped.
          </DialogDescription>
        </DialogHeader>

        {result ? (
          <div className="space-y-2 text-sm" data-testid="generate-all-result">
            <p>
              Queued {plural(result.keyframe_jobs, "keyframe job")} and{" "}
              {plural(result.video_jobs, "video job")}.
            </p>
            {result.skipped.length > 0 && (
              <ul className="text-muted-foreground max-h-48 space-y-0.5 overflow-y-auto text-xs">
                {result.skipped.map((s) => (
                  <li key={s.shot_id}>
                    Shot {numbers.get(s.shot_id) ?? "?"}: {s.reason}
                  </li>
                ))}
              </ul>
            )}
            <div className="flex justify-end">
              <Button onClick={onClose}>Done</Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3 text-sm">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={keyframes} onChange={(e) => setKeyframes(e.target.checked)} />
              Compose missing keyframes
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={videos} onChange={(e) => setVideos(e.target.checked)} />
              Generate videos
            </label>
            <label className="flex items-start gap-2">
              <input
                type="checkbox"
                className="mt-1"
                checked={autoApprove}
                disabled={!keyframes || !videos}
                onChange={(e) => setAutoApprove(e.target.checked)}
              />
              <span>
                Use each new keyframe right away
                <span className="text-muted-foreground block text-xs">
                  Otherwise those shots get 2 keyframe options; pick one, then run this again for the video.
                </span>
              </span>
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={extend} onChange={(e) => setExtend(e.target.checked)} />
              Extend long shots{longShots > 0 && ` (${longShots})`}
            </label>
            <label className="flex items-center gap-2">
              Camera moves
              <NativeSelect
                aria-label="Camera intensity"
                className="h-8 w-auto text-xs"
                value={intensity}
                onChange={(e) => setIntensity(e.target.value as CameraIntensity)}
              >
                {(["subtle", "normal", "strong"] as const).map((i) => (
                  <option key={i} value={i}>
                    {i}
                  </option>
                ))}
              </NativeSelect>
            </label>
            {error && <p className="text-destructive text-xs">{error.message}</p>}
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={onClose}>
                Cancel
              </Button>
              <Button
                disabled={run.isPending || (!keyframes && !videos) || withoutTake.length === 0}
                onClick={() => void submit()}
              >
                {run.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
                Queue
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
