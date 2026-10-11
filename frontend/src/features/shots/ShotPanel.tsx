import { Loader2Icon, PlusIcon, SparklesIcon, Trash2Icon, WandSparklesIcon, XIcon } from "lucide-react";
import { useState } from "react";
import type { Ingredient, Schemas, Shot } from "@/api/client";
import { rewriteShot, useBoardMutations } from "@/api/story";
import { DiffView } from "@/components/DiffView";
import { UsageLine } from "@/components/LlmPicker";
import { MentionTextarea } from "@/components/MentionTextarea";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge, Input, Label, NativeSelect, Textarea } from "@/components/ui/primitives";
import { ingredientOptions } from "@/features/characters/mentionOptions";
import { budgetWords, wordCount } from "@/features/timeline/narration";
import { cn } from "@/lib/utils";
import { AudioSection } from "./AudioSection";
import { CAMERA_MOVES, MODES, SHOT_TYPES } from "./constants";
import { KeyframeSection } from "./KeyframeSection";
import type { KeyframeRun, KeyframeWhich } from "./references";
import type { PromptField } from "./ReviewCheck";
import { VideoSection } from "./VideoSection";

type Draft = Pick<
  Shot,
  | "keyframe_prompt"
  | "video_prompt"
  | "negative_prompt"
  | "duration_s"
  | "shot_type"
  | "camera_move"
  | "generation_mode"
  | "ingredient_ids"
  | "dialogue"
  | "continuity_notes"
> & { end_keyframe_prompt: string; sfx_prompt: string; music_cue: string; narration: string };

function toDraft(s: Shot): Draft {
  return {
    keyframe_prompt: s.keyframe_prompt,
    video_prompt: s.video_prompt,
    negative_prompt: s.negative_prompt,
    end_keyframe_prompt: s.end_keyframe_prompt ?? "",
    duration_s: s.duration_s,
    shot_type: s.shot_type,
    camera_move: s.camera_move,
    generation_mode: s.generation_mode,
    ingredient_ids: s.ingredient_ids,
    dialogue: s.dialogue,
    narration: s.narration,
    sfx_prompt: s.sfx_prompt ?? "",
    music_cue: s.music_cue ?? "",
    continuity_notes: s.continuity_notes,
  };
}

function promptsText(p: { keyframe_prompt: string; video_prompt: string; negative_prompt?: string | null }) {
  return `Keyframe:\n${p.keyframe_prompt}\n\nVideo:\n${p.video_prompt}\n\nNegative:\n${p.negative_prompt ?? ""}`;
}

export function ShotPanel({
  shot,
  number,
  projectId,
  ingredients,
  locationId,
  maxClip,
  keyframeRuns,
  onKeyframeRun,
  onClose,
}: {
  shot: Shot;
  number: string;
  projectId: string;
  ingredients: Ingredient[];
  locationId: string | null | undefined;
  maxClip: number;
  keyframeRuns: Partial<Record<KeyframeWhich, KeyframeRun>>;
  onKeyframeRun: (run: KeyframeRun) => void;
  onClose: () => void;
}) {
  const m = useBoardMutations(projectId);
  const [draft, setDraft] = useState<Draft>(() => toDraft(shot));
  const [instruction, setInstruction] = useState("");
  const [rewriting, setRewriting] = useState(false);
  const [rewriteError, setRewriteError] = useState<string>();
  const [proposal, setProposal] = useState<Schemas["ShotRewriteResult"]>();
  const dirty = JSON.stringify(draft) !== JSON.stringify(toDraft(shot));
  const set = <K extends keyof Draft>(k: K, v: Draft[K]) => setDraft((d) => ({ ...d, [k]: v }));
  const characters = ingredients.filter((i) => i.kind === "character");
  const visible = ingredients.filter((i) => i.kind !== "style" && i.kind !== "location");
  const mentionOptions = ingredientOptions(ingredients);

  const saveBody = () => ({
    ...draft,
    end_keyframe_prompt: draft.generation_mode === "flf2v" ? draft.end_keyframe_prompt || null : null,
    sfx_prompt: draft.sfx_prompt || null,
    music_cue: draft.music_cue || null,
  });
  const save = () => m.updateShot.mutate({ id: shot.id, body: saveBody() });
  // A vision-review fix lands in the draft like an AI rewrite: the user sees it, then saves or regenerates.
  const applyFix = (field: PromptField, text: string) => set(field, text);

  const runRewrite = async () => {
    setRewriting(true);
    setRewriteError(undefined);
    try {
      setProposal(await rewriteShot(shot.id, instruction));
    } catch (e) {
      setRewriteError((e as Error).message);
    } finally {
      setRewriting(false);
    }
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent
        className="top-0 right-0 left-auto h-full max-h-full w-full max-w-xl translate-x-0 translate-y-0 content-start gap-4 overflow-y-auto rounded-none border-y-0 border-r-0"
        aria-describedby={undefined}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Shot {number}
            <Badge variant="outline">{shot.status.replace("_", " ")}</Badge>
          </DialogTitle>
          <DialogDescription className="sr-only">Edit the shot&apos;s prompts and settings</DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="space-y-1">
            <Label htmlFor="sh-dur">Duration (s)</Label>
            <Input
              id="sh-dur"
              type="number"
              min={0.5}
              step={0.5}
              value={draft.duration_s}
              onChange={(e) => set("duration_s", Number(e.target.value) || 0.5)}
              className={cn(draft.duration_s > maxClip && "border-warning")}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="sh-type">Shot type</Label>
            <NativeSelect
              id="sh-type"
              value={draft.shot_type}
              onChange={(e) => set("shot_type", e.target.value as Draft["shot_type"])}
            >
              {SHOT_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </NativeSelect>
          </div>
          <div className="space-y-1">
            <Label htmlFor="sh-cam">Camera</Label>
            <NativeSelect
              id="sh-cam"
              value={draft.camera_move}
              onChange={(e) => set("camera_move", e.target.value as Draft["camera_move"])}
            >
              {CAMERA_MOVES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </NativeSelect>
          </div>
          <div className="space-y-1">
            <Label htmlFor="sh-mode">Mode</Label>
            <NativeSelect
              id="sh-mode"
              value={draft.generation_mode}
              onChange={(e) => set("generation_mode", e.target.value as Draft["generation_mode"])}
            >
              {MODES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.value}
                </option>
              ))}
            </NativeSelect>
          </div>
        </div>
        {draft.duration_s > maxClip && (
          <p className="text-warning -mt-2 text-xs">
            Longer than the video model&apos;s {maxClip} s clips: the take is extended to fit (Video below),
            or split the shot.
          </p>
        )}

        <div className="space-y-1">
          <Label htmlFor="sh-kf">Keyframe prompt (still image; type @ to mention a character)</Label>
          <MentionTextarea
            id="sh-kf"
            rows={4}
            value={draft.keyframe_prompt}
            options={mentionOptions}
            onChange={(v) => set("keyframe_prompt", v)}
          />
        </div>
        {draft.generation_mode === "flf2v" && (
          <div className="space-y-1">
            <Label htmlFor="sh-ekf">End keyframe prompt</Label>
            <MentionTextarea
              id="sh-ekf"
              rows={3}
              value={draft.end_keyframe_prompt}
              options={mentionOptions}
              onChange={(v) => set("end_keyframe_prompt", v)}
            />
          </div>
        )}
        <div className="space-y-1">
          <Label htmlFor="sh-vp">Video prompt (motion)</Label>
          <MentionTextarea
            id="sh-vp"
            rows={3}
            value={draft.video_prompt}
            options={mentionOptions}
            onChange={(v) => set("video_prompt", v)}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="sh-neg">Negative prompt</Label>
          <Input
            id="sh-neg"
            value={draft.negative_prompt}
            onChange={(e) => set("negative_prompt", e.target.value)}
          />
        </div>

        {/* AI rewrite */}
        <div className="space-y-2 rounded-lg border p-3">
          <Label htmlFor="sh-rw" className="text-foreground flex items-center gap-1.5">
            <WandSparklesIcon className="size-3.5" /> AI rewrite
          </Label>
          <div className="flex gap-2">
            <Input
              id="sh-rw"
              placeholder="make it night · more dramatic · closer on her hands"
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && instruction.trim() && !rewriting && void runRewrite()}
            />
            <Button size="sm" disabled={!instruction.trim() || rewriting} onClick={() => void runRewrite()}>
              {rewriting ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
              Rewrite
            </Button>
          </div>
          {rewriteError && <p className="text-destructive text-xs">{rewriteError}</p>}
          {proposal && (
            <div className="space-y-2">
              <DiffView
                before={promptsText(draft)}
                after={promptsText(proposal.proposal)}
                className="max-h-60"
              />
              <UsageLine usage={proposal.usage} />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  onClick={() => {
                    const p = proposal.proposal;
                    setDraft((d) => ({
                      ...d,
                      keyframe_prompt: p.keyframe_prompt,
                      video_prompt: p.video_prompt,
                      negative_prompt: p.negative_prompt ?? d.negative_prompt,
                      end_keyframe_prompt: p.end_keyframe_prompt ?? d.end_keyframe_prompt,
                    }));
                    setProposal(undefined);
                  }}
                >
                  Apply to shot
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setProposal(undefined)}>
                  Discard
                </Button>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-1">
          <Label>In this shot</Label>
          <div className="flex flex-wrap gap-1.5">
            {visible.map((i) => {
              const on = draft.ingredient_ids.includes(i.id);
              return (
                <Button
                  key={i.id}
                  size="sm"
                  variant={on ? "default" : "outline"}
                  aria-pressed={on}
                  className="h-7 text-xs"
                  onClick={() =>
                    set(
                      "ingredient_ids",
                      on ? draft.ingredient_ids.filter((x) => x !== i.id) : [...draft.ingredient_ids, i.id],
                    )
                  }
                >
                  {i.name}
                </Button>
              );
            })}
            {visible.length === 0 && (
              <span className="text-muted-foreground text-xs">No characters or props yet.</span>
            )}
          </div>
        </div>

        <KeyframeSection
          shot={shot}
          projectId={projectId}
          ingredients={ingredients}
          ingredientIds={draft.ingredient_ids}
          locationId={locationId}
          runs={keyframeRuns}
          onRun={onKeyframeRun}
          beforeGenerate={
            dirty ? () => m.updateShot.mutateAsync({ id: shot.id, body: saveBody() }) : undefined
          }
          shotType={draft.shot_type}
          onSaveReferences={(ids) =>
            // One PATCH with the draft: the panel remounts when the shot changes, which would drop edits.
            m.updateShot.mutateAsync({ id: shot.id, body: { ...saveBody(), reference_asset_ids: ids } })
          }
          onApplyFix={applyFix}
        />

        <VideoSection
          shot={shot}
          projectId={projectId}
          beforeGenerate={
            dirty ? () => m.updateShot.mutateAsync({ id: shot.id, body: saveBody() }) : undefined
          }
          onApplyFix={applyFix}
        />

        <div className="space-y-1.5">
          <Label>Dialogue</Label>
          {draft.dialogue.map((d, idx) => (
            <div key={idx} className="flex gap-2">
              <NativeSelect
                aria-label="Speaker"
                className="w-36 shrink-0"
                value={d.character_ingredient_id ?? ""}
                onChange={(e) =>
                  set(
                    "dialogue",
                    draft.dialogue.map((x, j) =>
                      j === idx ? { ...x, character_ingredient_id: e.target.value || null } : x,
                    ),
                  )
                }
              >
                <option value="">Unknown</option>
                {characters.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </NativeSelect>
              <Input
                aria-label="Line"
                value={d.line}
                onChange={(e) =>
                  set(
                    "dialogue",
                    draft.dialogue.map((x, j) => (j === idx ? { ...x, line: e.target.value } : x)),
                  )
                }
              />
              <Input
                aria-label="Emotion"
                className="w-28 shrink-0"
                placeholder="emotion"
                value={d.emotion ?? ""}
                onChange={(e) =>
                  set(
                    "dialogue",
                    draft.dialogue.map((x, j) => (j === idx ? { ...x, emotion: e.target.value } : x)),
                  )
                }
              />
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label="Remove line"
                onClick={() =>
                  set(
                    "dialogue",
                    draft.dialogue.filter((_, j) => j !== idx),
                  )
                }
              >
                <XIcon />
              </Button>
            </div>
          ))}
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs"
            onClick={() =>
              set("dialogue", [
                ...draft.dialogue,
                { character_ingredient_id: characters[0]?.id ?? null, line: "", emotion: "" },
              ])
            }
          >
            <PlusIcon />
            Add line
          </Button>
        </div>

        <div className="space-y-1">
          <Label htmlFor="sh-narr">Narration (voice-over; leave empty for a silent shot)</Label>
          <Textarea
            id="sh-narr"
            rows={2}
            value={draft.narration}
            onChange={(e) => set("narration", e.target.value)}
          />
          <p className="text-muted-foreground text-xs">
            {wordCount(draft.narration)} / {budgetWords(draft.duration_s)} words for a {draft.duration_s} s
            shot. It is voiced from the Timeline tab.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1">
            <Label htmlFor="sh-sfx">Sound effects</Label>
            <Input id="sh-sfx" value={draft.sfx_prompt} onChange={(e) => set("sfx_prompt", e.target.value)} />
          </div>
          <div className="space-y-1">
            <Label htmlFor="sh-music">Music cue</Label>
            <Input id="sh-music" value={draft.music_cue} onChange={(e) => set("music_cue", e.target.value)} />
          </div>
        </div>
        <AudioSection
          shot={shot}
          projectId={projectId}
          beforeGenerate={
            dirty ? () => m.updateShot.mutateAsync({ id: shot.id, body: saveBody() }) : undefined
          }
        />

        <div className="space-y-1">
          <Label htmlFor="sh-cont">Continuity notes</Label>
          <Input
            id="sh-cont"
            value={draft.continuity_notes}
            onChange={(e) => set("continuity_notes", e.target.value)}
          />
        </div>

        <div className="flex items-center gap-2 border-t pt-3">
          <Button disabled={!dirty || m.updateShot.isPending} onClick={save}>
            {m.updateShot.isPending && <Loader2Icon className="animate-spin" />}
            Save shot
          </Button>
          {dirty && (
            <Button variant="ghost" onClick={() => setDraft(toDraft(shot))}>
              Discard
            </Button>
          )}
          <Button
            variant="ghost"
            className="text-muted-foreground ml-auto"
            onClick={() => {
              if (window.confirm(`Delete shot ${number}?`)) {
                m.deleteShot.mutate(shot.id);
                onClose();
              }
            }}
          >
            <Trash2Icon />
            Delete
          </Button>
        </div>
        {m.updateShot.error && <p className="text-destructive text-sm">{m.updateShot.error.message}</p>}
      </DialogContent>
    </Dialog>
  );
}
