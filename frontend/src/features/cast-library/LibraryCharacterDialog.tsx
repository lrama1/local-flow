import { Loader2Icon, Trash2Icon, UserRoundPlusIcon } from "lucide-react";
import { useState } from "react";
import {
  mediaUrl,
  useImportCharacter,
  useLibraryMutations,
  VIEW_LABELS,
  type LibraryCharacter,
} from "@/api/characters";
import { useProjects } from "@/api/queries";
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
import { VoicePicker } from "@/components/VoicePicker";
import { useUi } from "@/stores/ui";

function toForm(c: LibraryCharacter) {
  return {
    name: c.name,
    visual_prompt: c.visual_prompt,
    body_outfit_prompt: c.body_outfit_prompt,
    negative_prompt: c.negative_prompt,
    character_info: c.character_info,
    voice_hints: { gender: "", age: "", tone: "", ...c.voice_hints },
    voice_id: c.voice_id ?? null,
  };
}

/** Edit a Cast library character, cast it into a project, or delete it (Section 11.7.2). */
export function LibraryCharacterDialog({
  character: c,
  onClose,
}: {
  character: LibraryCharacter;
  onClose: () => void;
}) {
  const [form, setForm] = useState(() => toForm(c));
  const [target, setTarget] = useState("");
  const { update, remove } = useLibraryMutations();
  const cast = useImportCharacter();
  const { data: projects = [] } = useProjects();
  const toast = useUi((s) => s.pushToast);
  const dirty = JSON.stringify(form) !== JSON.stringify(toForm(c));
  const error = update.error ?? remove.error ?? cast.error;

  const castInto = () =>
    cast.mutate(
      { projectId: target, body: { library_character_id: c.id } },
      {
        onSuccess: (ing) => {
          const project = projects.find((p) => p.id === target)?.name ?? "the project";
          toast({ title: `${c.name} cast into ${project} as @${ing.key}`, variant: "success" });
        },
      },
    );

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{c.name}</DialogTitle>
          <DialogDescription>
            @{c.key}. Projects get a copy when you cast it; edits here show up there as an available update.
          </DialogDescription>
        </DialogHeader>
        <div className="flex gap-2 overflow-x-auto pb-1">
          {c.images.map((img) => (
            <a key={img.n} href={mediaUrl(img.url)} target="_blank" rel="noreferrer" className="shrink-0">
              <img
                src={mediaUrl(img.thumb_url)}
                alt={`${c.name} ${img.view}`}
                className="bg-muted h-28 rounded-md"
              />
              <span className="text-muted-foreground block text-center text-[11px]">
                {VIEW_LABELS[img.view] ?? img.view}
              </span>
            </a>
          ))}
        </div>
        <div className="grid gap-3">
          <div className="space-y-1">
            <Label htmlFor="lc-name">Name</Label>
            <Input
              id="lc-name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="lc-visual">Visual prompt</Label>
            <Textarea
              id="lc-visual"
              rows={3}
              value={form.visual_prompt}
              onChange={(e) => setForm({ ...form, visual_prompt: e.target.value })}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="lc-outfit">Body and outfit</Label>
            <Textarea
              id="lc-outfit"
              rows={2}
              value={form.body_outfit_prompt}
              onChange={(e) => setForm({ ...form, body_outfit_prompt: e.target.value })}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="lc-info">Character info</Label>
            <Textarea
              id="lc-info"
              rows={2}
              value={form.character_info}
              onChange={(e) => setForm({ ...form, character_info: e.target.value })}
            />
          </div>
          <div className="grid grid-cols-3 gap-2">
            {(["gender", "age", "tone"] as const).map((k) => (
              <div key={k} className="space-y-1">
                <Label htmlFor={`lc-voice-${k}`} className="capitalize">
                  Voice {k}
                </Label>
                <Input
                  id={`lc-voice-${k}`}
                  value={form.voice_hints[k]}
                  onChange={(e) =>
                    setForm({ ...form, voice_hints: { ...form.voice_hints, [k]: e.target.value } })
                  }
                />
              </div>
            ))}
          </div>
          <div className="space-y-1">
            <Label htmlFor="lc-voice">Voice</Label>
            <VoicePicker
              id="lc-voice"
              value={form.voice_id}
              hints={form.voice_hints}
              onChange={(v) => setForm({ ...form, voice_id: v })}
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 rounded-lg border p-2">
          <NativeSelect
            aria-label="Project to cast into"
            className="h-8 w-auto flex-1 text-sm"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">Cast into project…</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </NativeSelect>
          <Button size="sm" variant="outline" disabled={!target || cast.isPending} onClick={castInto}>
            {cast.isPending ? <Loader2Icon className="animate-spin" /> : <UserRoundPlusIcon />}
            Cast
          </Button>
        </div>
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        <DialogFooter className="items-center">
          <Button
            variant="ghost"
            className="text-muted-foreground mr-auto"
            disabled={remove.isPending}
            onClick={() =>
              window.confirm(`Delete ${c.name} from the Cast library? Projects keep their copies.`) &&
              remove.mutate(c.id, { onSuccess: onClose })
            }
          >
            <Trash2Icon />
            Delete
          </Button>
          <Button
            disabled={!dirty || !form.name.trim() || update.isPending}
            onClick={() => update.mutate({ id: c.id, body: form })}
          >
            Save changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
