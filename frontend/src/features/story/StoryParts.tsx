import { HistoryIcon, PencilIcon, RotateCcwIcon } from "lucide-react";
import { useState } from "react";
import type { Ingredient, RefinedStory, StoryBible } from "@/api/client";
import { useRevision, useRevisions } from "@/api/story";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  Badge,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Label,
  Textarea,
} from "@/components/ui/primitives";
import { relativeTime } from "@/lib/utils";
import { formatBible, formatRefined } from "./format";

// --- refined story editor ------------------------------------------------------------------------
export function RefinedEditor({
  refined,
  saving,
  onSave,
}: {
  refined: RefinedStory;
  saving: boolean;
  onSave: (r: RefinedStory) => void;
}) {
  const [draft, setDraft] = useState<RefinedStory>(refined);
  const dirty = JSON.stringify(draft) !== JSON.stringify(refined);
  const set = <K extends keyof RefinedStory>(k: K, v: RefinedStory[K]) => setDraft((d) => ({ ...d, [k]: v }));

  return (
    <div className="space-y-3" data-testid="refined-editor">
      <div className="grid gap-3 sm:grid-cols-[1fr_140px]">
        <div className="space-y-1">
          <Label htmlFor="rf-title">Title</Label>
          <Input id="rf-title" value={draft.title} onChange={(e) => set("title", e.target.value)} />
        </div>
        <div className="space-y-1">
          <Label htmlFor="rf-runtime">Runtime (s)</Label>
          <Input
            id="rf-runtime"
            type="number"
            min={1}
            value={draft.estimated_runtime_s}
            onChange={(e) => set("estimated_runtime_s", Number(e.target.value) || 1)}
          />
        </div>
      </div>
      <div className="space-y-1">
        <Label htmlFor="rf-logline">Logline</Label>
        <Input id="rf-logline" value={draft.logline} onChange={(e) => set("logline", e.target.value)} />
      </div>
      <div className="space-y-1">
        <Label htmlFor="rf-synopsis">Synopsis</Label>
        <Textarea
          id="rf-synopsis"
          rows={3}
          value={draft.synopsis}
          onChange={(e) => set("synopsis", e.target.value)}
        />
      </div>
      <div className="space-y-1">
        <Label htmlFor="rf-story">Story (beat by beat)</Label>
        <Textarea
          id="rf-story"
          rows={10}
          value={draft.story}
          onChange={(e) => set("story", e.target.value)}
        />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1">
          <Label htmlFor="rf-tone">Tone</Label>
          <Input id="rf-tone" value={draft.tone ?? ""} onChange={(e) => set("tone", e.target.value)} />
        </div>
        <div className="space-y-1">
          <Label htmlFor="rf-style">Visual style</Label>
          <Input
            id="rf-style"
            value={draft.visual_style ?? ""}
            onChange={(e) => set("visual_style", e.target.value)}
          />
        </div>
      </div>
      {dirty && (
        <div className="flex gap-2">
          <Button size="sm" disabled={saving} onClick={() => onSave(draft)}>
            Save edits
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setDraft(refined)}>
            Discard
          </Button>
        </div>
      )}
    </div>
  );
}

// --- bible ---------------------------------------------------------------------------------------
export function BibleView({
  bible,
  ingredients,
  saving,
  onSave,
}: {
  bible: StoryBible;
  ingredients: Ingredient[];
  saving: boolean;
  onSave: (b: StoryBible) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [json, setJson] = useState("");
  const [error, setError] = useState<string>();
  const locked = new Set(ingredients.filter((i) => i.locked).map((i) => i.key));
  const style = bible.style ?? {};

  if (editing) {
    return (
      <div className="space-y-2">
        <Textarea
          aria-label="Bible JSON"
          className="font-mono text-xs"
          rows={18}
          value={json}
          onChange={(e) => setJson(e.target.value)}
        />
        {error && <p className="text-destructive text-xs">{error}</p>}
        <div className="flex gap-2">
          <Button
            size="sm"
            disabled={saving}
            onClick={() => {
              try {
                onSave(JSON.parse(json) as StoryBible);
                setEditing(false);
              } catch (e) {
                setError(`Invalid JSON: ${(e as Error).message}`);
              }
            }}
          >
            Save bible
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  const group = (title: string, items: { key: string; name: string; visual_prompt: string }[] | undefined) =>
    items && items.length > 0 ? (
      <div className="space-y-1.5">
        <h4 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">{title}</h4>
        {items.map((it) => (
          <div key={it.key} className="rounded-md border p-2">
            <div className="flex items-center gap-2 text-sm font-medium">
              {it.name}
              <Badge variant="outline">{it.key}</Badge>
              {locked.has(it.key) && <Badge variant="warning">locked</Badge>}
            </div>
            <p className="text-muted-foreground text-xs">{it.visual_prompt}</p>
          </div>
        ))}
      </div>
    ) : null;

  return (
    <div className="space-y-3" data-testid="bible-view">
      <div className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
        {Object.entries(style)
          .filter(([, v]) => v)
          .map(([k, v]) => (
            <div key={k}>
              <span className="text-muted-foreground capitalize">{k}:</span> {String(v)}
            </div>
          ))}
      </div>
      {group("Characters", bible.characters)}
      {group("Locations", bible.locations)}
      {group("Props", bible.props)}
      <Button
        size="sm"
        variant="outline"
        onClick={() => {
          setJson(JSON.stringify(bible, null, 2));
          setError(undefined);
          setEditing(true);
        }}
      >
        <PencilIcon />
        Edit bible JSON
      </Button>
    </div>
  );
}

// --- revisions -----------------------------------------------------------------------------------
const STEP_LABEL: Record<string, string> = {
  refine: "Refined",
  bible: "Bible",
  breakdown: "Breakdown",
  edit: "Manual edit",
  restore: "Restored",
};

export function RevisionsPanel({
  projectId,
  current,
  onRestore,
}: {
  projectId: string;
  current: number | null | undefined;
  onRestore: (revision: number) => void;
}) {
  const { data: revisions } = useRevisions(projectId);
  const [open, setOpen] = useState<number>();
  const { data: rev } = useRevision(projectId, open);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HistoryIcon className="size-4" /> Revisions
        </CardTitle>
      </CardHeader>
      <CardContent className="max-h-80 space-y-1 overflow-y-auto">
        {!revisions?.length && <p className="text-muted-foreground text-xs">No revisions yet.</p>}
        {revisions?.map((r) => (
          <button
            key={r.revision}
            onClick={() => setOpen(r.revision)}
            className="hover:bg-accent flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs"
          >
            <span className="text-muted-foreground font-mono">r{r.revision}</span>
            <span className="truncate">{STEP_LABEL[r.step] ?? r.step}</span>
            {r.revision === current && <Badge>current</Badge>}
            <span className="text-muted-foreground ml-auto shrink-0">{relativeTime(r.created_at)}</span>
          </button>
        ))}
      </CardContent>
      <Dialog open={open !== undefined} onOpenChange={(o) => !o && setOpen(undefined)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Revision {open}</DialogTitle>
            <DialogDescription>
              {rev
                ? `${STEP_LABEL[rev.step] ?? rev.step}${rev.llm_model ? ` · ${rev.llm_model}` : ""}`
                : "Loading…"}
            </DialogDescription>
          </DialogHeader>
          {rev && (
            <pre className="bg-muted max-h-[55vh] overflow-auto rounded-md p-3 text-xs whitespace-pre-wrap">
              {rev.raw_idea && `Idea:\n${rev.raw_idea}\n\n`}
              {formatRefined(rev.refined)}
              {rev.bible && `\n\n${formatBible(rev.bible)}`}
            </pre>
          )}
          {open !== undefined && open !== current && (
            <Button
              onClick={() => {
                onRestore(open);
                setOpen(undefined);
              }}
            >
              <RotateCcwIcon />
              Restore this revision
            </Button>
          )}
        </DialogContent>
      </Dialog>
    </Card>
  );
}
