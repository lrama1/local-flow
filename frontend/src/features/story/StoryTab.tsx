import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangleIcon, KeyRoundIcon, LayoutGridIcon, UsersRoundIcon } from "lucide-react";
import { useState, type Dispatch, type SetStateAction } from "react";
import { Link } from "react-router";
import type { LlmChoice, LlmProvider, Story, StoryOptions, StoryStepResult, UsageOut } from "@/api/client";
import { ApiError } from "@/api/client";
import {
  sk,
  storySteps,
  useBoard,
  useIngredients,
  useLlmProviders,
  useStory,
  useStoryInvalidate,
} from "@/api/story";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DiffView } from "@/components/DiffView";
import { EmptyState } from "@/components/EmptyState";
import { LlmPicker, UsageLine } from "@/components/LlmPicker";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, Input, Label, Skeleton, Textarea } from "@/components/ui/primitives";
import { ImportCharacterDialog } from "@/features/characters/ImportCharacterDialog";
import { plural } from "@/lib/utils";
import { formatBible, formatRefined } from "./format";
import { RunButton, StepCard } from "./StoryCards";
import { BibleView, RefinedEditor, RevisionsPanel } from "./StoryParts";
import { useStepRunner, type StepRunner } from "./useStepRunner";

export function StoryTab({ projectId, onOpenShots }: { projectId: string; onOpenShots: () => void }) {
  const { data: providers, isLoading } = useLlmProviders();
  if (isLoading) return <Skeleton className="h-60" />;
  const usable = (providers ?? []).filter((p) => p.configured && p.ok !== false);
  if (!usable.length) {
    const broken = (providers ?? []).filter((p) => p.configured && p.ok === false);
    return (
      <EmptyState
        icon={KeyRoundIcon}
        title="Add an LLM API key to write stories"
        action={
          <Button asChild variant="outline">
            <Link to="/settings">Open Settings</Link>
          </Button>
        }
      >
        Set <code>OPENAI_API_KEY</code>, <code>ANTHROPIC_API_KEY</code>, or <code>OPENROUTER_API_KEY</code> in
        your environment, or paste a key in Settings.
        {broken.map((p) => (
          <span key={p.provider} className="text-destructive mt-2 block">
            {p.label}: {p.error ?? "key rejected"}
          </span>
        ))}
      </EmptyState>
    );
  }
  return <StoryWorkspace projectId={projectId} onOpenShots={onOpenShots} />;
}

type Proposal = {
  step: "refine" | "bible";
  before: string;
  after: string;
  usage: UsageOut | null | undefined;
  apply: () => Promise<StoryStepResult>;
};

function toChoice(usage: UsageOut | null | undefined): LlmChoice | undefined {
  return usage ? { provider: usage.provider as LlmProvider, model: usage.model } : undefined;
}

interface SessionState {
  runner: StepRunner;
  results: Record<string, StoryStepResult>;
  setResults: Dispatch<SetStateAction<Record<string, StoryStepResult>>>;
  llm: Record<string, LlmChoice | null>;
  setLlm: (v: Record<string, LlmChoice | null>) => void;
}

function StoryWorkspace({ projectId, onOpenShots }: { projectId: string; onOpenShots: () => void }) {
  const { data: story } = useStory(projectId);
  // Session state survives the editor re-mounting on each new revision.
  const runner = useStepRunner();
  const [results, setResults] = useState<Record<string, StoryStepResult>>({});
  const [llm, setLlm] = useState<Record<string, LlmChoice | null>>({});
  if (!story) return <Skeleton className="h-60" />;
  // Re-mount the editors whenever a new revision lands so drafts start from the saved content.
  return (
    <StoryEditor
      key={story.revision ?? 0}
      story={story}
      projectId={projectId}
      onOpenShots={onOpenShots}
      session={{ runner, results, setResults, llm, setLlm }}
    />
  );
}

function StoryEditor({
  story,
  projectId,
  onOpenShots,
  session,
}: {
  story: Story;
  projectId: string;
  onOpenShots: () => void;
  session: SessionState;
}) {
  const qc = useQueryClient();
  const invalidate = useStoryInvalidate(projectId);
  const { data: board } = useBoard(projectId);
  const { data: ingredients = [] } = useIngredients(projectId);
  const { runner, results, setResults, llm, setLlm } = session;

  const [rawIdea, setRawIdea] = useState(story.raw_idea);
  const [options, setOptions] = useState<StoryOptions>(story.options ?? {});
  const [castKeys, setCastKeys] = useState<string[]>(story.cast_keys ?? []);
  const [importOpen, setImportOpen] = useState(false);
  const [target, setTarget] = useState<number | undefined>(
    story.options?.target_length_seconds ?? story.refined?.estimated_runtime_s ?? undefined,
  );
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [confirmReplace, setConfirmReplace] = useState(false);
  const [saving, setSaving] = useState(false);

  const characters = ingredients.filter((i) => i.kind === "character");
  const usageFor = (step: string) =>
    results[step]?.usage ?? (story.step === step ? (story.usage as UsageOut | null) : null);
  const saved = (step: string, res: StoryStepResult) => {
    qc.setQueryData(sk.story(projectId), res.story);
    setResults((r) => ({ ...r, [step]: res }));
    invalidate();
  };
  const patch = async (body: Parameters<typeof storySteps.patch>[1]) => {
    setSaving(true);
    try {
      saved("edit", await storySteps.patch(projectId, body));
    } finally {
      setSaving(false);
    }
  };

  const runRefine = () =>
    runner.run("refine", async (signal) => {
      const ideaFields = { raw_idea: rawIdea, options, cast_keys: castKeys };
      const dry = story.edited_fields.includes("refined") && !!story.refined;
      const res = await storySteps.refine(
        projectId,
        { ...ideaFields, llm: llm.refine ?? undefined, dry_run: dry },
        { signal },
      );
      if (res.saved) return saved("refine", res);
      setProposal({
        step: "refine",
        before: formatRefined(story.refined),
        after: formatRefined(res.story.refined),
        usage: res.usage,
        apply: () =>
          storySteps.patch(projectId, {
            ...ideaFields,
            refined: res.story.refined!,
            applied_from: "refine",
            llm: toChoice(res.usage),
          }),
      });
    });

  const runBible = () =>
    runner.run("bible", async (signal) => {
      const dry = story.edited_fields.includes("bible") && !!story.bible;
      const res = await storySteps.bible(
        projectId,
        { llm: llm.bible ?? undefined, dry_run: dry },
        { signal },
      );
      if (res.saved) return saved("bible", res);
      setProposal({
        step: "bible",
        before: formatBible(story.bible),
        after: formatBible(res.story.bible),
        usage: res.usage,
        apply: () =>
          storySteps.patch(projectId, {
            bible: res.story.bible!,
            applied_from: "bible",
            llm: toChoice(res.usage),
          }),
      });
    });

  const runBreakdown = (replace: boolean) =>
    runner.run("breakdown", async (signal) => {
      try {
        const res = await storySteps.breakdown(
          projectId,
          { target_length_seconds: target, replace_existing: replace, llm: llm.breakdown ?? undefined },
          { signal },
        );
        saved("breakdown", res);
      } catch (e) {
        if (e instanceof ApiError && e.code === "shots_exist") {
          setConfirmReplace(true);
          return;
        }
        throw e;
      }
    });

  const warnings = story.warnings ?? [];

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_280px]">
      <div className="min-w-0 space-y-4">
        {/* 1. Idea */}
        <StepCard
          n={1}
          title="Idea"
          subtitle="A few sentences are enough. LocalFlow turns it into a filmable story."
        >
          <Textarea
            aria-label="Story idea"
            rows={4}
            value={rawIdea}
            onChange={(e) => setRawIdea(e.target.value)}
            placeholder="A lonely lighthouse keeper finds a mechanical seagull washed up on the rocks…"
          />
          <div className="grid gap-3 sm:grid-cols-4">
            {(
              [
                ["genre", "Genre", "fantasy, comedy…"],
                ["tone", "Tone", "wistful, tense…"],
                ["audience", "Audience", "all ages"],
              ] as const
            ).map(([k, label, ph]) => (
              <div key={k} className="space-y-1">
                <Label htmlFor={`opt-${k}`}>{label}</Label>
                <Input
                  id={`opt-${k}`}
                  placeholder={ph}
                  value={options[k] ?? ""}
                  onChange={(e) => setOptions({ ...options, [k]: e.target.value || null })}
                />
              </div>
            ))}
            <div className="space-y-1">
              <Label htmlFor="opt-length">Target length (s)</Label>
              <Input
                id="opt-length"
                type="number"
                min={5}
                max={3600}
                placeholder="auto"
                value={options.target_length_seconds ?? ""}
                onChange={(e) =>
                  setOptions({
                    ...options,
                    target_length_seconds: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </div>
          </div>
          <div className="space-y-1">
            <Label htmlFor="opt-style">Style notes</Label>
            <Input
              id="opt-style"
              placeholder="hand-painted watercolor, muted colors…"
              value={options.style_notes ?? ""}
              onChange={(e) => setOptions({ ...options, style_notes: e.target.value || null })}
            />
          </div>
          <div className="space-y-1">
            <Label>Cast (must appear)</Label>
            <div className="flex flex-wrap gap-1.5">
              {characters.map((c) => {
                const on = castKeys.includes(c.key);
                return (
                  <Button
                    key={c.key}
                    size="sm"
                    variant={on ? "default" : "outline"}
                    aria-pressed={on}
                    onClick={() =>
                      setCastKeys(on ? castKeys.filter((k) => k !== c.key) : [...castKeys, c.key])
                    }
                  >
                    {c.name}
                  </Button>
                );
              })}
              <Button size="sm" variant="ghost" onClick={() => setImportOpen(true)}>
                <UsersRoundIcon />
                Add from Cast library
              </Button>
            </div>
            <ImportCharacterDialog
              projectId={projectId}
              libraryOnly
              open={importOpen}
              onOpenChange={setImportOpen}
              onImported={(ing) => setCastKeys((keys) => [...keys, ing.key])}
            />
          </div>
          <LlmPicker value={llm.refine ?? null} onChange={(v) => setLlm({ ...llm, refine: v })} />
          <RunButton
            runner={runner}
            step="refine"
            label={story.refined ? "Re-run refine" : "Refine"}
            disabled={!rawIdea.trim()}
            onClick={runRefine}
          />
          <UsageLine usage={usageFor("refine")} />
        </StepCard>

        {/* 2. Refined story */}
        <StepCard
          n={2}
          title="Refined story"
          disabled={!story.refined}
          subtitle={
            story.refined ? (
              <>
                Edit freely; edits are kept as a new revision.
                {story.edited_fields.includes("refined") && (
                  <Badge variant="warning" className="ml-2">
                    edited
                  </Badge>
                )}
              </>
            ) : (
              "Appears after you refine the idea."
            )
          }
        >
          {story.refined && (
            <RefinedEditor
              refined={story.refined}
              saving={saving}
              onSave={(refined) => void patch({ refined })}
            />
          )}
        </StepCard>

        {/* 3. Bible */}
        <StepCard
          n={3}
          title="Story bible"
          disabled={!story.refined}
          subtitle="Style, characters, locations, and props. They become ingredients (locked ones are never overwritten)."
        >
          {story.refined && (
            <>
              {story.bible && (
                <BibleView
                  bible={story.bible}
                  ingredients={ingredients}
                  saving={saving}
                  onSave={(bible) => void patch({ bible })}
                />
              )}
              <LlmPicker value={llm.bible ?? null} onChange={(v) => setLlm({ ...llm, bible: v })} />
              <RunButton
                runner={runner}
                step="bible"
                label={story.bible ? "Rebuild bible" : "Build bible"}
                onClick={runBible}
              />
              {results.bible?.ingredients && (
                <p className="text-muted-foreground text-xs">
                  Ingredients: {results.bible.ingredients.created} created,{" "}
                  {results.bible.ingredients.updated} updated, {results.bible.ingredients.locked} locked
                  (kept), {results.bible.ingredients.removed} removed.
                </p>
              )}
              <UsageLine usage={usageFor("bible")} />
            </>
          )}
        </StepCard>

        {/* 4. Breakdown */}
        <StepCard
          n={4}
          title="Shot breakdown"
          disabled={!story.bible}
          subtitle="Scenes and shots sized to the video model (clips up to 5 s for the default model)."
          actions={
            board && board.shot_count > 0 ? (
              <Button size="sm" variant="outline" onClick={onOpenShots}>
                <LayoutGridIcon />
                Shots board
              </Button>
            ) : undefined
          }
        >
          {story.bible && (
            <>
              <div className="flex flex-wrap items-end gap-3">
                <div className="space-y-1">
                  <Label htmlFor="bd-target">Target length (s)</Label>
                  <Input
                    id="bd-target"
                    type="number"
                    min={5}
                    className="w-32"
                    value={target ?? ""}
                    onChange={(e) => setTarget(e.target.value ? Number(e.target.value) : undefined)}
                  />
                </div>
                <LlmPicker
                  className="flex-1"
                  value={llm.breakdown ?? null}
                  onChange={(v) => setLlm({ ...llm, breakdown: v })}
                />
              </div>
              <RunButton
                runner={runner}
                step="breakdown"
                label={board?.shot_count ? "Re-run breakdown" : "Break down into shots"}
                onClick={() => (board?.shot_count ? setConfirmReplace(true) : void runBreakdown(false))}
              />
              {board && board.shot_count > 0 && (
                <p className="text-sm" data-testid="board-summary">
                  {plural(board.scenes.length, "scene")} · {plural(board.shot_count, "shot")} ·{" "}
                  {Math.round(board.total_duration_s)} s total
                </p>
              )}
              <UsageLine usage={usageFor("breakdown")} />
            </>
          )}
        </StepCard>
      </div>

      <aside className="space-y-4">
        {warnings.length > 0 && (
          <Card>
            <CardContent className="space-y-1.5 pt-4">
              {warnings.map((w) => (
                <p key={w} className="text-warning flex gap-2 text-xs">
                  <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
                  {w}
                </p>
              ))}
            </CardContent>
          </Card>
        )}
        <RevisionsPanel
          projectId={projectId}
          current={story.revision}
          onRestore={(rev) => void storySteps.restore(projectId, rev).then(() => invalidate())}
        />
      </aside>

      <ConfirmDialog
        open={!!proposal}
        className="max-w-3xl"
        title={proposal?.step === "refine" ? "Replace your edited story?" : "Replace your edited bible?"}
        description="You edited this since it was generated. Review the new version before it replaces your edits."
        confirmLabel="Use the new version"
        cancelLabel="Keep my edits"
        onCancel={() => setProposal(null)}
        onConfirm={() => {
          const p = proposal;
          setProposal(null);
          if (p) void p.apply().then((res) => saved(p.step, res));
        }}
      >
        {proposal && (
          <>
            <DiffView before={proposal.before} after={proposal.after} />
            <UsageLine usage={proposal.usage} />
          </>
        )}
      </ConfirmDialog>

      <ConfirmDialog
        open={confirmReplace}
        destructive
        title="Replace the current shots?"
        description={`This deletes ${plural(board?.scenes.length ?? 0, "scene")} and ${plural(board?.shot_count ?? 0, "shot")}, including any edits you made to them.`}
        confirmLabel="Replace shots"
        onCancel={() => setConfirmReplace(false)}
        onConfirm={() => {
          setConfirmReplace(false);
          void runBreakdown(true);
        }}
      />
    </div>
  );
}
