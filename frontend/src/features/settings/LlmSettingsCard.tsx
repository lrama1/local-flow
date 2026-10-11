import { useState } from "react";
import type { LlmChoice } from "@/api/client";
import { Button } from "@/components/ui/button";
import { useAppSettings, useLlmUsage, usePatchSettings } from "@/api/story";
import { LlmPicker } from "@/components/LlmPicker";
import { Card, CardContent, CardHeader, CardTitle, Skeleton } from "@/components/ui/primitives";
import { formatCost, relativeTime } from "@/lib/utils";

const PURPOSE_LABELS: Record<string, string> = {
  "*": "All story steps",
  refine: "Refine",
  bible: "Story bible",
  breakdown: "Shot breakdown",
  prompt_rewrite: "Prompt rewrite",
  character_expand: "Character from text",
  character_describe: "Describe a photo",
  voice_design: "Voice design",
  vision_review: "Consistency check",
};

function DefaultRow({
  purpose,
  saved,
  saving,
  onSave,
}: {
  purpose: string;
  saved: LlmChoice | null;
  saving: boolean;
  onSave: (v: LlmChoice | null) => void;
}) {
  const [draft, setDraft] = useState<LlmChoice | null>(saved);
  const dirty = JSON.stringify(draft) !== JSON.stringify(saved);
  return (
    <div className="flex items-center gap-3" data-testid={`default-${purpose}`}>
      <span className="w-32 shrink-0 text-sm">{PURPOSE_LABELS[purpose] ?? purpose}</span>
      <LlmPicker
        className="flex-1"
        compact
        defaultLabel={purpose === "*" ? "Automatic" : "Same as all steps"}
        value={draft}
        onChange={setDraft}
      />
      {dirty && (
        <Button
          size="sm"
          disabled={saving || (draft !== null && !draft.model.trim())}
          onClick={() => onSave(draft)}
        >
          Save
        </Button>
      )}
    </div>
  );
}

export function LlmSettingsCard() {
  const { data: settings } = useAppSettings();
  const patch = usePatchSettings();
  const { data: usage } = useLlmUsage();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Default models</CardTitle>
        <p className="text-muted-foreground text-xs">
          Used when a story step doesn&apos;t pick a model.
          {settings &&
            ` Built-in defaults: ${Object.entries(settings.default_llm_models)
              .map(([p, m]) => `${p}/${m}`)
              .join(", ")}.`}
        </p>
      </CardHeader>
      <CardContent className="space-y-2">
        {!settings ? (
          <Skeleton className="h-32" />
        ) : (
          settings.purposes.map((purpose) => (
            <DefaultRow
              key={`${purpose}:${JSON.stringify(settings.llm_defaults[purpose] ?? null)}`}
              purpose={purpose}
              saved={settings.llm_defaults[purpose] ?? null}
              saving={patch.isPending}
              onSave={(v) => patch.mutate({ llm_defaults: { [purpose]: v } })}
            />
          ))
        )}
        {patch.error && <p className="text-destructive text-xs">{patch.error.message}</p>}
        {usage && (
          <div className="pt-3">
            <h4 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">Usage</h4>
            <p className="text-sm">
              {usage.calls} calls · {(usage.prompt_tokens + usage.completion_tokens).toLocaleString()} tokens
              · {formatCost(usage.cost_usd)}
            </p>
            <ul className="text-muted-foreground mt-1 space-y-0.5 text-xs">
              {usage.recent.slice(0, 6).map((r, i) => (
                <li key={i}>
                  {relativeTime(r.created_at)} · {r.purpose} · {r.provider}/{r.model} ·{" "}
                  {(r.prompt_tokens + r.completion_tokens).toLocaleString()} tokens · {formatCost(r.cost_usd)}
                  {!r.ok && " · failed"}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
