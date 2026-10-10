import { AudioLinesIcon, PencilIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { languageName, useAudition, useVoiceMutations, useVoices, type Voice } from "@/api/voices";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Badge, Card, NativeSelect, Skeleton } from "@/components/ui/primitives";
import { VoicePlayButton } from "@/components/VoicePicker";
import { VoiceDesigner } from "./VoiceDesigner";

/** Voices tab of the Cast library: custom voices (designed from built-ins) and the built-in voices. */
export function VoicesPanel() {
  const { data: voices = [], isLoading } = useVoices();
  const m = useVoiceMutations();
  const audition = useAudition();
  const [editing, setEditing] = useState<Voice | null>(null);
  const [creating, setCreating] = useState(false);
  const [lang, setLang] = useState("en-US");
  const custom = voices.filter((v) => v.kind === "custom");
  const builtins = voices.filter((v) => v.kind === "builtin" && v.available);
  const langs = [...new Set(builtins.map((v) => v.language))];
  const shown = builtins.filter((v) => v.language === (langs.includes(lang) ? lang : langs[0]));

  if (isLoading) return <Skeleton className="h-60" />;
  if (builtins.length === 0 && custom.length === 0) {
    return (
      <EmptyState icon={AudioLinesIcon} title="No voices yet">
        Install a text-to-speech model (Kokoro) on the{" "}
        <Link to="/models" className="text-primary underline">
          Models page
        </Link>{" "}
        to get its built-in voices.
      </EmptyState>
    );
  }
  return (
    <div className="space-y-6" data-testid="voices-panel">
      <section className="space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Custom voices</h2>
            <p className="text-muted-foreground text-xs">
              A built-in voice blended with others at your pace. Give one to a character in its card or the
              character wizard.
            </p>
          </div>
          <Button size="sm" onClick={() => setCreating(true)} disabled={builtins.length === 0}>
            <PlusIcon />
            New voice
          </Button>
        </div>
        {custom.length === 0 ? (
          <p className="text-muted-foreground text-sm">No custom voices yet.</p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {custom.map((v) => (
              <Card key={v.id} className="flex items-start gap-3 p-3" data-testid={`voice-${v.name}`}>
                <VoicePlayButton voiceId={v.available ? v.id : null} audition={audition} />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{v.name}</p>
                  <p className="text-muted-foreground truncate text-xs">
                    from {v.base_voice_name ?? "?"} · {languageName(v.language)}
                    {!v.available && " · model not installed"}
                  </p>
                  {v.description && (
                    <p className="text-muted-foreground line-clamp-2 text-xs">{v.description}</p>
                  )}
                </div>
                <Button
                  size="icon-sm"
                  variant="ghost"
                  aria-label={`Edit ${v.name}`}
                  onClick={() => setEditing(v)}
                >
                  <PencilIcon />
                </Button>
                <Button
                  size="icon-sm"
                  variant="ghost"
                  aria-label={`Delete ${v.name}`}
                  onClick={() => {
                    if (
                      window.confirm(`Delete the voice "${v.name}"? Characters using it lose their voice.`)
                    ) {
                      m.remove.mutate(v.id);
                    }
                  }}
                >
                  <Trash2Icon />
                </Button>
              </Card>
            ))}
          </div>
        )}
      </section>

      {builtins.length > 0 && (
        <section className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-sm font-semibold">Built-in voices</h2>
            <NativeSelect
              aria-label="Language"
              className="w-48"
              value={langs.includes(lang) ? lang : langs[0]}
              onChange={(e) => setLang(e.target.value)}
            >
              {langs.map((l) => (
                <option key={l} value={l}>
                  {languageName(l)}
                </option>
              ))}
            </NativeSelect>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {shown.map((v) => (
              <Card
                key={v.id}
                className="flex items-center gap-3 p-2.5"
                data-testid={`builtin-${v.native_id}`}
              >
                <VoicePlayButton voiceId={v.id} audition={audition} />
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-1.5 text-sm font-medium">
                    {v.name}
                    {v.quality && <Badge variant="outline">{v.quality}</Badge>}
                  </p>
                  <p className="text-muted-foreground truncate text-xs">
                    {v.gender} · {v.traits}
                  </p>
                </div>
              </Card>
            ))}
          </div>
          {audition.error && <p className="text-destructive text-xs">{audition.error}</p>}
        </section>
      )}

      {(creating || editing) && (
        <VoiceDesigner
          key={editing?.id ?? "new"}
          open
          voice={editing ?? undefined}
          onOpenChange={(o) => {
            if (!o) {
              setCreating(false);
              setEditing(null);
            }
          }}
        />
      )}
    </div>
  );
}
