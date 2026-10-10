import { Loader2Icon, PlayIcon, SquareIcon, WandSparklesIcon } from "lucide-react";
import { Link } from "react-router";
import { groupVoices, useAudition, useSuggestVoice, useVoices, voiceLabel } from "@/api/voices";
import { Button } from "@/components/ui/button";
import { NativeSelect } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

/** Play/stop the sample of a saved voice. */
export function VoicePlayButton({
  voiceId,
  text,
  audition,
  label = "Play sample",
}: {
  voiceId: string | null | undefined;
  text?: string;
  audition: ReturnType<typeof useAudition>;
  label?: string;
}) {
  const busy = !!voiceId && audition.active === voiceId;
  return (
    <Button
      type="button"
      size="icon-sm"
      variant="outline"
      aria-label={busy && audition.playing ? "Stop" : label}
      title={label}
      disabled={!voiceId}
      onClick={() =>
        busy && audition.playing
          ? audition.stop()
          : voiceId && void audition.start({ voiceId, text }, voiceId)
      }
    >
      {busy && audition.loading ? (
        <Loader2Icon className="animate-spin" />
      ) : busy && audition.playing ? (
        <SquareIcon />
      ) : (
        <PlayIcon />
      )}
    </Button>
  );
}

export function VoicePicker({
  id,
  value,
  onChange,
  hints,
  builtinOnly = false,
  allowNone = true,
  noneLabel = "No voice",
  onDesign,
  className,
}: {
  id?: string;
  value: string | null | undefined;
  onChange: (voiceId: string | null) => void;
  /** Character voice hints: offers the best matching built-in voice while none is chosen. */
  hints?: Record<string, string> | null;
  builtinOnly?: boolean;
  allowNone?: boolean;
  noneLabel?: string;
  /** Shows a "Design a voice" button. */
  onDesign?: () => void;
  className?: string;
}) {
  const { data: voices, isLoading } = useVoices();
  const audition = useAudition();
  const suggestion = useSuggestVoice(hints, !value);
  const list = voices ?? [];
  const groups = groupVoices(list, builtinOnly);
  const current = list.find((v) => v.id === value);
  const anyAvailable = groups.length > 0;

  if (!isLoading && !anyAvailable && !current) {
    return (
      <p className={cn("text-muted-foreground text-xs", className)} data-testid="no-voices">
        No voices yet: install a text-to-speech model (Kokoro) on the{" "}
        <Link to="/models" className="text-primary underline">
          Models page
        </Link>
        .
      </p>
    );
  }
  const suggested = !value ? suggestion.data?.voice : undefined;
  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="flex items-center gap-2">
        <NativeSelect
          id={id}
          aria-label={id ? undefined : "Voice"}
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value || null)}
        >
          {allowNone && <option value="">{noneLabel}</option>}
          {current && !current.available && (
            <option value={current.id}>{voiceLabel(current)} (model not installed)</option>
          )}
          {groups.map((g) => (
            <optgroup key={g.label} label={g.label}>
              {g.voices.map((v) => (
                <option key={v.id} value={v.id}>
                  {voiceLabel(v)}
                </option>
              ))}
            </optgroup>
          ))}
        </NativeSelect>
        <VoicePlayButton voiceId={current?.available ? current.id : null} audition={audition} />
        {onDesign && (
          <Button type="button" size="sm" variant="outline" className="shrink-0" onClick={onDesign}>
            <WandSparklesIcon />
            Design a voice
          </Button>
        )}
      </div>
      {current?.traits && <p className="text-muted-foreground text-xs">{current.traits}</p>}
      {suggested && (
        <p className="text-muted-foreground flex items-center gap-2 text-xs" data-testid="voice-suggestion">
          Suggested: {suggested.name} ({suggestion.data?.reason})
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-xs"
            onClick={() => onChange(suggested.id)}
          >
            Use
          </Button>
          <VoicePlayButton voiceId={suggested.id} audition={audition} label="Play suggested voice" />
        </p>
      )}
      {audition.error && <p className="text-destructive text-xs">{audition.error}</p>}
    </div>
  );
}
