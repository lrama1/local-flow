import { Loader2Icon, PlayIcon, PlusIcon, SparklesIcon, SquareIcon, XIcon } from "lucide-react";
import { useState } from "react";
import type { LlmChoice } from "@/api/client";
import {
  asParams,
  DEFAULT_PARAMS,
  groupVoices,
  useAudition,
  useVoiceMutations,
  useVoices,
  voiceLabel,
  type Voice,
  type VoiceParams,
} from "@/api/voices";
import { LlmPicker, UsageLine } from "@/components/LlmPicker";
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

const family = (lang: string) => lang.split("-")[0];

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <Label htmlFor={id} className="w-24 shrink-0">
        {label}
      </Label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        className="w-full"
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="text-muted-foreground w-12 text-right text-xs tabular-nums">{format(value)}</span>
    </div>
  );
}

/** Create or edit a custom voice: a built-in base voice, blended with up to two others, at a pace. "Design
 * with AI" maps a text description to those settings (Section 12.2); Audition plays them unsaved. */
export function VoiceDesigner({
  open,
  onOpenChange,
  voice,
  initialName = "",
  initialDescription = "",
  initialBaseId,
  onSaved,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Edit this custom voice; otherwise a new one. */
  voice?: Voice;
  initialName?: string;
  initialDescription?: string;
  initialBaseId?: string | null;
  onSaved?: (voice: Voice) => void;
}) {
  const { data: voices } = useVoices();
  const m = useVoiceMutations();
  const audition = useAudition();
  const builtins = (voices ?? []).filter((v) => v.kind === "builtin" && v.available);
  const firstBase = builtins.find((v) => v.language === "en-US") ?? builtins[0];
  const [name, setName] = useState(voice?.name ?? initialName);
  const [description, setDescription] = useState(voice?.description ?? initialDescription);
  const [baseId, setBaseId] = useState<string | undefined>(
    voice?.base_voice_id ?? initialBaseId ?? undefined,
  );
  const [params, setParams] = useState<VoiceParams>(voice ? asParams(voice.params) : DEFAULT_PARAMS);
  const [text, setText] = useState("");
  const [llm, setLlm] = useState<LlmChoice | null>(null);
  const base = builtins.find((v) => v.id === baseId) ?? firstBase;
  const blendable = builtins.filter(
    (v) => base && v.id !== base.id && family(v.language) === family(base.language),
  );
  const byNative = new Map(builtins.map((v) => [v.native_id, v]));
  const design = m.design.data;
  const saving = m.create.isPending || m.update.isPending;
  const error = m.design.error ?? m.create.error ?? m.update.error;

  const setMix = (mix: VoiceParams["mix"]) => setParams((p) => ({ ...p, mix }));
  const changeBase = (id: string) => {
    setBaseId(id);
    const next = builtins.find((v) => v.id === id);
    // Blend voices must share the base voice's language.
    if (next) {
      setMix(
        params.mix.filter(
          (x) =>
            family(byNative.get(x.voice)?.language ?? "") === family(next.language) &&
            x.voice !== next.native_id,
        ),
      );
    }
  };
  const runDesign = async () => {
    if (!base) return;
    const out = await m.design.mutateAsync({ base_voice_id: base.id, description, llm });
    setParams(asParams(out.params));
  };
  const save = async () => {
    if (!base) return;
    const body = {
      name: name.trim(),
      base_voice_id: base.id,
      description,
      params: params as unknown as Record<string, unknown>,
    };
    const saved = voice
      ? await m.update.mutateAsync({ id: voice.id, body })
      : await m.create.mutateAsync(body);
    onSaved?.(saved);
    onOpenChange(false);
  };
  const draftActive = audition.active === "draft";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{voice ? `Edit ${voice.name}` : "New voice"}</DialogTitle>
          <DialogDescription>
            Start from a built-in voice, then blend in others and set the pace, or describe the voice and let
            AI pick the settings.
          </DialogDescription>
        </DialogHeader>
        {!base ? (
          <p className="text-muted-foreground text-sm">Install a text-to-speech model (Kokoro) first.</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="vd-name">Name</Label>
                <Input id="vd-name" value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="vd-base">Base voice</Label>
                <NativeSelect id="vd-base" value={base.id} onChange={(e) => changeBase(e.target.value)}>
                  {groupVoices(builtins, true).map((g) => (
                    <optgroup key={g.label} label={g.label}>
                      {g.voices.map((v) => (
                        <option key={v.id} value={v.id}>
                          {voiceLabel(v)}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </NativeSelect>
              </div>
            </div>
            <div className="space-y-1">
              <Label htmlFor="vd-desc">Description</Label>
              <Textarea
                id="vd-desc"
                rows={2}
                placeholder="a weathered old fisherman; gravelly, slow and kind"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={!description.trim() || m.design.isPending}
                  onClick={() => void runDesign().catch(() => undefined)}
                >
                  {m.design.isPending ? <Loader2Icon className="animate-spin" /> : <SparklesIcon />}
                  Design with AI
                </Button>
                <LlmPicker value={llm} onChange={setLlm} compact />
              </div>
              {design && (
                <div className="text-muted-foreground space-y-0.5 text-xs" data-testid="design-notes">
                  {design.notes && <p>{design.notes}</p>}
                  {(design.warnings ?? []).map((w) => (
                    <p key={w} className="text-warning">
                      {w}
                    </p>
                  ))}
                  <UsageLine usage={design.usage} />
                </div>
              )}
            </div>

            <div className="space-y-2 rounded-lg border p-3">
              <Slider
                id="vd-base-weight"
                label={`${base.name} weight`}
                value={params.base_weight}
                min={0.2}
                max={1}
                step={0.05}
                format={(v) => v.toFixed(2)}
                onChange={(v) => setParams((p) => ({ ...p, base_weight: v }))}
              />
              {params.mix.map((x, i) => (
                <div key={i} className="flex items-center gap-2">
                  <NativeSelect
                    aria-label={`Blend voice ${i + 1}`}
                    className="w-40 shrink-0"
                    value={x.voice}
                    onChange={(e) =>
                      setMix(params.mix.map((y, j) => (j === i ? { ...y, voice: e.target.value } : y)))
                    }
                  >
                    {blendable.map((v) => (
                      <option key={v.id} value={v.native_id ?? ""}>
                        {voiceLabel(v)}
                      </option>
                    ))}
                  </NativeSelect>
                  <input
                    type="range"
                    aria-label={`Blend voice ${i + 1} weight`}
                    min={0.05}
                    max={1}
                    step={0.05}
                    value={x.weight}
                    className="w-full"
                    onChange={(e) =>
                      setMix(
                        params.mix.map((y, j) => (j === i ? { ...y, weight: Number(e.target.value) } : y)),
                      )
                    }
                  />
                  <span className="text-muted-foreground w-10 text-right text-xs tabular-nums">
                    {x.weight.toFixed(2)}
                  </span>
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="ghost"
                    aria-label="Remove blend voice"
                    onClick={() => setMix(params.mix.filter((_, j) => j !== i))}
                  >
                    <XIcon />
                  </Button>
                </div>
              ))}
              {params.mix.length < 2 && blendable.length > 0 && (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs"
                  onClick={() =>
                    setMix([...params.mix, { voice: blendable[0]!.native_id ?? "", weight: 0.3 }])
                  }
                >
                  <PlusIcon />
                  Blend in a voice
                </Button>
              )}
              <Slider
                id="vd-speed"
                label="Pace"
                value={params.speed}
                min={0.7}
                max={1.4}
                step={0.01}
                format={(v) => `${v.toFixed(2)}×`}
                onChange={(v) => setParams((p) => ({ ...p, speed: v }))}
              />
            </div>

            <div className="flex items-center gap-2">
              <Input
                aria-label="Sample line"
                placeholder="Sample line (optional)"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() =>
                  draftActive && audition.playing
                    ? audition.stop()
                    : void audition.start({ baseVoiceId: base.id, params, text }, "draft")
                }
              >
                {draftActive && audition.loading ? (
                  <Loader2Icon className="animate-spin" />
                ) : draftActive && audition.playing ? (
                  <SquareIcon />
                ) : (
                  <PlayIcon />
                )}
                Audition
              </Button>
            </div>
            {audition.error && <p className="text-destructive text-xs">{audition.error}</p>}
            {error && <p className="text-destructive text-xs">{error.message}</p>}
          </div>
        )}
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            disabled={!base || !name.trim() || saving}
            onClick={() => void save().catch(() => undefined)}
          >
            {saving && <Loader2Icon className="animate-spin" />}
            {voice ? "Save voice" : "Create voice"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
