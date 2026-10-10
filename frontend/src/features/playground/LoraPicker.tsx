import { XIcon } from "lucide-react";
import type { ModelEntry } from "@/api/models";
import { Button } from "@/components/ui/button";
import { Label, NativeSelect } from "@/components/ui/primitives";

const MAX_LORAS = 3;

export function LoraPicker({
  choices,
  value,
  onChange,
}: {
  choices: ModelEntry[];
  value: { id: string; weight: number }[];
  onChange: (v: { id: string; weight: number }[]) => void;
}) {
  const names = new Map(choices.map((c) => [c.id, c.name]));
  const remaining = choices.filter((c) => !value.some((v) => v.id === c.id));
  return (
    <div className="space-y-1.5">
      <Label>LoRAs</Label>
      {value.map((l) => (
        <div key={l.id} className="flex items-center gap-2 text-sm">
          <span className="min-w-0 flex-1 truncate" title={names.get(l.id)}>
            {names.get(l.id) ?? l.id}
          </span>
          <input
            type="range"
            aria-label={`${names.get(l.id) ?? "LoRA"} weight`}
            min={-2}
            max={2}
            step={0.05}
            value={l.weight}
            onChange={(e) =>
              onChange(value.map((x) => (x.id === l.id ? { ...x, weight: Number(e.target.value) } : x)))
            }
            className="w-28"
          />
          <span className="w-10 text-right text-xs tabular-nums">{l.weight.toFixed(2)}</span>
          <Button
            type="button"
            size="icon-sm"
            variant="ghost"
            aria-label="Remove LoRA"
            onClick={() => onChange(value.filter((x) => x.id !== l.id))}
          >
            <XIcon />
          </Button>
        </div>
      ))}
      {value.length < MAX_LORAS && remaining.length > 0 && (
        <NativeSelect
          aria-label="Add LoRA"
          value=""
          onChange={(e) => e.target.value && onChange([...value, { id: e.target.value, weight: 1 }])}
        >
          <option value="">Add a LoRA…</option>
          {remaining.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
              {c.base_model ? ` (${c.base_model})` : ""}
            </option>
          ))}
        </NativeSelect>
      )}
    </div>
  );
}
