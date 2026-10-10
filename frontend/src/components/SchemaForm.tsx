import { Input, Label } from "@/components/ui/primitives";

/** Minimal JSON-Schema → form renderer for adapter "Advanced" params (Section 17.2). */
type JsonSchema = {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  anyOf?: JsonSchema[];
  properties?: Record<string, JsonSchema>;
};

function baseType(s: JsonSchema): string | undefined {
  if (s.type) return s.type;
  return s.anyOf?.find((x) => x.type && x.type !== "null")?.type;
}

function bounds(s: JsonSchema): { min?: number; max?: number } {
  const inner = s.anyOf?.find((x) => x.type && x.type !== "null") ?? s;
  return { min: inner.minimum ?? s.minimum, max: inner.maximum ?? s.maximum };
}

export function SchemaForm({
  schema,
  value,
  onChange,
  idPrefix = "adv",
}: {
  schema: JsonSchema | undefined;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
  idPrefix?: string;
}) {
  const props = Object.entries(schema?.properties ?? {});
  if (!props.length)
    return <p className="text-muted-foreground text-xs">This adapter has no advanced options.</p>;

  const set = (key: string, v: unknown) => {
    const next = { ...value };
    if (v === undefined || v === "") delete next[key];
    else next[key] = v;
    onChange(next);
  };

  return (
    <div className="grid grid-cols-2 gap-3">
      {props.map(([key, s]) => {
        const type = baseType(s);
        const id = `${idPrefix}-${key}`;
        const label = s.title ?? key;
        const current = value[key];
        if (type === "boolean") {
          return (
            <label key={key} className="col-span-2 flex items-center gap-2 text-sm" title={s.description}>
              <input
                id={id}
                type="checkbox"
                checked={Boolean(current ?? s.default)}
                onChange={(e) => set(key, e.target.checked)}
              />
              {label}
            </label>
          );
        }
        const { min, max } = bounds(s);
        const numeric = type === "integer" || type === "number";
        return (
          <div key={key} className="space-y-1" title={s.description}>
            <Label htmlFor={id}>{label}</Label>
            <Input
              id={id}
              type={numeric ? "number" : "text"}
              step={type === "integer" ? 1 : "any"}
              min={min}
              max={max}
              placeholder={s.default != null ? String(s.default) : "default"}
              value={current == null ? "" : String(current)}
              onChange={(e) => {
                const raw = e.target.value;
                set(key, raw === "" ? undefined : numeric ? Number(raw) : raw);
              }}
            />
          </div>
        );
      })}
    </div>
  );
}
