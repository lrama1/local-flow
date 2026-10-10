import { useId } from "react";
import type { LlmChoice, LlmProvider, ProviderStatus, UsageOut } from "@/api/client";
import { useLlmModels, useLlmProviders } from "@/api/story";
import { Input, NativeSelect } from "@/components/ui/primitives";
import { cn, formatCost } from "@/lib/utils";

/** Provider + model picker. `null` means "use the default for this purpose" (Settings / config). */
export function LlmPicker({
  value,
  onChange,
  defaultLabel = "Default model",
  className,
  compact = false,
}: {
  value: LlmChoice | null;
  onChange: (v: LlmChoice | null) => void;
  defaultLabel?: string;
  className?: string;
  compact?: boolean;
}) {
  const { data: providers } = useLlmProviders();
  const usable = (providers ?? []).filter((p) => p.configured && p.ok !== false);
  const { data: models } = useLlmModels(value?.provider);
  const listId = useId();

  return (
    <div className={cn("flex min-w-0 items-center gap-2", className)}>
      <NativeSelect
        aria-label="LLM provider"
        className={cn("w-auto", compact && "h-8 text-xs")}
        value={value?.provider ?? ""}
        onChange={(e) => {
          const p = e.target.value as LlmProvider | "";
          if (!p) return onChange(null);
          const status = usable.find((u) => u.provider === p);
          onChange({ provider: p, model: status?.default_model ?? "" });
        }}
      >
        <option value="">{defaultLabel}</option>
        {usable.map((p: ProviderStatus) => (
          <option key={p.provider} value={p.provider}>
            {p.label}
          </option>
        ))}
      </NativeSelect>
      {value && (
        <>
          <Input
            aria-label="LLM model"
            list={listId}
            className={cn("min-w-40 flex-1", compact && "h-8 text-xs")}
            value={value.model}
            onChange={(e) => onChange({ ...value, model: e.target.value })}
            placeholder="model id"
          />
          <datalist id={listId}>
            {(models ?? []).slice(0, 400).map((m) => (
              <option key={m.id} value={m.id}>
                {m.name !== m.id ? m.name : undefined}
              </option>
            ))}
          </datalist>
        </>
      )}
    </div>
  );
}

export function UsageLine({ usage, className }: { usage: UsageOut | null | undefined; className?: string }) {
  if (!usage) return null;
  return (
    <p className={cn("text-muted-foreground text-xs", className)} data-testid="usage">
      {usage.provider}/{usage.model} · {usage.prompt_tokens.toLocaleString()} in /{" "}
      {usage.completion_tokens.toLocaleString()} out · {formatCost(usage.cost_usd)}
      {usage.attempts > 1 && ` · ${usage.attempts} attempts (repaired)`}
    </p>
  );
}
