import { BookOpenIcon, Loader2Icon, SparklesIcon, XIcon } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";
import type { StepRunner } from "./useStepRunner";

export function StepCard({
  n,
  title,
  subtitle,
  disabled,
  children,
  actions,
}: {
  n: number;
  title: string;
  subtitle?: ReactNode;
  disabled?: boolean;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <Card className={cn(disabled && "opacity-60")} data-testid={`step-${n}`}>
      <CardHeader className="flex-row items-start gap-3">
        <span className="bg-primary/15 text-primary flex size-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold">
          {n}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold">{title}</h3>
          {subtitle && <div className="text-muted-foreground text-xs">{subtitle}</div>}
        </div>
        {actions}
      </CardHeader>
      {children && <CardContent className="space-y-3 sm:pl-13">{children}</CardContent>}
    </Card>
  );
}

/** Primary action for a step: shows a spinner + Cancel while the LLM call runs. */
export function RunButton({
  runner,
  step,
  label,
  onClick,
  disabled,
}: {
  runner: StepRunner;
  step: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  const busy = runner.running === step;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button onClick={onClick} disabled={disabled || runner.running !== null} data-testid={`run-${step}`}>
        {busy ? (
          <Loader2Icon className="animate-spin" />
        ) : step === "refine" ? (
          <SparklesIcon />
        ) : (
          <BookOpenIcon />
        )}
        {busy ? "Working…" : label}
      </Button>
      {busy && (
        <Button variant="ghost" size="sm" onClick={runner.cancel}>
          <XIcon />
          Cancel
        </Button>
      )}
      {runner.error?.step === step && (
        <p className="text-destructive text-sm" role="alert">
          {runner.error.message}
        </p>
      )}
    </div>
  );
}
