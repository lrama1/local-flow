import { useMemo } from "react";
import { diffLines, hasChanges } from "@/lib/diff";
import { cn } from "@/lib/utils";

export function DiffView({
  before,
  after,
  className,
}: {
  before: string;
  after: string;
  className?: string;
}) {
  const ops = useMemo(() => diffLines(before, after), [before, after]);
  if (!hasChanges(ops)) {
    return <p className="text-muted-foreground text-sm">No differences.</p>;
  }
  return (
    <pre
      className={cn(
        "bg-muted/40 max-h-[55vh] overflow-auto rounded-md border p-3 text-xs leading-5",
        className,
      )}
      data-testid="diff"
    >
      {ops.map((op, i) => (
        <div
          key={i}
          className={cn(
            "px-1 whitespace-pre-wrap",
            op.type === "add" && "bg-success/15 text-success",
            op.type === "del" && "bg-destructive/15 text-destructive decoration-destructive/40 line-through",
          )}
        >
          <span className="text-muted-foreground mr-2 inline-block w-3 select-none">
            {op.type === "add" ? "+" : op.type === "del" ? "−" : " "}
          </span>
          {op.text || " "}
        </div>
      ))}
    </pre>
  );
}
