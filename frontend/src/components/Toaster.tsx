import { CheckCircle2Icon, InfoIcon, XCircleIcon, XIcon } from "lucide-react";
import { useEffect } from "react";
import { cn } from "@/lib/utils";
import { useUi, type Toast } from "@/stores/ui";

function ToastItem({ toast }: { toast: Toast }) {
  const dismiss = useUi((s) => s.dismissToast);
  useEffect(() => {
    const t = setTimeout(() => dismiss(toast.id), toast.variant === "error" ? 10_000 : 5_000);
    return () => clearTimeout(t);
  }, [toast.id, toast.variant, dismiss]);
  const Icon =
    toast.variant === "success" ? CheckCircle2Icon : toast.variant === "error" ? XCircleIcon : InfoIcon;
  return (
    <div
      role="status"
      className="bg-popover pointer-events-auto flex w-80 items-start gap-3 rounded-lg border p-3 shadow-lg"
    >
      <Icon
        className={cn(
          "mt-0.5 size-4 shrink-0",
          toast.variant === "success" && "text-success",
          toast.variant === "error" && "text-destructive",
        )}
      />
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium">{toast.title}</div>
        {toast.description && (
          <div className="text-muted-foreground line-clamp-3 text-xs">{toast.description}</div>
        )}
        {toast.action && (
          <button
            className="text-primary mt-1 text-xs font-medium hover:underline"
            onClick={() => {
              toast.action?.onClick();
              dismiss(toast.id);
            }}
          >
            {toast.action.label}
          </button>
        )}
      </div>
      <button aria-label="Dismiss" onClick={() => dismiss(toast.id)} className="text-muted-foreground">
        <XIcon className="size-3.5" />
      </button>
    </div>
  );
}

export function Toaster() {
  const toasts = useUi((s) => s.toasts);
  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} />
      ))}
    </div>
  );
}
