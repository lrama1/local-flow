import { ListChecksIcon, XIcon } from "lucide-react";
import { useEffect } from "react";
import { useJobs } from "@/api/queries";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useUi } from "@/stores/ui";
import { JobCard } from "./JobCard";

export function JobsDrawer() {
  const open = useUi((s) => s.jobsDrawerOpen);
  const setOpen = useUi((s) => s.setJobsDrawerOpen);
  const { data, isLoading } = useJobs();
  const items = data?.items ?? [];
  const active = items.filter((j) => j.status === "running" || j.status === "queued");
  const recent = items.filter((j) => j.status !== "running" && j.status !== "queued").slice(0, 25);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  return (
    <aside
      aria-label="Jobs"
      aria-hidden={!open}
      className={cn(
        "bg-background fixed top-0 right-0 z-40 flex h-full w-[380px] max-w-full flex-col border-l shadow-2xl transition-transform duration-200",
        open ? "translate-x-0" : "pointer-events-none translate-x-full",
      )}
    >
      <div className="flex h-14 items-center justify-between border-b px-4">
        <h2 className="font-semibold">Jobs</h2>
        <Button size="icon-sm" variant="ghost" aria-label="Close jobs" onClick={() => setOpen(false)}>
          <XIcon />
        </Button>
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto p-4">
        {!isLoading && items.length === 0 && (
          <EmptyState icon={ListChecksIcon} title="No jobs yet">
            Generations, downloads, and exports show up here with live progress.
          </EmptyState>
        )}
        {active.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
              Active · {active.length}
            </h3>
            {/* Running first, then queued in dispatch order (priority desc, FIFO). */}
            {active
              .slice()
              .sort((a, b) =>
                a.status !== b.status
                  ? a.status === "running"
                    ? -1
                    : 1
                  : b.priority - a.priority || a.id.localeCompare(b.id),
              )
              .map((j) => (
                <JobCard key={j.id} job={j} />
              ))}
          </section>
        )}
        {recent.length > 0 && (
          <section className="space-y-2">
            <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">Recent</h3>
            {recent.map((j) => (
              <JobCard key={j.id} job={j} compact />
            ))}
          </section>
        )}
      </div>
    </aside>
  );
}
