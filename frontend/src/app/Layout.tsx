import {
  BoxesIcon,
  FlaskConicalIcon,
  FolderKanbanIcon,
  GraduationCapIcon,
  ListChecksIcon,
  SettingsIcon,
  UsersRoundIcon,
} from "lucide-react";
import { NavLink, Outlet, useMatch } from "react-router";
import { useEventStream } from "@/api/events";
import { useProject } from "@/api/queries";
import { GpuStatusChip } from "@/components/GpuStatusChip";
import { Toaster } from "@/components/Toaster";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/primitives";
import { JobsDrawer } from "@/features/jobs/JobsDrawer";
import { useActiveJobCount } from "@/features/jobs/useActiveJobCount";
import { LicenseGate } from "@/features/models/LicenseGate";
import { cn } from "@/lib/utils";
import { useUi } from "@/stores/ui";

const NAV = [
  { to: "/projects", label: "Projects", icon: FolderKanbanIcon },
  { to: "/cast", label: "Cast library", icon: UsersRoundIcon },
  { to: "/playground", label: "Playground", icon: FlaskConicalIcon },
  { to: "/models", label: "Models", icon: BoxesIcon },
  { to: "/tutorials", label: "Tutorials", icon: GraduationCapIcon },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

function Sidebar() {
  const connection = useUi((s) => s.connection);
  return (
    <nav className="bg-sidebar flex w-56 shrink-0 flex-col border-r">
      <div className="flex h-14 items-center gap-2 px-4">
        <img src="/favicon.svg" alt="" className="size-6" />
        <span className="font-semibold tracking-tight">LocalFlow</span>
      </div>
      <ul className="flex-1 space-y-0.5 px-2 py-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <li key={to}>
            <NavLink
              to={to}
              className={({ isActive }) =>
                cn(
                  "text-muted-foreground hover:bg-accent hover:text-foreground flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive && "bg-accent text-foreground font-medium",
                )
              }
            >
              <Icon className="size-4" />
              {label}
            </NavLink>
          </li>
        ))}
      </ul>
      <div
        className="text-muted-foreground flex items-center gap-2 px-4 py-3 text-xs"
        data-testid="connection"
      >
        <span
          className={cn(
            "size-2 rounded-full",
            connection === "open"
              ? "bg-success"
              : connection === "connecting"
                ? "bg-warning"
                : "bg-destructive",
          )}
        />
        {connection === "open" ? "Connected" : connection === "connecting" ? "Connecting…" : "Disconnected"}
      </div>
    </nav>
  );
}

function TopBar() {
  const match = useMatch("/projects/:projectId/*");
  const { data: project } = useProject(match?.params.projectId);
  const { running, queued } = useActiveJobCount();
  const toggle = useUi((s) => s.toggleJobsDrawer);
  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b px-5">
      <div className="min-w-0 flex-1 truncate text-sm font-medium">{project?.name}</div>
      <GpuStatusChip />
      <Button variant="outline" size="sm" onClick={toggle} aria-label="Toggle jobs drawer">
        <ListChecksIcon />
        Jobs
        {running + queued > 0 && (
          <Badge data-testid="active-jobs">
            {running > 0 ? `${running} running` : ""}
            {running > 0 && queued > 0 ? " · " : ""}
            {queued > 0 ? `${queued} queued` : ""}
          </Badge>
        )}
      </Button>
    </header>
  );
}

export function Layout() {
  useEventStream();
  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
      <JobsDrawer />
      <LicenseGate />
      <Toaster />
    </div>
  );
}
