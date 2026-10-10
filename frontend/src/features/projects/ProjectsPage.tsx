import { FilmIcon, PlusIcon } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router";
import { withToken, type Schemas } from "@/api/client";
import { useCreateProject, useProjects } from "@/api/queries";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge, Input, Label, NativeSelect, Skeleton } from "@/components/ui/primitives";
import { relativeTime } from "@/lib/utils";

type AspectRatio = Schemas["ProjectCreate"]["aspect_ratio"];

function CreateProjectDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const create = useCreateProject();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [aspect, setAspect] = useState<AspectRatio>("16:9");
  const [fps, setFps] = useState(24);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    create.mutate(
      { name: name.trim(), aspect_ratio: aspect, fps },
      {
        onSuccess: (p) => {
          onOpenChange(false);
          setName("");
          void navigate(`/projects/${p.id}`);
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <form onSubmit={submit} className="space-y-4">
          <DialogHeader>
            <DialogTitle>New project</DialogTitle>
            <DialogDescription>
              A project holds one story, its ingredients, shots, and timeline.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5">
            <Label htmlFor="project-name">Name</Label>
            <Input
              id="project-name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My short film"
              required
              maxLength={200}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="project-aspect">Aspect ratio</Label>
              <NativeSelect
                id="project-aspect"
                value={aspect}
                onChange={(e) => setAspect(e.target.value as AspectRatio)}
              >
                <option value="16:9">16:9 landscape</option>
                <option value="9:16">9:16 vertical</option>
                <option value="1:1">1:1 square</option>
              </NativeSelect>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="project-fps">Frame rate</Label>
              <NativeSelect id="project-fps" value={fps} onChange={(e) => setFps(Number(e.target.value))}>
                {[16, 24, 25, 30].map((f) => (
                  <option key={f} value={f}>
                    {f} fps
                  </option>
                ))}
              </NativeSelect>
            </div>
          </div>
          {create.error && <p className="text-destructive text-sm">{create.error.message}</p>}
          <DialogFooter>
            <Button type="submit" disabled={!name.trim() || create.isPending}>
              Create project
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function ProjectsPage() {
  const { data: projects, isLoading } = useProjects();
  const [open, setOpen] = useState(false);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Projects</h1>
          <p className="text-muted-foreground text-sm">Turn a story idea into a short film, shot by shot.</p>
        </div>
        <Button onClick={() => setOpen(true)}>
          <PlusIcon />
          New project
        </Button>
      </div>

      {isLoading && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="aspect-video" />
          ))}
        </div>
      )}

      {projects && projects.length === 0 && (
        <EmptyState
          icon={FilmIcon}
          title="No projects yet"
          action={
            <Button onClick={() => setOpen(true)}>
              <PlusIcon />
              Create your first project
            </Button>
          }
        >
          Start with a rough story idea. LocalFlow refines it, breaks it into shots, and renders them on your
          GPU.
        </EmptyState>
      )}

      {projects && projects.length > 0 && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-4">
          {projects.map((p) => (
            <Link
              key={p.id}
              to={`/projects/${p.id}`}
              className="group bg-card hover:border-primary/60 overflow-hidden rounded-xl border transition-colors"
            >
              <div className="from-primary/25 via-muted to-background aspect-video bg-gradient-to-br">
                {p.cover_thumb_url && (
                  <img src={withToken(p.cover_thumb_url)} alt="" className="size-full object-cover" />
                )}
              </div>
              <div className="space-y-1 p-3">
                <div className="flex items-center gap-2">
                  <span className="truncate font-medium">{p.name}</span>
                  <Badge variant="outline" className="ml-auto">
                    {p.aspect_ratio}
                  </Badge>
                </div>
                <div className="text-muted-foreground text-xs">Updated {relativeTime(p.updated_at)}</div>
              </div>
            </Link>
          ))}
        </div>
      )}
      <CreateProjectDialog open={open} onOpenChange={setOpen} />
    </div>
  );
}
