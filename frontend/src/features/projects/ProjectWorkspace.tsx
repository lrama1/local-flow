import {
  BookOpenIcon,
  FilmIcon,
  ImagesIcon,
  LayoutGridIcon,
  type LucideIcon,
  SparklesIcon,
  Trash2Icon,
} from "lucide-react";
import { useNavigate, useParams } from "react-router";
import { useDeleteProject, useProject } from "@/api/queries";
import { EmptyState } from "@/components/EmptyState";
import { AssetLibrary } from "@/features/assets/AssetLibrary";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/primitives";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { IngredientsTab } from "@/features/ingredients/IngredientsTab";
import { ShotsBoard } from "@/features/shots/ShotsBoard";
import { StoryTab } from "@/features/story/StoryTab";
import { TimelineTab } from "@/features/timeline/TimelineTab";

const TABS: { value: string; label: string; icon: LucideIcon }[] = [
  { value: "story", label: "Story", icon: BookOpenIcon },
  { value: "ingredients", label: "Ingredients", icon: SparklesIcon },
  { value: "shots", label: "Shots board", icon: LayoutGridIcon },
  { value: "timeline", label: "Timeline", icon: FilmIcon },
  { value: "assets", label: "Assets", icon: ImagesIcon },
];

export function ProjectWorkspace() {
  const { projectId, tab = "story" } = useParams();
  const navigate = useNavigate();
  const { data: project, error } = useProject(projectId);
  const del = useDeleteProject();

  if (error) {
    return (
      <div className="p-6">
        <EmptyState icon={FilmIcon} title="Project not found">
          {error.message}
        </EmptyState>
      </div>
    );
  }
  if (!project) return <Skeleton className="m-6 h-40" />;

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">{project.name}</h1>
          <p className="text-muted-foreground text-sm">
            {project.aspect_ratio} · {project.fps} fps
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            if (window.confirm(`Delete "${project.name}" and all of its files? This can't be undone.`)) {
              del.mutate(project.id, { onSuccess: () => void navigate("/projects") });
            }
          }}
        >
          <Trash2Icon />
          Delete
        </Button>
      </div>
      <Tabs value={tab} onValueChange={(v) => void navigate(`/projects/${project.id}/${v}`)}>
        <TabsList>
          {TABS.map(({ value, label, icon: Icon }) => (
            <TabsTrigger key={value} value={value}>
              <Icon />
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
        {TABS.map(({ value }) => (
          <TabsContent key={value} value={value} className="pt-5">
            {value === "assets" ? (
              <AssetLibrary projectId={project.id} />
            ) : value === "story" ? (
              <StoryTab
                projectId={project.id}
                onOpenShots={() => void navigate(`/projects/${project.id}/shots`)}
              />
            ) : value === "ingredients" ? (
              <IngredientsTab projectId={project.id} />
            ) : value === "shots" ? (
              <ShotsBoard projectId={project.id} />
            ) : (
              <TimelineTab projectId={project.id} />
            )}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
