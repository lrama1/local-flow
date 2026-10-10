import {
  closestCorners,
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragOverEvent,
  type DragStartEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  horizontalListSortingStrategy,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  ClapperboardIcon,
  LayoutGridIcon,
  MessageSquareTextIcon,
  PlayIcon,
  PlusIcon,
  SparklesIcon,
  Trash2Icon,
  VideoIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import type { Ingredient, Scene, Shot } from "@/api/client";
import { thumbUrl } from "@/api/imagegen";
import { useBoard, useBoardMutations, useIngredients } from "@/api/story";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Badge, Card, Input, NativeSelect, Skeleton } from "@/components/ui/primitives";
import { Tooltip } from "@/components/ui/tooltip";
import { cn, formatDuration, plural } from "@/lib/utils";
import { CAMERA_MOVES, SHOT_TYPES, label } from "./constants";
import { GenerateAllDialog } from "./GenerateAllDialog";
import type { KeyframeRun, KeyframeWhich } from "./references";
import { ShotPanel } from "./ShotPanel";

type Containers = Record<string, string[]>;

const STATUS_VARIANT = {
  planned: "outline",
  keyframe_ready: "secondary",
  video_ready: "default",
  approved: "success",
} as const;

function initials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join("");
}

function ShotCardBody({
  shot,
  number,
  ingredients,
  maxClip,
  projectId,
}: {
  shot: Shot;
  number: string;
  ingredients: Map<string, Ingredient>;
  maxClip: number;
  projectId: string;
}) {
  const people = shot.ingredient_ids.map((id) => ingredients.get(id)).filter((i): i is Ingredient => !!i);
  const tooLong = shot.duration_s > maxClip + 1e-6;
  return (
    <>
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground font-mono text-xs">{number}</span>
        <Badge variant={STATUS_VARIANT[shot.status]}>{shot.status.replace("_", " ")}</Badge>
        <Badge variant={tooLong ? "warning" : "outline"} className="ml-auto">
          {formatDuration(shot.duration_s)}
          {shot.needs_extend && ` +${shot.extend_segments ?? 1}`}
        </Badge>
      </div>
      <div className="bg-muted relative flex aspect-video items-center justify-center overflow-hidden rounded-md">
        {shot.selected_video_asset_id || shot.selected_keyframe_asset_id ? (
          <img
            src={thumbUrl(projectId, shot.selected_video_asset_id ?? shot.selected_keyframe_asset_id)}
            alt={`Shot ${number} ${shot.selected_video_asset_id ? "take" : "keyframe"}`}
            draggable={false}
            className="size-full object-cover"
          />
        ) : (
          <ClapperboardIcon className="text-muted-foreground/60 size-6" />
        )}
        {shot.selected_video_asset_id && (
          <span
            className="absolute bottom-1 left-1 rounded bg-black/60 p-0.5 text-white"
            title="Has a video take"
            data-testid="card-take"
          >
            <PlayIcon className="size-3" />
          </span>
        )}
      </div>
      <p className="line-clamp-3 text-xs">
        {shot.keyframe_prompt || <em className="text-muted-foreground">No prompt</em>}
      </p>
      <div className="text-muted-foreground flex items-center gap-1.5 text-[11px]">
        <VideoIcon className="size-3" />
        {label(SHOT_TYPES, shot.shot_type)} · {label(CAMERA_MOVES, shot.camera_move)} · {shot.generation_mode}
        {shot.dialogue.length > 0 && <MessageSquareTextIcon className="ml-auto size-3" />}
      </div>
      {people.length > 0 && (
        <div className="flex -space-x-1.5">
          {people.map((p) => (
            <span
              key={p.id}
              title={p.name}
              className={cn(
                "border-card flex size-6 items-center justify-center rounded-full border-2 text-[10px] font-semibold",
                p.kind === "character"
                  ? "bg-primary/25 text-primary"
                  : "bg-secondary text-secondary-foreground",
              )}
            >
              {initials(p.name)}
            </span>
          ))}
        </div>
      )}
    </>
  );
}

function SortableShot(props: {
  shot: Shot;
  number: string;
  ingredients: Map<string, Ingredient>;
  maxClip: number;
  projectId: string;
  onOpen: () => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: props.shot.id,
  });
  return (
    <button
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      {...attributes}
      {...listeners}
      onClick={props.onOpen}
      aria-label={`Shot ${props.number}`}
      data-testid={`shot-${props.number}`}
      className={cn(
        "bg-card hover:border-primary/60 focus-visible:ring-ring flex w-56 shrink-0 cursor-grab flex-col gap-2 rounded-lg border p-2.5 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none active:cursor-grabbing",
        isDragging && "opacity-40",
      )}
    >
      <ShotCardBody {...props} />
    </button>
  );
}

function SceneRow({
  scene,
  position,
  sceneIds,
  ids,
  shotsById,
  ingredients,
  locations,
  maxClip,
  projectId,
  onOpen,
}: {
  scene: Scene;
  position: number;
  sceneIds: string[];
  ids: string[];
  shotsById: Map<string, Shot>;
  ingredients: Map<string, Ingredient>;
  locations: Ingredient[];
  maxClip: number;
  projectId: string;
  onOpen: (shotId: string) => void;
}) {
  const m = useBoardMutations(projectId);
  const { setNodeRef, isOver } = useDroppable({ id: scene.id });
  const [title, setTitle] = useState(scene.title);
  const duration = ids.reduce((sum, id) => sum + (shotsById.get(id)?.duration_s ?? 0), 0);

  return (
    <Card className="space-y-3 p-4" data-testid={`scene-${position + 1}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground font-mono text-sm">S{position + 1}</span>
        <Input
          aria-label="Scene title"
          className="hover:border-input h-8 max-w-sm border-transparent bg-transparent font-medium"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onBlur={() =>
            title.trim() && title !== scene.title && m.updateScene.mutate({ id: scene.id, body: { title } })
          }
        />
        <NativeSelect
          aria-label="Location"
          className="h-8 w-auto text-xs"
          value={scene.location_ingredient_id ?? ""}
          onChange={(e) =>
            m.updateScene.mutate({ id: scene.id, body: { location_ingredient_id: e.target.value || null } })
          }
        >
          <option value="">No location</option>
          {locations.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </NativeSelect>
        <span className="text-muted-foreground text-xs">
          {plural(ids.length, "shot")} · {formatDuration(duration)}
        </span>
        <div className="ml-auto flex gap-1">
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Move scene up"
            disabled={position === 0}
            onClick={() => m.reorderScenes.mutate(swap(position, position - 1))}
          >
            <ArrowUpIcon />
          </Button>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Move scene down"
            disabled={position === sceneIds.length - 1}
            onClick={() => m.reorderScenes.mutate(swap(position, position + 1))}
          >
            <ArrowDownIcon />
          </Button>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Delete scene"
            onClick={() =>
              window.confirm(`Delete scene "${scene.title}" and its ${plural(ids.length, "shot")}?`) &&
              m.deleteScene.mutate(scene.id)
            }
          >
            <Trash2Icon />
          </Button>
        </div>
      </div>
      {scene.summary && <p className="text-muted-foreground text-xs">{scene.summary}</p>}
      <SortableContext id={scene.id} items={ids} strategy={horizontalListSortingStrategy}>
        <div
          ref={setNodeRef}
          className={cn("flex min-h-24 gap-3 overflow-x-auto rounded-lg pb-2", isOver && "bg-accent/40")}
        >
          {ids.map((id, i) => {
            const shot = shotsById.get(id);
            return shot ? (
              <SortableShot
                key={id}
                shot={shot}
                number={`${position + 1}.${i + 1}`}
                ingredients={ingredients}
                maxClip={maxClip}
                projectId={projectId}
                onOpen={() => onOpen(id)}
              />
            ) : null;
          })}
          <Tooltip content="Add a shot to this scene">
            <button
              aria-label="Add shot"
              onClick={() =>
                m.createShot.mutate({ sceneId: scene.id, body: { duration_s: Math.min(4, maxClip) } })
              }
              className="text-muted-foreground hover:border-primary/60 hover:text-foreground flex w-24 shrink-0 items-center justify-center rounded-lg border border-dashed"
            >
              <PlusIcon className="size-5" />
            </button>
          </Tooltip>
        </div>
      </SortableContext>
    </Card>
  );

  function swap(a: number, b: number): string[] {
    const next = sceneIds.slice();
    [next[a], next[b]] = [next[b]!, next[a]!];
    return next;
  }
}

export function ShotsBoard({ projectId }: { projectId: string }) {
  const { data: board, isLoading } = useBoard(projectId);
  const { data: ingredientList = [] } = useIngredients(projectId);
  const m = useBoardMutations(projectId);
  const [openShot, setOpenShot] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [dragItems, setDragItems] = useState<Containers | null>(null);
  const [keyframeRuns, setKeyframeRuns] = useState<
    Record<string, Partial<Record<KeyframeWhich, KeyframeRun>>>
  >({});
  const [generateAll, setGenerateAll] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const ingredients = useMemo(() => new Map(ingredientList.map((i) => [i.id, i])), [ingredientList]);
  const shotsById = useMemo(
    () => new Map((board?.scenes ?? []).flatMap((s) => s.shots.map((sh) => [sh.id, sh] as const))),
    [board],
  );
  const serverItems: Containers = useMemo(
    () => Object.fromEntries((board?.scenes ?? []).map((s) => [s.id, s.shots.map((sh) => sh.id)])),
    [board],
  );
  const items = dragItems ?? serverItems;
  const sceneIds = (board?.scenes ?? []).map((s) => s.id);

  if (isLoading) return <Skeleton className="h-60" />;
  if (!board || board.scenes.length === 0) {
    return (
      <EmptyState
        icon={LayoutGridIcon}
        title="No scenes yet"
        action={
          <Button variant="outline" onClick={() => m.createScene.mutate({ title: "Scene 1" })}>
            <PlusIcon />
            Add a scene
          </Button>
        }
      >
        Run &quot;Break down into shots&quot; on the Story tab, or build the board by hand.
      </EmptyState>
    );
  }

  const findContainer = (id: string, source: Containers): string | undefined =>
    id in source ? id : Object.keys(source).find((k) => source[k]!.includes(id));

  const onDragStart = (e: DragStartEvent) => {
    setActiveId(String(e.active.id));
    setDragItems(serverItems);
  };

  const onDragOver = ({ active, over }: DragOverEvent) => {
    if (!over || !dragItems) return;
    const from = findContainer(String(active.id), dragItems);
    const to = findContainer(String(over.id), dragItems);
    if (!from || !to || from === to) return;
    setDragItems((prev) => {
      if (!prev) return prev;
      const fromItems = prev[from]!.filter((x) => x !== active.id);
      const toItems = prev[to]!.slice();
      const overIndex = toItems.indexOf(String(over.id));
      toItems.splice(overIndex >= 0 ? overIndex : toItems.length, 0, String(active.id));
      return { ...prev, [from]: fromItems, [to]: toItems };
    });
  };

  const onDragEnd = ({ active, over }: DragEndEvent) => {
    const current = dragItems;
    setActiveId(null);
    setDragItems(null);
    if (!over || !current) return;
    const id = String(active.id);
    const container = findContainer(id, current);
    const overContainer = findContainer(String(over.id), current);
    if (!container || container !== overContainer) return;
    let ids = current[container]!;
    const oldIndex = ids.indexOf(id);
    const newIndex = ids.indexOf(String(over.id));
    if (newIndex >= 0 && oldIndex !== newIndex) ids = arrayMove(ids, oldIndex, newIndex);
    const original = serverItems[container]!;
    if (ids.join() !== original.join()) m.reorderShots.mutate({ sceneId: container, ids });
  };

  const active = activeId ? shotsById.get(activeId) : undefined;
  const openShotData = openShot ? shotsById.get(openShot) : undefined;
  const openNumber = (() => {
    if (!openShotData) return "";
    const si = board.scenes.findIndex((s) => s.id === openShotData.scene_id);
    return `${si + 1}.${(board.scenes[si]?.shots.findIndex((sh) => sh.id === openShot) ?? 0) + 1}`;
  })();

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span data-testid="board-totals">
          {plural(board.scenes.length, "scene")} · {plural(board.shot_count, "shot")} ·{" "}
          {formatDuration(board.total_duration_s)} total
        </span>
        <span className="text-muted-foreground text-xs">
          Clips up to {board.max_clip_seconds} s · drag shots to reorder
        </span>
        <Button
          size="sm"
          variant="outline"
          className="ml-auto"
          onClick={() => m.createScene.mutate({ title: `Scene ${board.scenes.length + 1}` })}
        >
          <PlusIcon />
          Add scene
        </Button>
        <Button size="sm" disabled={board.shot_count === 0} onClick={() => setGenerateAll(true)}>
          <SparklesIcon />
          Generate all
        </Button>
      </div>
      {generateAll && (
        <GenerateAllDialog
          projectId={projectId}
          shots={board.scenes.flatMap((s) => s.shots)}
          numbers={
            new Map(
              board.scenes.flatMap((s, si) => s.shots.map((sh, i) => [sh.id, `${si + 1}.${i + 1}`] as const)),
            )
          }
          onClose={() => setGenerateAll(false)}
        />
      )}
      <DndContext
        sensors={sensors}
        collisionDetection={closestCorners}
        onDragStart={onDragStart}
        onDragOver={onDragOver}
        onDragEnd={onDragEnd}
        onDragCancel={() => {
          setActiveId(null);
          setDragItems(null);
        }}
      >
        {board.scenes.map((scene, i) => (
          <SceneRow
            key={`${scene.id}:${scene.title}`}
            scene={scene}
            position={i}
            sceneIds={sceneIds}
            ids={items[scene.id] ?? []}
            shotsById={shotsById}
            ingredients={ingredients}
            locations={ingredientList.filter((x) => x.kind === "location")}
            maxClip={board.max_clip_seconds}
            projectId={projectId}
            onOpen={setOpenShot}
          />
        ))}
        <DragOverlay>
          {active && (
            <div className="border-primary bg-card flex w-56 flex-col gap-2 rounded-lg border p-2.5 shadow-xl">
              <ShotCardBody
                shot={active}
                number=""
                ingredients={ingredients}
                maxClip={board.max_clip_seconds}
                projectId={projectId}
              />
            </div>
          )}
        </DragOverlay>
      </DndContext>
      {openShotData && (
        <ShotPanel
          key={`${openShotData.id}:${openShotData.updated_at}`}
          shot={openShotData}
          number={openNumber}
          projectId={projectId}
          ingredients={ingredientList}
          locationId={board.scenes.find((s) => s.id === openShotData.scene_id)?.location_ingredient_id}
          maxClip={board.max_clip_seconds}
          keyframeRuns={keyframeRuns[openShotData.id] ?? {}}
          onKeyframeRun={(run) =>
            setKeyframeRuns((all) => ({
              ...all,
              [openShotData.id]: { ...all[openShotData.id], [run.which]: run },
            }))
          }
          onClose={() => setOpenShot(null)}
        />
      )}
    </div>
  );
}
