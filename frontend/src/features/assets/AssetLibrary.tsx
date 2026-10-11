import {
  CheckIcon,
  FileAudioIcon,
  FolderIcon,
  FolderPlusIcon,
  ImageIcon,
  ImagesIcon,
  Loader2Icon,
  PencilIcon,
  PlayIcon,
  SearchIcon,
  StarIcon,
  TagIcon,
  Trash2Icon,
  UploadIcon,
  XIcon,
} from "lucide-react";
import { useDeferredValue, useRef, useState, type DragEvent, type FormEvent, type ReactNode } from "react";
import type { ApiError, Asset } from "@/api/client";
import {
  useAssetMutations,
  useCollectionMutations,
  useCollections,
  useFacets,
  useLibraryAssets,
  type AssetSource,
  type Collection,
  type LibraryFilters,
} from "@/api/library";
import { useBoard, useIngredients } from "@/api/story";
import { useUploadAsset } from "@/api/timeline";
import { EmptyState } from "@/components/EmptyState";
import { AssetThumb } from "@/components/MediaPreview";
import { Button } from "@/components/ui/button";
import { Badge, Input, NativeSelect, Skeleton } from "@/components/ui/primitives";
import { cn, formatDuration, plural } from "@/lib/utils";
import { useUi } from "@/stores/ui";
import { AssetDrawer } from "./AssetDrawer";
import { assetPrompt } from "./reuse";

/** "all", "favorites" or a collection id. */
type Scope = string;

const SOURCES: { value: AssetSource; label: string }[] = [
  { value: "generated", label: "Generated" },
  { value: "uploaded", label: "Uploaded" },
  { value: "extracted", label: "Extracted frames" },
  { value: "export", label: "Exports" },
];
const TYPE_ICON = { video: PlayIcon, image: ImageIcon, audio: FileAudioIcon } as const;

/** The project's asset library (Section 13): collections, filters, bulk actions and the detail drawer. */
export function AssetLibrary({ projectId }: { projectId: string }) {
  const [scope, setScope] = useState<Scope>("all");
  const [filters, setFilters] = useState<LibraryFilters>({ sort: "newest" });
  const [search, setSearch] = useState("");
  const q = useDeferredValue(search.trim());
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [openId, setOpenId] = useState<string | null>(null);

  const query: LibraryFilters = {
    ...filters,
    q,
    favorite: scope === "favorites" ? true : undefined,
    collection: scope !== "all" && scope !== "favorites" ? scope : undefined,
  };
  const lib = useLibraryAssets(projectId, query);
  const uploader = useUploader(projectId);
  const { data: facets } = useFacets(projectId);
  const { data: collections = [] } = useCollections(projectId);
  const assets = lib.data?.pages.flatMap((p) => p.items) ?? [];
  const total = lib.data?.pages[0]?.total ?? 0;
  // Only what's on screen can be selected; deleted or filtered-out assets drop out on their own.
  const selected = assets.filter((a) => picked.has(a.id)).map((a) => a.id);
  const filtered = Object.entries(filters).some(([k, v]) => k !== "sort" && v) || !!q;
  const set = (patch: LibraryFilters) => setFilters((f) => ({ ...f, ...patch }));
  const toggle = (id: string) =>
    setPicked((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  if (facets && facets.total === 0) {
    return (
      <div {...uploader.dropProps}>
        <EmptyState icon={ImagesIcon} title="No assets yet" action={<UploadButton uploader={uploader} />}>
          Generated and uploaded images, videos, and audio for this project appear here. Drop files to upload.
        </EmptyState>
      </div>
    );
  }

  return (
    <div
      className={cn("flex gap-5 rounded-lg", uploader.over && "ring-primary ring-2 ring-offset-4")}
      data-testid="asset-library"
      {...uploader.dropProps}
    >
      <CollectionsSidebar
        projectId={projectId}
        scope={scope}
        onScope={(s) => {
          setScope(s);
          setPicked(new Set());
        }}
        collections={collections}
        total={facets?.total}
        favorites={facets?.favorites}
      />
      <div className="min-w-0 flex-1 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-48 flex-1">
            <SearchIcon className="text-muted-foreground absolute top-2.5 left-2.5 size-4" />
            <Input
              aria-label="Search assets"
              placeholder="Search prompts, dialogue, file names…"
              className="pl-8"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <FilterSelect
            label="Type"
            value={filters.type}
            onChange={(v) => set({ type: v as LibraryFilters["type"] })}
            options={(
              [
                ["image", "Images"],
                ["video", "Videos"],
                ["audio", "Audio"],
              ] as const
            ).map(([t, name]) => ({
              value: t,
              label: `${name}${facets?.types[t] ? ` (${facets.types[t]})` : ""}`,
            }))}
          />
          <FilterSelect
            label="Source"
            value={filters.source}
            onChange={(v) => set({ source: v as AssetSource })}
            options={SOURCES.filter((s) => facets?.sources[s.value]).map((s) => ({
              value: s.value,
              label: `${s.label} (${facets?.sources[s.value]})`,
            }))}
          />
          <FilterSelect
            label="Model"
            value={filters.model}
            onChange={(v) => set({ model: v })}
            options={(facets?.models ?? []).map((m) => ({ value: m.model, label: `${m.name} (${m.count})` }))}
          />
          <FilterSelect
            label="Tag"
            value={filters.tag}
            onChange={(v) => set({ tag: v })}
            options={(facets?.tags ?? []).map((t) => ({ value: t.tag, label: `${t.tag} (${t.count})` }))}
          />
          <StoryFilters projectId={projectId} filters={filters} set={set} />
          <NativeSelect
            aria-label="Sort"
            className="w-auto"
            value={filters.sort}
            onChange={(e) => set({ sort: e.target.value as LibraryFilters["sort"] })}
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </NativeSelect>
          {filtered && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setFilters({ sort: filters.sort });
                setSearch("");
              }}
            >
              <XIcon />
              Clear
            </Button>
          )}
        </div>

        {selected.length > 0 ? (
          <SelectionBar
            projectId={projectId}
            ids={selected}
            scope={scope}
            collections={collections}
            onSelectAll={() => setPicked(new Set(assets.map((a) => a.id)))}
            onClear={() => setPicked(new Set())}
          />
        ) : (
          <div className="flex items-center gap-2">
            <p className="text-muted-foreground text-xs" data-testid="library-count">
              {lib.isLoading ? "Loading…" : plural(total, "asset")}
            </p>
            <span className="text-muted-foreground ml-auto hidden text-xs sm:inline">
              Drop files here to upload
            </span>
            <UploadButton uploader={uploader} />
          </div>
        )}

        {lib.isLoading ? (
          <Skeleton className="h-40" />
        ) : assets.length === 0 ? (
          <p className="text-muted-foreground rounded-lg border border-dashed p-8 text-center text-sm">
            {scope === "favorites"
              ? "No favorites yet. Star an asset to find it here."
              : scope !== "all" && !filtered
                ? "This collection is empty. Select assets and use “Add to collection”."
                : "Nothing matches these filters."}
          </p>
        ) : (
          <div
            className="grid grid-cols-[repeat(auto-fill,minmax(170px,1fr))] gap-3"
            data-testid="asset-grid"
          >
            {assets.map((a) => (
              <Tile
                key={a.id}
                asset={a}
                selected={picked.has(a.id)}
                selecting={selected.length > 0}
                onToggle={() => toggle(a.id)}
                onOpen={() => setOpenId(a.id)}
              />
            ))}
          </div>
        )}
        {lib.hasNextPage && (
          <div className="flex justify-center">
            <Button
              variant="outline"
              onClick={() => void lib.fetchNextPage()}
              disabled={lib.isFetchingNextPage}
            >
              Load more ({total - assets.length} left)
            </Button>
          </div>
        )}
      </div>
      <AssetDrawer
        projectId={projectId}
        assetId={openId}
        onOpen={setOpenId}
        onClose={() => setOpenId(null)}
      />
    </div>
  );
}

type Uploader = ReturnType<typeof useUploader>;

/** Uploads files one by one (images, videos, audio; the server converts videos to H.264 MP4 and lossless or
 * compressed audio to WAV) from the Upload button or a drop onto the library. */
function useUploader(projectId: string) {
  const upload = useUploadAsset(projectId);
  const toast = useUi((s) => s.pushToast);
  const [left, setLeft] = useState(0);
  const [over, setOver] = useState(false);

  const send = async (files: File[]) => {
    if (!files.length) return;
    setLeft(files.length);
    let done = 0;
    for (const file of files) {
      try {
        await upload.mutateAsync(file);
        done += 1;
      } catch (e) {
        toast({
          title: `Couldn't upload ${file.name}`,
          description: (e as ApiError).message,
          variant: "error",
        });
      }
      setLeft((n) => n - 1);
    }
    if (done) toast({ title: `Uploaded ${plural(done, "file")}`, variant: "success" });
  };

  const hasFiles = (e: DragEvent) => e.dataTransfer.types.includes("Files");
  const dropProps = {
    onDragOver: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setOver(true);
    },
    onDragLeave: (e: DragEvent) => {
      if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false);
    },
    onDrop: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setOver(false);
      void send([...e.dataTransfer.files]);
    },
  };
  return { send, left, over, dropProps };
}

function UploadButton({ uploader }: { uploader: Uploader }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={input}
        type="file"
        multiple
        accept="image/*,video/*,audio/*"
        className="hidden"
        aria-label="Upload files"
        onChange={(e) => {
          void uploader.send([...(e.target.files ?? [])]);
          e.target.value = "";
        }}
      />
      <Button size="sm" variant="outline" disabled={uploader.left > 0} onClick={() => input.current?.click()}>
        {uploader.left > 0 ? <Loader2Icon className="animate-spin" /> : <UploadIcon />}
        {uploader.left > 0 ? `Uploading ${uploader.left}…` : "Upload"}
      </Button>
    </>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string | undefined;
  onChange: (v: string | undefined) => void;
  options: { value: string; label: string }[];
}) {
  if (!options.length && !value) return null;
  return (
    <NativeSelect
      aria-label={label}
      className={cn("w-auto max-w-48", value && "border-primary")}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value || undefined)}
    >
      <option value="">{`Any ${label.toLowerCase()}`}</option>
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </NativeSelect>
  );
}

function StoryFilters({
  projectId,
  filters,
  set,
}: {
  projectId: string;
  filters: LibraryFilters;
  set: (patch: LibraryFilters) => void;
}) {
  const { data: board } = useBoard(projectId);
  const { data: ingredients } = useIngredients(projectId);
  return (
    <>
      <FilterSelect
        label="Scene"
        value={filters.scene_id}
        onChange={(v) => set({ scene_id: v })}
        options={(board?.scenes ?? []).map((s, i) => ({
          value: s.id,
          label: `${i + 1}. ${s.title || "Scene"}`,
        }))}
      />
      <FilterSelect
        label="Ingredient"
        value={filters.featuring}
        onChange={(v) => set({ featuring: v })}
        options={(ingredients ?? [])
          .filter((i) => !i.draft)
          .map((i) => ({ value: i.id, label: `@${i.key}` }))}
      />
    </>
  );
}

function Tile({
  asset,
  selected,
  selecting,
  onToggle,
  onOpen,
}: {
  asset: Asset;
  selected: boolean;
  selecting: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const Icon = TYPE_ICON[asset.type];
  const prompt = assetPrompt(asset);
  return (
    <div
      className={cn(
        "group bg-muted relative aspect-video overflow-hidden rounded-lg border-2",
        selected ? "border-primary" : "border-transparent",
      )}
      title={prompt}
    >
      <button
        type="button"
        // While selecting, a click adds to the selection instead of opening (like a photo library).
        onClick={selecting ? onToggle : onOpen}
        className="focus-visible:ring-ring size-full text-left focus-visible:ring-2 focus-visible:outline-none"
        aria-label={`Open ${asset.type} asset`}
      >
        <AssetThumb asset={asset} className="transition-transform group-hover:scale-[1.03]" />
        <div className="absolute inset-x-0 bottom-0 flex items-center gap-1.5 bg-gradient-to-t from-black/70 to-transparent p-2 text-[11px] text-white">
          <Icon className="size-3" />
          {asset.duration_s != null ? formatDuration(asset.duration_s) : `${asset.width}×${asset.height}`}
          {asset.tags.length > 0 && <TagIcon className="size-3 opacity-80" />}
          {asset.favorite && <StarIcon className="fill-warning text-warning ml-auto size-3" />}
        </div>
      </button>
      <button
        type="button"
        role="checkbox"
        aria-checked={selected}
        aria-label="Select asset"
        onClick={onToggle}
        className={cn(
          "absolute top-1.5 left-1.5 flex size-5 items-center justify-center rounded border border-white/70 bg-black/40 text-white transition-opacity",
          selected
            ? "bg-primary border-primary opacity-100"
            : "opacity-0 group-hover:opacity-100 focus:opacity-100",
          selecting && "opacity-100",
        )}
      >
        {selected && <CheckIcon className="size-3.5" />}
      </button>
    </div>
  );
}

function SelectionBar({
  projectId,
  ids,
  scope,
  collections,
  onSelectAll,
  onClear,
}: {
  projectId: string;
  ids: string[];
  scope: Scope;
  collections: Collection[];
  onSelectAll: () => void;
  onClear: () => void;
}) {
  const muts = useAssetMutations(projectId);
  const cols = useCollectionMutations(projectId);
  const toast = useUi((s) => s.pushToast);
  const [tag, setTag] = useState("");
  const inCollection = collections.find((c) => c.id === scope);
  const fail = (e: Error) =>
    toast({ title: "Couldn't update the assets", description: e.message, variant: "error" });

  const addTag = (e: FormEvent) => {
    e.preventDefault();
    const tags = tag
      .split(",")
      .map((t) => t.trim().toLowerCase())
      .filter(Boolean);
    if (!tags.length) return;
    muts.bulk.mutate({ asset_ids: ids, add_tags: tags }, { onSuccess: () => setTag(""), onError: fail });
  };

  const addTo = (value: string) => {
    if (value === "__new") {
      const name = window.prompt("New collection name")?.trim();
      if (name) cols.create.mutate({ name, asset_ids: ids }, { onError: fail });
    } else if (value) {
      cols.addAssets.mutate({ id: value, assetIds: ids }, { onError: fail });
    }
  };

  return (
    <div
      className="bg-muted/60 flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2 text-sm"
      data-testid="selection-bar"
    >
      <Badge>{ids.length} selected</Badge>
      <Button size="sm" variant="ghost" onClick={onSelectAll}>
        Select all
      </Button>
      <NativeSelect
        aria-label="Add to collection"
        className="h-8 w-auto"
        value=""
        onChange={(e) => addTo(e.target.value)}
      >
        <option value="">Add to collection…</option>
        {collections
          .filter((c) => c.id !== scope)
          .map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        <option value="__new">New collection…</option>
      </NativeSelect>
      {inCollection && (
        <Button
          size="sm"
          variant="outline"
          onClick={() => cols.removeAssets.mutate({ id: inCollection.id, assetIds: ids }, { onError: fail })}
        >
          Remove from “{inCollection.name}”
        </Button>
      )}
      <form onSubmit={addTag} className="flex items-center gap-1">
        <Input
          aria-label="Tag selected"
          placeholder="Add tag…"
          className="h-8 w-32"
          value={tag}
          onChange={(e) => setTag(e.target.value)}
        />
        <Button size="sm" variant="outline" type="submit" disabled={!tag.trim()}>
          <TagIcon />
          Tag
        </Button>
      </form>
      <Button
        size="sm"
        variant="outline"
        onClick={() => muts.bulk.mutate({ asset_ids: ids, favorite: true }, { onError: fail })}
      >
        <StarIcon />
        Favorite
      </Button>
      <Button
        size="sm"
        variant="ghost"
        className="text-destructive"
        onClick={() => {
          if (
            window.confirm(`Delete ${plural(ids.length, "asset")}? Shots and the timeline stop using them.`)
          ) {
            muts.bulk.mutate({ asset_ids: ids, delete: true }, { onSuccess: onClear, onError: fail });
          }
        }}
      >
        <Trash2Icon />
        Delete
      </Button>
      <Button
        size="icon-sm"
        variant="ghost"
        className="ml-auto"
        aria-label="Clear selection"
        onClick={onClear}
      >
        <XIcon />
      </Button>
    </div>
  );
}

function CollectionsSidebar({
  projectId,
  scope,
  onScope,
  collections,
  total,
  favorites,
}: {
  projectId: string;
  scope: Scope;
  onScope: (s: Scope) => void;
  collections: Collection[];
  total: number | undefined;
  favorites: number | undefined;
}) {
  const muts = useCollectionMutations(projectId);
  const toast = useUi((s) => s.pushToast);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const fail = (e: Error) =>
    toast({ title: "Couldn't save the collection", description: e.message, variant: "error" });

  const create = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    muts.create.mutate(
      { name: name.trim() },
      {
        onSuccess: (c) => {
          setName("");
          setCreating(false);
          onScope(c.id);
        },
        onError: fail,
      },
    );
  };

  const item = (value: Scope, label: string, icon: ReactNode, count?: number) => (
    <button
      type="button"
      onClick={() => onScope(value)}
      aria-current={scope === value ? "true" : undefined}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
        scope === value ? "bg-accent text-accent-foreground font-medium" : "hover:bg-accent/50",
      )}
    >
      {icon}
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {count != null && <span className="text-muted-foreground text-xs">{count}</span>}
    </button>
  );

  return (
    <nav className="w-52 shrink-0 space-y-1" aria-label="Collections">
      {item("all", "All assets", <ImagesIcon className="size-4" />, total)}
      {item("favorites", "Favorites", <StarIcon className="size-4" />, favorites)}
      <div className="text-muted-foreground flex items-center justify-between px-2 pt-3 pb-1 text-xs font-medium uppercase">
        Collections
        <Button
          size="icon-sm"
          variant="ghost"
          aria-label="New collection"
          onClick={() => setCreating((c) => !c)}
        >
          <FolderPlusIcon />
        </Button>
      </div>
      {creating && (
        <form onSubmit={create} className="px-1">
          <Input
            autoFocus
            aria-label="Collection name"
            placeholder="Name, then Enter"
            className="h-8"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setCreating(false)}
          />
        </form>
      )}
      {collections.map((c) => (
        <div key={c.id} className="group relative">
          {item(c.id, c.name, <FolderIcon className="size-4" />, c.asset_count)}
          {scope === c.id && (
            <div className="bg-accent absolute top-1 right-1 flex gap-0.5">
              <Button
                size="icon-sm"
                variant="ghost"
                className="size-6"
                aria-label={`Rename ${c.name}`}
                onClick={() => {
                  const next = window.prompt("Rename collection", c.name)?.trim();
                  if (next && next !== c.name)
                    muts.update.mutate({ id: c.id, body: { name: next } }, { onError: fail });
                }}
              >
                <PencilIcon />
              </Button>
              <Button
                size="icon-sm"
                variant="ghost"
                className="size-6"
                aria-label={`Delete ${c.name}`}
                onClick={() => {
                  if (window.confirm(`Delete the collection “${c.name}”? Its assets are kept.`)) {
                    muts.remove.mutate(c.id, { onSuccess: () => onScope("all"), onError: fail });
                  }
                }}
              >
                <Trash2Icon />
              </Button>
            </div>
          )}
        </div>
      ))}
      {!collections.length && !creating && (
        <p className="text-muted-foreground px-2 text-xs">Group shots, looks or takes into collections.</p>
      )}
    </nav>
  );
}
