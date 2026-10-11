import {
  CopyIcon,
  DownloadIcon,
  FastForwardIcon,
  ImageIcon,
  Loader2Icon,
  ScissorsIcon,
  StarIcon,
  Trash2Icon,
  XIcon,
} from "lucide-react";
import { useRef, useState, type FormEvent } from "react";
import { useNavigate } from "react-router";
import { withToken, type Asset } from "@/api/client";
import { EXTRA_VIEWS, VIEW_LABELS, useSetView, type CharacterView } from "@/api/characters";
import { useAsset, useSelectForShot, useSetPrimary } from "@/api/imagegen";
import {
  useAssetCollections,
  useAssetMutations,
  useCollectionMutations,
  useCollections,
} from "@/api/library";
import { useBoard, useIngredients } from "@/api/story";
import { useExtendShot } from "@/api/video";
import { AssetThumb, MediaPlayer } from "@/components/MediaPreview";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge, Input, Label, NativeSelect, Skeleton } from "@/components/ui/primitives";
import { cn, formatDuration } from "@/lib/utils";
import { useUi } from "@/stores/ui";
import { assetPrompt, assetTitle, reuseSettings, type Gen } from "./reuse";

/** The asset detail drawer (Section 13): preview, metadata, tags, collections and reuse actions. */
export function AssetDrawer({
  projectId,
  assetId,
  onOpen,
  onClose,
}: {
  projectId: string;
  assetId: string | null;
  onOpen: (id: string) => void;
  onClose: () => void;
}) {
  const { data: asset } = useAsset(assetId);
  if (!assetId) return null;
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-5xl" data-testid="asset-drawer">
        {asset ? (
          <DrawerBody key={asset.id} projectId={projectId} asset={asset} onOpen={onOpen} onClose={onClose} />
        ) : (
          <>
            <DialogTitle className="sr-only">Loading asset</DialogTitle>
            <Skeleton className="h-80" />
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

function DrawerBody({
  projectId,
  asset,
  onOpen,
  onClose,
}: {
  projectId: string;
  asset: Asset;
  onOpen: (id: string) => void;
  onClose: () => void;
}) {
  const gen = asset.generation as Gen;
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const muts = useAssetMutations(projectId);
  const toast = useUi((s) => s.pushToast);
  const reuse = reuseSettings(asset);
  const isVisual = asset.type !== "audio";

  /** The asset itself for images; for videos, a new image of the frame under the playhead. */
  const imageFor = async (): Promise<string> => {
    if (asset.type === "image") return asset.id;
    const frame = await muts.extractFrame.mutateAsync({
      id: asset.id,
      timeS: videoRef.current?.currentTime ?? 0,
    });
    return frame.id;
  };

  // Frames extracted in this session, shown inline: toasts sit outside the modal and can't be clicked.
  const [frames, setFrames] = useState<Asset[]>([]);
  const extractFrame = () =>
    muts.extractFrame.mutate(
      { id: asset.id, timeS: videoRef.current?.currentTime ?? 0 },
      {
        onSuccess: (frame) => setFrames((f) => [...f, frame]),
        onError: (e) =>
          toast({ title: "Couldn't extract the frame", description: e.message, variant: "error" }),
      },
    );

  return (
    <>
      <DialogHeader>
        <DialogTitle>{assetTitle(asset)}</DialogTitle>
        <DialogDescription className="line-clamp-3">{assetPrompt(asset) || "No prompt"}</DialogDescription>
      </DialogHeader>
      <div className="grid gap-5 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="min-w-0 space-y-3">
          <MediaPlayer asset={asset} videoRef={videoRef} className="max-h-[55vh]" />
          <div className="text-muted-foreground flex flex-wrap items-center gap-1.5 text-xs">
            {asset.width && asset.height && (
              <Badge variant="outline">
                {asset.width}×{asset.height}
              </Badge>
            )}
            {asset.duration_s != null && <Badge variant="outline">{formatDuration(asset.duration_s)}</Badge>}
            {asset.fps != null && <Badge variant="outline">{Math.round(asset.fps)} fps</Badge>}
            {asset.has_audio && asset.type === "video" && <Badge variant="outline">audio</Badge>}
            <Badge variant="outline">{asset.source}</Badge>
            {gen.model_name && <Badge variant="outline">{gen.model_name}</Badge>}
            {gen.seed != null && <Badge variant="outline">seed {gen.seed}</Badge>}
            {gen.normalized && <Badge variant="outline">{gen.normalized}</Badge>}
            <span className="ml-auto">{new Date(asset.created_at).toLocaleString()}</span>
          </div>
          {asset.parent_asset_ids.length > 0 && (
            <div className="space-y-1">
              <p className="text-muted-foreground text-xs">Made from</p>
              <div className="flex flex-wrap gap-1.5">
                {asset.parent_asset_ids.map((id) => (
                  <ParentThumb key={id} id={id} onOpen={onOpen} />
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="min-w-0 space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => muts.bulk.mutate({ asset_ids: [asset.id], favorite: !asset.favorite })}
            >
              <StarIcon className={cn(asset.favorite && "fill-warning text-warning")} />
              {asset.favorite ? "Favorited" : "Favorite"}
            </Button>
            <Button size="sm" variant="outline" asChild>
              <a href={withToken(asset.url)} download>
                <DownloadIcon />
                Download
              </a>
            </Button>
            {reuse && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  useUi.getState().setPlaygroundPrefill(reuse);
                  void navigate("/playground");
                }}
              >
                <CopyIcon />
                Reuse settings
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              className="text-destructive"
              onClick={() => {
                if (window.confirm("Delete this asset? Shots and the timeline stop using it.")) {
                  muts.remove.mutate(asset.id, { onSuccess: onClose });
                }
              }}
            >
              <Trash2Icon />
              Delete
            </Button>
          </div>

          {asset.type === "video" && (
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={extractFrame}
                disabled={muts.extractFrame.isPending}
              >
                {muts.extractFrame.isPending ? <Loader2Icon className="animate-spin" /> : <ScissorsIcon />}
                Extract frame
              </Button>
              <ExtendButton asset={asset} projectId={projectId} />
              {frames.length > 0 && (
                <div className="flex w-full flex-wrap items-center gap-1.5" data-testid="extracted-frames">
                  <span className="text-muted-foreground text-xs">Saved frames:</span>
                  {frames.map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => onOpen(f.id)}
                      aria-label={`Open frame at ${formatDuration(Number((f.generation as Gen).time_s ?? 0))}`}
                      className="bg-muted h-10 w-16 overflow-hidden rounded border"
                    >
                      <AssetThumb asset={f} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {isVisual && <UseAsKeyframe projectId={projectId} asset={asset} imageFor={imageFor} />}
          {isVisual && <UseAsReference projectId={projectId} asset={asset} imageFor={imageFor} />}
          <TagsEditor asset={asset} onSave={(tags) => muts.tags.mutate({ id: asset.id, tags })} />
          <CollectionToggles projectId={projectId} asset={asset} />

          <details className="text-xs">
            <summary className="text-muted-foreground cursor-pointer">Generation settings</summary>
            <pre
              className="bg-muted mt-2 max-h-60 overflow-auto rounded-md p-3"
              data-testid="generation-json"
            >
              {JSON.stringify(asset.generation, null, 2)}
            </pre>
          </details>
        </div>
      </div>
    </>
  );
}

function ParentThumb({ id, onOpen }: { id: string; onOpen: (id: string) => void }) {
  const { data: parent } = useAsset(id);
  return (
    <button
      type="button"
      onClick={() => onOpen(id)}
      aria-label="Open source asset"
      className="bg-muted size-14 overflow-hidden rounded-md border"
    >
      {parent && <AssetThumb asset={parent} />}
    </button>
  );
}

function ExtendButton({ asset, projectId }: { asset: Asset; projectId: string }) {
  const navigate = useNavigate();
  const extend = useExtendShot(projectId);
  const toast = useUi((s) => s.pushToast);
  const shotId = (asset.generation as Gen).target_ref?.shot_id;
  return (
    <Button
      size="sm"
      variant="secondary"
      disabled={extend.isPending}
      onClick={() => {
        if (shotId) {
          // A shot's take: extend it in place so the result lands back on the shot (Section 10.4).
          extend.mutate(
            { shotId, body: { asset_id: asset.id } },
            {
              onSuccess: () =>
                toast({ title: "Extending the shot", description: "Queued", variant: "success" }),
              onError: (e) => toast({ title: "Couldn't extend", description: e.message, variant: "error" }),
            },
          );
        } else {
          useUi.getState().setPlaygroundPrefill({
            kind: "extend",
            form: { prompt: assetPrompt(asset), assets: { video_asset_id: [asset.id] } },
          });
          void navigate("/playground");
        }
      }}
    >
      <FastForwardIcon />
      {shotId ? "Extend shot" : "Extend in Playground"}
    </Button>
  );
}

function UseAsKeyframe({
  projectId,
  asset,
  imageFor,
}: {
  projectId: string;
  asset: Asset;
  imageFor: () => Promise<string>;
}) {
  const { data: board } = useBoard(projectId);
  const select = useSelectForShot(projectId);
  const toast = useUi((s) => s.pushToast);
  const [shotId, setShotId] = useState("");
  const [slot, setSlot] = useState<"keyframe" | "end_keyframe">("keyframe");
  const [busy, setBusy] = useState(false);
  const shots =
    board?.scenes.flatMap((sc, si) =>
      sc.shots.map((sh, i) => ({
        id: sh.id,
        label: `Shot ${si + 1}.${i + 1}${sc.title ? ` · ${sc.title}` : ""}`,
      })),
    ) ?? [];
  if (!shots.length) return null;

  const apply = async (e: FormEvent) => {
    e.preventDefault();
    if (!shotId) return;
    setBusy(true);
    try {
      const assetId = await imageFor();
      await select.mutateAsync({ shotId, slot, assetId });
      const label = shots.find((s) => s.id === shotId)?.label ?? "the shot";
      toast({
        title: `Set as ${slot === "keyframe" ? "keyframe" : "end frame"} of ${label}`,
        variant: "success",
      });
    } catch (err) {
      toast({ title: "Couldn't set the keyframe", description: (err as Error).message, variant: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={(e) => void apply(e)} className="space-y-1.5" aria-label="Use as keyframe">
      <Label>Use {asset.type === "video" ? "this frame " : ""}as keyframe</Label>
      <div className="flex gap-2">
        <NativeSelect aria-label="Shot" value={shotId} onChange={(e) => setShotId(e.target.value)}>
          <option value="">Pick a shot…</option>
          {shots.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </NativeSelect>
        <NativeSelect
          aria-label="Frame"
          className="w-28"
          value={slot}
          onChange={(e) => setSlot(e.target.value as typeof slot)}
        >
          <option value="keyframe">Start</option>
          <option value="end_keyframe">End</option>
        </NativeSelect>
        <Button size="sm" type="submit" variant="outline" disabled={!shotId || busy}>
          {busy ? <Loader2Icon className="animate-spin" /> : <ImageIcon />}
          Use
        </Button>
      </div>
    </form>
  );
}

function UseAsReference({
  projectId,
  asset,
  imageFor,
}: {
  projectId: string;
  asset: Asset;
  imageFor: () => Promise<string>;
}) {
  const { data: ingredients } = useIngredients(projectId);
  const setPrimary = useSetPrimary(projectId);
  const setView = useSetView(projectId);
  const toast = useUi((s) => s.pushToast);
  const [ingredientId, setIngredientId] = useState("");
  const [view, setViewName] = useState<CharacterView>("primary");
  const [busy, setBusy] = useState(false);
  const list = (ingredients ?? []).filter((i) => !i.draft);
  if (!list.length) return null;
  const ing = list.find((i) => i.id === ingredientId);
  const views: CharacterView[] =
    ing?.kind === "character" ? ["primary", ...EXTRA_VIEWS.map((v) => v.view)] : [];

  const apply = async (e: FormEvent) => {
    e.preventDefault();
    if (!ing) return;
    setBusy(true);
    try {
      const assetId = await imageFor();
      if (view === "primary" || ing.kind !== "character")
        await setPrimary.mutateAsync({ id: ing.id, assetId });
      else await setView.mutateAsync({ id: ing.id, view, assetId });
      toast({ title: `${ing.name}: reference updated`, variant: "success" });
    } catch (err) {
      toast({ title: "Couldn't set the reference", description: (err as Error).message, variant: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={(e) => void apply(e)} className="space-y-1.5" aria-label="Use as reference">
      <Label>Use {asset.type === "video" ? "this frame " : ""}as ingredient reference</Label>
      <div className="flex gap-2">
        <NativeSelect
          aria-label="Ingredient"
          value={ingredientId}
          onChange={(e) => {
            setIngredientId(e.target.value);
            setViewName("primary");
          }}
        >
          <option value="">Pick an ingredient…</option>
          {list.map((i) => (
            <option key={i.id} value={i.id}>
              @{i.key} · {i.name}
            </option>
          ))}
        </NativeSelect>
        {views.length > 0 && (
          <NativeSelect
            aria-label="View"
            className="w-36"
            value={view}
            onChange={(e) => setViewName(e.target.value as CharacterView)}
          >
            {views.map((v) => (
              <option key={v} value={v}>
                {VIEW_LABELS[v] ?? v}
              </option>
            ))}
          </NativeSelect>
        )}
        <Button size="sm" type="submit" variant="outline" disabled={!ing || busy}>
          {busy ? <Loader2Icon className="animate-spin" /> : <ImageIcon />}
          Use
        </Button>
      </div>
    </form>
  );
}

function TagsEditor({ asset, onSave }: { asset: Asset; onSave: (tags: string[]) => void }) {
  const [draft, setDraft] = useState("");
  const add = (e: FormEvent) => {
    e.preventDefault();
    const fresh = draft
      .split(",")
      .map((t) => t.trim().toLowerCase())
      .filter((t) => t && !asset.tags.includes(t));
    if (fresh.length) onSave([...asset.tags, ...new Set(fresh)]);
    setDraft("");
  };
  return (
    <div className="space-y-1.5">
      <Label htmlFor="asset-tag-input">Tags</Label>
      <div className="flex flex-wrap items-center gap-1.5">
        {asset.tags.map((t) => (
          <Badge key={t} variant="secondary" className="gap-1">
            {t}
            <button
              type="button"
              aria-label={`Remove tag ${t}`}
              onClick={() => onSave(asset.tags.filter((x) => x !== t))}
            >
              <XIcon className="size-3" />
            </button>
          </Badge>
        ))}
        <form onSubmit={add} className="flex-1">
          <Input
            id="asset-tag-input"
            className="h-7 min-w-28 text-xs"
            placeholder="Add a tag…"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
        </form>
      </div>
    </div>
  );
}

function CollectionToggles({ projectId, asset }: { projectId: string; asset: Asset }) {
  const { data: collections } = useCollections(projectId);
  const { data: memberOf } = useAssetCollections(asset.id);
  const muts = useCollectionMutations(projectId);
  if (!collections?.length) return null;
  return (
    <fieldset className="space-y-1.5">
      <legend className="text-sm font-medium">Collections</legend>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {collections.map((c) => {
          const on = memberOf?.includes(c.id) ?? false;
          return (
            <label key={c.id} className="flex items-center gap-1.5 text-sm">
              <input
                type="checkbox"
                checked={on}
                onChange={() =>
                  (on ? muts.removeAssets : muts.addAssets).mutate({ id: c.id, assetIds: [asset.id] })
                }
              />
              {c.name}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
