import { DownloadIcon, FileAudioIcon, ImageIcon, PlayIcon, StarIcon } from "lucide-react";
import { useState } from "react";
import { withToken, type Asset } from "@/api/client";
import { useToggleFavorite } from "@/api/queries";
import { AssetThumb, MediaPlayer } from "@/components/MediaPreview";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/primitives";
import { JOB_TYPE_LABELS, cn, formatDuration } from "@/lib/utils";

export function AssetViewer({ asset, onClose }: { asset: Asset | null; onClose: () => void }) {
  const fav = useToggleFavorite();
  if (!asset) return null;
  const gen = asset.generation as { job_type?: string; params?: { prompt?: string; text?: string } };
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>{JOB_TYPE_LABELS[gen.job_type ?? ""] ?? asset.type}</DialogTitle>
          <DialogDescription className="line-clamp-2">
            {gen.params?.prompt || gen.params?.text || "No prompt"}
          </DialogDescription>
        </DialogHeader>
        <MediaPlayer asset={asset} className="max-h-[60vh]" />
        <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-xs">
          {asset.width && asset.height && (
            <Badge variant="outline">
              {asset.width}×{asset.height}
            </Badge>
          )}
          {asset.duration_s != null && <Badge variant="outline">{formatDuration(asset.duration_s)}</Badge>}
          {asset.fps != null && <Badge variant="outline">{Math.round(asset.fps)} fps</Badge>}
          {asset.has_audio && asset.type === "video" && <Badge variant="outline">audio</Badge>}
          <Badge variant="outline">{asset.source}</Badge>
          <div className="ml-auto flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => fav.mutate({ id: asset.id, favorite: !asset.favorite })}
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
          </div>
        </div>
        <details className="text-xs">
          <summary className="text-muted-foreground cursor-pointer">Generation settings</summary>
          <pre className="bg-muted mt-2 max-h-60 overflow-auto rounded-md p-3">
            {JSON.stringify(asset.generation, null, 2)}
          </pre>
        </details>
      </DialogContent>
    </Dialog>
  );
}

const TYPE_ICON = { video: PlayIcon, image: ImageIcon, audio: FileAudioIcon } as const;

export function AssetGrid({ assets }: { assets: Asset[] }) {
  const [selected, setSelected] = useState<Asset | null>(null);
  // Keep the open viewer in sync with refetched data (e.g. favorite toggles).
  const current = selected ? (assets.find((a) => a.id === selected.id) ?? selected) : null;
  return (
    <>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-3" data-testid="asset-grid">
        {assets.map((a) => {
          const Icon = TYPE_ICON[a.type];
          return (
            <button
              key={a.id}
              onClick={() => setSelected(a)}
              className="group bg-muted focus-visible:ring-ring relative aspect-video overflow-hidden rounded-lg border text-left focus-visible:ring-2 focus-visible:outline-none"
              aria-label={`Open ${a.type} asset`}
            >
              <AssetThumb asset={a} className="transition-transform group-hover:scale-[1.03]" />
              <div className="absolute inset-x-0 bottom-0 flex items-center gap-1.5 bg-gradient-to-t from-black/70 to-transparent p-2 text-[11px] text-white">
                <Icon className="size-3" />
                {a.duration_s != null ? formatDuration(a.duration_s) : `${a.width}×${a.height}`}
                {a.favorite && <StarIcon className="fill-warning text-warning ml-auto size-3" />}
              </div>
            </button>
          );
        })}
      </div>
      <AssetViewer asset={current} onClose={() => setSelected(null)} />
    </>
  );
}
