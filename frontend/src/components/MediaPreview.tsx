import { FileAudioIcon } from "lucide-react";
import { withToken, type Asset } from "@/api/client";
import { cn } from "@/lib/utils";

/** Full-size playable/viewable media for an asset. */
export function MediaPlayer({ asset, className }: { asset: Asset; className?: string }) {
  const src = withToken(asset.url);
  if (asset.type === "video") {
    return (
      <video
        key={asset.id}
        src={src}
        poster={withToken(asset.thumb_url)}
        controls
        autoPlay
        loop
        className={cn("w-full rounded-lg bg-black", className)}
        data-testid="video-player"
      />
    );
  }
  if (asset.type === "image") {
    return <img src={src} alt="" className={cn("w-full rounded-lg object-contain", className)} />;
  }
  return (
    <div className={cn("bg-muted flex flex-col items-center gap-3 rounded-lg p-6", className)}>
      <FileAudioIcon className="text-muted-foreground size-10" />
      <audio src={src} controls className="w-full" />
    </div>
  );
}

/** Small thumbnail tile. */
export function AssetThumb({ asset, className }: { asset: Asset; className?: string }) {
  if (asset.thumb_url) {
    return (
      <img
        src={withToken(asset.thumb_url)}
        alt=""
        loading="lazy"
        className={cn("size-full object-cover", className)}
      />
    );
  }
  return (
    <div className={cn("bg-muted flex size-full items-center justify-center", className)}>
      <FileAudioIcon className="text-muted-foreground size-8" />
    </div>
  );
}
