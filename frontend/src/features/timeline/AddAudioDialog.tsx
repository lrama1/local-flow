import { Loader2Icon, PlusIcon, UploadIcon } from "lucide-react";
import { useRef } from "react";
import type { Asset } from "@/api/client";
import { fileUrl } from "@/api/imagegen";
import { useUploadAsset } from "@/api/timeline";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { formatDuration } from "@/lib/utils";
import { assetLabel, timecode, type Track } from "./math";
import { MusicGenerator } from "./MusicGenerator";

export function AddAudioDialog({
  track,
  media,
  projectId,
  playhead,
  fps,
  onAdd,
  onClose,
}: {
  track: Track;
  media: Map<string, Asset> | undefined;
  projectId: string;
  playhead: number;
  fps: number;
  onAdd: (asset: Asset) => void;
  onClose: () => void;
}) {
  const upload = useUploadAsset(projectId);
  const input = useRef<HTMLInputElement>(null);
  const audio = [...(media?.values() ?? [])]
    .filter((a) => a.type === "audio" && a.duration_s)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Add audio to {track.name}</DialogTitle>
          <DialogDescription>
            The clip starts at the playhead ({timecode(playhead, fps)}); drag it on the track to move it.
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-80 space-y-2 overflow-y-auto" data-testid="audio-choices">
          {audio.map((a) => (
            <div key={a.id} className="flex items-center gap-2 rounded-md border p-2">
              <div className="min-w-0 flex-1 space-y-1">
                <p className="truncate text-xs font-medium">{assetLabel(a)}</p>
                <audio src={fileUrl(projectId, a.id)} controls preload="none" className="h-7 w-full" />
              </div>
              <span className="text-muted-foreground text-xs tabular-nums">
                {formatDuration(a.duration_s)}
              </span>
              <Button size="sm" aria-label={`Add ${assetLabel(a)}`} onClick={() => onAdd(a)}>
                <PlusIcon />
                Add
              </Button>
            </div>
          ))}
          {audio.length === 0 && (
            <p className="text-muted-foreground text-sm">
              No audio in this project yet. Upload a file below, or generate dialogue and sound effects in a
              shot&apos;s Audio section (Shots board).
            </p>
          )}
        </div>
        {track.id === "a_music" && <MusicGenerator projectId={projectId} />}
        <div className="flex items-center gap-2 border-t pt-3">
          <input
            ref={input}
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac"
            className="hidden"
            aria-label="Audio file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) upload.mutate(file, { onSuccess: (a) => a.type === "audio" && onAdd(a) });
              e.target.value = "";
            }}
          />
          <Button
            variant="outline"
            size="sm"
            disabled={upload.isPending}
            onClick={() => input.current?.click()}
          >
            {upload.isPending ? <Loader2Icon className="animate-spin" /> : <UploadIcon />}
            Upload audio
          </Button>
          {upload.error && <p className="text-destructive text-xs">{upload.error.message}</p>}
        </div>
      </DialogContent>
    </Dialog>
  );
}
