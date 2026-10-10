import { Loader2Icon, MusicIcon } from "lucide-react";
import { useState } from "react";
import { useGenerateMusic } from "@/api/audio";
import { useBoard } from "@/api/story";
import { JobProgress } from "@/components/JobProgress";
import { Button } from "@/components/ui/button";
import { Input, Label, NativeSelect } from "@/components/ui/primitives";
import { gateLicense } from "@/features/models/gateLicense";

/** Music for the whole film or one scene (Section 12.1), from the board's music cues unless a prompt is
 * given. The finished track lands on the Music track at the film's or scene's start. */
export function MusicGenerator({ projectId }: { projectId: string }) {
  const { data: board } = useBoard(projectId);
  const music = useGenerateMusic(projectId);
  const [scope, setScope] = useState("film"); // "film" | scene id
  const [prompt, setPrompt] = useState("");
  const scenes = board?.scenes ?? [];
  const inScope = scope === "film" ? scenes : scenes.filter((s) => s.id === scope);
  const cues = [
    ...new Set(inScope.flatMap((s) => s.shots.map((sh) => sh.music_cue?.trim()).filter(Boolean))),
  ];
  const out = music.data;

  const run = async () => {
    try {
      await music.mutateAsync({
        scope: scope === "film" ? "film" : "scene",
        scene_id: scope === "film" ? null : scope,
        prompt: prompt.trim() || null,
      });
    } catch (e) {
      gateLicense(e, () => void run());
    }
  };

  return (
    <div className="space-y-2 rounded-lg border p-3" data-testid="music-generator">
      <Label className="text-foreground flex items-center gap-1.5">
        <MusicIcon className="size-3.5" /> Generate music
      </Label>
      <div className="flex gap-2">
        <NativeSelect
          aria-label="Music for"
          className="w-44 shrink-0"
          value={scope}
          onChange={(e) => setScope(e.target.value)}
        >
          <option value="film">The whole film</option>
          {scenes.map((s, i) => (
            <option key={s.id} value={s.id}>
              Scene {i + 1}: {s.title}
            </option>
          ))}
        </NativeSelect>
        <Input
          aria-label="Music prompt"
          placeholder={cues.length ? cues.slice(0, 3).join("; ") : "sparse piano, melancholic"}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <Button size="sm" disabled={music.isPending} onClick={() => void run()}>
          {music.isPending ? <Loader2Icon className="animate-spin" /> : <MusicIcon />}
          Generate
        </Button>
      </div>
      <p className="text-muted-foreground text-xs">
        {prompt.trim()
          ? "Uses your prompt."
          : cues.length
            ? "Uses the shots' music cues."
            : "No music cues on the board yet."}{" "}
        It&apos;s added to the Music track when it&apos;s done.
      </p>
      {out && (
        <div className="space-y-1 text-xs">
          <p className="text-muted-foreground">
            {out.duration_s.toFixed(1)} s of “{out.prompt}”{out.model_name ? ` with ${out.model_name}` : ""}.
          </p>
          {out.notes.map((n) => (
            <p key={n} className="text-warning">
              {n}
            </p>
          ))}
          <JobProgress jobId={out.job.id} showPreview={false} />
        </div>
      )}
      {music.error && <p className="text-destructive text-xs">{music.error.message}</p>}
    </div>
  );
}
