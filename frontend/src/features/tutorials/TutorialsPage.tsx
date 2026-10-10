import { GraduationCapIcon, PlayIcon } from "lucide-react";
import { useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { EmptyState } from "@/components/EmptyState";
import { Badge, Card, Skeleton } from "@/components/ui/primitives";
import { cn, formatDuration } from "@/lib/utils";
import { currentChapter, useTutorials, type Tutorial } from "./tutorials";

function clock(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function TutorialList({
  tutorials,
  current,
  onSelect,
}: {
  tutorials: Tutorial[];
  current: Tutorial;
  onSelect: (id: string) => void;
}) {
  return (
    <ul className="space-y-2" aria-label="Tutorial list">
      {tutorials.map((t, i) => {
        const on = t.id === current.id;
        return (
          <li key={t.id}>
            <button
              type="button"
              aria-current={on ? "true" : undefined}
              onClick={() => onSelect(t.id)}
              className={cn(
                "hover:bg-accent flex w-full gap-3 rounded-lg border p-2 text-left transition-colors",
                on && "border-primary bg-accent",
              )}
            >
              <div className="bg-muted relative aspect-video w-28 shrink-0 overflow-hidden rounded-md">
                {t.poster && <img src={t.poster} alt="" className="size-full object-cover" />}
                <span className="absolute right-1 bottom-1 rounded bg-black/70 px-1 text-[10px] text-white tabular-nums">
                  {formatDuration(t.duration_s)}
                </span>
              </div>
              <div className="min-w-0 space-y-0.5">
                <div className="text-muted-foreground text-[11px]">Tutorial {i + 1}</div>
                <div className="line-clamp-2 text-sm font-medium">{t.title}</div>
                {on && (
                  <Badge variant="secondary" className="gap-1">
                    <PlayIcon className="size-3" />
                    Selected
                  </Badge>
                )}
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function Player({ tutorial }: { tutorial: Tutorial }) {
  const video = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const active = currentChapter(tutorial.chapters, time);

  const seek = (t: number) => {
    const v = video.current;
    if (!v) return;
    v.currentTime = t;
    setTime(t);
    void v.play()?.catch(() => undefined); // autoplay may be blocked until the user interacts
  };

  return (
    <div className="space-y-4">
      <video
        key={tutorial.id}
        ref={video}
        controls
        preload="metadata"
        poster={tutorial.poster}
        className="aspect-video w-full rounded-lg bg-black"
        onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
        data-testid="tutorial-video"
      >
        <source src={tutorial.video} type="video/mp4" />
        {tutorial.captions && (
          <track kind="captions" src={tutorial.captions} srcLang="en" label="English" default />
        )}
      </video>
      <div className="space-y-1">
        <h2 className="text-lg font-semibold">{tutorial.title}</h2>
        <p className="text-muted-foreground text-sm">{tutorial.summary}</p>
      </div>
      {tutorial.chapters.length > 0 && (
        <Card className="p-2">
          <h3 className="text-muted-foreground px-2 py-1 text-xs font-medium tracking-wide uppercase">
            Chapters
          </h3>
          <ol className="grid gap-0.5 sm:grid-cols-2" aria-label="Chapters">
            {tutorial.chapters.map((c, i) => (
              <li key={`${c.time_s}-${c.title}`}>
                <button
                  type="button"
                  onClick={() => seek(c.time_s)}
                  aria-current={i === active ? "true" : undefined}
                  className={cn(
                    "hover:bg-accent flex w-full items-baseline gap-3 rounded-md px-2 py-1.5 text-left text-sm",
                    i === active && "bg-accent font-medium",
                  )}
                >
                  <span className="text-muted-foreground w-10 shrink-0 text-xs tabular-nums">
                    {clock(c.time_s)}
                  </span>
                  {c.title}
                </button>
              </li>
            ))}
          </ol>
        </Card>
      )}
    </div>
  );
}

export function TutorialsPage() {
  const { data, isLoading } = useTutorials();
  const [params, setParams] = useSearchParams();
  const tutorials = data?.tutorials ?? [];
  const current = tutorials.find((t) => t.id === params.get("t")) ?? tutorials[0];

  return (
    <div className="mx-auto max-w-7xl space-y-5 p-6">
      <div>
        <h1 className="flex items-center gap-2 text-xl font-semibold">
          <GraduationCapIcon className="size-5" /> Tutorials
        </h1>
        <p className="text-muted-foreground text-sm">
          Narrated walkthroughs recorded in LocalFlow itself. Captions are on by default.
        </p>
      </div>
      {isLoading && <Skeleton className="aspect-video w-full" />}
      {!isLoading && !current && (
        <EmptyState icon={GraduationCapIcon} title="No tutorials yet">
          Record them with <code>uv run scripts/tutorials/make_tutorials.py</code> while LocalFlow is running.
        </EmptyState>
      )}
      {current && (
        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <Player tutorial={current} />
          <TutorialList
            tutorials={tutorials}
            current={current}
            onSelect={(id) => setParams({ t: id }, { replace: true })}
          />
        </div>
      )}
    </div>
  );
}
