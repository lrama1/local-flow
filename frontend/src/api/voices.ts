import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, unwrap, withToken, type Job, type Schemas } from "./client";
import { useJob } from "./queries";

export type Voice = Schemas["VoiceOut"];
export type VoiceCreate = Schemas["VoiceCreate"];
export type VoiceUpdate = Schemas["VoiceUpdate"];
export type VoiceDesignOut = Schemas["VoiceDesignOut"];
export type PreviewOut = Schemas["PreviewOut"];
/** Custom-voice params for blend-capable TTS (Kokoro, FakeEngine): `VoiceParams` in the backend. */
export interface VoiceParams {
  mix: { voice: string; weight: number }[];
  base_weight: number;
  speed: number;
}
export const DEFAULT_PARAMS: VoiceParams = { mix: [], base_weight: 1, speed: 1 };

export const vok = {
  voices: ["voices"] as const,
  suggest: (hints: Record<string, string>) => ["voices", "suggest", hints] as const,
};

const LANGUAGES: Record<string, string> = {
  "en-US": "English (US)",
  "en-GB": "English (UK)",
  es: "Spanish",
  fr: "French",
  hi: "Hindi",
  it: "Italian",
  "pt-BR": "Portuguese (BR)",
};
export const languageName = (code: string) => LANGUAGES[code] ?? code;

export function asParams(raw: Record<string, unknown> | null | undefined): VoiceParams {
  const p = (raw ?? {}) as Partial<VoiceParams>;
  return {
    mix: Array.isArray(p.mix) ? p.mix : [],
    base_weight: typeof p.base_weight === "number" ? p.base_weight : 1,
    speed: typeof p.speed === "number" ? p.speed : 1,
  };
}

/** "Old Tom · custom" / "Heart · English (US) · female · A". */
export function voiceLabel(v: Voice): string {
  if (v.kind === "custom") return `${v.name} · custom${v.base_voice_name ? ` (${v.base_voice_name})` : ""}`;
  return [v.name, v.gender, v.quality].filter(Boolean).join(" · ");
}

export function useVoices() {
  return useQuery({
    queryKey: vok.voices,
    queryFn: () => unwrap(api.GET("/api/voices")),
    staleTime: 30_000,
  });
}

/** The best available built-in voice for a character's voice hints. */
export function useSuggestVoice(hints: Record<string, string> | null | undefined, enabled = true) {
  const h = { gender: hints?.gender ?? "", age: hints?.age ?? "", tone: hints?.tone ?? "" };
  return useQuery({
    queryKey: vok.suggest(h),
    enabled: enabled && !!(h.gender || h.age || h.tone),
    queryFn: () => unwrap(api.GET("/api/voices/suggest", { params: { query: h } })),
    staleTime: 60_000,
  });
}

export function useVoiceMutations() {
  const qc = useQueryClient();
  const done = () => qc.invalidateQueries({ queryKey: vok.voices });
  return {
    create: useMutation({
      mutationFn: (body: VoiceCreate) => unwrap(api.POST("/api/voices", { body })),
      onSuccess: done,
    }),
    update: useMutation({
      mutationFn: ({ id, body }: { id: string; body: VoiceUpdate }) =>
        unwrap(api.PATCH("/api/voices/{voice_id}", { params: { path: { voice_id: id } }, body })),
      onSuccess: done,
    }),
    remove: useMutation({
      mutationFn: (id: string) =>
        unwrap(api.DELETE("/api/voices/{voice_id}", { params: { path: { voice_id: id } } })),
      onSuccess: () => {
        void done();
        void qc.invalidateQueries({ queryKey: ["ingredients"] });
      },
    }),
    design: useMutation({
      mutationFn: (body: Schemas["VoiceDesignBody"]) => unwrap(api.POST("/api/voices/design", { body })),
    }),
  };
}

/** What to say with which voice: a saved voice (cached sample), or unsaved settings on a base voice. */
export type AuditionTarget =
  { voiceId: string; text?: string } | { baseVoiceId: string; params: VoiceParams; text?: string };

function jobAudioUrl(job: Job): string | undefined {
  const id = (job.result?.["asset_ids"] as string[] | undefined)?.[0];
  return id && job.project_id ? `/api/files/${job.project_id}/${id}` : undefined;
}

/** Plays voice samples: cached ones at once, otherwise when the TTS job finishes. One at a time. */
export function useAudition() {
  const audio = useRef<HTMLAudioElement | null>(null);
  const handled = useRef<string | null>(null); // the job whose audio already played
  const [key, setKey] = useState<string | null>(null); // what's loading/playing
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const { data: job } = useJob(jobId ?? undefined);
  const mine = job && job.id === jobId ? job : undefined;
  const finished = !!mine && ["completed", "failed", "cancelled"].includes(mine.status);
  const loading = !!jobId && !finished;
  const jobError =
    mine && (mine.status === "failed" || mine.status === "cancelled")
      ? (mine.error ?? "The voice sample failed.")
      : null;

  const play = useCallback((url: string) => {
    audio.current?.pause();
    const el = new Audio(withToken(url));
    audio.current = el;
    el.onplay = () => setPlaying(true);
    el.onended = () => setPlaying(false);
    el.onpause = () => setPlaying(false);
    void el.play().catch(() => setPlaying(false));
  }, []);

  useEffect(() => {
    if (!mine || mine.status !== "completed" || handled.current === mine.id) return;
    handled.current = mine.id;
    const url = jobAudioUrl(mine);
    if (url) play(url);
  }, [mine, play]);

  useEffect(() => () => audio.current?.pause(), []);

  const start = useCallback(
    async (target: AuditionTarget, id: string) => {
      setError(null);
      setJobId(null);
      setKey(id);
      audio.current?.pause();
      try {
        const out =
          "voiceId" in target
            ? await unwrap(
                api.POST("/api/voices/{voice_id}/preview", {
                  params: { path: { voice_id: target.voiceId } },
                  body: { text: target.text || null },
                }),
              )
            : await unwrap(
                api.POST("/api/voices/audition", {
                  body: {
                    base_voice_id: target.baseVoiceId,
                    params: target.params as unknown as Record<string, unknown>,
                    text: target.text || null,
                  },
                }),
              );
        if (out.url) play(out.url);
        else if (out.job) setJobId(out.job.id);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [play],
  );

  const stop = useCallback(() => {
    audio.current?.pause();
    setPlaying(false);
  }, []);

  return {
    start,
    stop,
    /** Which target is loading or playing (the `id` passed to start). */
    active: loading || playing ? key : null,
    loading,
    playing,
    progress: mine?.progress ?? 0,
    error: error ?? jobError,
  };
}

/** Groups available voices: custom first, then built-ins by language. */
export function groupVoices(voices: Voice[], builtinOnly = false): { label: string; voices: Voice[] }[] {
  const ok = voices.filter((v) => v.available && (!builtinOnly || v.kind === "builtin"));
  const groups: { label: string; voices: Voice[] }[] = [];
  const custom = ok.filter((v) => v.kind === "custom");
  if (custom.length) groups.push({ label: "Custom voices", voices: custom });
  const langs = [...new Set(ok.filter((v) => v.kind === "builtin").map((v) => v.language))];
  for (const lang of langs) {
    groups.push({
      label: languageName(lang),
      voices: ok.filter((v) => v.kind === "builtin" && v.language === lang),
    });
  }
  return groups;
}
