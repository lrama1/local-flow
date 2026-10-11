import type { Asset } from "@/api/client";
import type { GenerateKind } from "@/api/queries";
import { KINDS, SIZE_PRESETS, type PlaygroundForm, type SizePreset } from "@/features/playground/kinds";
import { JOB_TYPE_LABELS } from "@/lib/utils";

/** The parts of `asset.generation` the library reads. */
export type Gen = {
  job_type?: string;
  model_name?: string;
  seed?: number;
  prompt_raw?: string;
  uploaded_name?: string;
  normalized?: string;
  time_s?: number;
  target_ref?: { shot_id?: string; slot?: string };
  params?: { prompt?: string; text?: string };
};

export function assetTitle(asset: Asset): string {
  const gen = asset.generation as Gen;
  if (gen.job_type) return JOB_TYPE_LABELS[gen.job_type] ?? gen.job_type;
  if (asset.source === "uploaded") return `Uploaded ${asset.type}`;
  if (asset.source === "extracted") return "Extracted frame";
  return { image: "Image", video: "Video", audio: "Audio" }[asset.type];
}

export function assetPrompt(asset: Asset): string {
  const gen = asset.generation as Gen;
  return gen.prompt_raw || gen.params?.prompt || gen.params?.text || gen.uploaded_name || "";
}

/** The size preset whose aspect ratio is closest to width×height. */
export function nearestPreset(width: number, height: number): SizePreset {
  const ratio = width / height;
  let best: SizePreset = "16:9";
  let gap = Infinity;
  for (const [key, v] of Object.entries(SIZE_PRESETS) as [SizePreset, (typeof SIZE_PRESETS)[SizePreset]][]) {
    const [w, h] = v.video;
    const d = Math.abs(Math.log(ratio / (w / h)));
    if (d < gap) [best, gap] = [key, d];
  }
  return best;
}

/** Playground settings that reproduce a generated asset ("Reuse settings", Section 13), or null when it
 * wasn't made by a Playground-capable job (uploads, exports, extracted frames). */
export function reuseSettings(asset: Asset): { kind: GenerateKind; form: Partial<PlaygroundForm> } | null {
  const gen = (asset.generation ?? {}) as Record<string, unknown>;
  const kind = gen.job_type as GenerateKind | undefined;
  if (!kind || !(kind in KINDS)) return null;
  const p = (gen.params ?? {}) as Record<string, unknown>;
  const cfg = KINDS[kind];
  const num = (v: unknown) => (typeof v === "number" ? v : undefined);
  const lines = Array.isArray(p.lines) ? (p.lines as { text?: string }[]) : [];
  // The raw prompt (with @mentions) only for Playground outputs: their mentions are Cast library keys,
  // which the Playground resolves again. A project's shots and ingredients mention project keys the
  // Playground doesn't know, and their sent prompt adds the camera move, style and cast details, so the
  // resolved prompt is what reproduces them.
  const playgroundMade = !gen.target_ref;
  const prompt =
    (playgroundMade && typeof gen.prompt_raw === "string" && gen.prompt_raw) ||
    (cfg.promptField === "text" && lines.length ? lines.map((l) => l.text ?? "").join("\n") : "") ||
    String(p[cfg.promptField] ?? p.prompt ?? "");
  const form: Partial<PlaygroundForm> = { prompt, negative: String(p.negative_prompt ?? "") };
  const seed = num(gen.seed) ?? num(p.seed); // the seed this output actually used
  if (seed !== undefined) form.seed = String(seed);
  if (typeof gen.adapter_key === "string") form.adapterKey = gen.adapter_key;
  if (cfg.voice && typeof p.voice_id === "string") form.voiceId = p.voice_id;
  const width = num(p.width) ?? asset.width ?? undefined;
  const height = num(p.height) ?? asset.height ?? undefined;
  if (cfg.size && width && height) form.preset = nearestPreset(width, height);
  const images = num(p.num_images);
  if (cfg.numImages && images) form.numImages = images;
  const fps = num(p.fps);
  if (cfg.duration) {
    const frames = num(p.num_frames);
    const duration = num(p.duration_s) ?? (frames && fps ? frames / fps : (asset.duration_s ?? undefined));
    if (duration) form.duration = Math.round(duration * 100) / 100;
    if (fps && cfg.output === "video") form.fps = fps;
  }
  if (cfg.withAudio && typeof p.with_audio === "boolean") form.withAudio = p.with_audio;
  if (typeof p.model_entry_id === "string") form.modelId = p.model_entry_id;
  if (Array.isArray(p.loras)) {
    form.loras = (p.loras as { model_entry_id: string; weight?: number }[]).map((l) => ({
      id: l.model_entry_id,
      weight: l.weight ?? 1,
    }));
  }
  if (p.extra && typeof p.extra === "object") form.extra = p.extra as Record<string, unknown>;
  const assets: Record<string, string[]> = {};
  for (const input of cfg.assets ?? []) {
    const v = p[input.field];
    const ids = Array.isArray(v) ? (v as string[]) : typeof v === "string" ? [v] : [];
    if (ids.length) assets[input.field] = ids;
  }
  if (Object.keys(assets).length) form.assets = assets;
  return { kind, form };
}
