import type { GenerateKind } from "@/api/queries";

export interface AssetInput {
  field: string;
  label: string;
  type: "image" | "video";
  multiple?: boolean;
}

export interface KindConfig {
  label: string;
  output: "image" | "video" | "audio";
  promptField: "prompt" | "text";
  promptRequired?: boolean;
  negative?: boolean;
  size?: boolean;
  duration?: boolean;
  numImages?: boolean;
  withAudio?: boolean;
  assets?: AssetInput[];
}

export const KINDS: Record<GenerateKind, KindConfig> = {
  t2i: {
    label: "Text → image",
    output: "image",
    promptField: "prompt",
    negative: true,
    size: true,
    numImages: true,
  },
  image_edit: {
    label: "Image edit",
    output: "image",
    promptField: "prompt",
    negative: true,
    size: true,
    numImages: true,
    assets: [{ field: "reference_asset_ids", label: "Reference images", type: "image", multiple: true }],
  },
  t2v: {
    label: "Text → video",
    output: "video",
    promptField: "prompt",
    negative: true,
    size: true,
    duration: true,
    withAudio: true,
  },
  i2v: {
    label: "Image → video",
    output: "video",
    promptField: "prompt",
    negative: true,
    size: true,
    duration: true,
    withAudio: true,
    assets: [{ field: "image_asset_id", label: "Start image", type: "image" }],
  },
  flf2v: {
    label: "First/last frame",
    output: "video",
    promptField: "prompt",
    negative: true,
    size: true,
    duration: true,
    assets: [
      { field: "first_asset_id", label: "First frame", type: "image" },
      { field: "last_asset_id", label: "Last frame", type: "image" },
    ],
  },
  ref2v: {
    label: "Reference → video",
    output: "video",
    promptField: "prompt",
    negative: true,
    size: true,
    duration: true,
    assets: [{ field: "reference_asset_ids", label: "Reference images", type: "image", multiple: true }],
  },
  extend: {
    label: "Extend",
    output: "video",
    promptField: "prompt",
    size: true,
    duration: true,
    assets: [{ field: "video_asset_id", label: "Video to extend", type: "video" }],
  },
  tts: { label: "Speech", output: "audio", promptField: "text", promptRequired: true },
  v2a: {
    label: "Video → audio",
    output: "audio",
    promptField: "prompt",
    assets: [{ field: "video_asset_id", label: "Video", type: "video" }],
  },
  music: { label: "Music", output: "audio", promptField: "prompt", duration: true },
};

export const KIND_ORDER: GenerateKind[] = [
  "t2v",
  "t2i",
  "i2v",
  "flf2v",
  "extend",
  "image_edit",
  "tts",
  "v2a",
  "music",
];

export const SIZE_PRESETS = {
  "16:9": { image: [1344, 768], video: [1280, 720] },
  "9:16": { image: [768, 1344], video: [720, 1280] },
  "1:1": { image: [1024, 1024], video: [960, 960] },
} as const;
export type SizePreset = keyof typeof SIZE_PRESETS;

export interface PlaygroundForm {
  prompt: string;
  negative: string;
  preset: SizePreset;
  numImages: number;
  duration: number;
  fps: number;
  withAudio: boolean;
  seed: string;
  adapterKey: string;
  /** Model registry entry (image and video kinds); empty = the server's default model. */
  modelId: string;
  loras: { id: string; weight: number }[];
  extra: Record<string, unknown>;
  assets: Record<string, string[]>;
}

export const DEFAULT_FORM: PlaygroundForm = {
  prompt: "",
  negative: "",
  preset: "16:9",
  numImages: 1,
  duration: 4,
  fps: 24,
  withAudio: false,
  seed: "",
  adapterKey: "",
  modelId: "",
  loras: [],
  extra: {},
  assets: {},
};

/** Builds the request body for `POST /api/generate/{kind}`; returns an error string if incomplete. */
export function buildBody(
  kind: GenerateKind,
  form: PlaygroundForm,
  projectId: string | undefined,
): { body: Record<string, unknown> } | { error: string } {
  const cfg = KINDS[kind];
  const body: Record<string, unknown> = { project_id: projectId, extra: form.extra };
  const text = form.prompt.trim();
  if (cfg.promptRequired && !text) return { error: "Enter some text" };
  body[cfg.promptField] = text;
  if (cfg.negative && form.negative.trim()) body.negative_prompt = form.negative.trim();
  if (cfg.size) {
    const [w, h] = SIZE_PRESETS[form.preset][cfg.output === "image" ? "image" : "video"];
    body.width = w;
    body.height = h;
  }
  if (cfg.numImages) body.num_images = form.numImages;
  if (cfg.duration) body.duration_s = form.duration;
  if (cfg.duration && cfg.output === "video") body.fps = form.fps;
  if (cfg.withAudio) body.with_audio = form.withAudio;
  if (form.seed.trim()) {
    const seed = Number(form.seed);
    if (!Number.isInteger(seed) || seed < 0) return { error: "Seed must be a non-negative integer" };
    body.seed = seed;
  }
  if (cfg.output !== "audio" && form.modelId) body.model_entry_id = form.modelId;
  else if (form.adapterKey) body.adapter_key = form.adapterKey;
  if (cfg.output === "image" && form.loras.length) {
    body.loras = form.loras.map((l) => ({ model_entry_id: l.id, weight: l.weight }));
  }
  for (const input of cfg.assets ?? []) {
    const ids = form.assets[input.field] ?? [];
    if (!ids.length) return { error: `Pick ${input.label.toLowerCase()}` };
    body[input.field] = input.multiple ? ids : ids[0];
  }
  return { body };
}
