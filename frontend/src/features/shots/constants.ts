import type { CameraMove, GenerationMode, ShotType } from "@/api/client";

export const SHOT_TYPES: { value: ShotType; label: string }[] = [
  { value: "wide", label: "Wide" },
  { value: "medium", label: "Medium" },
  { value: "close", label: "Close-up" },
  { value: "extreme_close", label: "Extreme close-up" },
  { value: "insert", label: "Insert" },
  { value: "pov", label: "POV" },
  { value: "over_shoulder", label: "Over the shoulder" },
];

export const CAMERA_MOVES: { value: CameraMove; label: string }[] = [
  { value: "static", label: "Static" },
  { value: "pan_left", label: "Pan left" },
  { value: "pan_right", label: "Pan right" },
  { value: "tilt_up", label: "Tilt up" },
  { value: "tilt_down", label: "Tilt down" },
  { value: "dolly_in", label: "Dolly in" },
  { value: "dolly_out", label: "Dolly out" },
  { value: "truck_left", label: "Truck left" },
  { value: "truck_right", label: "Truck right" },
  { value: "orbit", label: "Orbit" },
  { value: "handheld", label: "Handheld" },
  { value: "crane_up", label: "Crane up" },
  { value: "crane_down", label: "Crane down" },
  { value: "zoom_in", label: "Zoom in" },
  { value: "zoom_out", label: "Zoom out" },
];

export const MODES: { value: GenerationMode; label: string }[] = [
  { value: "i2v", label: "Keyframe → video (i2v)" },
  { value: "t2v", label: "Text → video (t2v)" },
  { value: "flf2v", label: "First/last frame (flf2v)" },
  { value: "ref2v", label: "References → video (ref2v)" },
];

export const label = <T extends string>(list: { value: T; label: string }[], v: T): string =>
  list.find((x) => x.value === v)?.label ?? v;
