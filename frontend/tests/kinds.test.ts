import { buildBody, DEFAULT_FORM } from "@/features/playground/kinds";
import { formatDuration, formatMb, relativeTime } from "@/lib/utils";

test("t2v body uses the video size preset, duration, and fps", () => {
  const r = buildBody("t2v", { ...DEFAULT_FORM, prompt: "  a fox  ", preset: "9:16", duration: 3 }, "p1");
  expect(r).toEqual({
    body: {
      project_id: "p1",
      extra: {},
      prompt: "a fox",
      width: 720,
      height: 1280,
      duration_s: 3,
      fps: 24,
      with_audio: false,
    },
  });
});

test("i2v requires a start image and sends a single id", () => {
  expect(buildBody("i2v", DEFAULT_FORM, "p1")).toEqual({ error: "Pick start image" });
  const r = buildBody("i2v", { ...DEFAULT_FORM, assets: { image_asset_id: ["img1"] } }, "p1");
  expect("body" in r && r.body.image_asset_id).toBe("img1");
});

test("multi-reference inputs are sent as arrays", () => {
  const r = buildBody("image_edit", { ...DEFAULT_FORM, assets: { reference_asset_ids: ["a", "b"] } }, "p1");
  expect("body" in r && r.body.reference_asset_ids).toEqual(["a", "b"]);
});

test("tts requires text and has no size fields", () => {
  expect(buildBody("tts", DEFAULT_FORM, "p1")).toEqual({ error: "Enter some text" });
  const r = buildBody("tts", { ...DEFAULT_FORM, prompt: "hello" }, "p1");
  expect(r).toEqual({ body: { project_id: "p1", extra: {}, text: "hello" } });
});

test("seed is validated", () => {
  expect(buildBody("t2i", { ...DEFAULT_FORM, seed: "-3" }, "p1")).toEqual({
    error: "Seed must be a non-negative integer",
  });
  const r = buildBody("t2i", { ...DEFAULT_FORM, seed: "42" }, "p1");
  expect("body" in r && r.body.seed).toBe(42);
});

test("formatting helpers", () => {
  expect(formatDuration(4)).toBe("4.0s");
  expect(formatDuration(75)).toBe("1:15");
  expect(formatDuration(null)).toBe("–");
  expect(formatMb(16303)).toBe("15.9 GB");
  const now = Date.parse("2026-01-01T12:00:00Z");
  expect(relativeTime("2026-01-01T11:59:50Z", now)).toBe("just now");
  expect(relativeTime("2026-01-01T11:30:00Z", now)).toBe("30m ago");
});
