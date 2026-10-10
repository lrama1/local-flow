import type { Board, Shot } from "@/api/client";
import { moveShots } from "@/api/story";
import { formatRefined } from "@/features/story/format";
import { diffLines, hasChanges } from "@/lib/diff";

test("diffLines marks added, removed, and unchanged lines", () => {
  const ops = diffLines("a\nb\nc", "a\nB\nc\nd");
  expect(ops).toEqual([
    { type: "same", text: "a" },
    { type: "del", text: "b" },
    { type: "add", text: "B" },
    { type: "same", text: "c" },
    { type: "add", text: "d" },
  ]);
  expect(hasChanges(diffLines("x", "x"))).toBe(false);
});

const shot = (id: string, scene_id: string, index: number) => ({ id, scene_id, index }) as unknown as Shot;

function board(): Board {
  return {
    scenes: [
      { id: "s1", shots: [shot("a", "s1", 0), shot("b", "s1", 1)] },
      { id: "s2", shots: [shot("c", "s2", 0), shot("d", "s2", 1)] },
    ],
    total_duration_s: 0,
    shot_count: 4,
    max_clip_seconds: 5,
  } as unknown as Board;
}

test("moveShots reorders within a scene", () => {
  const b = moveShots(board(), "s1", ["b", "a"]);
  expect(b.scenes[0]!.shots.map((s) => [s.id, s.index])).toEqual([
    ["b", 0],
    ["a", 1],
  ]);
  expect(b.scenes[1]!.shots.map((s) => s.id)).toEqual(["c", "d"]);
});

test("moveShots moves a shot across scenes and reindexes the source", () => {
  const b = moveShots(board(), "s1", ["a", "c", "b"]);
  expect(b.scenes[0]!.shots.map((s) => [s.id, s.scene_id, s.index])).toEqual([
    ["a", "s1", 0],
    ["c", "s1", 1],
    ["b", "s1", 2],
  ]);
  expect(b.scenes[1]!.shots.map((s) => [s.id, s.index])).toEqual([["d", 0]]);
});

test("formatRefined is stable for diffing", () => {
  const text = formatRefined({
    title: "T",
    logline: "L",
    synopsis: "S",
    story: "Beat 1\n\nBeat 2",
    estimated_runtime_s: 30,
    tone: "calm",
    visual_style: "ink",
  });
  expect(text.split("\n").slice(0, 3)).toEqual(["Title: T", "Logline: L", "Runtime: 30 s"]);
  expect(text.endsWith("Beat 1\n\nBeat 2")).toBe(true);
});
