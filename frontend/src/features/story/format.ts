import type { RefinedStory, StoryBible } from "@/api/client";

export function formatRefined(r: RefinedStory | null | undefined): string {
  if (!r) return "";
  return [
    `Title: ${r.title}`,
    `Logline: ${r.logline}`,
    `Runtime: ${r.estimated_runtime_s} s`,
    `Tone: ${r.tone ?? ""}`,
    `Visual style: ${r.visual_style ?? ""}`,
    "",
    "Synopsis:",
    r.synopsis,
    "",
    "Story:",
    r.story,
  ].join("\n");
}

export function formatBible(b: StoryBible | null | undefined): string {
  if (!b) return "";
  const style = b.style ?? {};
  const lines = [
    "Style:",
    ...Object.entries(style)
      .filter(([, v]) => v)
      .map(([k, v]) => `  ${k}: ${String(v)}`),
  ];
  const section = (title: string, items: { key: string; name: string; visual_prompt: string }[] = []) => {
    if (!items.length) return;
    lines.push("", `${title}:`);
    for (const it of items) lines.push(`  ${it.name} (${it.key}): ${it.visual_prompt}`);
  };
  section("Characters", b.characters);
  section("Locations", b.locations);
  section("Props", b.props);
  return lines.join("\n");
}
