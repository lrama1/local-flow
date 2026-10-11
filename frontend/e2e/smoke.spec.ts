import { expect, test, type Page } from "@playwright/test";

// The fake LLM's canned breakdown (backend/localflow/services/llm/demo.py) has 5 shots in 2 scenes.
const SHOTS = 5;

async function api<T>(page: Page, path: string): Promise<T> {
  const token = await page.locator('meta[name="localflow-token"]').getAttribute("content");
  const res = await page.request.get(path, { headers: { "X-LocalFlow-Token": token ?? "" } });
  expect(res.ok(), `${path}: ${res.status()}`).toBeTruthy();
  return (await res.json()) as T;
}

type Board = { shot_count: number; scenes: { shots: { selected_video_asset_id: string | null }[] }[] };
type AssetPage = { items: { id: string; type: string; source: string; duration_s: number | null }[] };

test("idea → story → shots → generate all → timeline → export", async ({ page }) => {
  await page.goto("/projects");
  await expect(page.getByText("Connected")).toBeVisible();

  // Create a project: it opens on the Story tab.
  await page.getByRole("button", { name: "New project" }).first().click();
  const dialog = page.getByRole("dialog", { name: "New project" });
  await dialog.getByRole("textbox", { name: "Name" }).fill("Smoke test");
  await dialog.getByRole("button", { name: "Create project" }).click();
  const idea = page.getByRole("textbox", { name: "Story idea" });
  await expect(idea).toBeVisible();
  const projectId = /\/projects\/([^/]+)/.exec(page.url())?.[1];
  expect(projectId).toBeTruthy();

  // Story pipeline (fake LLM): refine → bible → breakdown.
  await idea.fill("A lighthouse keeper repairs a mechanical seagull during a storm.");
  await page.getByRole("button", { name: "Refine", exact: true }).click();
  await expect(page.getByTestId("refined-editor")).toBeVisible();
  await page.getByRole("button", { name: "Build bible" }).click();
  await expect(page.getByTestId("bible-view")).toBeVisible();
  await page.getByRole("button", { name: "Break down into shots" }).click();
  await expect(page.getByTestId("board-summary")).toContainText(`${SHOTS} shots`);

  // Shots board: generate everything, using each first keyframe right away.
  await page.getByRole("tab", { name: "Shots board" }).click();
  await expect(page.getByTestId("shot-1.1")).toBeVisible();
  await page.getByRole("button", { name: "Generate all" }).click();
  const gen = page.getByRole("dialog", { name: "Generate all" });
  await gen.getByRole("checkbox", { name: /Use each new keyframe right away/ }).check();
  await gen.getByRole("button", { name: "Queue" }).click();
  await expect(gen.getByTestId("generate-all-result")).toContainText(`Queued ${SHOTS} keyframe jobs`);
  await gen.getByRole("button", { name: "Done" }).click();

  await expect
    .poll(
      async () => {
        const board = await api<Board>(page, `/api/projects/${projectId}/scenes`);
        return board.scenes.flatMap((s) => s.shots).filter((s) => s.selected_video_asset_id).length;
      },
      { timeout: 180_000, intervals: [1000] },
    )
    .toBe(SHOTS);

  // Timeline: assemble the selected takes, then export an MP4.
  await page.getByRole("tab", { name: "Timeline" }).click();
  await page.getByRole("button", { name: "Auto-assemble" }).first().click();
  await expect(page.getByText(`${SHOTS} clips placed`)).toBeVisible();
  const exportPanel = page.getByTestId("export-panel");
  await exportPanel.getByRole("button", { name: "Export MP4" }).click();
  await expect(exportPanel.getByTestId("export-history").getByRole("listitem")).toHaveCount(1, {
    timeout: 120_000,
  });

  const exports = await api<AssetPage>(page, `/api/projects/${projectId}/assets?source=export`);
  expect(exports.items).toHaveLength(1);
  const video = exports.items[0]!;
  expect(video.type).toBe("video");
  expect(video.duration_s ?? 0).toBeGreaterThan(10); // 5 shots of 3–5 s

  // The export is in the asset library too.
  await page.getByRole("tab", { name: "Assets" }).click();
  await page.getByLabel("Source").selectOption("export");
  await expect(page.getByTestId("asset-grid").getByRole("button", { name: "Open video asset" })).toHaveCount(
    1,
  );
});
