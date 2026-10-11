import { defineConfig, devices } from "@playwright/test";

const port = process.env.E2E_PORT ?? "8799";

/** Smoke test (Section 21): the built app against a real backend with the FakeEngine and fake LLM.
 * Run with `npm run test:e2e` (builds first). */
export default defineConfig({
  testDir: "./e2e",
  timeout: 240_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    video: "retain-on-failure",
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 900 },
  },
  webServer: {
    command: "node e2e/serve.mjs",
    url: `http://127.0.0.1:${port}/api/health`,
    timeout: 180_000, // the first run may sync the backend's Python environment
    reuseExistingServer: false, // every run starts from an empty data dir
    stdout: "ignore",
    stderr: "pipe",
    env: { E2E_PORT: port },
  },
});
