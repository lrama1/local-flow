// Starts the LocalFlow server for the Playwright smoke test: a fresh data dir, the FakeEngine, canned LLM
// responses, and no access to the developer's API keys or OS keyring. Serves the built frontend (dist/).
import { spawn } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const port = process.env.E2E_PORT ?? "8799";
const dataDir = process.env.E2E_DATA_DIR ?? join(tmpdir(), "localflow-e2e");
if (!/^\d+$/.test(port)) throw new Error(`E2E_PORT must be a number, got ${port}`);

rmSync(dataDir, { recursive: true, force: true });
mkdirSync(dataDir, { recursive: true });

// One command string through the shell, which finds `uv` on PATH the way a terminal does.
const child = spawn(`uv run localflow serve --port ${port}`, {
  cwd: join(here, "..", "..", "backend"),
  stdio: "inherit",
  shell: true,
  env: {
    ...process.env,
    LOCALFLOW_DATA_DIR: dataDir,
    LOCALFLOW_ENGINE: "fake",
    LOCALFLOW_FAKE_LLM: "true",
    LOCALFLOW_FAKE_STEP_DELAY_S: "0.02",
    LOCALFLOW_SECRETS_FROM_ENV: "false",
    LOCALFLOW_SECRET_BACKEND: "memory",
  },
});
child.on("exit", (code) => process.exit(code ?? 0));
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => child.kill(signal));
