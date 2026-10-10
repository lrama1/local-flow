// Runs the backend (engine `auto`: the real GPU engine when available) and the Vite dev server
// together, and restarts the backend when Python files change. Generates one per-launch API token for
// both (Section 18).
//
// Backend reload is done here rather than with `uvicorn --reload`: on Windows uvicorn restarts by
// sending a console-wide Ctrl+C, which would also kill Vite.
import { spawn, spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { readdirSync, statSync, watch } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const isWin = process.platform === "win32";
const token = process.env.LOCALFLOW_API_TOKEN || randomBytes(24).toString("base64url");
const env = {
  ...process.env,
  LOCALFLOW_API_TOKEN: token,
  LOCALFLOW_DEV_MODE: "true",
  VITE_LOCALFLOW_TOKEN: token,
  PYTHONUNBUFFERED: "1",
};

const colors = { api: "\x1b[36m", web: "\x1b[35m" };
const procs = {};
let shuttingDown = false;
let restartingApi = false;

function log(name, line) {
  process.stdout.write(`${colors[name]}[${name}]\x1b[0m ${line}\n`);
}

function run(name, cmd, args, cwd, onExit) {
  // On Windows, npm/uv may be .cmd shims, so go through cmd.exe explicitly (args are fixed literals).
  const [file, argv] = isWin ? ["cmd.exe", ["/d", "/s", "/c", cmd, ...args]] : [cmd, args];
  const child = spawn(file, argv, { cwd: join(root, cwd), env, stdio: ["ignore", "pipe", "pipe"] });
  for (const stream of [child.stdout, child.stderr]) {
    let buf = "";
    stream.on("data", (chunk) => {
      buf += chunk.toString();
      const lines = buf.split(/\r?\n/);
      buf = lines.pop() ?? "";
      for (const line of lines) log(name, line);
    });
  }
  child.on("exit", (code) => onExit(code));
  procs[name] = child;
  return child;
}

function kill(child) {
  if (!child || child.exitCode !== null || !child.pid) return;
  if (isWin) spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  else child.kill("SIGTERM");
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  Object.values(procs).forEach(kill);
  setTimeout(() => process.exit(code), 300);
}

function startApi() {
  run("api", "uv", ["run", "localflow", "serve", "--dev"], "backend", (code) => {
    if (shuttingDown) return;
    if (restartingApi) {
      restartingApi = false;
      startApi();
      return;
    }
    log("api", `exited with code ${code}`);
    shutdown(code ?? 1);
  });
}

// On Windows fs.watch also reports last-access/attribute changes, so merely importing a module would
// restart the backend (and kill a GPU job). Only a changed mtime, a new file, or a deletion counts.
const pkg = join(root, "backend", "localflow");
const mtimes = new Map();
for (const f of readdirSync(pkg, { recursive: true })) {
  if (f.endsWith(".py") && !f.includes("__pycache__")) mtimes.set(f, statSync(join(pkg, f)).mtimeMs);
}

function modified(file) {
  let mtime = null;
  try {
    mtime = statSync(join(pkg, file)).mtimeMs;
  } catch {
    // deleted
  }
  const before = mtimes.get(file);
  if (mtime === null) mtimes.delete(file);
  else mtimes.set(file, mtime);
  return mtime !== before;
}

let debounce;
watch(pkg, { recursive: true }, (_event, file) => {
  if (!file || !file.endsWith(".py") || file.includes("__pycache__")) return;
  if (!modified(file)) return;
  clearTimeout(debounce);
  debounce = setTimeout(() => {
    if (shuttingDown || restartingApi) return;
    log("api", `change detected in ${file}; restarting backend…`);
    restartingApi = true;
    kill(procs.api);
  }, 300);
});

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

startApi();
run("web", "npm", ["run", "dev"], "frontend", (code) => {
  log("web", `exited with code ${code}`);
  shutdown(code ?? 1);
});
process.stdout.write("\nLocalFlow dev: open http://127.0.0.1:5173 (API on http://127.0.0.1:8765)\n\n");
