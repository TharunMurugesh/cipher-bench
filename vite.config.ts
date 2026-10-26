import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { copyFile, mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const rootDir = fileURLToPath(new URL(".", import.meta.url));
type RunState = { id: string; directory: string; status: "running" | "completed" | "failed"; logs: string[]; error?: string; child?: ChildProcessWithoutNullStreams };
let currentRun: RunState | null = null;
type BenchmarkState = { id: string; directory: string; task: string; status: "running" | "completed" | "failed"; logs: string[]; error?: string; child?: ChildProcessWithoutNullStreams; result?: unknown };
let currentBenchmark: BenchmarkState | null = null;
const artifactFiles = [
  "published_reference.json",
  "baseline_results.json",
  "ckks_validation.json",
  "ckks_final_test.json",
  "ckks_smoke.json",
  "phase3_ablation.json",
  "phase3_ckks_validation.json",
  "phase3_splits.json",
  "baseline_predictions.csv",
  "phase3_predictions.csv",
];
function localResearchFiles(): Plugin {
  const allow = new Set(artifactFiles.map((name) => `artifacts/${name}`));
  return {
    name: "cipherbench-local-research-files",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const pathname = new URL(req.url ?? "/", "http://localhost").pathname;
        const key = pathname.replace(/^\//, "");
        if (!allow.has(key)) return next();
        try {
          const filename = resolve(rootDir, key);
          const content = await readFile(filename);
          res.statusCode = 200;
          res.setHeader("Content-Type", key.endsWith(".json") ? "application/json; charset=utf-8" : "text/csv; charset=utf-8");
          res.setHeader("Cache-Control", "no-store");
          res.end(content);
        } catch {
          res.statusCode = 500;
          res.end("Unable to read the local research artifact.");
        }
      });
    },
    async writeBundle(output) {
      const outDir = resolve(rootDir, output.dir ?? "dist");
      const artifactDir = resolve(outDir, "artifacts");
      await mkdir(artifactDir, { recursive: true });
      await Promise.all([
        ...artifactFiles.map((name) => copyFile(resolve(rootDir, "artifacts", name), resolve(artifactDir, name))),
      ]);
    },
  };
}

function localExperimentRunner(): Plugin {
  const allowed = new Set(artifactFiles);
  const reply = (res: import("node:http").ServerResponse, status: number, value: unknown) => {
    res.statusCode = status;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.setHeader("Cache-Control", "no-store");
    res.end(JSON.stringify(value));
  };
  const state = () => currentRun ? { available: true, status: currentRun.status, runId: currentRun.id, logs: currentRun.logs, error: currentRun.error } : { available: true, status: "idle", logs: [] };
  return {
    name: "cipherbench-local-experiment-runner",
    configureServer(server) {
      server.httpServer?.once("close", () => {
        const run = currentRun;
        if (run?.child) run.child.kill();
        if (run) void rm(run.directory, { recursive: true, force: true });
        currentRun = null;
        const benchmark = currentBenchmark;
        if (benchmark?.child) benchmark.child.kill();
        if (benchmark) void rm(benchmark.directory, { recursive: true, force: true });
        currentBenchmark = null;
      });
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url ?? "/", "http://127.0.0.1");
        if (url.pathname.startsWith("/api/benchmark/")) {
          if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) return reply(res, 403, { error: "Cross-origin requests are not allowed." });
          if (url.pathname === "/api/benchmark/dataset" && req.method === "GET") {
            const task = url.searchParams.get("task") ?? "digits_0_vs_1";
            if (!["digits_0_vs_1", "digits_2_vs_7", "digits_3_vs_8", "digits_4_vs_9"].includes(task)) return reply(res, 400, { error: "Choose one of the supported digit datasets." });
            const pyEnv = process.env.CIPHERBENCH_PYTHON;
            const candidates = [pyEnv, join(rootDir, ".venv", "Scripts", "python.exe"), join(rootDir, ".venv", "bin", "python"), "python", "python3"].filter((item): item is string => Boolean(item));
            const python = candidates.find((item) => item === pyEnv || item === "python" || item === "python3" || existsSync(item)) ?? "python";
            const child = spawn(python, ["-u", "scripts/inspect_dataset.py", "--task", task], { cwd: rootDir, windowsHide: true });
            let output = "";
            let errorOutput = "";
            child.stdout.setEncoding("utf8").on("data", (chunk: string) => { output += chunk; });
            child.stderr.setEncoding("utf8").on("data", (chunk: string) => { errorOutput += chunk; });
            child.on("error", (error) => reply(res, 500, { error: error.message }));
            child.on("close", (code) => {
              if (res.writableEnded) return;
              if (code !== 0) return reply(res, 500, { error: errorOutput.trim() || "Could not inspect the selected dataset." });
              try { return reply(res, 200, JSON.parse(output)); }
              catch { return reply(res, 500, { error: "Dataset preview was not valid JSON." }); }
            });
            return;
          }
          if (url.pathname === "/api/benchmark/status" && req.method === "GET") {
            const run = currentBenchmark;
            return reply(res, 200, run ? { available: true, status: run.status, runId: run.id, task: run.task, logs: run.logs, error: run.error, result: run.result } : { available: true, status: "idle", logs: [] });
          }
          if (url.pathname === "/api/benchmark/run" && req.method === "POST") {
            if (currentBenchmark?.status === "running") return reply(res, 409, { error: "A model training run is already in progress." });
            const task = url.searchParams.get("task") ?? "digits_0_vs_1";
            if (!["digits_0_vs_1", "digits_2_vs_7", "digits_3_vs_8", "digits_4_vs_9"].includes(task)) return reply(res, 400, { error: "Choose one of the supported digit tasks." });
            if (currentBenchmark) await rm(currentBenchmark.directory, { recursive: true, force: true });
            const id = randomUUID();
            const directory = await mkdtemp(join(tmpdir(), "cipherbench-model-lab-"));
            const run: BenchmarkState = { id, directory, task, status: "running", logs: ["Training the Paper Model and CipherBench Model on the selected dataset.", "Both models use the same stratified held-out test split."] };
            currentBenchmark = run;
            const pyEnv = process.env.CIPHERBENCH_PYTHON;
            const candidates = [pyEnv, join(rootDir, ".venv", "Scripts", "python.exe"), join(rootDir, ".venv", "bin", "python"), "python", "python3"].filter((item): item is string => Boolean(item));
            const python = candidates.find((item) => item === pyEnv || item === "python" || item === "python3" || existsSync(item)) ?? "python";
            try {
              const child = spawn(python, ["-u", "scripts/run_dashboard_benchmark.py", "--task", task, "--output", join(directory, "result.json")], { cwd: rootDir, env: { ...process.env, PYTHONUNBUFFERED: "1" }, windowsHide: true });
              run.child = child;
              const collect = (chunk: Buffer) => { run.logs.push(...chunk.toString("utf8").split(/\r?\n/).filter(Boolean)); if (run.logs.length > 40) run.logs.splice(0, run.logs.length - 40); };
              child.stdout.on("data", collect);
              child.stderr.on("data", collect);
              child.on("error", (error) => { run.status = "failed"; run.error = error.message; run.child = undefined; });
              child.on("close", async (code) => {
                run.child = undefined;
                if (code !== 0) { run.status = "failed"; run.error = `Training process exited with code ${code}. Review the run log.`; return; }
                try { run.result = JSON.parse(await readFile(join(directory, "result.json"), "utf8")); run.status = "completed"; run.logs.push("Training and held-out evaluation completed successfully."); }
                catch (error) { run.status = "failed"; run.error = error instanceof Error ? error.message : "Could not load the benchmark result."; }
              });
            } catch (error) { run.status = "failed"; run.error = error instanceof Error ? error.message : "Unable to start the Python runner."; }
            return reply(res, 202, { available: true, status: run.status, runId: run.id, task: run.task, logs: run.logs });
          }
          if (url.pathname === "/api/benchmark/run" && req.method === "DELETE") {
            if (currentBenchmark?.status === "running") return reply(res, 409, { error: "Wait for training to finish before clearing its results." });
            if (currentBenchmark) await rm(currentBenchmark.directory, { recursive: true, force: true });
            currentBenchmark = null;
            return reply(res, 200, { status: "idle" });
          }
          return reply(res, 404, { error: "Unknown model-lab endpoint." });
        }
        if (!url.pathname.startsWith("/api/demo/")) return next();
        if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) return reply(res, 403, { error: "Cross-origin requests are not allowed." });
        if (url.pathname === "/api/demo/status" && req.method === "GET") return reply(res, 200, state());
        if (url.pathname === "/api/demo/run" && req.method === "POST") {
          if (currentRun?.status === "running") return reply(res, 409, { error: "A training run is already in progress." });
          if (currentRun) await rm(currentRun.directory, { recursive: true, force: true });
          const id = randomUUID();
          const directory = await mkdtemp(join(tmpdir(), "cipherbench-demo-"));
          const run: RunState = { id, directory, status: "running", logs: ["Started the complete Phase 3 training, validation, CKKS diagnostic and held-out test pipeline.", "Run outputs are isolated in temporary storage; repository artifacts are unchanged."] };
          currentRun = run;
          const pyEnv = process.env.CIPHERBENCH_PYTHON;
          const candidates = [pyEnv, join(rootDir, ".venv", "Scripts", "python.exe"), join(rootDir, ".venv", "bin", "python"), "python", "python3"].filter((item): item is string => Boolean(item));
          const python = candidates.find((item) => item === pyEnv || item === "python" || item === "python3" || existsSync(item)) ?? "python";
          try {
            const child = spawn(python, ["-u", "scripts/run_phase3_ablation.py"], { cwd: rootDir, env: { ...process.env, CIPHERBENCH_ARTIFACT_DIR: directory, PYTHONUNBUFFERED: "1" }, windowsHide: true });
            run.child = child;
            const collect = (chunk: Buffer) => {
              const lines = chunk.toString("utf8").split(/\r?\n/).filter(Boolean);
              run.logs.push(...lines);
              if (run.logs.length > 100) run.logs.splice(0, run.logs.length - 100);
            };
            child.stdout.on("data", collect);
            child.stderr.on("data", collect);
            child.on("error", (error) => { run.status = "failed"; run.error = error.message; run.child = undefined; });
            child.on("close", (code) => {
              run.child = undefined;
              run.status = code === 0 ? "completed" : "failed";
              if (code !== 0) run.error = `Training process exited with code ${code}. Review the run log.`;
              if (code === 0) run.logs.push("Training and held-out evaluation completed successfully.");
            });
          } catch (error) {
            run.status = "failed";
            run.error = error instanceof Error ? error.message : "Unable to start the Python runner.";
          }
          return reply(res, 202, state());
        }
        if (url.pathname === "/api/demo/run" && req.method === "DELETE") {
          if (currentRun?.status === "running") return reply(res, 409, { error: "Wait for the active run to finish before clearing its temporary results." });
          if (currentRun) await rm(currentRun.directory, { recursive: true, force: true });
          currentRun = null;
          return reply(res, 200, { cacheCleared: true, status: "idle" });
        }
        const match = url.pathname.match(/^\/api\/demo\/artifacts\/([0-9a-f-]+)\/([^/]+)$/i);
        if (match && req.method === "GET") {
          const [, runId, filename] = match;
          if (!currentRun || currentRun.id !== runId || currentRun.status !== "completed" || !allowed.has(filename)) return reply(res, 404, { error: "Run artifact not found." });
          try {
            let content: Buffer;
            try { content = await readFile(join(currentRun.directory, filename)); }
            catch { content = await readFile(resolve(rootDir, "artifacts", filename)); }
            res.statusCode = 200;
            res.setHeader("Content-Type", filename.endsWith(".json") ? "application/json; charset=utf-8" : "text/csv; charset=utf-8");
            res.setHeader("Cache-Control", "no-store");
            res.end(content);
          } catch { return reply(res, 404, { error: "The run did not produce this artifact." }); }
          return;
        }
        return reply(res, 404, { error: "Unknown local demo endpoint." });
      });
    },
  };
}

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  plugins: [react(), localResearchFiles(), localExperimentRunner()],
  server: { strictPort: true, host: "127.0.0.1" },
  preview: { strictPort: true },
});
