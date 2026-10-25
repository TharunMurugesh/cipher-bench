import { copyFile, mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const rootDir = fileURLToPath(new URL(".", import.meta.url));
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
const reportFiles = [
  "BASELINE_REPORT.md",
  "CKKS_DIAGNOSTIC_REPORT.md",
  "ABLATION_REPORT.md",
  "NOVELTY_REVIEW.md",
  "ENVIRONMENT_REPORT.md",
];

function localResearchFiles(): Plugin {
  const allow = new Set([
    ...artifactFiles.map((name) => `artifacts/${name}`),
    ...reportFiles.map((name) => `reports/${name}`),
  ]);
  return {
    name: "cipherbench-local-research-files",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const pathname = new URL(req.url ?? "/", "http://localhost").pathname;
        const key = pathname.replace(/^\//, "");
        if (!allow.has(key)) return next();
        try {
          const filename = key.startsWith("artifacts/")
            ? resolve(rootDir, key)
            : resolve(rootDir, key.replace("reports/", ""));
          const content = await readFile(filename);
          res.statusCode = 200;
          res.setHeader("Content-Type", key.endsWith(".json") ? "application/json; charset=utf-8" :
            key.endsWith(".csv") ? "text/csv; charset=utf-8" : "text/markdown; charset=utf-8");
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
      const reportDir = resolve(outDir, "reports");
      await mkdir(artifactDir, { recursive: true });
      await mkdir(reportDir, { recursive: true });
      await Promise.all([
        ...artifactFiles.map((name) => copyFile(resolve(rootDir, "artifacts", name), resolve(artifactDir, name))),
        ...reportFiles.map((name) => copyFile(resolve(rootDir, name), resolve(reportDir, name))),
      ]);
    },
  };
}

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  plugins: [react(), localResearchFiles()],
  server: { strictPort: true },
  preview: { strictPort: true },
});
