import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { candidateArchitecture } from "./architecture";
import { exportJson, filterRecords, formatDeltaPp, formatPercent, toCsv } from "./compare";
import { buildDashboardData, loadDashboardData, validateArtifactBundle } from "./transform";
import type { ArtifactBundle } from "./types";

const artifact = (name: string) => JSON.parse(readFileSync(`artifacts/${name}`, "utf-8")) as unknown;
const bundle = (): ArtifactBundle => ({
  published_reference: artifact("published_reference.json"),
  baseline_results: artifact("baseline_results.json"),
  ckks_validation: artifact("ckks_validation.json"),
  ckks_final_test: artifact("ckks_final_test.json"),
  ckks_smoke: artifact("ckks_smoke.json"),
  phase3_ablation: artifact("phase3_ablation.json"),
  phase3_ckks_validation: artifact("phase3_ckks_validation.json"),
  phase3_splits: artifact("phase3_splits.json"),
  baseline_predictions: readFileSync("artifacts/baseline_predictions.csv", "utf-8"),
  phase3_predictions: readFileSync("artifacts/phase3_predictions.csv", "utf-8"),
});

describe("dashboard artifact data", () => {
  it("validates and maps real published, baseline, CKKS and Phase 3 artifacts", () => {
    const b = validateArtifactBundle(bundle());
    const data = buildDashboardData(b);
    expect(data.records.some((r) => r.provenance === "Published" && r.variant.includes("Logistic Regression"))).toBe(true);
    expect(data.records.some((r) => r.sourceArtifact === "baseline_results.json" && r.variant === "TinyCNN baseline")).toBe(true);
    expect(data.records.some((r) => r.variant === "A3 · RC-TinyCNN" && r.task === "digit 3 vs digit 8" && r.predictions?.length === 108)).toBe(true);
    expect(data.records.some((r) => r.executionMode === "CKKS features + plaintext head" && r.agreement?.count === 108)).toBe(true);
    expect(data.records.some((r) => r.executionMode === "CKKS validation partial path" && r.agreement?.count === 38)).toBe(true);
  });

  it("rejects missing and malformed artifact schemas", () => {
    expect(() => validateArtifactBundle({})).toThrow("Required artifact is missing");
    const broken = { ...bundle(), baseline_results: { models: [] } };
    expect(() => validateArtifactBundle(broken)).toThrow("baseline_results.models must be an object");
    const brokenCsv = { ...bundle(), phase3_predictions: "task\n0\n" };
    expect(() => buildDashboardData(brokenCsv)).toThrow("phase3_predictions.csv is missing column");
  });

  it("combines fresh run outputs with checked-in artifacts that the run does not regenerate", async () => {
    const requested: string[] = [];
    const fetcher: typeof fetch = async (input) => {
      const url = String(input);
      requested.push(url);
      const name = url.split("/").pop() ?? "";
      if (url.startsWith("/api/demo/artifacts/")) {
        if (name === "phase3_ablation.json" || name === "phase3_ckks_validation.json" || name === "phase3_splits.json" || name === "phase3_predictions.csv") {
          return new Response(readFileSync(`artifacts/${name}`, "utf-8"), { status: 200 });
        }
        return new Response("missing from this run", { status: 404 });
      }
      return new Response(readFileSync(`artifacts/${name}`, "utf-8"), { status: 200 });
    };
    const data = await loadDashboardData(fetcher, "/api/demo/artifacts/test-run/", "/artifacts/");
    expect(data.records.some((record) => record.provenance === "Published")).toBe(true);
    expect(data.records.some((record) => record.variant === "A3 · RC-TinyCNN" && record.sourceArtifact.startsWith("phase3_ablation.json"))).toBe(true);
    expect(requested).toContain("/api/demo/artifacts/test-run/published_reference.json");
    expect(requested).toContain("/artifacts/published_reference.json");
  });

  it("filters by task and execution mode without mixing benchmark tasks", () => {
    const records = buildDashboardData(bundle()).records;
    const secondary = filterRecords(records, "digit 3 vs digit 8", "Local plaintext");
    expect(secondary.length).toBeGreaterThan(0);
    expect(secondary.every((record) => record.task === "digit 3 vs digit 8" && record.executionMode === "Local plaintext")).toBe(true);
    const ckks = filterRecords(records, "digit 0 vs digit 1", "CKKS validation partial path");
    expect(ckks.length).toBe(4);
    expect(ckks.every((record) => record.splitLabel.startsWith("Validation"))).toBe(true);
  });

  it("keeps unpublished metrics explicit and formats percentage-point changes", () => {
    const data = buildDashboardData(bundle());
    const paperTinyCnn = data.records.find((record) => record.provenance === "Published" && record.variant.includes("Conv + square"));
    expect(paperTinyCnn?.metrics.accuracy).toBe(0.98);
    expect(paperTinyCnn?.metrics.support).toBeNull();
    expect(paperTinyCnn?.metrics.precision).toBeNull();
    expect(formatPercent(null)).toBe("Not reported");
    expect(formatDeltaPp(3.703)).toBe("+3.70 pp");
    expect(formatDeltaPp(-1)).toBe("-1.00 pp");
  });

  it("exports valid JSON and quoted CSV from normalized artifact records", () => {
    const records = buildDashboardData(bundle()).records.slice(0, 3);
    const json = JSON.parse(exportJson(records)) as { records: unknown[] };
    expect(json.records).toHaveLength(3);
    const csv = toCsv(records);
    expect(csv.split("\r\n")).toHaveLength(4);
    expect(csv.startsWith('"task","variant"')).toBe(true);
  });

  it("labels the range loss as training-only and makes no inference-stage claim", () => {
    const stages = candidateArchitecture("pixel_0_1", 1, 0.001);
    const loss = stages.find((stage) => stage.id === "range-loss");
    expect(loss?.execution).toBe("training-only");
    expect(stages.find((stage) => stage.id === "head")?.execution).toBe("plaintext");
    expect(stages.some((stage) => stage.execution === "training-only" && /not an inference step/i.test(stage.detail))).toBe(true);
  });
});
