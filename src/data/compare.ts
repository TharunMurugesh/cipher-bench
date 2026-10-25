import type { ExperimentRecord } from "./types";

export const primaryTask = "digit 0 vs digit 1";
export const secondaryTask = "digit 3 vs digit 8";
export const executionModes = [
  "All modes", "Published plaintext", "Published CKKS", "Published BFV", "Local plaintext",
  "CKKS features + plaintext head", "CKKS validation partial path", "Diagnostic / unavailable",
] as const;

export function filterRecords(records: ExperimentRecord[], task: string, mode: string): ExperimentRecord[] {
  return records.filter((record) => (task === "All tasks" || record.task === task) && (mode === "All modes" || record.executionMode === mode));
}

export function formatPercent(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "Not reported" : `${(value * 100).toFixed(2)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "Unavailable" : value.toFixed(digits);
}

export function formatDeltaPp(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Unavailable";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)} pp`;
}

export function selectedAblations(records: ExperimentRecord[], task: string): ExperimentRecord[] {
  return records.filter((record) => record.task === task && /^A[0-4]/.test(record.variant));
}

export function confusionMatrixCells(matrix: number[][] | null): string[][] {
  if (!matrix || matrix.length !== 2 || matrix.some((row) => row.length !== 2)) return [["Not reported", "Not reported"], ["Not reported", "Not reported"]];
  return matrix.map((row) => row.map((value) => Number.isFinite(value) ? String(value) : "Unavailable"));
}

export function toCsv(records: ExperimentRecord[]): string {
  const columns = ["task", "variant", "executionMode", "status", "provenance", "accuracy", "precision", "recall", "f1", "support", "runId", "seed", "splitId", "sourceArtifact"] as const;
  const quote = (value: unknown) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return [columns.map(quote).join(","), ...records.map((record) => columns.map((column) => quote(column in record.metrics ? record.metrics[column as keyof typeof record.metrics] : record[column as keyof ExperimentRecord])).join(","))].join("\r\n");
}

export function exportJson(records: ExperimentRecord[]): string {
  return JSON.stringify({ exportedAt: new Date().toISOString(), metricNote: "Fractions are 0-1; pp deltas are percentage points; null means unavailable.", records }, null, 2);
}
