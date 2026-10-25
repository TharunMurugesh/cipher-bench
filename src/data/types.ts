export type MetricSet = {
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  macroPrecision: number | null;
  macroRecall: number | null;
  macroF1: number | null;
  support: number | null;
  confusion: number[][] | null;
  perClass: Record<string, { precision: number | null; recall: number | null; f1: number | null; support: number | null }> | null;
};

export type ExecutionMode = "Published plaintext" | "Published CKKS" | "Published BFV" | "Local plaintext" | "CKKS features + plaintext head" | "CKKS validation partial path" | "Diagnostic / unavailable";
export type Provenance = "Published" | "Measured" | "Provisional" | "Unavailable";

export type ExperimentRecord = {
  id: string;
  task: string;
  variant: string;
  executionMode: ExecutionMode;
  status: string;
  provenance: Provenance;
  runId: string | null;
  timestamp: string | null;
  seed: number | null;
  splitId: string | null;
  splitLabel: string;
  sourceArtifact: string;
  metrics: MetricSet;
  deltasVsA0?: Partial<Record<"accuracy" | "precision" | "recall" | "f1", number | null>>;
  timings: Record<string, number | null>;
  ciphertextSizes: Record<string, number | null>;
  agreement: { count: number; total: number; flips: number } | null;
  numericalErrors: Record<string, { meanMae: number | null; worstMax: number | null }> | null;
  privacyBoundary: string;
  configuration: Record<string, unknown>;
  activationDiagnostics: Record<string, unknown> | null;
  missingMetrics: string[];
  predictions?: number[];
  labels?: number[];
};

export type ArtifactBundle = {
  published_reference: unknown;
  baseline_results: unknown;
  ckks_validation: unknown;
  ckks_final_test: unknown;
  ckks_smoke: unknown;
  phase3_ablation: unknown;
  phase3_ckks_validation: unknown;
  phase3_splits: unknown;
  baseline_predictions: unknown;
  phase3_predictions: unknown;
};

export type DashboardData = {
  records: ExperimentRecord[];
  publishedPaper: string;
  sourceReports: string[];
  phase3RunId: string;
  phase3Timestamp: string | null;
  taskSplits: Record<string, { splitId: string | null; seed: number | null; sizes: Record<string, number> }>;
};
