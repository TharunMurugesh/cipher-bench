import type { ArtifactBundle, DashboardData, ExperimentRecord, ExecutionMode, MetricSet } from "./types";

type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj => typeof value === "object" && value !== null && !Array.isArray(value);
const isArray = (value: unknown): value is unknown[] => Array.isArray(value);
const num = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const str = (value: unknown): string | null => typeof value === "string" && value.length ? value : null;
const obj = (value: unknown): Obj => isObj(value) ? value : {};
const arr = (value: unknown): unknown[] => isArray(value) ? value : [];
const metricsUnavailable = ["precision", "recall", "f1", "macro metrics", "confusion matrix"];

function metricSet(raw: unknown): MetricSet {
  const m = obj(raw);
  const positive = obj(m.positive_class_metrics);
  const macro = obj(m.macro_metrics);
  const perClassRaw = obj(m.per_class);
  const perClassEntries = Object.entries(perClassRaw).filter(([, value]) => isObj(value));
  const perClass = perClassEntries.length ? Object.fromEntries(perClassEntries.map(([label, rawClass]) => {
    const c = obj(rawClass);
    return [label, { precision: num(c.precision), recall: num(c.recall), f1: num(c.f1), support: num(c.support) }];
  })) : null;
  const confusionRaw = m.confusion_matrix_labels_0_1;
  const confusion = isArray(confusionRaw) && confusionRaw.length === 2 && confusionRaw.every((row) =>
    isArray(row) && row.length === 2 && row.every((cell) => num(cell) !== null))
    ? confusionRaw.map((row) => (row as number[]).slice()) : null;
  const totalSupport = confusion ? confusion.flat().reduce((sum, value) => sum + value, 0) : null;
  return {
    accuracy: num(m.accuracy),
    precision: num(positive.precision ?? m.precision),
    recall: num(positive.recall ?? m.recall),
    f1: num(positive.f1 ?? m.f1),
    macroPrecision: num(macro.precision),
    macroRecall: num(macro.recall),
    macroF1: num(macro.f1),
    support: num(m.test_support) ?? totalSupport,
    confusion,
    perClass,
  };
}

function baseRecord(partial: Omit<ExperimentRecord, "metrics" | "timings" | "ciphertextSizes" | "agreement" | "numericalErrors" | "configuration" | "activationDiagnostics" | "missingMetrics"> & Partial<Pick<ExperimentRecord, "metrics" | "timings" | "ciphertextSizes" | "agreement" | "numericalErrors" | "configuration" | "activationDiagnostics" | "missingMetrics">>): ExperimentRecord {
  const metrics = partial.metrics ?? metricSet(null);
  return {
    ...partial,
    metrics,
    timings: partial.timings ?? {},
    ciphertextSizes: partial.ciphertextSizes ?? {},
    agreement: partial.agreement ?? null,
    numericalErrors: partial.numericalErrors ?? null,
    configuration: partial.configuration ?? {},
    activationDiagnostics: partial.activationDiagnostics ?? null,
    missingMetrics: partial.missingMetrics ?? [
      ...(metrics.accuracy === null ? ["accuracy"] : []),
      ...(metrics.precision === null ? ["positive-class precision"] : []),
      ...(metrics.recall === null ? ["positive-class recall"] : []),
      ...(metrics.f1 === null ? ["positive-class F1"] : []),
      ...(metrics.confusion === null ? ["confusion matrix"] : []),
    ],
  };
}

function requireObj(value: unknown, label: string): Obj {
  if (!isObj(value)) throw new Error(`${label} must be a JSON object.`);
  return value;
}

export function validateArtifactBundle(raw: unknown): ArtifactBundle {
  const b = requireObj(raw, "Artifact bundle");
  const expected = ["published_reference", "baseline_results", "ckks_validation", "ckks_final_test", "ckks_smoke", "phase3_ablation", "phase3_ckks_validation", "phase3_splits", "baseline_predictions", "phase3_predictions"] as const;
  for (const key of expected) if (!(key in b)) throw new Error(`Required artifact is missing: ${key}.`);
  const published = requireObj(b.published_reference, "published_reference");
  const baseline = requireObj(b.baseline_results, "baseline_results");
  const phase3 = requireObj(b.phase3_ablation, "phase3_ablation");
  if (!isArray(published.records)) throw new Error("published_reference.records must be an array.");
  if (!isObj(baseline.models)) throw new Error("baseline_results.models must be an object.");
  if (!isObj(phase3.benchmarks)) throw new Error("phase3_ablation.benchmarks must be an object.");
  for (const [task, benchmark] of Object.entries(phase3.benchmarks)) {
    if (!isObj(benchmark) || !isObj(benchmark.models)) throw new Error(`phase3_ablation benchmark ${task} is malformed.`);
  }
  if (typeof b.baseline_predictions !== "string" || typeof b.phase3_predictions !== "string") throw new Error("Prediction CSV artifacts must be text.");
  return b as unknown as ArtifactBundle;
}

function parseCsv(text: string, label: string): Obj[] {
  const rows: string[][] = [];
  let row: string[] = []; let field = ""; let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { row.push(field); field = ""; }
    else if (char === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += char;
  }
  if (quoted) throw new Error(`${label} contains an unterminated quoted field.`);
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  const header = rows.shift()?.map((cell) => cell.trim());
  if (!header?.length || header.some((cell) => !cell)) throw new Error(`${label} has no valid header row.`);
  if (new Set(header).size !== header.length) throw new Error(`${label} contains duplicate column names.`);
  const dataRows = rows.filter((values) => values.some((value) => value !== ""));
  if (dataRows.some((values) => values.length !== header.length)) throw new Error(`${label} has rows with a different number of fields than its header.`);
  const parsed = dataRows.map((values) => Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""])));
  if (!parsed.length) throw new Error(`${label} contains no prediction rows.`);
  return parsed;
}

function predictionValues(rows: Obj[], taskKey: string | null, predictionColumn: string): { labels?: number[]; predictions?: number[] } {
  const selected = rows.filter((row) => taskKey === null || row.task === taskKey).sort((a, b) => (num(a.test_position) ?? 0) - (num(b.test_position) ?? 0));
  if (!selected.length) return {};
  const labels = selected.map((row) => Number.parseInt(String(row.y_true), 10));
  const predictions = selected.map((row) => Number.parseInt(String(row[predictionColumn]), 10));
  if (labels.some((value) => value !== 0 && value !== 1) || predictions.some((value) => value !== 0 && value !== 1)) throw new Error(`Prediction CSV has invalid binary values in column ${predictionColumn}.`);
  return { labels, predictions };
}

function mappedMode(mode: unknown): ExecutionMode {
  if (mode === "plaintext") return "Published plaintext";
  if (mode === "CKKS") return "Published CKKS";
  if (mode === "BFV") return "Published BFV";
  return "Diagnostic / unavailable";
}

function recordsFromPublished(raw: unknown): ExperimentRecord[] {
  const p = obj(raw);
  const records = arr(p.records);
  return records.filter(isObj).map((r, index) => {
    const paperMetrics = {
      accuracy: num(r.accuracy), precision: num(r.precision), recall: num(r.recall), f1: num(r.f1),
      macro_metrics: {}, test_support: num(r.test_support),
    };
    const support = num(r.test_support);
    const metrics = metricSet(paperMetrics);
    metrics.support = support;
    return baseRecord({
      id: `paper-${index}`, task: str(r.task) ?? "Unknown task", variant: str(r.model) ?? "Published model",
      executionMode: mappedMode(r.execution_mode), status: "Reported by paper", provenance: "Published",
      runId: null, timestamp: null, seed: null, splitId: null,
      splitLabel: support === null ? "Test support not reported" : `Reported test support n=${support}`,
      sourceArtifact: `published_reference.json · ${str(r.source_location) ?? "source location not recorded"}`,
      metrics, timings: { latency_ms_per_sample: num(r.latency_ms_per_sample) },
      ciphertextSizes: { ciphertext_size_bytes: num(r.ciphertext_size_bytes) }, privacyBoundary:
        r.execution_mode === "CKKS" && String(r.model).toLowerCase().includes("convolution")
          ? "Convolution features encrypted; pooled features are decrypted before the plaintext Logistic Regression head (paper-described path)."
          : "Execution boundary is as described in the published table; not independently reimplemented here.",
      configuration: { publishedThroughputSamplesPerSecond: num(r.throughput_samples_per_second), sourceLocation: r.source_location },
      missingMetrics: metricsUnavailable.filter((name) => {
        if (name === "precision") return metrics.precision === null;
        if (name === "recall") return metrics.recall === null;
        if (name === "f1") return metrics.f1 === null;
        if (name === "confusion matrix") return metrics.confusion === null;
        return metrics.macroF1 === null;
      }),
    });
  });
}

function recordsFromBaseline(raw: unknown, predictionsCsv: Obj[]): ExperimentRecord[] {
  const b = obj(raw); const models = obj(b.models);
  return Object.entries(models).filter(([, v]) => isObj(v)).map(([key, value]) => {
    const m = obj(value); const metrics = metricSet(m.metrics);
    const name = key === "tinycnn" ? "TinyCNN baseline" : key === "logistic_regression" ? "Logistic Regression" : key === "majority" ? "Majority baseline" : "Plaintext-equivalent polynomial graph";
    const csvColumn: Record<string, string> = { majority: "majority_prediction", logistic_regression: "logistic_regression_prediction", tinycnn: "tinycnn_prediction", plaintext_equivalent_polynomial_graph: "polynomial_graph_prediction" };
    const predicted = predictionValues(predictionsCsv, null, csvColumn[key] ?? "");
    return baseRecord({
      id: `baseline-${key}`, task: str(b.task) ?? "digit 0 vs digit 1", variant: name,
      executionMode: "Local plaintext", status: str(b.status) ?? "Recorded", provenance: "Measured",
      runId: str(b.run_id), timestamp: str(b.timestamp_utc), seed: num(b.seed), splitId: str(b.split_id),
      splitLabel: `Held-out test · ${metrics.support ?? "support unavailable"} samples`,
      sourceArtifact: "baseline_results.json", metrics,
      timings: { training_seconds: num(m.training_seconds), inference_ms_per_sample: num(m.test_batch_model_inference_ms_per_sample), preprocessing_ms_per_sample: num(m.test_preprocessing_ms_per_sample) },
      privacyBoundary: "Plaintext execution; no encrypted evaluation.",
      configuration: { architecture: m.architecture, threshold: m.threshold, solver: m.solver, selection: m.selection },
      labels: predicted.labels ?? (isArray(b.test_labels) ? b.test_labels.map((x) => num(x) ?? 0) : undefined),
      predictions: predicted.predictions ?? (isArray(m.predictions) ? m.predictions.map((x) => num(x) ?? 0) : undefined),
    });
  });
}

function recordsFromPhase2(raw: unknown): ExperimentRecord[] {
  const d = obj(raw); const result = obj(obj(d.parameter_results).depth3_40bit_exploratory);
  if (!Object.keys(result).length) return [];
  const samples = arr(result.samples).filter(isObj);
  const numericMean = (key: string) => {
    const values = samples.map((sample) => num(obj(sample.timings_seconds)[key])).filter((v): v is number => v !== null);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const modelPredictions = arr(result.predictions).filter(isObj);
  const firstSample = obj(samples[0]);
  const errors = obj(firstSample.errors);
  const agreement = obj(result.prediction_agreement);
  return [baseRecord({
    id: `phase2-${str(d.mode) ?? "ckks"}`, task: "digit 0 vs digit 1", variant: "TinyCNN · supported CKKS feature path",
    executionMode: "CKKS features + plaintext head", status: samples.length ? "Completed" : "Recorded",
    provenance: "Measured", runId: str(d.run_id), timestamp: str(d.timestamp_utc), seed: num(d.seed), splitId: str(d.split_id),
    splitLabel: `${str(d.mode) ?? "CKKS"} · ${samples.length} samples`, sourceArtifact: `${str(d.mode) === "final-test" ? "ckks_final_test.json" : "ckks_validation.json"}`,
    metrics: metricSet(result.metrics), timings: { encrypt_seconds: numericMean("encrypt"), encrypted_convolution_seconds: numericMean("encrypted_convolution"), encrypted_square_seconds: numericMean("encrypted_square"), encrypted_pooling_seconds: numericMean("encrypted_pooling"), decrypt_seconds: numericMean("decrypt_intermediates"), total_feature_extraction_seconds: numericMean("total_feature_extraction") },
    agreement: { count: num(agreement.agreement_count) ?? 0, total: num(agreement.sample_count) ?? 0, flips: num(agreement.flipped_predictions) ?? 0 },
    numericalErrors: Object.fromEntries(Object.entries(errors).filter(([, value]) => isObj(value)).map(([key, value]) => [key, { meanMae: num(obj(value).mean_absolute_error), worstMax: num(obj(value).max_absolute_error) }])),
    ciphertextSizes: {}, privacyBoundary: str(d.head_execution) ?? "Pooled features are decrypted before the plaintext head.",
    configuration: { parameters: result.parameters, contextFlags: result.context_flags, executionMode: d.mode },
    predictions: modelPredictions.map((item) => num(obj(item).encrypted_prediction) ?? 0),
    labels: modelPredictions.map((item) => num(obj(item).label) ?? 0),
  })];
}

const phase3Keys: Record<string, string> = {
  A0_standard_scaler_control: "A0 · StandardScaler control",
  A1_pixel_scale_only: "A1 · Pixel scaling only",
  A2_standard_scaler_plus_range_penalty: "A2 · Range penalty only",
  A3_pixel_scale_plus_range_penalty_RC_TinyCNN: "A3 · RC-TinyCNN",
};

function recordsFromPhase3(raw: unknown, predictionsCsv: Obj[]): ExperimentRecord[] {
  const p = obj(raw); const output: ExperimentRecord[] = [];
  for (const [taskKey, rawTask] of Object.entries(obj(p.benchmarks))) {
    const task = obj(rawTask); const models = obj(task.models); const splitSizes = obj(task.split_sizes);
    const taskLabel = taskKey === "digits_0_vs_1" ? "digit 0 vs digit 1" : taskKey === "digits_3_vs_8" ? "digit 3 vs digit 8" : str(task.task) ?? taskKey;
    for (const [key, value] of Object.entries(models)) {
      const m = obj(value); const cfg = obj(m.configuration);
      if (!key.startsWith("A")) continue;
      const csvColumn: Record<string, string> = { A0_standard_scaler_control: "A0_standard_scaler_control", A1_pixel_scale_only: "A1_pixel_scale_only", A2_standard_scaler_plus_range_penalty: "A2_standard_scaler_plus_range_penalty", A3_pixel_scale_plus_range_penalty_RC_TinyCNN: "A3_pixel_scale_plus_range_penalty_RC_TinyCNN" };
      const predictionColumn = key === "A3_pixel_scale_plus_range_penalty_RC_TinyCNN" ? "A3_pixel_scale_plus_range_penalty_RC_TinyCNN" : csvColumn[key];
      const predicted = predictionValues(predictionsCsv, taskKey, predictionColumn);
      const common = {
        task: taskLabel, executionMode: "Local plaintext" as const, status: "Completed · final test once after freeze",
        provenance: "Measured" as const, runId: str(p.run_id), timestamp: str(p.completed_utc) ?? str(p.timestamp_utc),
        seed: num(cfg.seed), splitId: str(task.split_id),
        splitLabel: `Train ${num(splitSizes.train) ?? "?"} / test ${num(splitSizes.test) ?? "?"} · seed ${num(cfg.seed) ?? "?"}`,
        sourceArtifact: "phase3_ablation.json + phase3_predictions.csv", timings: { training_seconds: num(m.total_model_training_seconds) },
        privacyBoundary: "Plaintext training and inference. No encrypted evaluation in this record.",
        configuration: cfg, labels: predicted.labels ?? arr(obj(task.final_test_evaluation).test_labels).map((x) => num(x) ?? 0),
        activationDiagnostics: isObj(m.test_activation_diagnostics) ? m.test_activation_diagnostics : null,
      };
      const metrics = metricSet(m.test_metrics_threshold_zero);
      const a0 = metricSet(obj(models.A0_standard_scaler_control).test_metrics_threshold_zero);
      const delta = (value: number | null, base: number | null) => value === null || base === null ? null : (value - base) * 100;
      const pred = arr(m.test_predictions_threshold_zero).map((x) => num(x) ?? 0);
      output.push(baseRecord({
        ...common, id: `${taskKey}-${key}`, variant: phase3Keys[key] ?? key, metrics,
        predictions: predicted.predictions ?? pred, labels: common.labels,
        deltasVsA0: { accuracy: delta(metrics.accuracy, a0.accuracy), precision: delta(metrics.precision, a0.precision), recall: delta(metrics.recall, a0.recall), f1: delta(metrics.f1, a0.f1) },
      }));
      if (key === "A3_pixel_scale_plus_range_penalty_RC_TinyCNN") {
        const thresholdMetrics = metricSet(m.A4_test_metrics_validation_threshold);
        output.push(baseRecord({
          ...common, id: `${taskKey}-A4-validation-threshold`, variant: "A4 · validation-selected threshold",
          metrics: thresholdMetrics, predictions: predictionValues(predictionsCsv, taskKey, "A4_RC_validation_threshold").predictions ?? arr(m.A4_test_predictions_validation_threshold).map((x) => num(x) ?? 0),
          labels: common.labels,
          deltasVsA0: { accuracy: delta(thresholdMetrics.accuracy, a0.accuracy), precision: delta(thresholdMetrics.precision, a0.precision), recall: delta(thresholdMetrics.recall, a0.recall), f1: delta(thresholdMetrics.f1, a0.f1) },
          configuration: { ...cfg, thresholdSelection: obj(task.A4_validation_selected_threshold) },
        }));
      }
    }
  }
  return output;
}

function recordsFromPhase3CKKS(raw: unknown): ExperimentRecord[] {
  const d = obj(raw); const models = obj(d.models);
  return Object.entries(models).filter(([, value]) => isObj(value)).map(([key, rawModel]) => {
    const m = obj(rawModel); const agreement = obj(m.prediction_agreement_with_plaintext_graph);
    const errors = obj(m.numerical_error_summary); const timing = obj(m.timing_summary);
    const first = obj(arr(m.per_sample_results)[0]);
    const mode: ExecutionMode = "CKKS validation partial path";
    const modelName = str(m.model) ?? key;
    const timings = Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, num(obj(v).mean_seconds)]));
    return baseRecord({
      id: `phase3-ckks-${key}`, task: "digit 0 vs digit 1", variant: `${modelName} · ${str(m.method) ?? "CKKS"}`,
      executionMode: mode, status: "Completed · primary validation only", provenance: "Measured",
      runId: "phase3-ckks-validation", timestamp: null, seed: 2026, splitId: "digits01-seed2026-e1b8b781ea09ee9f",
      splitLabel: `Validation · ${num(agreement.sample_count) ?? "?"} samples`, sourceArtifact: "phase3_ckks_validation.json",
      metrics: metricSet(m.metrics), timings, agreement: { count: num(agreement.agreement_count) ?? 0, total: num(agreement.sample_count) ?? 0, flips: num(agreement.flipped_predictions) ?? 0 },
      numericalErrors: Object.fromEntries(Object.entries(errors).filter(([, v]) => isObj(v)).map(([k, v]) => [k, { meanMae: num(obj(v).mean_of_sample_mae), worstMax: num(obj(v).worst_sample_max_abs_error) }])),
      ciphertextSizes: (() => {
        const rawSizes = obj(first.serialized_ciphertext_bytes);
        return Object.fromEntries(Object.entries(rawSizes).filter(([name]) => name.endsWith("_bytes") || (m.method === "im2col_partial" && ["input", "convolution", "square"].includes(name))).map(([name, value]) => [name.endsWith("_bytes") ? name : `${name}_bytes`, num(value)]));
      })(),
      privacyBoundary: str(m.privacy_boundary) ?? "Encrypted path boundary unavailable.",
      configuration: { method: m.method, parameterSet: m.parameter_set, parameters: m.parameters, threadCount: d.thread_count, batchSize: d.batch_size, warmupCount: d.warmup_count },
    });
  });
}

function recordsFromSmoke(raw: unknown): ExperimentRecord[] {
  const d = obj(raw); const results = obj(d.parameter_results);
  return Object.entries(results).filter(([, value]) => isObj(value)).map(([setName, rawResult]) => {
    const r = obj(rawResult); const exceptions = arr(r.exceptions);
    return baseRecord({
      id: `smoke-${setName}`, task: "digit 0 vs digit 1", variant: `CKKS smoke · ${setName}`,
      executionMode: "Diagnostic / unavailable", status: exceptions.length ? `Completed with ${exceptions.length} recorded exception(s)` : "Completed smoke test",
      provenance: "Measured", runId: str(d.run_id), timestamp: str(d.timestamp_utc), seed: num(d.seed), splitId: str(d.split_id),
      splitLabel: "Diagnostic sample · not a benchmark evaluation", sourceArtifact: "ckks_smoke.json",
      configuration: { parameters: r.parameters, contextCreated: r.context_created, exceptions },
      privacyBoundary: "Diagnostic only; not a full classification benchmark.",
      missingMetrics: ["accuracy", "positive-class precision", "positive-class recall", "positive-class F1", "confusion matrix"],
    });
  });
}

export function buildDashboardData(raw: unknown): DashboardData {
  const bundle = validateArtifactBundle(raw);
  const baselinePredictions = parseCsv(String(bundle.baseline_predictions), "baseline_predictions.csv");
  const phase3Predictions = parseCsv(String(bundle.phase3_predictions), "phase3_predictions.csv");
  for (const column of ["y_true", "majority_prediction", "logistic_regression_prediction", "tinycnn_prediction", "polynomial_graph_prediction"]) {
    if (!(column in baselinePredictions[0])) throw new Error(`baseline_predictions.csv is missing column ${column}.`);
  }
  for (const column of ["task", "test_position", "y_true", "A0_standard_scaler_control", "A1_pixel_scale_only", "A2_standard_scaler_plus_range_penalty", "A3_pixel_scale_plus_range_penalty_RC_TinyCNN", "A4_RC_validation_threshold"]) {
    if (!(column in phase3Predictions[0])) throw new Error(`phase3_predictions.csv is missing column ${column}.`);
  }
  const phase3 = obj(bundle.phase3_ablation);
  const phase3Splits = obj(bundle.phase3_splits);
  const taskSplits: DashboardData["taskSplits"] = {};
  for (const [task, split] of Object.entries(phase3Splits)) {
    const s = obj(split); const splitIndices = obj(s.indices);
    taskSplits[task] = { splitId: str(s.split_id), seed: num(s.seed), sizes: Object.fromEntries(Object.entries(splitIndices).map(([name, indices]) => [name, arr(indices).length])) };
  }
  const records = [
    ...recordsFromPublished(bundle.published_reference),
    ...recordsFromBaseline(bundle.baseline_results, baselinePredictions),
    ...recordsFromPhase2(bundle.ckks_validation),
    ...recordsFromPhase2(bundle.ckks_final_test),
    ...recordsFromPhase3(bundle.phase3_ablation, phase3Predictions),
    ...recordsFromPhase3CKKS(bundle.phase3_ckks_validation),
    ...recordsFromSmoke(bundle.ckks_smoke),
  ];
  return {
    records,
    publishedPaper: str(obj(bundle.published_reference).paper) ?? "Published paper title unavailable",
    phase3RunId: str(phase3.run_id) ?? "Run ID unavailable",
    phase3Timestamp: str(phase3.completed_utc) ?? str(phase3.timestamp_utc),
    taskSplits,
  };
}

export const artifactPaths = [
  "published_reference.json", "baseline_results.json", "ckks_validation.json", "ckks_final_test.json",
  "ckks_smoke.json", "phase3_ablation.json", "phase3_ckks_validation.json", "phase3_splits.json",
] as const;
const predictionCsvPaths = ["baseline_predictions.csv", "phase3_predictions.csv"] as const;

export async function loadDashboardData(fetcher: typeof fetch = fetch, artifactBase = `${import.meta.env.BASE_URL}artifacts/`, fallbackArtifactBase = `${import.meta.env.BASE_URL}artifacts/`): Promise<DashboardData> {
  const loadResponse = async (name: string) => {
    let response = await fetcher(`${artifactBase}${name}`, { cache: "no-store" });
    if (!response.ok && artifactBase !== fallbackArtifactBase) response = await fetcher(`${fallbackArtifactBase}${name}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not load artifacts/${name} (HTTP ${response.status}).`);
    return response;
  };
  const jsonEntries = await Promise.all(artifactPaths.map(async (name) => {
    const response = await loadResponse(name);
    try { return [name.replace(".json", ""), await response.json()] as const; }
    catch { throw new Error(`artifacts/${name} is not valid JSON.`); }
  }));
  const csvEntries = await Promise.all(predictionCsvPaths.map(async (name) => {
    const response = await loadResponse(name);
    return [name.replace(".csv", ""), await response.text()] as const;
  }));
  return buildDashboardData(Object.fromEntries([...jsonEntries, ...csvEntries]));
}
