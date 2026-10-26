import { useEffect, useMemo, useState } from "react";
import { executionModes, filterRecords, formatPercent, primaryTask, secondaryTask } from "../data/compare";
import { loadDashboardData } from "../data/transform";
import type { DashboardData, ExperimentRecord, MetricSet } from "../data/types";

type Page = "Model Lab" | "Results" | "Optimization & Architecture";
type BenchmarkResult = { run_id: string; completed_at: string; task: string; dataset: string; class_mapping: string; split: { method: string; seed: number; fit_samples: number; validation_samples: number; test_samples: number }; models: Record<string, { name: string; description: string; metrics: Record<string, unknown>; training: { epochs: number; best_epoch: number; seconds: number; fit_samples: number; validation_samples: number; test_samples: number; range_bound: number | null; range_penalty: number }; } > };
type BenchmarkRun = { available: boolean; status: "idle" | "running" | "completed" | "failed"; runId?: string; task?: string; logs: string[]; error?: string; result?: BenchmarkResult };
type DatasetInfo = { dataset: string; task: string; class_mapping: Record<string, number>; source_url: string; description: string; sample_count: number; feature_count: number; class_counts: Record<string, number>; columns: string[]; preview: (number | string)[][]; csv: string };
const digitPairs: Record<string, [number, number]> = { digits_0_vs_1: [0, 1], digits_2_vs_7: [2, 7], digits_3_vs_8: [3, 8], digits_4_vs_9: [4, 9] };
function datasetInfoFromCsv(task: string, csv: string): DatasetInfo {
  const rows = csv.trim().split(/\r?\n/).map((line) => line.split(",").map((cell) => Number(cell)));
  const columns = csv.trim().split(/\r?\n/, 1)[0].split(",");
  const [low, high] = digitPairs[task];
  const dataRows = rows.slice(1);
  return { dataset: "scikit-learn Digits", task, class_mapping: { [String(low)]: 0, [String(high)]: 1 }, source_url: "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html", description: "8×8 grayscale handwritten digit images. Pixel values are integers from 0 to 16. class_label is the binary label used by the benchmark; digit is the original digit.", sample_count: dataRows.length, feature_count: 64, class_counts: { "0": dataRows.filter((row) => row[1] === 0).length, "1": dataRows.filter((row) => row[1] === 1).length }, columns, preview: dataRows.slice(0, 10), csv };
}
const pages: { title: Page; icon: string; eyebrow: string }[] = [
  { title: "Model Lab", icon: "◉", eyebrow: "TRAIN + EVALUATE" },
  { title: "Results", icon: "◫", eyebrow: "RESULTS" },
  { title: "Optimization & Architecture", icon: "⇄", eyebrow: "MODEL DESIGN" },
];

function BenchmarkMetric({ label, value, help }: { label: string; value: unknown; help?: string }) {
  const numeric = typeof value === "number" ? value : null;
  return <div className="benchmark-metric"><span>{label}</span><strong>{numeric === null ? "Not reported" : formatPercent(numeric)}</strong>{help && <small>{help}</small>}</div>;
}

function ModelLab({ run, busy, task, onTask, onRun, onClear }: { run: BenchmarkRun | null; busy: boolean; task: string; onTask: (task: string) => void; onRun: () => void; onClear: () => void }) {
  const [modelTab, setModelTab] = useState("paper_model");
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [datasetInfoError, setDatasetInfoError] = useState("");
  const result = run?.result?.task === task ? run.result : undefined;
  const model = result?.models[modelTab];
  const metrics = model?.metrics ?? {};
  const confusion = metrics.confusion_matrix as number[][] | undefined;
  const taskLabel = `Digits ${task.replace("digits_", "").replace("_vs_", " vs ")}`;
  useEffect(() => {
    let active = true;
    setDatasetInfo(null);
    setDatasetInfoError("");
    fetch(`/api/benchmark/dataset?task=${encodeURIComponent(task)}`, { cache: "no-store" })
      .then(async (response) => { if (response.ok) return response.json() as Promise<DatasetInfo>; const staticResponse = await fetch(`${import.meta.env.BASE_URL}datasets/${task}.csv`); if (!staticResponse.ok) throw new Error("Could not load the dataset preview."); return datasetInfoFromCsv(task, await staticResponse.text()); })
      .then((info) => { if (active) setDatasetInfo(info); })
      .catch((error) => { if (active) setDatasetInfoError(error instanceof Error ? error.message : "Dataset preview is unavailable."); });
    return () => { active = false; };
  }, [task]);
  const downloadDataset = () => {
    if (!datasetInfo) return;
    const url = URL.createObjectURL(new Blob([datasetInfo.csv], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${task}.csv`; anchor.click(); URL.revokeObjectURL(url);
  };
  return <>
    <section className="page-intro model-lab-intro"><div><div className="eyebrow">TRAINING + EVALUATION</div><h1>Model Lab</h1><p>Train both models on the same dataset and compare their results on unseen test examples.</p></div><div className="model-lab-pick"><label htmlFor="benchmark-task">Choose dataset</label><select id="benchmark-task" value={task} onChange={(event) => onTask(event.target.value)} disabled={run?.status === "running"}><option value="digits_0_vs_1">Digits 0 vs 1</option><option value="digits_2_vs_7">Digits 2 vs 7</option><option value="digits_3_vs_8">Digits 3 vs 8</option><option value="digits_4_vs_9">Digits 4 vs 9</option></select></div></section>
    <details className="dataset-inspector" open><summary>Inspect dataset <span>Class balance and links to the full data</span></summary><div className="dataset-inspector-body">{datasetInfo ? <><div className="dataset-summary"><div><b>{datasetInfo.dataset} · {taskLabel}</b><p>{datasetInfo.description}</p></div><div className="dataset-actions"><a className="button subtle" href={datasetInfo.source_url} target="_blank" rel="noreferrer">Dataset source ↗</a><button className="button subtle" onClick={downloadDataset}>Download all rows · CSV</button></div></div><div className="dataset-facts"><span><b>{datasetInfo.sample_count.toLocaleString()}</b> examples</span><span><b>{datasetInfo.feature_count}</b> pixel features</span><span><b>Class 0:</b> {datasetInfo.class_counts["0"]}</span><span><b>Class 1:</b> {datasetInfo.class_counts["1"]}</span><span><b>Mapping:</b> {Object.entries(datasetInfo.class_mapping).map(([digit, label]) => `${digit} → ${label}`).join(" · ")}</span></div></> : <div className="dataset-loading">{datasetInfoError || "Loading dataset details…"} <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html" target="_blank" rel="noreferrer">View dataset documentation ↗</a></div>}</div></details>
    <section className="panel benchmark-start"><div><div className="eyebrow">ONE CLICK RUN</div><h2>{run?.status === "running" ? "Training both models…" : run?.status === "completed" ? "Evaluation complete" : "Train and evaluate the models"}</h2><p>The same training, validation, and held-out test split is used for both models. Metrics below are calculated from the test split only.</p></div><div className="benchmark-start-actions"><button className="button primary" disabled={busy || run?.status === "running" || !run?.available} onClick={onRun}>{run?.status === "running" || busy ? "Training…" : "Train + evaluate"}</button>{result && <button className="button subtle" disabled={busy} onClick={onClear}>Clear results</button>}</div></section>
    {!run?.available && <div className="callout caution-callout">Training runs on your local dashboard because GitHub Pages cannot start Python jobs. Start the app with <code>npm run dev</code> to train models here.</div>}
    {run?.error && <div className="demo-error benchmark-error" role="alert">{run.error}</div>}
    {run?.status === "running" && <div className="panel benchmark-progress"><span className="live-dot" />Training the paper-style model and searching the optimized model settings. This can take a few minutes.</div>}
    {run?.logs?.length && (run.status === "running" || run.status === "failed") ? <details className="demo-log benchmark-log" open><summary>Run progress</summary><pre>{run.logs.join("\n")}</pre></details> : null}
    <section className="benchmark-model-panel panel">
      <div className="benchmark-panel-head"><div><div className="eyebrow">MODEL RESULTS · {result ? taskLabel : "READY WHEN YOU RUN"}</div><h2>{model?.name ?? "Choose a model tab"}</h2><p>{model?.description ?? "Train the models to show their held-out test metrics here."}</p></div>{result && <span className="status-pill measured"><i />Test set · {result.split.test_samples} samples</span>}</div>
      <div className="benchmark-tabs" role="tablist" aria-label="Model results">
        <button role="tab" aria-selected={modelTab === "paper_model"} className={modelTab === "paper_model" ? "active" : ""} onClick={() => setModelTab("paper_model")}>Paper Model <small>Base model</small></button>
        <button role="tab" aria-selected={modelTab === "cipherbench_model"} className={modelTab === "cipherbench_model" ? "active" : ""} onClick={() => setModelTab("cipherbench_model")}>CipherBench Model <small>Optimized model</small></button>
      </div>
      {!result ? <div className="empty-state benchmark-empty">Select a dataset, then choose <b>Train + evaluate</b>. The two model tabs will show separate results.</div> : model ? <>
        <div className="benchmark-metrics-grid"><BenchmarkMetric label="Accuracy" value={metrics.accuracy} help="Correct predictions"/><BenchmarkMetric label="Precision" value={(metrics.positive_class_metrics as Record<string, unknown> | undefined)?.precision} help="Positive class"/><BenchmarkMetric label="Recall (R1)" value={(metrics.positive_class_metrics as Record<string, unknown> | undefined)?.recall} help="Recall for class 1"/><BenchmarkMetric label="F1 score" value={(metrics.positive_class_metrics as Record<string, unknown> | undefined)?.f1} help="Positive class"/><BenchmarkMetric label="ROC-AUC" value={metrics.roc_auc} help="Ability to separate the two classes"/></div>
        <div className="benchmark-detail-grid"><div className="benchmark-confusion"><h3>Confusion matrix</h3><p>Rows are the actual class. Columns are the model’s prediction.</p><div className="confusion-wrap"><div className="confusion-axis">Predicted class</div><div className="confusion-grid"><div className="confusion-side">Actual</div><div className="confusion-corner"/><div className="confusion-colhead">0</div><div className="confusion-colhead">1</div><div className="confusion-rowhead">0</div>{[0,1].map((col)=><div key={`0-${col}`} className={`confusion-cell ${col===0?"tn":"fp"}`}>{confusion?.[0]?.[col] ?? "—"}</div>)}<div className="confusion-rowhead">1</div>{[0,1].map((col)=><div key={`1-${col}`} className={`confusion-cell ${col===0?"fn":"tp"}`}>{confusion?.[1]?.[col] ?? "—"}</div>)}</div><div className="confusion-legend"><span><i className="true"/>Correct</span><span><i className="false"/>Incorrect</span></div></div></div><div className="benchmark-run-info"><h3>How this run was made</h3><p><b>Dataset:</b> {result.dataset} · {result.class_mapping}</p><p><b>Training examples:</b> {model.training.fit_samples}</p><p><b>Validation examples:</b> {model.training.validation_samples}</p><p><b>Test examples:</b> {model.training.test_samples}</p><p><b>Training time:</b> {model.training.seconds}s</p>{modelTab === "cipherbench_model" && <p><b>Selected range settings:</b> {model.training.range_bound} · strength {model.training.range_penalty}</p>}<small>Test examples were kept separate from model training and settings selection.</small></div></div>
      </> : <div className="empty-state benchmark-empty">No results are available for this model.</div>}
    </section>
  </>;
}

function Status({ value }: { value: string }) {
  const kind = value.toLowerCase().includes("published") || value.toLowerCase().includes("paper") ? "published" : value.toLowerCase().includes("validation") || value.toLowerCase().includes("provisional") ? "provisional" : value.toLowerCase().includes("unavailable") ? "unavailable" : "measured";
  return <span className={`status-pill ${kind}`}><i />{value}</span>;
}

function metricFor(records: ExperimentRecord[], task: string, variant: string): ExperimentRecord | undefined {
  const matches = records.filter((record) => record.task === task && record.variant === variant);
  return matches.find((record) => record.sourceArtifact.includes("phase3_ablation.json")) ?? matches.find((record) => record.sourceArtifact === "baseline_results.json") ?? matches[0];
}

type CompareMetric = "accuracy" | "precision" | "recall" | "f1";
function AccuracyChart({ records, selectedId, metric, onSelect }: { records: ExperimentRecord[]; selectedId?: string; metric: CompareMetric; onSelect: (r: ExperimentRecord) => void }) {
  const rows = records.filter((r) => r.metrics[metric] !== null && r.executionMode !== "Diagnostic / unavailable");
  const width = 950; const rowH = 42; const labelX = 0; const barX = 265; const barMax = 520; const valueX = 810; const height = Math.max(150, rows.length * rowH + 35);
  if (!rows.length) return <div className="empty-state">No accuracy values are available for this task and execution mode.</div>;
  return <div className="chart-scroll"><svg className="accuracy-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Accuracy comparison by model. Validation-only CKKS results are excluded.">
    {[0, .25, .5, .75, 1].map((tick) => <g key={tick}><line x1={barX + tick * barMax} x2={barX + tick * barMax} y1="12" y2={height - 14} className="chart-grid" /><text x={barX + tick * barMax} y="10" textAnchor="middle" className="chart-axis">{Math.round(tick * 100)}%</text></g>)}
    {rows.map((r, i) => {
      const y = 24 + i * rowH; const value = r.metrics[metric] ?? 0; const paper = r.provenance === "Published"; const selected = r.id === selectedId;
      const color = paper ? "#d9a353" : r.executionMode.includes("CKKS") ? "#67c5bb" : r.variant.startsWith("A3") ? "#9d8bff" : "#5a9eef";
      return <g key={r.id} className="chart-row" role="button" tabIndex={0} aria-label={`${r.variant}, ${metric}: ${formatPercent(value)}, ${r.splitLabel}`} onClick={() => onSelect(r)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onSelect(r); }}><title>{`${r.variant} · ${metric} ${formatPercent(value)} · ${r.splitLabel} · ${r.sourceArtifact}`}</title>
        <text x={labelX} y={y + 14} className={selected ? "chart-label selected" : "chart-label"}>{r.variant.length > 33 ? `${r.variant.slice(0, 31)}…` : r.variant}</text>
        <rect x={barX} y={y} width={barMax} height="22" rx="5" className="bar-track" />
        <rect x={barX} y={y} width={barMax * value} height="22" rx="5" fill={color} className={selected ? "bar selected" : "bar"} />
        <text x={valueX} y={y + 15} className="chart-value">{formatPercent(value)}</text>
        <text x={barX} y={y + 35} className="chart-caption">{r.provenance} · {r.executionMode} · {r.splitLabel}</text>
      </g>;
    })}
  </svg></div>;
}

function Metric({ label, value, helper, accent = "" }: { label: string; value: string; helper: string; accent?: string }) {
  return <article className={`metric-card ${accent}`}><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-helper">{helper}</div></article>;
}

function Confusion({ metrics }: { metrics: MetricSet }) {
  const matrix = metrics.confusion;
  return <div className="confusion-wrap">
    <div className="confusion-axis">Predicted class</div>
    <div className="confusion-grid">
      <div className="confusion-side">Actual</div><div className="confusion-corner" />
      <div className="confusion-colhead">0</div><div className="confusion-colhead">1</div>
      <div className="confusion-rowhead">0</div>{[0, 1].map((col) => <div key={`a${col}`} className={`confusion-cell ${col === 0 ? "tn" : "fp"}`}>{matrix?.[0]?.[col] ?? "Not reported"}</div>)}
      <div className="confusion-rowhead">1</div>{[0, 1].map((col) => <div key={`b${col}`} className={`confusion-cell ${col === 0 ? "fn" : "tp"}`}>{matrix?.[1]?.[col] ?? "Not reported"}</div>)}
    </div>
    <div className="confusion-legend"><span><i className="true" />Correct</span><span><i className="false" />Misclassified</span></div>
  </div>;
}

function ClassMetrics({ metrics }: { metrics: MetricSet }) {
  if (!metrics.perClass) return <div className="empty-state compact">Per-class metrics: Not reported.</div>;
  return <table className="mini-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{Object.entries(metrics.perClass).map(([label, value]) => <tr key={label}><td>{label}</td><td>{formatPercent(value.precision)}</td><td>{formatPercent(value.recall)}</td><td>{formatPercent(value.f1)}</td><td>{value.support ?? "Unavailable"}</td></tr>)}</tbody></table>;
}

function Overview({ data, task, mode, variant, onTask, onMode, onVariant, selected, onSelect }: { data: DashboardData; task: string; mode: string; variant: string; onTask: (v: string) => void; onMode: (v: string) => void; onVariant: (v: string) => void; selected: ExperimentRecord | undefined; onSelect: (r: ExperimentRecord) => void }) {
  const [chartMetric, setChartMetric] = useState<CompareMetric>("accuracy");
  const filtered = filterRecords(data.records, task, mode).filter((r) => !r.variant.includes("smoke"));
  const chartRecords = filtered.filter((r) => r.metrics.accuracy !== null && r.executionMode !== "Diagnostic / unavailable" && (mode === "CKKS validation partial path" || (!r.executionMode.includes("validation") && !r.splitLabel.toLowerCase().includes("validation"))));
  const selectedMatchesFilters = selected && selected.task === task && (mode === "All modes" || selected.executionMode === mode) ? selected : undefined;
  const focused = selectedMatchesFilters ?? metricFor(filtered, task, variant) ?? filtered[0];
  const variants = [...new Set(data.records.filter((r) => r.task === task && r.metrics.accuracy !== null).map((r) => r.variant))];
  const ablationTaskKey = task === primaryTask ? "digits_0_vs_1" : "digits_3_vs_8";
  const ckksFidelity = data.records.find((r) => r.task === primaryTask && r.variant.includes("A3") && r.executionMode === "CKKS validation partial path");
  return <>
    <section className="hero-panel">
      <div className="hero-copy"><div className="eyebrow">INDEPENDENT RESEARCH / CKKS INFERENCE</div><h1>Homomorphic inference,<br /><em>measured honestly.</em></h1><p>Can input calibration and range-aware training improve shallow CNN inference under CKKS—without trading away accuracy or practical cost?</p>
        <div className="hero-chips"><span><i className="dot mint" />Evidence-linked</span><span><i className="dot amber" />Small-sample study</span><span><i className="dot violet" />Encrypted path partial</span></div>
      </div>
      <div className="hero-paper"><div className="paper-icon">▤</div><div className="eyebrow">PUBLISHED REFERENCE</div><strong>Homomorphic Encryption for Privacy-Preserving Machine Learning Inference</strong><a href="https://doi.org/10.56082/annalsarsciinfo.2025.2.50" target="_blank" rel="noreferrer">Deonise et al. · 2025 · DOI ↗</a><div className="paper-stats"><div><b>49%</b><small>paper TinyCNN · CKKS</small></div><div><b>98%</b><small>paper TinyCNN · plaintext</small></div></div></div>
    </section>
    <section className="filter-row" aria-label="Dashboard filters">
      <label>Dataset task<select value={task} onChange={(e) => onTask(e.target.value)}><option value={primaryTask}>{primaryTask}</option><option value={secondaryTask}>{secondaryTask}</option></select></label>
      <label>Model variant<select value={variant} onChange={(e) => onVariant(e.target.value)}><option value="All variants">All variants</option>{variants.map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
      <label>Execution mode<select value={mode} onChange={(e) => onMode(e.target.value)}>{executionModes.map((m) => <option key={m} value={m}>{m}</option>)}</select></label>
      <div className="filter-note">Task-scoped comparison · {mode === "CKKS validation partial path" ? "validation slice shown separately from test runs" : "validation-only CKKS rows are separated from held-out test comparisons"}</div>
    </section>
    <section className="kpi-grid">
      <Metric label="Selected accuracy" value={formatPercent(focused?.metrics.accuracy)} helper={focused ? `${focused.variant} · ${focused.metrics.support ?? focused.splitLabel}` : "Select a result to inspect"} accent="mint-accent" />
      <Metric label="Positive-class F1" value={formatPercent(focused?.metrics.f1)} helper="Positive class follows the selected task mapping" />
      <Metric label="CKKS parity" value={ckksFidelity?.agreement ? `${ckksFidelity.agreement.count}/${ckksFidelity.agreement.total}` : "Unavailable"} helper="Primary validation · prediction agreement" accent="violet-accent" />
      <Metric label="Candidate run" value={data.phase3RunId.replace("phase3-", "")} helper={`Completed ${data.phase3Timestamp ? new Date(data.phase3Timestamp).toLocaleString() : "timestamp unavailable"}`} />
    </section>
    <section className="content-grid overview-grid">
      <div className="panel chart-panel"><div className="panel-heading"><div><div className="eyebrow">CLASSIFICATION / {mode === "CKKS validation partial path" ? "VALIDATION RESULTS" : "HELD-OUT RESULTS"}</div><h2>{mode === "CKKS validation partial path" ? `Validation ${chartMetric} by CKKS path` : `${chartMetric[0].toUpperCase()}${chartMetric.slice(1)} by model`}</h2></div><div className="chart-head-actions"><select className="chart-metric-select" aria-label="Comparison metric" value={chartMetric} onChange={(e) => setChartMetric(e.target.value as CompareMetric)}><option value="accuracy">Accuracy</option><option value="precision">Positive precision</option><option value="recall">Positive recall</option><option value="f1">Positive F1</option></select><span className="legend-chip"><i className="dot blue" />Measured <i className="dot amber" />Published</span></div></div>
        <AccuracyChart records={chartRecords} selectedId={selected?.id} metric={chartMetric} onSelect={onSelect} />
        <div className="chart-footnote">{mode === "CKKS validation partial path" ? task === primaryTask ? "These are primary-task validation scores on 38 rows, not test accuracy. Select a bar for numerical-fidelity details and the decryption boundary." : "No CKKS partial-path validation record exists for this secondary task. The saved CKKS comparison is primary-task-only." : "Published and local values are kept distinct. Paper TinyCNN support is not reported. Validation-only CKKS rows are shown separately. Select a bar to inspect its metrics and confusion matrix."}</div>
      </div>
      <div className="panel selected-panel"><div className="panel-heading"><div><div className="eyebrow">SELECTED RESULT</div><h2>{focused?.variant ?? "Choose a model"}</h2></div>{focused && <Status value={focused.provenance} />}</div>
        {focused ? <><div className="selected-meta"><span>{focused.task}</span><span>{focused.executionMode}</span></div>
          <div className="metric-strip"><div><small>Precision</small><b>{formatPercent(focused.metrics.precision)}</b></div><div><small>Recall</small><b>{formatPercent(focused.metrics.recall)}</b></div><div><small>F1</small><b>{formatPercent(focused.metrics.f1)}</b></div></div>
          <div className="macro-metrics"><span>Macro averages</span><b>P {formatPercent(focused.metrics.macroPrecision)}</b><b>R {formatPercent(focused.metrics.macroRecall)}</b><b>F1 {formatPercent(focused.metrics.macroF1)}</b></div>
          <div className="selected-info"><p><span>Run</span>{focused.runId ?? "Published reference"}</p><p><span>Split</span>{focused.splitId ?? focused.splitLabel}</p><p><span>Source</span>{focused.sourceArtifact}</p><p><span>Privacy boundary</span>{focused.privacyBoundary}</p></div>
          <div className="mini-heading">Confusion matrix · class order [0, 1]</div><Confusion metrics={focused.metrics} /><ClassMetrics metrics={focused.metrics} />
        </> : <div className="empty-state">No record matches these filters.</div>}
      </div>
    </section>
    <section className="insight-grid">
      <div className="insight-card positive"><div className="insight-symbol">↗</div><div><div className="eyebrow">OBSERVED / {ablationTaskKey.replaceAll("_", " ")}</div><h3>{task === primaryTask ? "A0–A3 tie on this split" : "Pixel scaling leads this split"}</h3><p>{task === primaryTask ? "A0–A3 each classify 107/108 held-out samples correctly. One sample changes class across variants; this is not a measurable accuracy gain." : "A1 and A3 score 107/108. Since A1 matches A3, the result does not show an added benefit from the range penalty."}</p></div></div>
      <div className="insight-card caution"><div className="insight-symbol">!</div><div><div className="eyebrow">UNRESOLVED / PRIVACY</div><h3>CKKS feature extraction is not end-to-end</h3><p>{ckksFidelity?.privacyBoundary ?? "Pooled features are decrypted before the linear classification head. The im2col experiment decrypts even earlier."}</p></div></div>
    </section>
    <div className="citation-line"><span>Published reference: {data.publishedPaper}</span><a href="https://doi.org/10.56082/annalsarsciinfo.2025.2.50" target="_blank" rel="noreferrer">Open paper ↗</a></div>
  </>;
}

function ModelComparisonPage() {
  const baseSteps = ["8×8 digit image", "StandardScaler input", "3×3 convolution", "Square activation", "2×2 average pooling", "Linear classification head"];
  const optimizedSteps = ["8×8 digit image", "Fixed pixel scaling · divide by 16", "3×3 convolution", "Square activation", "2×2 average pooling", "Linear classification head"];
  const techniques = [
    { title: "Fixed pixel scaling", detail: "Maps pixel intensities from 0–16 into 0–1 with a simple divide-by-16 step." },
    { title: "Range-aware training", detail: "Adds a soft training penalty when convolution activations exceed a selected range. It does not add an inference layer." },
    { title: "Validation-based selection", detail: "Chooses range settings using validation examples, then reports performance on a separate test split." },
  ];
  return <>
    <section className="page-intro"><div><div className="eyebrow">MODEL DESIGN</div><h1>How CipherBench is optimized</h1><p>Compare the paper-style base model with the modified model and see what changed.</p></div></section>
    <div className="model-comparison-note"><span>Same core CNN</span><p>The convolution, square activation, pooling, and classification head stay the same. CipherBench changes input scaling and adds a training-only range penalty.</p></div>
    <section className="model-architecture-compare" aria-label="Base and modified model architecture comparison">
      {[{ title: "Base Model", caption: "Paper-style reconstruction", steps: baseSteps, tone: "base" }, { title: "CipherBench Model", caption: "Modified and optimized", steps: optimizedSteps, tone: "modified" }].map((column) => <article className={`architecture-compare-card ${column.tone}`} key={column.title}><div className="architecture-card-heading"><div><span>{column.caption}</span><h2>{column.title}</h2></div><span className="architecture-badge">{column.tone === "base" ? "REFERENCE" : "OPTIMIZED"}</span></div><ol className="architecture-steps">{column.steps.map((step, index) => <li key={step}><span className="architecture-step-number">{index + 1}</span><div><b>{step}</b>{index === 0 ? <small>64 image values</small> : index === 1 ? <small>{column.tone === "base" ? "Per-feature scaling" : "Pixels ÷ 16 → [0,1]"}</small> : null}</div></li>)}</ol></article>)}
    </section>
    <section className="optimization-section"><div className="eyebrow">WHAT WAS OPTIMIZED</div><h2>Three focused changes</h2><div className="optimization-cards">{techniques.map((technique, index) => <article className="optimization-card" key={technique.title}><span className="optimization-number">0{index + 1}</span><div><h3>{technique.title}</h3><p>{technique.detail}</p></div></article>)}</div></section>
    <p className="model-comparison-footnote">The base model is a local retraining of the paper-style architecture; the paper’s original model weights and exact training recipe were not provided.</p>
  </>;
}
export function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState("");
  const [page, setPage] = useState<Page>("Model Lab");
  const [task, setTask] = useState(primaryTask);
  const [mode, setMode] = useState<string>("All modes");
  const [variant, setVariant] = useState("A3 · RC-TinyCNN");
  const [selectedId, setSelectedId] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const [benchmarkTask, setBenchmarkTask] = useState("digits_0_vs_1");
  const [benchmarkRun, setBenchmarkRun] = useState<BenchmarkRun | null>(null);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  useEffect(() => { loadDashboardData().then(setData).catch((e: unknown) => setError(e instanceof Error ? e.message : "Unknown artifact loading error.")); }, []);
  useEffect(() => { let alive = true; fetch("/api/benchmark/status", { cache: "no-store" }).then(async (response) => { if (!response.ok) throw new Error(); return response.json() as Promise<BenchmarkRun>; }).then((run) => { if (alive) { setBenchmarkRun(run); if (run.task) setBenchmarkTask(run.task); } }).catch(() => { if (alive) setBenchmarkRun({ available: false, status: "idle", logs: [] }); }); return () => { alive = false; }; }, []);
  useEffect(() => {
    if (benchmarkRun?.status !== "running") return;
    let alive = true;
    const timer = window.setInterval(() => { fetch("/api/benchmark/status", { cache: "no-store" }).then((response) => response.json() as Promise<BenchmarkRun>).then((next) => { if (alive) setBenchmarkRun(next); }).catch(() => undefined); }, 1200);
    return () => { alive = false; window.clearInterval(timer); };
  }, [benchmarkRun?.status, benchmarkRun?.runId]);
  const startBenchmark = async () => {
    setBenchmarkBusy(true);
    try { const response = await fetch(`/api/benchmark/run?task=${encodeURIComponent(benchmarkTask)}`, { method: "POST" }); const body = await response.json() as BenchmarkRun & { error?: string }; if (!response.ok) throw new Error(body.error ?? "Could not start training."); setBenchmarkRun(body); }
    catch (error) { setBenchmarkRun((previous) => ({ ...(previous ?? { available: true, logs: [] }), status: "failed", error: error instanceof Error ? error.message : "Could not start training." })); }
    finally { setBenchmarkBusy(false); }
  };
  const clearBenchmark = async () => {
    setBenchmarkBusy(true);
    try { const response = await fetch("/api/benchmark/run", { method: "DELETE" }); const body = await response.json() as { error?: string }; if (!response.ok) throw new Error(body.error ?? "Could not clear the results."); setBenchmarkRun({ available: true, status: "idle", logs: [] }); }
    catch (error) { setBenchmarkRun((previous) => previous ? { ...previous, error: error instanceof Error ? error.message : "Could not clear the results." } : previous); }
    finally { setBenchmarkBusy(false); }
  };
  const taskRecords = useMemo(() => data?.records.filter((r) => r.task === task) ?? [], [data, task]);
  const variantNames = [...new Set(taskRecords.map((r) => r.variant))];
  useEffect(() => { if (!variantNames.includes(variant)) setVariant(variantNames.find((v) => v.startsWith("A3")) ?? "All variants"); }, [task, data]);
  const selectedRecord = data?.records.find((r) => r.id === selectedId);
  const focusedRecord = selectedRecord?.task === task ? selectedRecord : metricFor(data?.records ?? [], task, variant);
  const onSelectRecord = (record: ExperimentRecord) => { setSelectedId(record.id); setVariant(record.variant); };
  const chooseTask = (value: string) => { setTask(value); setSelectedId(""); setVariant("A3 · RC-TinyCNN"); };
  if (error) return <div className="app-shell error-shell"><div className="error-mark">!</div><div className="eyebrow">ARTIFACT LOAD ERROR</div><h1>Research results unavailable</h1><p>{error}</p><p className="muted">Start the dashboard from the project root so the local artifact routes are available.</p><button className="button primary" onClick={() => location.reload()}>Retry loading</button></div>;
  if (!data) return <div className="app-shell loading-shell"><div className="brand-mark">C</div><div className="loading-track"><i /></div><p>Loading verified experiment artifacts…</p></div>;
  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}><div className="brand"><div className="brand-mark">C</div><div><b>CipherBench</b><small>MODEL WORKSPACE</small></div><button className="mobile-close" aria-label="Close navigation" onClick={() => setMobileNav(false)}>×</button></div>
      <nav aria-label="Primary navigation">{pages.map((item) => <button key={item.title} className={`nav-item ${page === item.title ? "active" : ""}`} onClick={() => { setPage(item.title); setMobileNav(false); }}><span className="nav-icon">{item.icon}</span><span><small>{item.eyebrow}</small>{item.title}</span>{page === item.title && <i />}</button>)}</nav>
    </aside>
    {mobileNav && <button aria-label="Close navigation overlay" className="nav-backdrop" onClick={() => setMobileNav(false)} />}
    <main className="main-content"><header className="topbar"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNav(true)}>☰</button><div className="breadcrumb">CipherBench <span>/</span> <b>{page}</b></div></header>
      <div className="page-content">{page === "Model Lab" && <ModelLab run={benchmarkRun} busy={benchmarkBusy} task={benchmarkTask} onTask={setBenchmarkTask} onRun={startBenchmark} onClear={clearBenchmark} />}{page === "Results" && <Overview data={data} task={task} mode={mode} variant={variant} onTask={chooseTask} onMode={(value) => { setMode(value); setSelectedId(""); }} onVariant={(value) => { setVariant(value); setSelectedId(""); }} selected={focusedRecord} onSelect={onSelectRecord} />}{page === "Optimization & Architecture" && <ModelComparisonPage />}</div>
    </main>
  </div>;
}
