import { useEffect, useMemo, useState } from "react";
import { baselineArchitecture, candidateArchitecture } from "../data/architecture";
import { executionModes, filterRecords, formatDeltaPp, formatNumber, formatPercent, primaryTask, secondaryTask, toCsv, exportJson } from "../data/compare";
import { loadDashboardData } from "../data/transform";
import type { DashboardData, ExperimentRecord, MetricSet } from "../data/types";

type Page = "Overview" | "Architecture & Ablation" | "Experiments & Reproducibility";
const pages: { title: Page; icon: string; eyebrow: string }[] = [
  { title: "Overview", icon: "◫", eyebrow: "RESULTS" },
  { title: "Architecture & Ablation", icon: "⌘", eyebrow: "METHOD" },
  { title: "Experiments & Reproducibility", icon: "◴", eyebrow: "RUNS" },
];

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

function ArchitectureDiagram({ stages }: { stages: ReturnType<typeof candidateArchitecture> }) {
  return <div className="architecture-flow" aria-label="Model computation graph">
    {stages.map((stage, i) => <div className="stage-wrap" key={stage.id}>
      <div className={`stage-card ${stage.execution}`}>
        <div className="stage-type">{stage.execution === "encrypted" ? "CIPHERTEXT" : stage.execution === "training-only" ? "TRAINING ONLY" : "PLAINTEXT"}</div>
        <strong>{stage.title}</strong><span>{stage.detail}</span>
      </div>
      {i < stages.length - 1 && <div className="flow-arrow" aria-hidden="true">→</div>}
    </div>)}
  </div>;
}

function TrainingObjective({ stage }: { stage: ReturnType<typeof candidateArchitecture>[number] | undefined }) {
  if (!stage) return null;
  return <div className="training-objective"><span>TRAINING OBJECTIVE</span><b>{stage.title}</b><p>{stage.detail}</p><small>Applied while fitting the candidate weights only; not a node in the inference graph.</small></div>;
}

function Overview({ data, task, mode, variant, onTask, onMode, onVariant, selected, onSelect }: { data: DashboardData; task: string; mode: string; variant: string; onTask: (v: string) => void; onMode: (v: string) => void; onVariant: (v: string) => void; selected: ExperimentRecord | undefined; onSelect: (r: ExperimentRecord) => void }) {
  const [chartMetric, setChartMetric] = useState<CompareMetric>("accuracy");
  const filtered = filterRecords(data.records, task, mode).filter((r) => !r.variant.includes("smoke"));
  const chartRecords = filtered.filter((r) => r.metrics.accuracy !== null && r.executionMode !== "Diagnostic / unavailable" && (mode === "CKKS validation partial path" || (!r.executionMode.includes("validation") && !r.splitLabel.toLowerCase().includes("validation"))));
  const selectedMatchesFilters = selected && selected.task === task && (mode === "All modes" || selected.executionMode === mode) ? selected : undefined;
  const focused = selectedMatchesFilters ?? metricFor(filtered, task, variant) ?? filtered[0];
  const variants = [...new Set(data.records.filter((r) => r.task === task && r.metrics.accuracy !== null).map((r) => r.variant))];
  const phase3TaskKey = task === primaryTask ? "digits_0_vs_1" : "digits_3_vs_8";
  const ckksFidelity = data.records.find((r) => r.task === primaryTask && r.variant.includes("A3") && r.executionMode === "CKKS validation partial path");
  return <>
    <section className="hero-panel">
      <div className="hero-copy"><div className="eyebrow">RESEARCH PROJECT / PHASE 4</div><h1>Homomorphic inference,<br /><em>measured honestly.</em></h1><p>Can input calibration and range-aware training improve shallow CNN inference under CKKS—without trading away accuracy or practical cost?</p>
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
      <Metric label="CKKS parity" value={ckksFidelity?.agreement ? `${ckksFidelity.agreement.count}/${ckksFidelity.agreement.total}` : "Unavailable"} helper="Phase 3 primary validation · prediction agreement" accent="violet-accent" />
      <Metric label="Candidate run" value={data.phase3RunId.replace("phase3-", "")} helper={`Completed ${data.phase3Timestamp ? new Date(data.phase3Timestamp).toLocaleString() : "timestamp unavailable"}`} />
    </section>
    <section className="content-grid overview-grid">
      <div className="panel chart-panel"><div className="panel-heading"><div><div className="eyebrow">CLASSIFICATION / {mode === "CKKS validation partial path" ? "VALIDATION RESULTS" : "HELD-OUT RESULTS"}</div><h2>{mode === "CKKS validation partial path" ? `Validation ${chartMetric} by CKKS path` : `${chartMetric[0].toUpperCase()}${chartMetric.slice(1)} by model`}</h2></div><div className="chart-head-actions"><select className="chart-metric-select" aria-label="Comparison metric" value={chartMetric} onChange={(e) => setChartMetric(e.target.value as CompareMetric)}><option value="accuracy">Accuracy</option><option value="precision">Positive precision</option><option value="recall">Positive recall</option><option value="f1">Positive F1</option></select><span className="legend-chip"><i className="dot blue" />Measured <i className="dot amber" />Published</span></div></div>
        <AccuracyChart records={chartRecords} selectedId={selected?.id} metric={chartMetric} onSelect={onSelect} />
        <div className="chart-footnote">{mode === "CKKS validation partial path" ? task === primaryTask ? "These are primary-task validation scores on 38 rows, not test accuracy. Select a bar for numerical-fidelity details and the decryption boundary." : "No CKKS partial-path validation record exists for this secondary task. The saved CKKS comparison is primary-task-only." : "Published and local values are kept distinct. Paper TinyCNN support is not reported. Validation-only CKKS rows are shown separately. Select a bar to inspect its report and confusion matrix."}</div>
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
      <div className="insight-card positive"><div className="insight-symbol">↗</div><div><div className="eyebrow">OBSERVED / {phase3TaskKey.replaceAll("_", " ")}</div><h3>{task === primaryTask ? "All Phase 3 variants tie on accuracy" : "Pixel scaling leads this split"}</h3><p>{task === primaryTask ? "A0–A3 each classify 107/108 held-out samples correctly. One sample changes class across variants; this is not a measurable accuracy gain." : "A1 and A3 score 107/108. Since A1 matches A3, the result does not show an added benefit from the range penalty."}</p></div></div>
      <div className="insight-card caution"><div className="insight-symbol">!</div><div><div className="eyebrow">UNRESOLVED / PRIVACY</div><h3>CKKS feature extraction is not end-to-end</h3><p>{ckksFidelity?.privacyBoundary ?? "Pooled features are decrypted before the linear classification head. The im2col experiment decrypts even earlier."}</p></div></div>
    </section>
    <div className="citation-line"><span>Published reference: {data.publishedPaper}</span><a href={`${import.meta.env.BASE_URL}reports/BASELINE_REPORT.md`}>Read baseline report ↗</a></div>
  </>;
}

function AblationTable({ records, task }: { records: ExperimentRecord[]; task: string }) {
  const rows = records.filter((r) => r.task === task && r.sourceArtifact.includes("phase3_ablation.json"));
  const fidelity = records.filter((r) => r.executionMode === "CKKS validation partial path");
  const getCkksRecord = (variant: string) => {
    if (!variant.startsWith("A0") && !variant.startsWith("A3")) return undefined;
    const candidate = variant.startsWith("A3");
    return fidelity.find((r) => candidate ? r.variant.includes("A3") && r.variant.includes("mask_dot") : r.variant.includes("A0") && r.variant.includes("mask_dot"));
  };
  const getFidelity = (variant: string) => {
    const rec = getCkksRecord(variant);
    return rec?.numericalErrors?.logit;
  };
  return <div className="table-scroll"><table className="data-table ablation-table"><thead><tr><th>Variant</th><th>Accuracy</th><th>Δ Acc.</th><th>Precision</th><th>Δ Prec.</th><th>Recall</th><th>Δ Recall</th><th>F1</th><th>Δ F1</th><th>CKKS logit MAE¹</th><th>CKKS mask mean seconds²</th><th>Mode / run</th></tr></thead><tbody>{rows.map((r) => {
    const error = getFidelity(r.variant);
    const cost = getCkksRecord(r.variant)?.timings.total_until_plaintext_prediction;
    return <tr key={r.id}><td><strong>{r.variant}</strong><small>{String(r.configuration.preprocessing ?? "")}{r.configuration.range_bound ? ` · B=${String(r.configuration.range_bound)}, λ=${String(r.configuration.range_penalty_lambda)}` : ""}</small></td>
      <td>{formatPercent(r.metrics.accuracy)}</td><td>{formatDeltaPp(r.deltasVsA0?.accuracy)}</td><td>{formatPercent(r.metrics.precision)}</td><td>{formatDeltaPp(r.deltasVsA0?.precision)}</td><td>{formatPercent(r.metrics.recall)}</td><td>{formatDeltaPp(r.deltasVsA0?.recall)}</td><td>{formatPercent(r.metrics.f1)}</td><td>{formatDeltaPp(r.deltasVsA0?.f1)}</td><td>{error ? `${error.meanMae === null ? "Unavailable" : error.meanMae.toExponential(2)} mean` : "Unavailable"}</td><td>{cost === null || cost === undefined ? "Unavailable" : `${formatNumber(cost, 4)} s`}</td><td><Status value={r.status} /><small>{r.runId ?? "Run unavailable"}</small></td></tr>;
  })}</tbody></table><div className="table-note">¹ CKKS fidelity and ² measured CKKS partial-path time are available only for A0/A3 on the primary validation set; the path includes diagnostic decryption before a plaintext head. These are not test accuracy or end-to-end latency values. Plaintext Phase 3 inference time was not recorded. A4 threshold-tuned row is included when present in source artifact.</div></div>;
}

function ArchitecturePage({ data, task, mode, selectedRecord, onSelectRecord }: { data: DashboardData; task: string; mode: string; selectedRecord: ExperimentRecord | undefined; onSelectRecord: (record: ExperimentRecord) => void }) {
  const candidates = data.records.filter((r) => r.task === task && r.sourceArtifact.includes("phase3_ablation.json") && r.variant.startsWith("A"));
  const candidate = selectedRecord && selectedRecord.task === task && selectedRecord.sourceArtifact.includes("phase3_ablation.json") ? selectedRecord : candidates.find((r) => r.variant.startsWith("A3"));
  const conf = candidate?.configuration ?? {};
  const allStages = candidateArchitecture(String(conf.preprocessing ?? "pixel_0_1"), typeof conf.range_bound === "number" ? conf.range_bound : null, typeof conf.range_penalty_lambda === "number" ? conf.range_penalty_lambda : 0);
  const stages = allStages.filter((stage) => stage.execution !== "training-only");
  const rangeObjective = allStages.find((stage) => stage.execution === "training-only");
  const baseline = baselineArchitecture;
  const a1 = candidates.find((r) => r.variant.startsWith("A1")); const a3 = candidates.find((r) => r.variant.startsWith("A3"));
  const a2 = candidates.find((r) => r.variant.startsWith("A2"));
  const tied = Boolean(a1 && a3 && a1.metrics.accuracy === a3.metrics.accuracy);
  const a3Range = objValue(a3?.activationDiagnostics, "pre_square");
  const a2Range = objValue(a2?.activationDiagnostics, "pre_square");
  return <>
    <section className="page-intro"><div><div className="eyebrow">METHOD / COMPUTATION GRAPH</div><h1>Architecture & Ablation</h1><p>Explore where the reconstructed graph leaves ciphertext and how each controlled change performed on the selected task.</p></div><div className="intro-badge">Task: {task} <span>·</span> Mode filter: {mode}</div></section>
    <section className="panel architecture-panel"><div className="panel-heading"><div><div className="eyebrow">A0 / RECONSTRUCTED BASELINE</div><h2>Feature extraction, then plaintext classification</h2></div><span className="boundary-key"><i className="key-encrypted" />Encrypted <i className="key-plain" />Plaintext</span></div><ArchitectureDiagram stages={baseline} /><div className="warning-line"><span>!</span><p>Decrypted pooled features are passed to a plaintext Logistic Regression head. The diagram reflects the reported CKKS path; local Phase 3 classification ablations were evaluated in plaintext.</p></div></section>
    <section className="panel architecture-panel"><div className="panel-heading"><div><div className="eyebrow">RC-TINYCNN / SELECTED CONFIGURATION</div><h2>{candidate?.variant ?? "No candidate selected"}</h2></div><label className="inline-select">Candidate<select value={candidate?.id ?? ""} onChange={(e) => { const match = candidates.find((r) => r.id === e.target.value); if (match) onSelectRecord(match); }}>{candidates.map((r) => <option key={r.id} value={r.id}>{r.variant}</option>)}</select></label></div><ArchitectureDiagram stages={stages} /><TrainingObjective stage={rangeObjective} /><div className="boundary-note"><b>Execution boundary:</b> This shows the graph placement used by the CKKS diagnostics. Phase 3 classification ablations were evaluated in plaintext; CKKS parity measurements exist separately for A0/A3 on primary validation, and both leave the head plaintext.</div></section>
    <section className="panel"><div className="panel-heading"><div><div className="eyebrow">CONTROLLED CHANGES</div><h2>Ablation comparison</h2><p className="subtext">Test metrics from the saved Phase 3 run. Deltas are percentage points relative to A0.</p></div><Status value="Measured" /></div><AblationTable records={data.records} task={task} />
      {tied && <div className="callout violet-callout"><b>A1 and A3 tied at {formatPercent(a3?.metrics.accuracy ?? null)} on this task.</b> The combined candidate did not add measurable accuracy beyond pixel scaling alone.</div>}
    </section>
    <section className="content-grid component-grid"><div className="panel"><div className="eyebrow">COMPONENT DETAIL</div><h2>Why each ablation exists</h2><div className="detail-list">
      <div><b>A0 · StandardScaler</b><p>Control matching the Phase 1 reconstruction. Per-feature means/scales are fitted on training rows. Reference range; no range loss.</p><small>Observed: {formatPercent(candidates.find((r) => r.variant.startsWith("A0"))?.metrics.accuracy)} accuracy.</small></div>
      <div><b>A1 · Pixel scaling</b><p>Fixed raw pixel /16 maps the documented 0–16 intensity domain into [0,1]. Tests whether input calibration avoids amplified standardized border features.</p><small>Observed: {formatPercent(a1?.metrics.accuracy)}. Same classification score as A3 here.</small></div>
      <div><b>A2 · Range penalty</b><p>Squared hinge loss penalizes convolution pre-activations outside selected B. B and λ are selected only using development validation data.</p><small>Observed: {formatPercent(a2?.metrics.accuracy)}; max |z| {formatNumber(safeNum(objValue(a2Range, "max_abs")), 2)}, outside B {formatPercent(safeNum(objValue(a2Range, "bound_violation_rate")))}. Penalty is soft, not a hard bound.</small></div>
      <div><b>A3 · RC-TinyCNN</b><p>Combines pixel scaling and the selected range penalty; same square activation and pooling structure.</p><small>Observed: {formatPercent(a3?.metrics.accuracy)}; max |z| {formatNumber(safeNum(objValue(a3Range, "max_abs")), 2)}, outside B {formatPercent(safeNum(objValue(a3Range, "bound_violation_rate")))}. No incremental gain beyond A1 in this run.</small></div>
      <div><b>A4 · Threshold</b><p>Threshold is selected to maximize validation positive-class F1, then validation accuracy, with a tie preference nearest zero.</p><small>Observed threshold: {formatNumber(Number(objValue(candidate?.configuration.thresholdSelection, "threshold_logit")), 2)}. Report separately from architecture changes.</small></div>
    </div></div><div className="panel"><div className="eyebrow">INTERPRETATION</div><h2>Accuracy ≠ CKKS agreement</h2><p className="body-copy">Classification accuracy compares predictions with ground-truth labels. CKKS agreement compares encrypted-path predictions with the same model’s float64 polynomial graph. A path can agree perfectly with plaintext and still misclassify; agreement is a numerical fidelity check, not an accuracy score.</p><div className="pair-stat"><span>CKKS agreement · primary validation</span><strong>38 / 38</strong></div><div className="pair-stat"><span>CKKS agreement · Phase 2 test</span><strong>108 / 108</strong></div><div className="callout caution-callout">The pooled feature vector is decrypted before the head. These are partial encrypted-inference measurements.</div></div></section>
    <div className="citation-line"><span>Values from {data.phase3RunId}; no training or test evaluation was triggered by this page.</span><a href={`${import.meta.env.BASE_URL}reports/ABLATION_REPORT.md`}>Full ablation report ↗</a></div>
  </>;
}

function objValue(value: unknown, key: string): unknown { return value && typeof value === "object" ? (value as Record<string, unknown>)[key] : null; }
function safeNum(value: unknown): number | null { return typeof value === "number" && Number.isFinite(value) ? value : null; }

function ExperimentsPage({ data, task, mode, selectedId, onSelectId }: { data: DashboardData; task: string; mode: string; selectedId: string; onSelectId: (id: string) => void }) {
  const [scope, setScope] = useState("All records");
  const [exportMessage, setExportMessage] = useState("");
  const filtered = useMemo(() => filterRecords(data.records, task, mode), [data.records, task, mode]);
  const shown = scope === "All records" ? filtered : filtered.filter((r) => r.provenance === scope);
  const selected = shown.find((r) => r.id === selectedId) ?? shown[0];
  const ckksOnly = selected?.executionMode.includes("CKKS") ?? false;
  function download(filename: string, content: string, type: string) {
    const url = URL.createObjectURL(new Blob([content], { type })); const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url); setExportMessage(`${filename} exported from the filtered artifact records.`);
  }
  return <>
    <section className="page-intro"><div><div className="eyebrow">RUNS / REPRODUCIBILITY</div><h1>Experiments & Reproducibility</h1><p>Every result links back to its artifact, split, configuration, and execution boundary.</p></div><div className="intro-actions"><button className="button subtle" onClick={() => download("cipherbench-experiments.json", exportJson(shown), "application/json")}>↓ Export JSON</button><button className="button subtle" onClick={() => download("cipherbench-experiments.csv", toCsv(shown), "text/csv")}>↓ Export CSV</button></div></section>
    <section className="run-toolbar"><div className="segmented"><button className={scope === "All records" ? "active" : ""} onClick={() => setScope("All records")}>All records</button><button className={scope === "Measured" ? "active" : ""} onClick={() => setScope("Measured")}>Measured</button><button className={scope === "Published" ? "active" : ""} onClick={() => setScope("Published")}>Published</button></div><span>{shown.length} records · filters inherited from task and execution mode</span></section>
    {exportMessage && <div className="inline-success" role="status">{exportMessage}</div>}
    <section className="panel run-table-panel"><div className="table-scroll"><table className="data-table run-table"><thead><tr><th>Variant / run</th><th>Task</th><th>Execution mode</th><th>Accuracy</th><th>Status</th><th>Split / seed</th><th>Source</th></tr></thead><tbody>{shown.map((r) => <tr key={r.id} className={r.id === selected?.id ? "selected-row" : ""} onClick={() => onSelectId(r.id)} tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter") onSelectId(r.id); }}><td><strong>{r.variant}</strong><small>{r.runId ?? "Paper reference"}</small></td><td>{r.task}</td><td>{r.executionMode}</td><td>{formatPercent(r.metrics.accuracy)}</td><td><Status value={r.provenance} /><small>{r.status}</small></td><td>{r.splitLabel}<small>Seed: {r.seed ?? "Not reported"}</small></td><td><code>{r.sourceArtifact}</code></td></tr>)}</tbody></table>{!shown.length && <div className="empty-state">No experiment records match the selected filters.</div>}</div></section>
    {selected && <section className="content-grid run-detail-grid"><div className="panel"><div className="panel-heading"><div><div className="eyebrow">RESULT DETAIL</div><h2>{selected.variant}</h2></div><Status value={selected.status} /></div>
        <div className="detail-metrics"><Metric label="Accuracy" value={formatPercent(selected.metrics.accuracy)} helper={`Support ${selected.metrics.support ?? selected.splitLabel}`} /><Metric label="Positive precision" value={formatPercent(selected.metrics.precision)} helper="Positive class is task mapping" /><Metric label="Recall" value={formatPercent(selected.metrics.recall)} helper="Positive class" /><Metric label="F1" value={formatPercent(selected.metrics.f1)} helper="Positive class" /></div><div className="macro-metrics detail-macro"><span>Macro averages</span><b>P {formatPercent(selected.metrics.macroPrecision)}</b><b>R {formatPercent(selected.metrics.macroRecall)}</b><b>F1 {formatPercent(selected.metrics.macroF1)}</b></div>
        <div className="detail-columns"><div><h3>Confusion matrix</h3><Confusion metrics={selected.metrics} /><ClassMetrics metrics={selected.metrics} /></div><div><h3>Configuration & provenance</h3><div className="selected-info"><p><span>Run ID</span>{selected.runId ?? "Not reported"}</p><p><span>Timestamp</span>{selected.timestamp ?? "Not reported"}</p><p><span>Seed</span>{selected.seed ?? "Not reported"}</p><p><span>Split ID</span>{selected.splitId ?? "Not reported"}</p><p><span>Source</span>{selected.sourceArtifact}</p><p><span>Privacy boundary</span>{selected.privacyBoundary}</p></div><pre className="json-preview">{JSON.stringify(selected.configuration, null, 2)}</pre></div></div>
      </div>
      <div className="panel"><div className="eyebrow">NUMERICAL FIDELITY & COST</div><h2>Execution details</h2>{selected.agreement ? <div className="fidelity-card"><div><b>{selected.agreement.count}/{selected.agreement.total}</b><small>Prediction agreement</small></div><div><b>{selected.agreement.flips}</b><small>Flipped predictions</small></div></div> : <div className="empty-state compact">Prediction agreement: Unavailable for this record.</div>}
        {selected.numericalErrors ? <div className="error-list">{Object.entries(selected.numericalErrors).map(([key, v]) => <div key={key}><span>{key}</span><b>MAE {v.meanMae === null ? "Unavailable" : v.meanMae.toExponential(2)}</b><small>Worst max abs {v.worstMax === null ? "Unavailable" : v.worstMax.toExponential(2)}</small></div>)}</div> : <div className="empty-state compact">Numerical error: Not reported for this record.</div>}
        <h3>Timing · seconds unless stated</h3><div className="timing-list">{Object.entries(selected.timings).filter(([, value]) => value !== null).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><b>{formatNumber(value, 5)}</b></div>)}{!Object.values(selected.timings).some((v) => v !== null) && <span className="muted">Timing unavailable.</span>}</div>
        <h3>Ciphertext sizes</h3><div className="timing-list">{Object.entries(selected.ciphertextSizes).filter(([, value]) => value !== null).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><b>{value?.toLocaleString()} B</b></div>)}{!Object.values(selected.ciphertextSizes).some((v) => v !== null) && <span className="muted">Ciphertext size unavailable for this execution.</span>}</div>
        {ckksOnly && <div className="callout caution-callout">Do not interpret this as fully encrypted classification. The specific decryption boundary is listed above.</div>}
      </div>
    </section>}
    <section className="panel action-panel"><div className="action-icon">⊘</div><div><div className="eyebrow">RUN EXPERIMENT</div><h2>Dashboard execution is disabled</h2><p>The verified ablation command takes several minutes and writes to the frozen Phase 3 result paths. Launching it here could overwrite historical artifacts, so this UI does not start runs or show simulated progress. Use the documented CLI workflow only against a preserved copy.</p><code>python scripts/run_phase3_ablation.py</code></div><Status value="Unsupported in dashboard" /></section>
    <section className="panel reports-panel"><div className="panel-heading"><div><div className="eyebrow">RESEARCH NOTES</div><h2>Reports & methodology</h2></div></div><div className="report-links">{data.sourceReports.map((name) => <a key={name} href={`${import.meta.env.BASE_URL}reports/${name}`} target="_blank" rel="noreferrer"><span>▤</span><div><b>{name.replaceAll("_", " ").replace(".md", "")}</b><small>Markdown report · project workspace</small></div><strong>↗</strong></a>)}</div></section>
  </>;
}

export function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState("");
  const [page, setPage] = useState<Page>("Overview");
  const [task, setTask] = useState(primaryTask);
  const [mode, setMode] = useState<string>("All modes");
  const [variant, setVariant] = useState("A3 · RC-TinyCNN");
  const [selectedId, setSelectedId] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  useEffect(() => { loadDashboardData().then(setData).catch((e: unknown) => setError(e instanceof Error ? e.message : "Unknown artifact loading error.")); }, []);
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
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}><div className="brand"><div className="brand-mark">C</div><div><b>CipherBench</b><small>RESEARCH WORKSPACE</small></div><button className="mobile-close" aria-label="Close navigation" onClick={() => setMobileNav(false)}>×</button></div>
      <div className="side-project"><span className="project-icon">⌬</span><div><small>ACTIVE PROJECT</small><b>CKKS TinyCNN study</b></div><span className="side-chevron">⌄</span></div>
      <nav aria-label="Primary navigation">{pages.map((item) => <button key={item.title} className={`nav-item ${page === item.title ? "active" : ""}`} onClick={() => { setPage(item.title); setMobileNav(false); }}><span className="nav-icon">{item.icon}</span><span><small>{item.eyebrow}</small>{item.title}</span>{page === item.title && <i />}</button>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-status"><span className="live-dot" /><div><b>Artifacts connected</b><small>Local files · no cloud sync</small></div></div><div className="sidebar-user"><div className="avatar">CB</div><div><b>Research workspace</b><small>Local session</small></div><span>···</span></div></div>
    </aside>
    {mobileNav && <button aria-label="Close navigation overlay" className="nav-backdrop" onClick={() => setMobileNav(false)} />}
    <main className="main-content"><header className="topbar"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNav(true)}>☰</button><div className="breadcrumb">CipherBench <span>/</span> <b>{page}</b></div><div className="topbar-right"><div className="top-run"><span className="live-dot" />Latest run <b>{data.phase3RunId.replace("phase3-", "")}</b></div><div className="top-avatar">CB</div></div></header>
      <div className="page-content">{page === "Overview" && <Overview data={data} task={task} mode={mode} variant={variant} onTask={chooseTask} onMode={(value) => { setMode(value); setSelectedId(""); }} onVariant={(value) => { setVariant(value); setSelectedId(""); }} selected={focusedRecord} onSelect={onSelectRecord} />}{page === "Architecture & Ablation" && <ArchitecturePage data={data} task={task} mode={mode} selectedRecord={selectedRecord} onSelectRecord={onSelectRecord} />}{page === "Experiments & Reproducibility" && <ExperimentsPage data={data} task={task} mode={mode} selectedId={selectedId} onSelectId={(id) => { setSelectedId(id); const record = data.records.find((item) => item.id === id); if (record) setVariant(record.variant); }} />}</div>
      <footer className="app-footer"><span>Data loaded from project artifacts · results retain source and split labels</span><span>Research dashboard <i>•</i> {data.records.length} artifact-backed records</span></footer>
    </main>
  </div>;
}
