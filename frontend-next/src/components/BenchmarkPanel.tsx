"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, XCircle, RefreshCw, FlaskConical, Clock } from "lucide-react";
import { fetchBenchmark, type BenchmarkReport } from "@/lib/api";

function Metric({ label, value, unit = "" }: { label: string; value: string | number; unit?: string }) {
  return (
    <div className="p-3 rounded-xl" style={{ background: "#090e1c" }}>
      <p className="label-meta mb-1" style={{ fontSize: "0.58rem" }}>{label}</p>
      <p className="text-sm font-bold text-on-surface mono">{typeof value === "number" ? value.toFixed(4) : value}{unit}</p>
    </div>
  );
}

function GateBadge({ label, item }: { label: string; item: { value: number; target: string | number; pass: boolean } }) {
  return (
    <div className={`flex items-center justify-between p-3 rounded-xl border ${item.pass ? "border-success/20 bg-success/5" : "border-error/20 bg-error/5"}`}>
      <div className="flex items-center gap-2">
        {item.pass
          ? <CheckCircle2 size={14} className="text-success shrink-0" />
          : <XCircle size={14} className="text-error shrink-0" />}
        <span className="text-xs text-on-surface">{label.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</span>
      </div>
      <div className="text-right">
        <p className="mono text-xs font-bold" style={{ color: item.pass ? "#4ade80" : "#f87171" }}>
          {item.value.toFixed(4)}
        </p>
        <p className="mono text-xs text-on-surface-variant" style={{ fontSize: "0.6rem" }}>
          target: {item.target}
        </p>
      </div>
    </div>
  );
}

export function BenchmarkPanel() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  async function load() {
    setLoading(true);
    try {
      const r = await fetchBenchmark();
      if (r.available && r.report) {
        setReport(r.report);
        setUnavailable(false);
        setLastRefreshed(new Date());
      } else {
        setUnavailable(true);
      }
    } catch {
      setUnavailable(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  if (loading) {
    return (
      <div className="space-y-3">
        {[0, 1, 2].map(i => <div key={i} className="skeleton h-16 rounded-xl" />)}
      </div>
    );
  }

  if (unavailable || !report) {
    return (
      <div className="glass-card p-10 text-center space-y-4">
        <FlaskConical size={32} className="mx-auto text-outline opacity-40" />
        <div>
          <p className="font-semibold text-on-surface">No Benchmark Report Found</p>
          <p className="text-xs text-on-surface-variant mt-1">Run the evaluation pipeline to generate results.</p>
        </div>
        <pre className="text-xs text-on-surface-variant p-3 rounded-lg text-left overflow-auto" style={{ background: "#090e1c" }}>
          {`cd "D:\\Projects\\Final Project"\n.venv_local\\Scripts\\python.exe -m benchmark.run_publication_eval --profile quick`}
        </pre>
        <button className="btn-primary px-4 py-2 rounded-xl text-sm flex items-center gap-2 mx-auto" onClick={load}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>
    );
  }

  const gateEntries = Object.entries(report.publication_gate);
  const allPass = report.ready_for_publication;

  return (
    <div className="space-y-4 animate-fade-up">
      {/* Header */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <FlaskConical size={18} className={allPass ? "text-success" : "text-warning"} />
            <h3 className="font-bold text-on-surface text-lg">
              {allPass ? "✅ Publication Ready" : "⚠️ Not Yet Publication Ready"}
            </h3>
          </div>
          <button onClick={load} className="text-outline hover:text-on-surface transition-colors">
            <RefreshCw size={14} />
          </button>
        </div>
        <div className="flex gap-4 text-xs text-on-surface-variant mt-2">
          <span className="flex items-center gap-1 mono"><Clock size={11} /> {report.runtime_seconds.toFixed(1)}s runtime</span>
          <span className="mono">Profile: {report.profile}</span>
          <span className="mono">{new Date(report.timestamp).toLocaleString()}</span>
          {lastRefreshed && <span className="mono text-outline">Fetched {lastRefreshed.toLocaleTimeString()}</span>}
        </div>
      </div>

      {/* Publication gate */}
      <div className="glass-card p-5">
        <p className="label-meta mb-3">Publication Gate ({gateEntries.filter(([,v]) => v.pass).length}/{gateEntries.length} pass)</p>
        <div className="grid grid-cols-2 gap-2">
          {gateEntries.map(([key, val]) => (
            <GateBadge key={key} label={key} item={val} />
          ))}
        </div>
      </div>

      {/* OCR metrics */}
      <div className="glass-card p-5">
        <p className="label-meta mb-3">OCR Pipeline ({report.ocr.count} cases)</p>
        <div className="grid grid-cols-3 gap-2">
          <Metric label="Mean CER" value={report.ocr.mean_cer} />
          <Metric label="Mean WER" value={report.ocr.mean_wer} />
          <Metric label="Field F1" value={report.ocr.field_f1} />
          <Metric label="Precision" value={report.ocr.field_precision} />
          <Metric label="Recall" value={report.ocr.field_recall} />
          <Metric label="QA Accuracy" value={report.ocr.qa_accuracy} />
          <Metric label="Avg Latency" value={report.ocr.mean_latency_ms.toFixed(0)} unit="ms" />
        </div>
      </div>

      {/* VQA & Routing */}
      <div className="grid grid-cols-2 gap-3">
        <div className="glass-card p-4">
          <p className="label-meta mb-3">VQA ({report.vqa.count} cases)</p>
          <div className="space-y-2">
            <Metric label="Fine-tuned EM" value={report.vqa.ft_exact_match} />
            <Metric label="Base EM" value={report.vqa.base_exact_match} />
          </div>
        </div>
        <div className="glass-card p-4">
          <p className="label-meta mb-3">Mode Routing</p>
          <div className="space-y-2">
            <Metric label="Accuracy" value={report.mode_routing.accuracy} />
            <Metric label="Cases" value={report.mode_routing.count} />
          </div>
        </div>
      </div>

      {/* Satellite & Calibration */}
      <div className="grid grid-cols-2 gap-3">
        <div className="glass-card p-4">
          <p className="label-meta mb-3">Satellite ({report.satellite.count} cases)</p>
          <div className="space-y-2">
            <Metric label="Feature F1" value={report.satellite.feature_f1} />
            <Metric label="Precision" value={report.satellite.feature_precision} />
            <Metric label="Recall" value={report.satellite.feature_recall} />
          </div>
        </div>
        <div className="glass-card p-4">
          <p className="label-meta mb-3">Calibration</p>
          <div className="space-y-2">
            <Metric label="ECE" value={report.calibration.ece} />
            <Metric label="Brier Score" value={report.calibration.brier} />
            <Metric label="Samples" value={report.calibration.samples} />
          </div>
        </div>
      </div>

      {/* Ablation */}
      <div className="glass-card p-5">
        <p className="label-meta mb-3">Doc QA Ablation ({report.doc_qa_ablation.count} docs)</p>
        <div className="grid grid-cols-3 gap-2">
          <Metric label="OCR-only EM" value={report.doc_qa_ablation.ocr_only_em} />
          <Metric label="VQA-only EM" value={report.doc_qa_ablation.vqa_only_em} />
          <Metric label="Hybrid EM" value={report.doc_qa_ablation.hybrid_em} />
        </div>
      </div>
    </div>
  );
}
