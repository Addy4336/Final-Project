"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Activity, Brain, ClipboardList, Compass, FileImage, FlaskConical, ScanText, Sparkles, WandSparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { classifyImage, fetchHealth, fetchStats, runVqa, type ModelStat, type UiMode, type VqaResponse } from "@/lib/api";
import { ControlPanel } from "./ControlPanel";
import { ResultCard } from "./ResultCard";
import { EvidencePanel } from "./EvidencePanel";
import { StatsPanel } from "./StatsPanel";
import { HistoryPanel } from "./HistoryPanel";
import { BenchmarkPanel } from "./BenchmarkPanel";

type HealthState = { status: "checking" | "online" | "offline"; version?: string };

export type QueryHistoryItem = {
  id: string;
  question: string;
  mode: UiMode;
  createdAt: string;
  result: VqaResponse;
};

type ResultTab = "results" | "evidence" | "stats" | "benchmark" | "history";

function normalizeMode(raw: string | null | undefined): UiMode {
  const t = String(raw || "").trim().toUpperCase();
  if (t === "DOC" || t === "DOCUMENT" || t === "OCR") return "OCR";
  if (t === "SATELLITE" || t === "SAT") return "SATELLITE";
  if (t === "VQA") return "VQA";
  return "AUTO";
}

function classifyToUiMode(raw: string | undefined): UiMode {
  const t = String(raw || "").trim().toLowerCase();
  if (t === "doc" || t === "document" || t === "ocr") return "OCR";
  if (t === "satellite" || t === "sat") return "SATELLITE";
  if (t === "vqa") return "VQA";
  return "AUTO";
}

export function VisionQueryDashboard() {
  const [health, setHealth] = useState<HealthState>({ status: "checking" });
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [mode, setMode] = useState<UiMode>("AUTO");
  const [recommendedMode, setRecommendedMode] = useState<UiMode | null>(null);
  const [lang, setLang] = useState("en-US");
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [activeTab, setActiveTab] = useState<ResultTab>("results");
  const [stats, setStats] = useState<ModelStat[]>([]);

  const selectedItem = useMemo(() => {
    if (!history.length) return null;
    return history.find((i) => i.id === selectedId) || history[0];
  }, [history, selectedId]);

  const selectedResult = selectedItem?.result;

  useEffect(() => {
    let alive = true;
    fetchHealth()
      .then((r) => { if (alive) setHealth({ status: "online", version: r.version }); })
      .catch(() => { if (alive) setHealth({ status: "offline" }); });
    fetchStats()
      .then((s) => { if (alive) setStats(s.models); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!imageFile) { setImageUrl(""); return; }
    const url = URL.createObjectURL(imageFile);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  async function setImage(file: File) {
    setImageFile(file);
    setError("");
    try {
      const cls = await classifyImage(file);
      const suggested = classifyToUiMode(cls.mode);
      setRecommendedMode(suggested);
      setMode(suggested);
    } catch { setRecommendedMode(null); }
  }

  async function submitQuestion(nextQ?: string) {
    const q = (nextQ ?? question).trim();
    if (!imageFile || !q || loading) return;
    setError("");
    setLoading(true);
    try {
      const result = await runVqa({ image: imageFile, question: q, mode, lang, brightness, contrast });
      const resolvedMode = normalizeMode(result.auto_detected_mode || result.model);
      if (result.auto_detected_mode || result.model) setMode(resolvedMode);
      const item: QueryHistoryItem = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        question: q, mode, createdAt: new Date().toISOString(), result,
      };
      setHistory((p) => [item, ...p]);
      setSelectedId(item.id);
      setQuestion("");
      setActiveTab("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const modeColor: Record<UiMode, string> = {
    AUTO: "#c0c1ff", OCR: "#2fd9f4", SATELLITE: "#4ade80", VQA: "#d0bcff",
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ── */}
      <motion.header
        initial={{ opacity: 0, y: -14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="sticky top-0 z-50 flex items-center justify-between px-6 py-3 border-b border-outline-variant/20"
        style={{ background: "rgba(14,19,34,0.85)", backdropFilter: "blur(16px)" }}
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-secondary-container grid place-items-center animate-float shadow-glow-primary">
            <WandSparkles size={18} className="text-white" />
          </div>
          <div>
            <p className="label-meta text-on-surface-variant" style={{ fontSize: "0.6rem" }}>PhD Defense Console</p>
            <h1 className="text-lg font-bold text-on-surface gradient-text">VisionQuery Nexus</h1>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Mode badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full glass">
            <div className="w-2 h-2 rounded-full" style={{ background: modeColor[mode], boxShadow: `0 0 6px ${modeColor[mode]}88` }} />
            <span className="label-meta" style={{ fontSize: "0.65rem" }}>{mode}</span>
          </div>
          {/* Backend status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full glass">
            <div className={`status-dot ${health.status}`} />
            <div>
              <p className="label-meta" style={{ fontSize: "0.6rem" }}>Backend</p>
              <p className="text-xs font-semibold text-on-surface mono">
                {health.status === "checking" ? "…" : health.status === "online" ? `v${health.version || "?"}` : "Offline"}
              </p>
            </div>
          </div>
        </div>
      </motion.header>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Control Panel */}
        <motion.aside
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.08 }}
          className="w-80 shrink-0 border-r border-outline-variant/15 overflow-y-auto p-4 space-y-4"
          style={{ background: "#161b2b" }}
        >
          <ControlPanel
            imageFile={imageFile} imageUrl={imageUrl}
            mode={mode} setMode={setMode}
            lang={lang} setLang={setLang}
            brightness={brightness} setBrightness={setBrightness}
            contrast={contrast} setContrast={setContrast}
            question={question} setQuestion={setQuestion}
            recommendedMode={recommendedMode}
            loading={loading} error={error}
            onSetImage={setImage}
            onClear={() => { setImageFile(null); setRecommendedMode(null); setHistory([]); setSelectedId(""); }}
            onSubmit={() => submitQuestion()}
            suggestedQueries={selectedResult?.suggested_queries}
          />
        </motion.aside>

        {/* Center: Results */}
        <motion.main
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.45, delay: 0.14 }}
          className="flex-1 overflow-y-auto p-5 space-y-4"
        >
          {/* Tab nav */}
          <div className="flex items-center gap-1 p-1 rounded-xl w-fit" style={{ background: "#090e1c" }}>
            {(["results", "evidence", "stats", "benchmark", "history"] as ResultTab[]).map((tab) => (
              <button key={tab} className={`tab-btn ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab === "results" && <><Sparkles size={12} className="inline mr-1.5" />Results</>}
                {tab === "evidence" && <><ScanText size={12} className="inline mr-1.5" />Evidence</>}
                {tab === "stats" && <><Activity size={12} className="inline mr-1.5" />Stats</>}
                {tab === "benchmark" && <><FlaskConical size={12} className="inline mr-1.5" />Benchmark</>}
                {tab === "history" && <><ClipboardList size={12} className="inline mr-1.5" />History</>}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {activeTab === "results" && (
              <motion.div key="results" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                {loading ? (
                  <SkeletonResults />
                ) : !selectedResult ? (
                  <EmptyState />
                ) : (
                  <ResultCard result={selectedResult} loading={loading} onSuggest={submitQuestion} />
                )}
              </motion.div>
            )}
            {activeTab === "evidence" && (
              <motion.div key="evidence" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                {selectedResult ? (
                  <EvidencePanel result={selectedResult} />
                ) : (
                  <EmptyStateTab label="Run an analysis to see evidence" icon={<ScanText size={32} />} />
                )}
              </motion.div>
            )}
            {activeTab === "stats" && (
              <motion.div key="stats" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StatsPanel stats={stats} />
              </motion.div>
            )}
            {activeTab === "benchmark" && (
              <motion.div key="benchmark" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <BenchmarkPanel />
              </motion.div>
            )}
            {activeTab === "history" && (
              <motion.div key="history" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <div className="glass-card p-5">
                  <HistoryPanel
                    history={history}
                    selectedId={selectedItem?.id || ""}
                    onSelect={setSelectedId}
                    onClear={() => { setHistory([]); setSelectedId(""); }}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.main>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="glass-card p-10 text-center space-y-4">
      <div className="w-16 h-16 rounded-2xl mx-auto bg-gradient-to-br from-primary/20 to-secondary-container/20 grid place-items-center">
        <Brain size={28} className="text-primary" />
      </div>
      <div>
        <p className="label-meta text-on-surface-variant mb-1">Ready for Analysis</p>
        <h2 className="text-xl font-bold text-on-surface">Professional Research Interface</h2>
        <p className="text-sm text-on-surface-variant mt-2 max-w-lg mx-auto leading-relaxed">
          Upload an image, ask a question, and compare base vs fine-tuned reasoning with OCR and visual evidence grounded in the hybrid pipeline.
        </p>
      </div>
    </div>
  );
}

function EmptyStateTab({ label, icon }: { label: string; icon: React.ReactNode }) {
  return (
    <div className="glass-card p-12 text-center">
      <div className="text-outline mx-auto mb-3 w-fit opacity-40">{icon}</div>
      <p className="text-sm text-on-surface-variant">{label}</p>
    </div>
  );
}

function SkeletonResults() {
  return (
    <div className="space-y-4">
      <div className="glass-card p-6 space-y-4">
        <div className="skeleton h-4 w-28 rounded" />
        <div className="skeleton h-8 w-3/4 rounded" />
        <div className="skeleton h-4 w-full rounded" />
        <div className="skeleton h-4 w-2/3 rounded" />
        <div className="skeleton h-2 w-full rounded-full mt-4" />
      </div>
      <div className="grid grid-cols-3 gap-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="glass-card p-4 space-y-3">
            <div className="skeleton h-4 w-20 rounded" />
            <div className="skeleton h-12 w-full rounded" />
            <div className="skeleton h-3 w-16 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}
