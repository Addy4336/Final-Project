"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Bot,
  Brain,
  ClipboardList,
  Compass,
  FileImage,
  ImagePlus,
  LoaderCircle,
  ScanText,
  SendHorizonal,
  Sparkles,
  UploadCloud,
  WandSparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  classifyImage,
  fetchHealth,
  runVqa,
  type UiMode,
  type VqaResponse,
} from "@/lib/api";

type HealthState = {
  status: "checking" | "online" | "offline";
  version?: string;
};

type QueryHistoryItem = {
  id: string;
  question: string;
  mode: UiMode;
  createdAt: string;
  result: VqaResponse;
};

const MODE_OPTIONS: Array<{ value: UiMode; label: string }> = [
  { value: "AUTO", label: "Auto" },
  { value: "OCR", label: "OCR" },
  { value: "SATELLITE", label: "Satellite" },
  { value: "VQA", label: "VQA" },
];

const SAMPLE_QUESTIONS: Record<UiMode, string[]> = {
  AUTO: [
    "What is the main content in this image?",
    "Can you classify this scene?",
    "Give a concise analysis summary.",
  ],
  OCR: [
    "What is the name?",
    "What is the emp code?",
    "Read all text from this document.",
  ],
  SATELLITE: [
    "Are roads visible in this area?",
    "How much vegetation is present?",
    "Describe the land-use pattern.",
  ],
  VQA: [
    "What color is dominant in this image?",
    "How many main objects are visible?",
    "What is happening in this scene?",
  ],
};

function normalizeMode(rawMode: string | null | undefined): UiMode {
  const token = String(rawMode || "").trim().toUpperCase();
  if (token === "DOC" || token === "DOCUMENT" || token === "OCR" || token === "DOC") {
    return "OCR";
  }
  if (token === "SATELLITE" || token === "SAT") {
    return "SATELLITE";
  }
  if (token === "VQA") {
    return "VQA";
  }
  return "AUTO";
}

function classifyToUiMode(rawMode: string | undefined): UiMode {
  const token = String(rawMode || "").trim().toLowerCase();
  if (token === "doc" || token === "document" || token === "ocr") {
    return "OCR";
  }
  if (token === "satellite" || token === "sat") {
    return "SATELLITE";
  }
  if (token === "vqa") {
    return "VQA";
  }
  return "AUTO";
}

function metricColor(confidence: number): string {
  if (confidence >= 75) return "bg-emerald-500";
  if (confidence >= 45) return "bg-amber-500";
  return "bg-rose-500";
}

export function VisionQueryDashboard() {
  const [health, setHealth] = useState<HealthState>({ status: "checking" });
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [dragActive, setDragActive] = useState(false);

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

  const selectedItem = useMemo(() => {
    if (!history.length) return null;
    return history.find((item) => item.id === selectedId) || history[0];
  }, [history, selectedId]);

  const selectedResult = selectedItem?.result;

  useEffect(() => {
    let alive = true;

    fetchHealth()
      .then((res) => {
        if (!alive) return;
        setHealth({ status: "online", version: res.version });
      })
      .catch(() => {
        if (!alive) return;
        setHealth({ status: "offline" });
      });

    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!imageFile) {
      setImageUrl("");
      return;
    }

    const nextUrl = URL.createObjectURL(imageFile);
    setImageUrl(nextUrl);

    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [imageFile]);

  const sampleQuestions = SAMPLE_QUESTIONS[mode];

  async function setImage(file: File) {
    setImageFile(file);
    setError("");

    try {
      const cls = await classifyImage(file);
      const suggested = classifyToUiMode(cls.mode);
      setRecommendedMode(suggested);
      setMode(suggested);
    } catch {
      setRecommendedMode(null);
    }
  }

  async function submitQuestion(nextQuestion?: string) {
    const q = (nextQuestion ?? question).trim();
    if (!imageFile || !q || loading) return;

    setError("");
    setLoading(true);

    try {
      const result = await runVqa({
        image: imageFile,
        question: q,
        mode,
        lang,
        brightness,
        contrast,
      });

      const resolvedMode = normalizeMode(result.auto_detected_mode || result.model);
      if (result.auto_detected_mode || result.model) {
        setMode(resolvedMode);
      }

      const item: QueryHistoryItem = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        question: q,
        mode,
        createdAt: new Date().toISOString(),
        result,
      };

      setHistory((prev) => [item, ...prev]);
      setSelectedId(item.id);
      setQuestion("");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="surface-grid min-h-screen px-4 py-5 md:px-8 md:py-8">
      <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-6">
        <motion.header
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="flex flex-wrap items-center justify-between gap-4 rounded-3xl border border-slate-200/80 bg-white/80 px-6 py-4 shadow-soft backdrop-blur"
        >
          <div className="flex items-center gap-3">
            <div className="float-loop grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-accent-500 to-coral-500 text-white shadow-panel">
              <WandSparkles size={20} />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">PhD Defense Console</p>
              <h1 className="text-2xl font-bold text-slate-900">VisionQuery Nexus</h1>
            </div>
          </div>

          <div className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-2">
            <Activity size={16} className={health.status === "online" ? "text-emerald-600" : "text-rose-600"} />
            <div>
              <p className="text-xs uppercase tracking-[0.15em] text-slate-500">Backend Status</p>
              <p className="metric-font text-sm font-semibold text-slate-900">
                {health.status === "checking"
                  ? "Checking"
                  : health.status === "online"
                    ? `Online v${health.version || "?"}`
                    : "Offline"}
              </p>
            </div>
          </div>
        </motion.header>

        <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
          <motion.section
            initial={{ opacity: 0, x: -18 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45, delay: 0.06 }}
            className="space-y-5 rounded-3xl border border-slate-200/80 bg-white/85 p-5 shadow-panel backdrop-blur"
          >
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Image Input</p>
              <div
                className={`mt-3 rounded-2xl border-2 border-dashed p-4 transition ${
                  dragActive
                    ? "border-accent-500 bg-accent-50"
                    : "border-slate-300 bg-slate-50/80"
                }`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                  const file = event.dataTransfer.files?.[0];
                  if (file) setImage(file);
                }}
              >
                {imageUrl ? (
                  <div className="space-y-3">
                    <img
                      src={imageUrl}
                      alt="Preview"
                      className="h-56 w-full rounded-xl object-contain bg-white"
                      style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
                    />
                    <div className="flex gap-2">
                      <label className="flex-1 cursor-pointer rounded-xl bg-accent-600 px-3 py-2 text-center text-sm font-semibold text-white transition hover:bg-accent-700">
                        Replace Image
                        <input
                          type="file"
                          className="hidden"
                          accept="image/*"
                          onChange={(event) => {
                            const file = event.target.files?.[0];
                            if (file) setImage(file);
                          }}
                        />
                      </label>
                      <button
                        type="button"
                        className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-slate-700 transition hover:bg-slate-100"
                        onClick={() => {
                          setImageFile(null);
                          setRecommendedMode(null);
                          setHistory([]);
                          setSelectedId("");
                        }}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="grid place-items-center gap-3 py-8 text-center">
                    <div className="grid h-14 w-14 place-items-center rounded-2xl bg-white text-accent-600 shadow">
                      <UploadCloud size={24} />
                    </div>
                    <div>
                      <p className="font-semibold text-slate-900">Drop an image or browse</p>
                      <p className="text-sm text-slate-500">Document, satellite scene, or real-world photo</p>
                    </div>
                    <label className="cursor-pointer rounded-xl bg-coral-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-coral-700">
                      Select Image
                      <input
                        type="file"
                        className="hidden"
                        accept="image/*"
                        onChange={(event) => {
                          const file = event.target.files?.[0];
                          if (file) setImage(file);
                        }}
                      />
                    </label>
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <label className="text-sm font-medium text-slate-700">
                Mode
                <select
                  className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm"
                  value={mode}
                  onChange={(event) => setMode(event.target.value as UiMode)}
                >
                  {MODE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="text-sm font-medium text-slate-700">
                Language
                <select
                  className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm"
                  value={lang}
                  onChange={(event) => setLang(event.target.value)}
                >
                  <option value="en-US">English</option>
                  <option value="hi-IN">Hindi</option>
                  <option value="gu-IN">Gujarati</option>
                </select>
              </label>
            </div>

            <div className="space-y-3 rounded-2xl border border-slate-200 bg-white px-4 py-3">
              <div>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <span className="text-slate-700">Brightness</span>
                  <span className="metric-font text-slate-500">{brightness}%</span>
                </div>
                <input
                  className="w-full accent-accent-600"
                  type="range"
                  min={50}
                  max={150}
                  value={brightness}
                  onChange={(event) => setBrightness(Number(event.target.value))}
                />
              </div>
              <div>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <span className="text-slate-700">Contrast</span>
                  <span className="metric-font text-slate-500">{contrast}%</span>
                </div>
                <input
                  className="w-full accent-accent-600"
                  type="range"
                  min={50}
                  max={150}
                  value={contrast}
                  onChange={(event) => setContrast(Number(event.target.value))}
                />
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50/70 px-4 py-3">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Suggested Questions</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {sampleQuestions.map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-700 transition hover:border-accent-400 hover:text-accent-700"
                    onClick={() => setQuestion(sample)}
                  >
                    {sample}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white px-4 py-4">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-sm font-semibold text-slate-900">Analysis Prompt</p>
                {recommendedMode ? (
                  <span className="rounded-full bg-accent-100 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-accent-700">
                    Recommended: {recommendedMode}
                  </span>
                ) : null}
              </div>
              <textarea
                className="h-24 w-full resize-none rounded-xl border border-slate-300 px-3 py-2 text-sm text-slate-800 outline-none ring-accent-400 transition focus:ring-2"
                placeholder="Ask your question about this image..."
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
              />
              <button
                type="button"
                disabled={!imageFile || !question.trim() || loading}
                className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent-600 to-coral-500 px-4 py-2.5 font-semibold text-white transition hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-45"
                onClick={() => submitQuestion()}
              >
                {loading ? <LoaderCircle size={16} className="animate-spin" /> : <SendHorizonal size={16} />}
                {loading ? "Running analysis..." : "Analyze"}
              </button>
              {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
            </div>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, x: 18 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45, delay: 0.1 }}
            className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]"
          >
            <div className="space-y-6">
              <AnimatePresence mode="wait">
                {!selectedResult ? (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="rounded-3xl border border-slate-200/80 bg-white/85 p-8 shadow-panel"
                  >
                    <div className="flex items-start gap-4">
                      <div className="grid h-14 w-14 place-items-center rounded-2xl bg-accent-100 text-accent-700">
                        <Sparkles size={22} />
                      </div>
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Ready for Demo</p>
                        <h2 className="mt-1 text-2xl font-bold text-slate-900">Professional Presentation Interface</h2>
                        <p className="mt-2 max-w-2xl text-slate-600">
                          Upload any image, ask a question, and compare base vs fine-tuned reasoning with OCR and visual evidence grounded in your backend pipeline.
                        </p>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key={selectedItem?.id}
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-5"
                  >
                    <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-6 shadow-panel">
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Final Response</p>
                          <h3 className="mt-2 text-2xl font-bold text-slate-900">
                            {selectedResult.hybrid?.answer || selectedResult.answer || "No answer"}
                          </h3>
                          <p className="mt-2 max-w-3xl text-sm text-slate-600">
                            {selectedResult.hybrid?.explanation || "No additional explanation available."}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-right">
                          <p className="metric-font text-xs uppercase tracking-[0.14em] text-slate-500">Confidence</p>
                          <p className="metric-font text-2xl font-semibold text-slate-900">
                            {(selectedResult.confidence || selectedResult.hybrid?.confidence || 0).toFixed(1)}%
                          </p>
                          <p className="metric-font text-xs text-slate-500">
                            {(selectedResult.time_ms || selectedResult.latency_ms || 0).toFixed(0)} ms
                          </p>
                        </div>
                      </div>

                      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-200">
                        <div
                          className={`h-full ${metricColor(selectedResult.confidence || selectedResult.hybrid?.confidence || 0)}`}
                          style={{ width: `${Math.max(2, Math.min(100, selectedResult.confidence || selectedResult.hybrid?.confidence || 0))}%` }}
                        />
                      </div>

                      <div className="mt-4 flex flex-wrap gap-2 text-xs">
                        <span className="rounded-full bg-slate-900 px-2.5 py-1 uppercase tracking-[0.12em] text-white">
                          {selectedResult.model || mode}
                        </span>
                        {selectedResult.answer_type ? (
                          <span className="rounded-full bg-accent-100 px-2.5 py-1 uppercase tracking-[0.12em] text-accent-800">
                            {selectedResult.answer_type}
                          </span>
                        ) : null}
                      </div>
                    </div>

                    <div className="grid gap-4 md:grid-cols-3">
                      <MetricCard
                        title="Baseline"
                        icon={<Bot size={16} />}
                        answer={selectedResult.base_answers?.[0]?.answer || "--"}
                        confidence={selectedResult.base_answers?.[0]?.confidence || 0}
                        tone="slate"
                      />
                      <MetricCard
                        title="Fine-Tuned"
                        icon={<Brain size={16} />}
                        answer={selectedResult.ft_answers?.[0]?.answer || "--"}
                        confidence={selectedResult.ft_answers?.[0]?.confidence || 0}
                        tone="accent"
                      />
                      <MetricCard
                        title="Hybrid"
                        icon={<Sparkles size={16} />}
                        answer={selectedResult.hybrid?.answer || selectedResult.answer || "--"}
                        confidence={selectedResult.hybrid?.confidence || selectedResult.confidence || 0}
                        tone="coral"
                      />
                    </div>

                    {(selectedResult.ocr_metadata?.fields || selectedResult.ocr_extracted_text) && (
                      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-5 shadow-panel">
                        <div className="mb-4 flex items-center gap-2 text-slate-900">
                          <ScanText size={18} className="text-accent-700" />
                          <h4 className="text-lg font-semibold">OCR Evidence</h4>
                        </div>

                        {selectedResult.ocr_metadata?.fields ? (
                          <div className="grid gap-2 sm:grid-cols-2">
                            {Object.entries(selectedResult.ocr_metadata.fields)
                              .filter(([, value]) => Boolean(value))
                              .map(([key, value]) => (
                                <div key={key} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
                                  <p className="text-xs uppercase tracking-[0.14em] text-slate-500">{key}</p>
                                  <p className="text-sm font-medium text-slate-800">{value}</p>
                                </div>
                              ))}
                          </div>
                        ) : null}

                        {selectedResult.ocr_extracted_text ? (
                          <pre className="mt-4 max-h-56 overflow-auto rounded-xl border border-slate-200 bg-slate-950 p-3 text-xs text-slate-100">
                            {selectedResult.ocr_extracted_text}
                          </pre>
                        ) : null}
                      </div>
                    )}

                    {selectedResult.detection_image ? (
                      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-5 shadow-panel">
                        <div className="mb-4 flex items-center gap-2 text-slate-900">
                          <FileImage size={18} className="text-coral-700" />
                          <h4 className="text-lg font-semibold">Visual Grounding</h4>
                        </div>
                        <img
                          src={`data:image/png;base64,${selectedResult.detection_image}`}
                          alt="Detection result"
                          className="w-full rounded-2xl border border-slate-200"
                        />
                      </div>
                    ) : null}

                    {selectedResult.satellite?.features?.length ? (
                      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-5 shadow-panel">
                        <div className="mb-4 flex items-center gap-2 text-slate-900">
                          <Compass size={18} className="text-coral-700" />
                          <h4 className="text-lg font-semibold">Satellite Feature Summary</h4>
                        </div>
                        <div className="grid gap-3 md:grid-cols-2">
                          {selectedResult.satellite.features.map((feature, index) => (
                            <div key={`${feature.type || "feature"}-${index}`} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
                              <p className="text-sm font-semibold text-slate-800">{feature.label || feature.type || "Feature"}</p>
                              <p className="metric-font text-xs text-slate-500">
                                coverage {Number(feature.coverage_pct || 0).toFixed(1)}% | conf {Number(feature.confidence || 0).toFixed(1)}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {selectedResult.suggested_queries?.length ? (
                      <div className="rounded-3xl border border-slate-200/80 bg-white/90 p-5 shadow-panel">
                        <div className="mb-3 flex items-center gap-2 text-slate-900">
                          <ImagePlus size={18} className="text-accent-700" />
                          <h4 className="text-lg font-semibold">Follow-up Prompts</h4>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {selectedResult.suggested_queries.map((suggestion) => (
                            <button
                              key={suggestion}
                              className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-700 transition hover:border-accent-500 hover:text-accent-700"
                              onClick={() => setQuestion(suggestion)}
                              type="button"
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="rounded-3xl border border-slate-200/80 bg-white/85 p-4 shadow-panel backdrop-blur">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2 text-slate-900">
                  <ClipboardList size={18} className="text-accent-700" />
                  <h3 className="text-lg font-semibold">Query History</h3>
                </div>
                <button
                  className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500 transition hover:text-slate-800"
                  type="button"
                  onClick={() => {
                    setHistory([]);
                    setSelectedId("");
                  }}
                >
                  Clear
                </button>
              </div>

              {!history.length ? (
                <p className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm text-slate-500">
                  Your analysis history appears here after the first query.
                </p>
              ) : (
                <div className="max-h-[720px] space-y-2 overflow-auto pr-1">
                  {history.map((item) => {
                    const isActive = item.id === selectedItem?.id;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => setSelectedId(item.id)}
                        className={`w-full rounded-xl border px-3 py-2 text-left transition ${
                          isActive
                            ? "border-accent-500 bg-accent-50"
                            : "border-slate-200 bg-white hover:border-slate-300"
                        }`}
                      >
                        <p className="line-clamp-2 text-sm font-semibold text-slate-900">{item.question}</p>
                        <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
                          <span className="uppercase tracking-[0.12em]">{item.result.model || item.mode}</span>
                          <span className="metric-font">{(item.result.confidence || item.result.hybrid?.confidence || 0).toFixed(1)}%</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </motion.section>
        </div>
      </div>
    </main>
  );
}

type MetricCardProps = {
  title: string;
  icon: React.ReactNode;
  answer: string;
  confidence: number;
  tone: "slate" | "accent" | "coral";
};

function MetricCard({ title, icon, answer, confidence, tone }: MetricCardProps) {
  const toneClasses: Record<MetricCardProps["tone"], string> = {
    slate: "border-slate-200 bg-white",
    accent: "border-accent-200 bg-accent-50/60",
    coral: "border-coral-300 bg-coral-100/50",
  };

  return (
    <div className={`rounded-2xl border p-4 shadow-sm ${toneClasses[tone]}`}>
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-900">
        {icon}
        <span>{title}</span>
      </div>
      <p className="min-h-16 text-sm text-slate-700">{answer || "--"}</p>
      <p className="metric-font mt-2 text-xs text-slate-500">confidence {confidence.toFixed(1)}%</p>
    </div>
  );
}
