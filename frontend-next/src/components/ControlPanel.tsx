"use client";

import { LoaderCircle, SendHorizonal, Sun, Contrast, UploadCloud, X } from "lucide-react";
import type { UiMode } from "@/lib/api";

const MODES: UiMode[] = ["AUTO", "OCR", "SATELLITE", "VQA"];
const LANGS = [
  { value: "en-US", label: "English" },
  { value: "hi-IN", label: "हिंदी" },
  { value: "gu-IN", label: "ગુજરાતી" },
];
const SAMPLE_Q: Record<UiMode, string[]> = {
  AUTO: ["What is the main content?", "Classify this scene", "Give an analysis summary"],
  OCR: ["What is the name?", "What is the emp code?", "Read all text"],
  SATELLITE: ["Are roads visible?", "How much vegetation?", "Describe land-use"],
  VQA: ["What color is dominant?", "How many objects?", "What is happening?"],
};

interface Props {
  imageFile: File | null; imageUrl: string;
  mode: UiMode; setMode: (m: UiMode) => void;
  lang: string; setLang: (l: string) => void;
  brightness: number; setBrightness: (v: number) => void;
  contrast: number; setContrast: (v: number) => void;
  question: string; setQuestion: (q: string) => void;
  recommendedMode: UiMode | null;
  loading: boolean; error: string;
  onSetImage: (f: File) => void;
  onClear: () => void;
  onSubmit: () => void;
  suggestedQueries?: string[];
}

export function ControlPanel(p: Props) {
  const [dragActive, setDragActive] = useState(false);

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (file) p.onSetImage(file);
  }

  return (
    <div className="space-y-4">
      {/* Image input */}
      <div>
        <p className="label-meta mb-2" style={{ fontSize: "0.6rem" }}>Image Input</p>
        <div
          className={`rounded-xl border-2 border-dashed transition-all duration-200 ${
            dragActive ? "border-primary bg-primary/5" : "border-outline-variant/40 bg-surface-lowest"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); }}
        >
          {p.imageUrl ? (
            <div className="p-3 space-y-2">
              <img
                src={p.imageUrl} alt="Preview"
                className="w-full max-h-40 object-contain rounded-lg"
                style={{ background: "#090e1c", filter: `brightness(${p.brightness}%) contrast(${p.contrast}%)` }}
              />
              <div className="flex gap-2">
                <label className="flex-1 text-center py-2 rounded-lg text-xs font-semibold cursor-pointer btn-primary">
                  Replace
                  <input type="file" className="hidden" accept="image/*" onChange={(e) => handleFiles(e.target.files)} />
                </label>
                <button className="px-3 py-2 rounded-lg text-xs border border-outline-variant/40 text-on-surface-variant hover:border-error hover:text-error transition-colors" onClick={p.onClear}>
                  <X size={14} />
                </button>
              </div>
            </div>
          ) : (
            <label className="flex flex-col items-center gap-2 py-8 px-4 text-center cursor-pointer">
              <div className="w-12 h-12 rounded-xl bg-surface-high grid place-items-center">
                <UploadCloud size={22} className="text-primary" />
              </div>
              <div>
                <p className="text-sm font-semibold text-on-surface">Drop image or browse</p>
                <p className="text-xs text-on-surface-variant mt-0.5">Document · Satellite · Photo</p>
              </div>
              <span className="mt-1 px-4 py-1.5 rounded-full text-xs font-semibold btn-primary">Select File</span>
              <input type="file" className="hidden" accept="image/*" onChange={(e) => handleFiles(e.target.files)} />
            </label>
          )}
        </div>
      </div>

      {/* Mode pills */}
      <div>
        <p className="label-meta mb-2" style={{ fontSize: "0.6rem" }}>Analysis Mode</p>
        <div className="flex flex-wrap gap-2">
          {MODES.map((m) => (
            <button key={m} className={`mode-pill ${p.mode === m ? "active" : ""}`} onClick={() => p.setMode(m)}>
              {m}
              {p.recommendedMode === m && <span className="ml-1 opacity-70">★</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Language */}
      <div>
        <p className="label-meta mb-1.5" style={{ fontSize: "0.6rem" }}>Language</p>
        <select
          className="input-dark w-full px-3 py-2 text-sm"
          value={p.lang} onChange={(e) => p.setLang(e.target.value)}
        >
          {LANGS.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
        </select>
      </div>

      {/* Sliders */}
      <div className="space-y-3 p-3 rounded-xl" style={{ background: "#090e1c" }}>
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1 text-on-surface-variant"><Sun size={12} /> Brightness</span>
            <span className="mono text-primary">{p.brightness}%</span>
          </div>
          <input type="range" min={50} max={150} value={p.brightness} className="w-full" onChange={(e) => p.setBrightness(Number(e.target.value))} />
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1.5">
            <span className="flex items-center gap-1 text-on-surface-variant"><Contrast size={12} /> Contrast</span>
            <span className="mono text-primary">{p.contrast}%</span>
          </div>
          <input type="range" min={50} max={150} value={p.contrast} className="w-full" onChange={(e) => p.setContrast(Number(e.target.value))} />
        </div>
      </div>

      {/* Suggested questions */}
      {(p.suggestedQueries?.length ? p.suggestedQueries : SAMPLE_Q[p.mode]).length > 0 && (
        <div>
          <p className="label-meta mb-2" style={{ fontSize: "0.6rem" }}>Suggested Questions</p>
          <div className="flex flex-wrap gap-1.5">
            {(p.suggestedQueries?.length ? p.suggestedQueries : SAMPLE_Q[p.mode]).map((q) => (
              <button key={q} className="question-chip" onClick={() => p.setQuestion(q)}>{q}</button>
            ))}
          </div>
        </div>
      )}

      {/* Prompt + submit */}
      <div className="space-y-2">
        <p className="label-meta" style={{ fontSize: "0.6rem" }}>Analysis Prompt</p>
        <textarea
          className="input-dark w-full h-24 resize-none px-3 py-2 text-sm"
          placeholder="Ask a question about this image…"
          value={p.question}
          onChange={(e) => p.setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) p.onSubmit(); }}
        />
        <button
          disabled={!p.imageFile || !p.question.trim() || p.loading}
          className="btn-primary w-full flex items-center justify-center gap-2 rounded-xl py-2.5 text-sm"
          onClick={p.onSubmit}
        >
          {p.loading ? <LoaderCircle size={16} className="animate-spin" /> : <SendHorizonal size={16} />}
          {p.loading ? "Analyzing…" : "Analyze"}
        </button>
        {p.error && <p className="text-xs text-error">{p.error}</p>}
      </div>
    </div>
  );
}

// useState import needed inside this file
import { useState } from "react";
