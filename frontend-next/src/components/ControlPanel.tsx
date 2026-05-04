"use client";

import { LoaderCircle, SendHorizonal, Sun, Contrast } from "lucide-react";
import type { UiMode } from "@/lib/api";
import { useState } from "react";

const MODES: UiMode[] = ["AUTO", "OCR", "SATELLITE", "VQA"];
const LANGS = [
  { value: "en-US", label: "English" },
  { value: "hi-IN", label: "हिंदी" },
  { value: "gu-IN", label: "ગુજરાતી" },
];
const SAMPLE_Q: Record<UiMode, string[]> = {
  AUTO: ["Classify this scene", "Give an analysis summary"],
  OCR: ["What is the name?", "What is the emp code?", "Read all text"],
  SATELLITE: ["Are roads visible?", "How much vegetation?", "Describe land-use"],
  VQA: ["What color is dominant?", "How many objects?", "What is happening?"],
};

interface Props {
  mode: UiMode; setMode: (m: UiMode) => void;
  lang: string; setLang: (l: string) => void;
  brightness: number; setBrightness: (v: number) => void;
  contrast: number; setContrast: (v: number) => void;
  question: string; setQuestion: (q: string) => void;
  recommendedMode: UiMode | null;
  loading: boolean; error: string;
  onSubmit: () => void;
  suggestedQueries?: string[];
}

export function ControlPanel(p: Props) {
  return (
    <div className="space-y-6">
      {/* Mode pills */}
      <div>
        <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30 mb-3">Analysis Mode</p>
        <div className="flex flex-wrap gap-2">
          {MODES.map((m) => (
            <button 
              key={m} 
              className={`px-3 py-1.5 rounded-sm font-mono text-[10px] uppercase tracking-widest transition-all border ${
                p.mode === m 
                  ? "border-tertiary bg-tertiary/10 text-tertiary shadow-[0_0_15px_rgba(47,217,244,0.15)]" 
                  : "border-white/5 bg-transparent text-white/40 hover:text-white/80 hover:border-white/20"
              }`} 
              onClick={() => p.setMode(m)}
            >
              {m}
              {p.recommendedMode === m && <span className="ml-1.5 opacity-50 text-tertiary">★</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Language */}
      <div>
        <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30 mb-3">Language Target</p>
        <select
          className="w-full bg-[#02040a] border border-white/10 rounded-sm px-3 py-2 text-xs font-mono text-white/70 focus:border-tertiary focus:outline-none transition-colors"
          value={p.lang} onChange={(e) => p.setLang(e.target.value)}
        >
          {LANGS.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
        </select>
      </div>

      {/* Sliders */}
      <div className="space-y-4 pt-2">
        <div>
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-widest mb-2">
            <span className="flex items-center gap-1.5 text-white/40"><Sun size={12} strokeWidth={1.5} /> Brightness</span>
            <span className="text-tertiary">{p.brightness}%</span>
          </div>
          <input type="range" min={50} max={150} value={p.brightness} className="w-full" onChange={(e) => p.setBrightness(Number(e.target.value))} />
        </div>
        <div>
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-widest mb-2">
            <span className="flex items-center gap-1.5 text-white/40"><Contrast size={12} strokeWidth={1.5} /> Contrast</span>
            <span className="text-tertiary">{p.contrast}%</span>
          </div>
          <input type="range" min={50} max={150} value={p.contrast} className="w-full" onChange={(e) => p.setContrast(Number(e.target.value))} />
        </div>
      </div>

      {/* Suggested questions */}
      {(p.suggestedQueries?.length ? p.suggestedQueries : SAMPLE_Q[p.mode]).length > 0 && (
        <div className="pt-2">
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30 mb-3">Predicted Parameters</p>
          <div className="flex flex-wrap gap-2">
            {(p.suggestedQueries?.length ? p.suggestedQueries : SAMPLE_Q[p.mode]).map((q) => (
              <button 
                key={q} 
                className="px-2.5 py-1 text-[10px] font-mono border border-white/5 bg-white/[0.02] text-white/40 hover:text-tertiary hover:border-tertiary/30 transition-colors" 
                onClick={() => p.setQuestion(q)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Prompt + submit */}
      <div className="pt-2 relative">
        <textarea
          className="w-full h-24 resize-none bg-[#02040a] border border-white/10 rounded-sm px-4 py-3 text-sm font-sans text-white placeholder-white/20 focus:border-tertiary focus:outline-none transition-colors"
          placeholder="Execute Query..."
          value={p.question}
          onChange={(e) => p.setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) p.onSubmit(); }}
        />
        <button
          disabled={!p.question.trim() || p.loading}
          className="absolute bottom-3 right-3 bg-tertiary/10 hover:bg-tertiary/20 text-tertiary border border-tertiary/30 rounded-sm px-3 py-1.5 flex items-center gap-2 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          onClick={p.onSubmit}
        >
          {p.loading ? <LoaderCircle size={14} className="animate-spin" /> : <SendHorizonal size={14} />}
          <span className="font-mono text-[10px] uppercase tracking-widest">{p.loading ? "Running" : "Execute"}</span>
        </button>
        {p.error && <p className="text-[10px] font-mono text-rose-400 mt-2">{p.error}</p>}
      </div>
    </div>
  );
}
