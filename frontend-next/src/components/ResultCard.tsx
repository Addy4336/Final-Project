"use client";

import type { UiMode, VqaResponse } from "@/lib/api";

interface Props { 
  data: VqaResponse; 
  mode: UiMode; 
  question: string;
  imageUrl?: string; 
}

function confColor(v: number) {
  if (v >= 75) return "#22d3ee"; // Cyan
  if (v >= 45) return "#818cf8"; // Indigo
  return "#fb7185"; // Rose
}

export function ResultCard({ data, mode, question }: Props) {
  const conf = data.confidence || data.hybrid?.confidence || 0;
  const answer = data.hybrid?.answer || data.answer || "Unable to determine";
  const explanation = data.hybrid?.explanation || "";
  const color = confColor(conf);

  return (
    <div className="animate-fade-up w-full max-w-5xl mx-auto flex flex-col gap-16">
      
      {/* ── META HEADER ── */}
      <div className="flex justify-between items-start border-b border-white/5 pb-4">
        <div>
          <p className="font-mono text-[10px] tracking-[0.3em] uppercase text-white/40 mb-1">Query Parameter</p>
          <p className="font-serif text-xl text-white/80 italic">&quot;{question}&quot;</p>
        </div>
        <div className="text-right">
          <p className="font-mono text-[10px] tracking-[0.3em] uppercase text-white/40 mb-1">Execution Mode</p>
          <p className="font-mono text-xs uppercase tracking-widest text-tertiary">{mode}</p>
        </div>
      </div>

      {/* ── THE MAJESTIC METRIC ── */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-12 relative">
        <div className="absolute inset-0 bg-tertiary/5 blur-[80px] rounded-full pointer-events-none -z-10" />
        
        <div className="flex-1">
          <p className="font-mono text-[10px] tracking-[0.4em] uppercase text-white/30 mb-6">Primary Synthesis</p>
          <h2 className="font-serif text-5xl md:text-[5rem] leading-[1.1] tracking-tight text-white mb-6">
            {answer}
          </h2>
          {explanation && (
            <p className="font-sans text-sm text-white/50 leading-relaxed max-w-xl border-l border-tertiary/20 pl-4">
              {explanation}
            </p>
          )}
        </div>

        <div className="shrink-0 flex flex-col items-center justify-center relative">
          <svg width="200" height="200" viewBox="0 0 200 200" className="-rotate-90">
            <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(255,255,255,0.02)" strokeWidth="2" />
            <circle
              cx="100" cy="100" r="90" fill="none"
              stroke={color} strokeWidth="1" strokeLinecap="square"
              strokeDasharray={2 * Math.PI * 90} 
              strokeDashoffset={(2 * Math.PI * 90) - ((conf / 100) * (2 * Math.PI * 90))}
              className="transition-all duration-1000 ease-out"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-serif text-5xl font-light tracking-tighter" style={{ color }}>
              {conf.toFixed(0)}<span className="text-2xl text-white/30">%</span>
            </span>
            <span className="font-mono text-[9px] uppercase tracking-[0.4em] text-white/30 mt-2">Confidence</span>
          </div>
        </div>
      </div>

      {/* ── DATA VISUALIZATION / ENTITIES ── */}
      {data.ocr_metadata?.fields && Object.keys(data.ocr_metadata.fields).length > 0 && (
        <div className="mt-8">
          <p className="font-mono text-[10px] tracking-[0.3em] uppercase text-white/30 mb-6">Extracted Entities</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-white/5 border border-white/5 rounded-xl overflow-hidden">
            {Object.entries(data.ocr_metadata.fields).map(([label, value], idx) => (
              <div key={idx} className="bg-[#02040a] p-5 group hover:bg-white/[0.02] transition-colors relative overflow-hidden">
                <div className="absolute bottom-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-tertiary/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/30 mb-2 truncate">
                  {label.replace(/_/g, " ")}
                </p>
                <p className="font-serif text-lg text-white/90 truncate">{value}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── TECHNICAL READOUTS ── */}
      <div className="mt-8 flex gap-8 border-t border-white/5 pt-8">
        <div>
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30 mb-1">Latency</p>
          <p className="font-mono text-xs text-white/60">{data.time_ms != null ? `${data.time_ms}ms` : "—"}</p>
        </div>
        <div>
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30 mb-1">Routing Logic</p>
          <p className="font-mono text-xs text-white/60">{data.answer_type || "Standard"}</p>
        </div>
        <div>
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30 mb-1">Engine</p>
          <p className="font-mono text-xs text-white/60">{data.model || "Nexus Core"}</p>
        </div>
      </div>
    </div>
  );
}
