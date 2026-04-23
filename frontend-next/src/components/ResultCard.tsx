"use client";

import { Brain, Compass, FileImage, ScanText, Sparkles } from "lucide-react";
import type { VqaResponse } from "@/lib/api";

interface Props { result: VqaResponse; loading?: boolean; onSuggest?: (q: string) => void; }

function ConfidenceRing({ value, color }: { value: number; color: string }) {
  const r = 44;
  const circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;
  return (
    <svg width="104" height="104" viewBox="0 0 104 104">
      <circle cx="52" cy="52" r={r} fill="none" stroke="#2f3445" strokeWidth="8" />
      <circle
        cx="52" cy="52" r={r} fill="none"
        stroke={color} strokeWidth="8" strokeLinecap="round"
        strokeDasharray={circ} strokeDashoffset={offset}
        className="conf-ring"
        style={{ transform: "rotate(-90deg)", transformOrigin: "52px 52px", transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)" }}
      />
      <text x="52" y="52" textAnchor="middle" dominantBaseline="central" fill={color} fontSize="15" fontWeight="700" fontFamily="JetBrains Mono, monospace">
        {value.toFixed(0)}%
      </text>
    </svg>
  );
}

function confColor(v: number) {
  if (v >= 75) return "#4ade80";
  if (v >= 45) return "#fbbf24";
  return "#f87171";
}

function ModelBadge({ label, answer, conf, accent }: { label: string; answer: string; conf: number; accent: string }) {
  return (
    <div className="glass-card p-4 space-y-2" style={{ borderColor: `${accent}22` }}>
      <p className="label-meta" style={{ color: accent, fontSize: "0.6rem" }}>{label}</p>
      <p className="text-sm text-on-surface leading-relaxed min-h-[3rem]">{answer || "—"}</p>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 rounded-full bg-surface-highest overflow-hidden">
          <div className="h-full rounded-full transition-all duration-700" style={{ width: `${conf}%`, background: `linear-gradient(90deg, ${accent}88, ${accent})` }} />
        </div>
        <span className="mono text-xs" style={{ color: accent }}>{conf.toFixed(1)}%</span>
      </div>
    </div>
  );
}

export function ResultCard({ result, onSuggest }: Props) {
  const conf = result.confidence || result.hybrid?.confidence || 0;
  const answer = result.hybrid?.answer || result.answer || "Unable to determine";
  const explanation = result.hybrid?.explanation || "";
  const color = confColor(conf);

  return (
    <div className="space-y-4 animate-fade-up">
      {/* Main answer card */}
      <div className="glass-card p-6">
        <div className="flex items-start justify-between gap-6">
          <div className="flex-1 min-w-0">
            <p className="label-meta mb-2">Final Response</p>
            <h2 className="text-2xl font-bold text-on-surface leading-tight break-words">{answer}</h2>
            {explanation && <p className="mt-3 text-sm text-on-surface-variant leading-relaxed">{explanation}</p>}
            <div className="flex flex-wrap gap-2 mt-4">
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-surface-highest text-on-surface-variant mono uppercase tracking-wide">
                {result.model || "AUTO"}
              </span>
              {result.answer_type && (
                <span className="px-2.5 py-1 rounded-full text-xs font-semibold" style={{ background: "rgba(192,193,255,0.12)", color: "#c0c1ff" }}>
                  {result.answer_type}
                </span>
              )}
              {result.time_ms != null && (
                <span className="px-2.5 py-1 rounded-full text-xs bg-surface-highest text-on-surface-variant mono">
                  {result.time_ms}ms
                </span>
              )}
            </div>
          </div>
          <div className="shrink-0 text-center">
            <ConfidenceRing value={conf} color={color} />
            <p className="label-meta mt-1" style={{ color, fontSize: "0.6rem" }}>
              {conf >= 75 ? "High" : conf >= 45 ? "Medium" : "Low"}
            </p>
          </div>
        </div>
      </div>

      {/* Model comparison */}
      <div className="grid grid-cols-3 gap-3">
        <ModelBadge label="Baseline" answer={result.base_answers?.[0]?.answer || "—"} conf={result.base_answers?.[0]?.confidence || 0} accent="#908fa0" />
        <ModelBadge label="Fine-Tuned" answer={result.ft_answers?.[0]?.answer || "—"} conf={result.ft_answers?.[0]?.confidence || 0} accent="#c0c1ff" />
        <ModelBadge label="Hybrid" answer={answer} conf={conf} accent="#2fd9f4" />
      </div>

      {/* Suggested queries */}
      {result.suggested_queries?.length ? (
        <div className="glass-card p-4">
          <p className="label-meta mb-3">Follow-up Prompts</p>
          <div className="flex flex-wrap gap-2">
            {result.suggested_queries.map((q) => (
              <button key={q} className="question-chip" onClick={() => onSuggest?.(q)}>{q}</button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
