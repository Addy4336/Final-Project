"use client";

import { Activity } from "lucide-react";
import type { ModelStat } from "@/lib/api";

interface Props { stats: ModelStat[]; }

export function StatsPanel({ stats }: Props) {
  if (!stats.length) {
    return (
      <div className="glass-card p-10 text-center">
        <Activity size={32} className="mx-auto text-outline opacity-40 mb-3" />
        <p className="text-sm text-on-surface-variant">Loading model statistics…</p>
      </div>
    );
  }

  const maxAcc = Math.max(...stats.map((s) => s.accuracy_overall), 1);

  return (
    <div className="space-y-4 animate-fade-up">
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-6">
          <Activity size={18} className="text-primary" />
          <h3 className="font-semibold text-on-surface text-lg">Model Accuracy Overview</h3>
          <span className="ml-auto label-meta text-on-surface-variant" style={{ fontSize: "0.6rem" }}>Overall Accuracy %</span>
        </div>

        <div className="space-y-5">
          {stats.map((model) => (
            <div key={model.type} className="space-y-1.5">
              <div className="flex justify-between items-center">
                <div>
                  <span className="text-sm font-semibold text-on-surface">{model.name}</span>
                  <p className="text-xs text-on-surface-variant mt-0.5">{model.note}</p>
                </div>
                <span className="mono text-sm font-bold" style={{ color: model.color }}>
                  {model.accuracy_overall.toFixed(1)}%
                </span>
              </div>
              <div className="h-2.5 rounded-full overflow-hidden" style={{ background: "#090e1c" }}>
                <div
                  className="h-full rounded-full transition-all duration-1000 ease-out"
                  style={{
                    width: `${(model.accuracy_overall / maxAcc) * 100}%`,
                    background: `linear-gradient(90deg, ${model.color}88, ${model.color})`,
                    boxShadow: `0 0 8px ${model.color}66`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Hybrid gains summary */}
      <div className="grid grid-cols-2 gap-3">
        <div className="glass-card p-4">
          <p className="label-meta mb-2" style={{ fontSize: "0.6rem" }}>VQA Improvement</p>
          <div className="text-xl font-bold text-primary">+46.5 pp</div>
          <p className="text-xs text-on-surface-variant mt-1">Base → Fine-tuned</p>
        </div>
        <div className="glass-card p-4">
          <p className="label-meta mb-2" style={{ fontSize: "0.6rem" }}>OCR Recovery</p>
          <div className="text-xl font-bold text-tertiary">+12%</div>
          <p className="text-xs text-on-surface-variant mt-1">4-strategy preprocessing</p>
        </div>
      </div>

      {/* Publication gate hint */}
      <div className="glass-card p-4">
        <p className="label-meta mb-3">Publication Gate Targets</p>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            { label: "OCR Field F1", target: "≥ 0.80" },
            { label: "OCR Mean CER", target: "≤ 0.35" },
            { label: "VQA Fine-tuned EM", target: "≥ 0.80" },
            { label: "Mode Routing Acc.", target: "≥ 0.80" },
            { label: "Satellite F1", target: "≥ 0.70" },
            { label: "Calibration ECE", target: "≤ 0.25" },
          ].map(({ label, target }) => (
            <div key={label} className="flex justify-between items-center p-2 rounded-lg" style={{ background: "#090e1c" }}>
              <span className="text-on-surface-variant">{label}</span>
              <span className="mono font-semibold text-primary">{target}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
