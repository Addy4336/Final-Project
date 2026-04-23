"use client";

import { ClipboardList, Trash2 } from "lucide-react";
import type { QueryHistoryItem } from "./VisionQueryDashboard";

interface Props {
  history: QueryHistoryItem[];
  selectedId: string;
  onSelect: (id: string) => void;
  onClear: () => void;
}

const MODE_DOT: Record<string, string> = {
  AUTO: "#c0c1ff", OCR: "#2fd9f4", SATELLITE: "#4ade80", VQA: "#d0bcff",
};

function confColor(v: number) {
  if (v >= 75) return "#4ade80";
  if (v >= 45) return "#fbbf24";
  return "#f87171";
}

export function HistoryPanel({ history, selectedId, onSelect, onClear }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ClipboardList size={16} className="text-primary" />
          <h3 className="font-semibold text-on-surface text-sm">Query History</h3>
        </div>
        {history.length > 0 && (
          <button
            className="flex items-center gap-1 text-xs text-outline hover:text-error transition-colors"
            onClick={onClear}
          >
            <Trash2 size={12} />
            Clear
          </button>
        )}
      </div>

      {!history.length ? (
        <div className="text-center py-10 px-3">
          <ClipboardList size={28} className="mx-auto text-outline opacity-30 mb-2" />
          <p className="text-xs text-on-surface-variant">Analysis history appears here after the first query.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {history.map((item) => {
            const isActive = item.id === selectedId;
            const conf = item.result.confidence || item.result.hybrid?.confidence || 0;
            const modeDot = MODE_DOT[item.mode] || "#908fa0";
            return (
              <button
                key={item.id}
                onClick={() => onSelect(item.id)}
                className={`w-full text-left rounded-xl p-3 transition-all duration-150 border ${
                  isActive
                    ? "border-primary/30 bg-surface-high"
                    : "border-transparent hover:border-outline-variant/20 hover:bg-surface-high/50"
                }`}
                style={isActive ? { boxShadow: "0 0 12px rgba(192,193,255,0.08)" } : {}}
              >
                <p className="text-xs font-semibold text-on-surface line-clamp-2 leading-relaxed">{item.question}</p>
                <div className="flex items-center justify-between mt-2">
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: modeDot }} />
                    <span className="mono text-xs text-on-surface-variant uppercase tracking-wide" style={{ fontSize: "0.6rem" }}>
                      {item.result.model || item.mode}
                    </span>
                  </div>
                  <span className="mono text-xs font-semibold" style={{ color: confColor(conf) }}>
                    {conf.toFixed(1)}%
                  </span>
                </div>
                <p className="text-xs text-outline mt-1" style={{ fontSize: "0.6rem" }}>
                  {new Date(item.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </p>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
