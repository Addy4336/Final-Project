"use client";

import { Compass, FileImage, ScanText } from "lucide-react";
import type { VqaResponse } from "@/lib/api";

interface Props { result: VqaResponse; }

function HighlightedText({ text, highlights }: { text: string; highlights: string[] }) {
  if (!text) return null;
  const validHighlights = highlights.filter((h) => h && h.trim().length > 2);
  if (!validHighlights.length) return <>{text}</>;

  // Escape special characters and join with OR
  const escaped = validHighlights.map((h) => h.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`(${escaped.join("|")})`, "gi");

  const parts = text.split(regex);
  return (
    <>
      {parts.map((part, i) => {
        const isMatch = validHighlights.some((h) => h.toLowerCase() === part.toLowerCase());
        return isMatch ? (
          <mark key={i} className="bg-tertiary/20 text-tertiary rounded px-1 -mx-1 font-bold">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        );
      })}
    </>
  );
}

export function EvidencePanel({ result }: Props) {
  const hasOcr = !!(result.ocr_metadata?.fields && Object.keys(result.ocr_metadata.fields).length > 0) || !!result.ocr_extracted_text;
  const hasDetection = !!result.detection_image;
  const hasSatellite = !!(result.satellite?.features?.length);

  if (!hasOcr && !hasDetection && !hasSatellite) {
    return (
      <div className="glass-card p-10 text-center">
        <ScanText size={32} className="mx-auto text-outline opacity-40 mb-3" />
        <p className="text-sm text-on-surface-variant">No evidence data available for this result.</p>
        <p className="text-xs text-outline mt-1">OCR and visual grounding appear when applicable.</p>
      </div>
    );
  }

  // Get field values and the main answer to highlight
  const highlights = [];
  if (result.ocr_metadata?.fields) {
    highlights.push(...Object.values(result.ocr_metadata.fields));
  }
  if (result.answer && !result.answer.includes("No readable text")) {
    highlights.push(result.answer);
  }

  return (
    <div className="space-y-4 animate-fade-up">
      {/* OCR fields */}
      {result.ocr_metadata?.fields && Object.keys(result.ocr_metadata.fields).length > 0 && (
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <ScanText size={16} className="text-tertiary" />
            <h3 className="font-semibold text-on-surface">OCR Extracted Fields</h3>
            {result.ocr_metadata.quality && (
              <span className="ml-auto label-meta px-2 py-0.5 rounded-full"
                style={{ background: "rgba(47,217,244,0.1)", color: "#2fd9f4", fontSize: "0.6rem" }}>
                Quality: {result.ocr_metadata.quality}
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {Object.entries(result.ocr_metadata.fields)
              .filter(([, v]) => Boolean(v))
              .map(([key, value]) => (
                <div key={key} className="p-3 rounded-xl" style={{ background: "#090e1c" }}>
                  <p className="label-meta mb-1" style={{ fontSize: "0.6rem" }}>{key}</p>
                  <p className="text-sm font-medium text-on-surface">{value}</p>
                </div>
              ))}
          </div>
          {result.ocr_metadata.char_count != null && (
            <div className="flex gap-4 mt-3 pt-3 border-t border-outline-variant/15">
              <span className="mono text-xs text-on-surface-variant">{result.ocr_metadata.char_count} chars</span>
              <span className="mono text-xs text-on-surface-variant">{result.ocr_metadata.line_count} lines</span>
            </div>
          )}
        </div>
      )}

      {/* OCR raw text */}
      {result.ocr_extracted_text && (
        <div className="glass-card p-5">
          <p className="label-meta mb-3">Raw OCR Text (Grounded Spans Highlighted)</p>
          <pre className="text-xs text-on-surface leading-relaxed overflow-auto max-h-48 font-mono p-3 rounded-lg whitespace-pre-wrap" style={{ background: "#090e1c" }}>
            <HighlightedText text={result.ocr_extracted_text} highlights={highlights} />
          </pre>
        </div>
      )}

      {/* Visual grounding / YOLO */}
      {hasDetection && (
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <FileImage size={16} className="text-secondary" />
            <h3 className="font-semibold text-on-surface">
              {result.detection_meta?.mode === "satellite_grounding" ? "Satellite Grounding Overlay" : "Object Detection"}
            </h3>
          </div>
          <img
            src={`data:image/png;base64,${result.detection_image}`}
            alt="Detection result"
            className="w-full rounded-xl border border-outline-variant/15"
          />
        </div>
      )}

      {/* Satellite features */}
      {hasSatellite && (
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Compass size={16} className="text-tertiary" />
            <h3 className="font-semibold text-on-surface">Satellite Feature Analysis</h3>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {result.satellite!.features!.map((f, i) => (
              <div key={i} className="p-3 rounded-xl space-y-2" style={{ background: "#090e1c" }}>
                <p className="text-sm font-semibold text-on-surface">{f.label || f.type || "Feature"}</p>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-on-surface-variant">
                    <span>Coverage</span>
                    <span className="mono">{Number(f.coverage_pct || 0).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-surface-highest overflow-hidden">
                    <div className="h-full rounded-full bg-gradient-to-r from-tertiary-container to-tertiary"
                      style={{ width: `${Math.min(100, f.coverage_pct || 0)}%` }} />
                  </div>
                  <div className="flex justify-between text-xs text-on-surface-variant">
                    <span>Confidence</span>
                    <span className="mono">{Number(f.confidence || 0).toFixed(1)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
