"use client";

import { useEffect, useState, useRef } from "react";
import { classifyImage, runVqa, type UiMode, type VqaResponse } from "@/lib/api";
import { UploadCloud, Sparkles, Map, ScanText, Activity, ChevronRight, RefreshCcw, CheckCircle2 } from "lucide-react";

type Step = "UPLOAD" | "CONFIGURE" | "PROCESSING" | "RESULTS";

const UI_MODES: { value: UiMode; label: string; icon: any; desc: string }[] = [
  { value: "OCR", label: "Text Extraction", icon: ScanText, desc: "Extract structured data & text" },
  { value: "VQA", label: "Visual Reasoning", icon: Activity, desc: "Ask questions about the image" },
  { value: "SATELLITE", label: "Satellite Analysis", icon: Map, desc: "Geospatial feature detection" },
];

export function VisionQueryDashboard() {
  const [step, setStep] = useState<Step>("UPLOAD");
  
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [mode, setMode] = useState<UiMode>("OCR");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<VqaResponse | null>(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!imageFile) { setImageUrl(""); return; }
    const url = URL.createObjectURL(imageFile);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  const handleUpload = async (file: File) => {
    setImageFile(file);
    setStep("PROCESSING"); 
    try {
      const cls = await classifyImage(file);
      const raw = String(cls.mode || "").trim().toLowerCase();
      
      if (raw === "doc" || raw === "document" || raw === "ocr") setMode("OCR");
      else if (raw === "satellite" || raw === "sat") setMode("SATELLITE");
      else setMode("VQA");

      setStep("CONFIGURE");
    } catch {
      setMode("VQA");
      setStep("CONFIGURE");
    }
  };

  const handleExecute = async () => {
    if (!imageFile || !question.trim()) return;
    setStep("PROCESSING");
    setError("");
    
    try {
      const resp = await runVqa({
        image: imageFile,
        question: question.trim(),
        mode,
        lang: "en-US",
        brightness: 100,
        contrast: 100,
      });
      setResult(resp);
      setStep("RESULTS");
    } catch (err: any) {
      setError(err.message || "An error occurred");
      setStep("CONFIGURE");
    }
  };

  const reset = () => {
    setImageFile(null);
    setResult(null);
    setQuestion("");
    setError("");
    setStep("UPLOAD");
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-white/20 flex flex-col">
      
      {/* Top Header Navigation */}
      <header className="w-full px-8 py-6 flex items-center justify-between border-b border-white/10 bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Sparkles size={20} className="text-white opacity-80" />
          <span className="font-display italic text-2xl tracking-wide">VisionQuery</span>
        </div>
        <div className="hidden md:flex items-center gap-8">
          {["Upload", "Configure", "Results"].map((label, i) => {
            const stepNum = i + 1;
            const currentNum = step === "UPLOAD" ? 1 : step === "CONFIGURE" || step === "PROCESSING" && !result ? 2 : 3;
            const isActive = currentNum === stepNum;
            const isPast = currentNum > stepNum;

            return (
              <div key={label} className="flex items-center gap-3">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-500 ${
                  isActive ? "bg-white text-black shadow-[0_0_15px_rgba(255,255,255,0.4)]" :
                  isPast ? "bg-neutral-800 text-white" : "border border-neutral-700 text-neutral-500"
                }`}>
                  {isPast ? <CheckCircle2 size={14} /> : stepNum}
                </div>
                <span className={`text-sm font-medium tracking-wide ${isActive ? "text-white" : "text-neutral-500"}`}>
                  {label}
                </span>
                {i < 2 && <div className="w-8 h-px bg-neutral-800 ml-5" />}
              </div>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-6 py-12 md:py-20 flex flex-col items-center">
        
        {/* ── STEP 1: UPLOAD ── */}
        {step === "UPLOAD" && (
          <div className="w-full max-w-3xl text-center space-y-12 animate-fade-in-up">
            <div className="space-y-6">
              <h1 className="font-display italic text-5xl md:text-7xl tracking-tight leading-tight">
                Initialize Research
              </h1>
              <p className="text-neutral-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
                Provide a visual data source to commence multimodal synthesis.
              </p>
            </div>

            <div 
              className="group relative flex flex-col items-center justify-center p-20 border-dashed border-2 border-neutral-800 hover:border-white/40 bg-neutral-900/30 hover:bg-neutral-900/60 transition-all duration-300 cursor-pointer overflow-hidden rounded-3xl"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const file = e.dataTransfer.files?.[0];
                if (file) handleUpload(file);
              }}
            >
              <div className="relative z-10 flex flex-col items-center gap-6">
                <div className="w-24 h-24 rounded-full bg-neutral-800 flex items-center justify-center shadow-2xl group-hover:scale-110 group-hover:bg-white group-hover:text-black transition-all duration-500">
                  <UploadCloud size={40} className="opacity-90" />
                </div>
                <div className="space-y-2">
                  <p className="text-2xl font-medium tracking-wide">Click or drag image to upload</p>
                  <p className="text-neutral-500">Supports high-res JPG, PNG, WEBP</p>
                </div>
              </div>
              <input 
                ref={fileInputRef} 
                type="file" 
                className="hidden" 
                accept="image/*" 
                onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])} 
              />
            </div>
          </div>
        )}

        {/* ── STEP 2: CONFIGURE ── */}
        {step === "CONFIGURE" && (
          <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 animate-fade-in-up">
            
            {/* Left: Image Preview */}
            <div className="bg-neutral-900/50 border border-white/10 p-4 flex items-center justify-center min-h-[400px] lg:h-[600px] relative overflow-hidden rounded-3xl group shadow-2xl">
              <img src={imageUrl} alt="Target" className="max-w-full max-h-full object-contain rounded-2xl opacity-90 transition-opacity group-hover:opacity-100" />
              <button 
                onClick={reset}
                className="absolute top-6 right-6 bg-black/60 backdrop-blur-md border border-white/20 px-4 py-2 rounded-full text-xs font-medium hover:bg-white hover:text-black transition-all shadow-lg"
              >
                Change Image
              </button>
            </div>

            {/* Right: Configuration Form */}
            <div className="flex flex-col justify-center space-y-10">
              <div>
                <h2 className="text-4xl md:text-5xl font-display italic mb-4">Analysis Parameters</h2>
                <p className="text-neutral-400 text-lg">Configure the multimodal engine for this source.</p>
              </div>

              <div className="space-y-4">
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  <label className="text-xs font-mono uppercase tracking-widest text-[#888]">Auto-Detected Engine</label>
                  <div style={{ display: "flex", alignItems: "center", gap: "16px", padding: "16px", borderRadius: "12px", border: "1px solid #333", backgroundColor: "#111" }}>
                    {mode === "OCR" && <ScanText size={24} className="text-[#58a6ff]" />}
                    {mode === "VQA" && <Activity size={24} className="text-[#d2a8ff]" />}
                    {mode === "SATELLITE" && <Map size={24} className="text-[#3fb950]" />}
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <p className="font-medium text-white text-lg tracking-wide" style={{ margin: 0 }}>
                        {mode === "OCR" ? "Text Extraction Engine" : 
                         mode === "SATELLITE" ? "Satellite Analysis Engine" : "Visual Reasoning Engine"}
                      </p>
                      <p className="text-[#888] text-sm" style={{ margin: 0 }}>Optimized for this data source</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                  <label className="text-xs font-mono uppercase tracking-widest text-[#888]">Define Query</label>
                  
                  {/* Suggestions */}
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                    {(mode === "OCR" ? [
                      "Extract all named entities", "What is the document type?", "Read all text"
                    ] : mode === "SATELLITE" ? [
                      "Describe land coverage", "Are roads visible?", "Estimate vegetation density"
                    ] : [
                      "What is happening here?", "Describe the main object", "What are the colors?"
                    ]).map((sug) => (
                      <button
                        key={sug}
                        onClick={() => setQuestion(sug)}
                        style={{ 
                          padding: "6px 12px", 
                          fontSize: "12px", 
                          backgroundColor: "#222", 
                          border: "1px solid #444", 
                          borderRadius: "100px",
                          color: "#ccc",
                          cursor: "pointer"
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = "#333"}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = "#222"}
                      >
                        {sug}
                      </button>
                    ))}
                  </div>

                  <input
                    type="text"
                    placeholder="e.g., Extract all named entities..."
                    className="w-full bg-[#111] border border-[#444] rounded-xl px-6 py-5 text-xl text-white focus:outline-none focus:border-white transition-all shadow-inner placeholder:text-[#666]"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleExecute()}
                  />
                </div>
              </div>

              {error && (
                <div className="p-5 rounded-xl bg-[#310000] border border-[#ff4444] text-[#ff8888] text-sm">
                  {error}
                </div>
              )}

              <button
                onClick={handleExecute}
                disabled={!question.trim()}
                className="w-full bg-white text-black font-semibold text-xl py-5 rounded-xl hover:bg-[#e5e5e5] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
              >
                Commence Synthesis
                <ChevronRight size={24} />
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 3: PROCESSING ── */}
        {step === "PROCESSING" && (
          <div className="w-full max-w-4xl bg-neutral-900/40 border border-white/10 rounded-3xl p-16 flex flex-col items-center justify-center text-center animate-fade-in-up min-h-[600px] relative overflow-hidden shadow-2xl">
            {imageUrl && (
              <div className="absolute inset-0 opacity-10 blur-3xl pointer-events-none">
                <img src={imageUrl} alt="" className="w-full h-full object-cover" />
              </div>
            )}
            
            <div className="relative z-10 flex flex-col items-center">
              <div className="relative w-32 h-32 mb-10">
                <div className="absolute inset-0 border-2 border-white/10 rounded-full" />
                <div className="absolute inset-0 border-t-2 border-white rounded-full animate-spin" style={{ animationDuration: '1.5s' }} />
                <div className="absolute inset-0 flex items-center justify-center animate-pulse">
                  <Activity size={40} className="text-white" />
                </div>
              </div>
              <h2 className="text-4xl font-display italic mb-6">Synthesizing Data</h2>
              <div className="font-mono text-base text-neutral-400 flex items-center gap-4">
                <span className="w-3 h-3 bg-white rounded-full animate-ping" />
                Running {mode} engine models...
              </div>
            </div>

            <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-white to-transparent opacity-50 animate-[scanline_2.5s_ease-in-out_infinite]" />
          </div>
        )}

        {/* ── STEP 4: RESULTS ── */}
        {step === "RESULTS" && result && (
          <div style={{ width: "100%", animation: "fade-in-up 0.5s ease-out", display: "flex", flexDirection: "column", gap: "32px" }}>
            
            {/* Header */}
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "24px" }}>
              <div>
                <h2 className="font-display italic text-white" style={{ fontSize: "36px", margin: 0 }}>Synthesis Complete</h2>
                <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "8px" }}>
                  <span style={{ width: "10px", height: "10px", backgroundColor: "#22c55e", borderRadius: "50%", boxShadow: "0 0 10px rgba(34,197,94,0.5)" }} />
                  <p style={{ margin: 0, color: "#a3a3a3", fontSize: "16px" }}>Processed via {result.model || mode} in {result.time_ms || 0}ms</p>
                </div>
              </div>
              <button 
                onClick={reset}
                style={{ display: "flex", alignItems: "center", gap: "8px", backgroundColor: "#171717", border: "1px solid #404040", padding: "12px 24px", borderRadius: "12px", color: "white", fontSize: "14px", fontWeight: "bold", cursor: "pointer", transition: "all 0.2s" }}
                onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "white"; e.currentTarget.style.color = "black"; }}
                onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = "#171717"; e.currentTarget.style.color = "white"; }}
              >
                <RefreshCcw size={18} />
                New Analysis
              </button>
            </div>

            {/* Main Content Grid */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "32px", alignItems: "flex-start" }}>
              
              {/* Left: Image Viewer (40%) */}
              <div style={{ flex: "1 1 400px", position: "relative", backgroundColor: "#111", border: "1px solid #333", borderRadius: "24px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", padding: "16px", minHeight: "400px" }}>
                 <img 
                  src={result.detection_image ? `data:image/png;base64,${result.detection_image}` : imageUrl} 
                  alt="Result" 
                  style={{ maxWidth: "100%", maxHeight: "700px", objectFit: "contain", borderRadius: "16px" }}
                 />
                 {result.detection_image && (
                   <div style={{ position: "absolute", bottom: "24px", left: "24px", backgroundColor: "rgba(0,0,0,0.8)", backdropFilter: "blur(8px)", border: "1px solid rgba(255,255,255,0.2)", padding: "8px 16px", borderRadius: "100px", fontFamily: "monospace", fontSize: "10px", textTransform: "uppercase", letterSpacing: "2px", color: "white", boxShadow: "0 10px 20px rgba(0,0,0,0.5)" }}>
                     YOLO Overlays Active
                   </div>
                 )}
              </div>

              {/* Right: Data Output (60%) */}
              <div style={{ flex: "1.5 1 500px", display: "flex", flexDirection: "column", gap: "32px" }}>
                
                {/* Answer Card */}
                <div style={{ position: "relative", backgroundColor: "#1a1a1a", border: "1px solid #333", padding: "40px", borderRadius: "24px", overflow: "hidden", boxShadow: "0 20px 40px rgba(0,0,0,0.5)" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, width: "6px", height: "100%", background: "linear-gradient(to bottom, #fff, #555)" }} />
                  <p style={{ fontFamily: "monospace", fontSize: "12px", textTransform: "uppercase", letterSpacing: "2px", color: "#888", marginBottom: "20px", marginTop: 0 }}>Primary Conclusion</p>
                  
                  <p className="font-display italic" style={{ fontSize: "42px", lineHeight: "1.2", color: "white", margin: "0 0 24px 0" }}>
                    {result.hybrid?.answer || result.answer || "No conclusion drawn."}
                  </p>
                  
                  {result.hybrid?.explanation && (
                    <p style={{ color: "#a3a3a3", fontSize: "18px", lineHeight: "1.6", borderLeft: "2px solid #333", paddingLeft: "16px", margin: "0 0 32px 0" }}>
                      {result.hybrid.explanation}
                    </p>
                  )}
                  
                  <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                    <div style={{ backgroundColor: "rgba(0,0,0,0.5)", padding: "8px 16px", borderRadius: "12px", border: "1px solid #333", display: "flex", alignItems: "center", gap: "12px" }}>
                      <span style={{ fontSize: "12px", color: "#888", textTransform: "uppercase", letterSpacing: "1px", fontFamily: "monospace" }}>Confidence</span>
                      <span style={{ fontFamily: "monospace", fontWeight: "bold", color: "#4ade80", fontSize: "18px", textShadow: "0 0 10px rgba(74,222,128,0.3)" }}>
                        {((result.confidence || result.hybrid?.confidence || 0)).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Detailed Data Container */}
                <div style={{ backgroundColor: "#111", border: "1px solid #333", borderRadius: "24px", padding: "32px" }}>
                  
                  {/* OCR Fields */}
                  {result.ocr_metadata?.fields && Object.keys(result.ocr_metadata.fields).length > 0 && (
                    <div style={{ marginBottom: "40px" }}>
                      <p style={{ fontSize: "12px", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "2px", color: "#888", borderBottom: "1px solid #333", paddingBottom: "12px", marginBottom: "24px" }}>Structured Data</p>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "20px" }}>
                        {Object.entries(result.ocr_metadata.fields).filter(([,v]) => Boolean(v)).map(([k, v]) => (
                          <div key={k} style={{ backgroundColor: "#1a1a1a", border: "1px solid #333", padding: "20px", borderRadius: "16px" }}>
                            <p style={{ fontSize: "10px", fontFamily: "monospace", color: "#888", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "8px", marginTop: 0 }}>{k.replace(/_/g, " ")}</p>
                            <p style={{ fontWeight: 500, fontSize: "16px", color: "white", margin: 0, wordBreak: "break-word" }}>{v}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Satellite Features */}
                  {result.satellite?.features && result.satellite.features.length > 0 && (
                    <div style={{ marginBottom: "40px" }}>
                      <p style={{ fontSize: "12px", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "2px", color: "#888", borderBottom: "1px solid #333", paddingBottom: "12px", marginBottom: "24px" }}>Geospatial Features</p>
                      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                        {result.satellite.features.map((f, i) => (
                          <div key={i} style={{ backgroundColor: "#1a1a1a", border: "1px solid #333", padding: "24px", borderRadius: "16px" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                              <span style={{ fontWeight: 500, fontSize: "18px", color: "white" }}>{f.label || f.type || "Feature"}</span>
                              <span style={{ fontFamily: "monospace", fontSize: "18px", color: "white", fontWeight: "bold" }}>{Number(f.coverage_pct || 0).toFixed(1)}%</span>
                            </div>
                            <div style={{ height: "12px", backgroundColor: "#111", borderRadius: "100px", overflow: "hidden", border: "1px solid #333" }}>
                              <div style={{ height: "100%", backgroundColor: "white", borderRadius: "100px", width: `${Math.min(100, f.coverage_pct || 0)}%`, boxShadow: "0 0 10px rgba(255,255,255,0.5)", transition: "width 1s ease-out" }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Raw Dump */}
                  {result.ocr_extracted_text && (
                    <div>
                      <p style={{ fontSize: "12px", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "2px", color: "#888", borderBottom: "1px solid #333", paddingBottom: "12px", marginBottom: "24px" }}>Raw Text Dump</p>
                      <div style={{ backgroundColor: "#000", border: "1px solid #333", padding: "24px", borderRadius: "16px", fontFamily: "monospace", fontSize: "14px", color: "#a3a3a3", whiteSpace: "pre-wrap", lineHeight: "1.6", boxShadow: "inset 0 0 20px rgba(0,0,0,0.5)" }}>
                        {result.ocr_extracted_text}
                      </div>
                    </div>
                  )}

                </div>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
