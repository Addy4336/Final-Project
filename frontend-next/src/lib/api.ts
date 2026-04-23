export type UiMode = "AUTO" | "OCR" | "SATELLITE" | "VQA";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export interface ClassifyResponse {
  mode: "doc" | "satellite" | "vqa" | string;
  confidence: number;
  scores?: Record<string, number>;
  method?: string;
}

export interface VqaAnswer {
  answer: string;
  confidence: number;
  confidence_level?: string;
}

export interface VqaResponse {
  question?: string;
  answer?: string;
  answer_type?: string;
  model?: string;
  model_name?: string;
  confidence?: number;
  time_ms?: number;
  latency_ms?: number;
  auto_detected_mode?: string | null;
  suggested_queries?: string[];
  hybrid?: {
    answer?: string;
    confidence?: number;
    explanation?: string;
    type?: string;
  };
  base_answers?: VqaAnswer[];
  ft_answers?: VqaAnswer[];
  answers?: VqaAnswer[];
  ocr_extracted_text?: string;
  ocr_metadata?: {
    quality?: string;
    char_count?: number;
    line_count?: number;
    fields?: Record<string, string>;
    diagnostics?: Record<string, unknown>;
  } | null;
  detection_image?: string | null;
  detection_meta?: {
    ran?: boolean;
    mode?: string;
  } | null;
  satellite?: {
    land_use?: unknown;
    density?: unknown;
    features?: Array<{
      type?: string;
      label?: string;
      coverage_pct?: number;
      confidence?: number;
    }>;
  } | null;
  error?: boolean;
  errorMessage?: string;
}

export interface ModelStat {
  name: string;
  type: string;
  accuracy_overall: number;
  note: string;
  color: string;
}

export interface StatsResponse {
  models: ModelStat[];
  improvement_vqa: string;
  improvement_ocr: string;
  last_updated: string;
}

export interface OcrHealthResponse {
  easyocr_loaded: boolean;
  easyocr_attempted: boolean;
  tesseract_enabled: boolean;
  tesseract_cmd: string;
  error?: string;
}

export async function fetchHealth() {
  const res = await fetch(apiUrl("/api/health"), { cache: "no-store" });
  if (!res.ok) throw new Error("Backend health check failed");
  return res.json() as Promise<{ status: string; version: string }>;
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(apiUrl("/api/stats"), { cache: "no-store" });
  if (!res.ok) throw new Error("Stats fetch failed");
  return res.json() as Promise<StatsResponse>;
}

export async function fetchOcrHealth(): Promise<OcrHealthResponse> {
  const res = await fetch(apiUrl("/api/ocr/health"), { cache: "no-store" });
  if (!res.ok) throw new Error("OCR health fetch failed");
  return res.json() as Promise<OcrHealthResponse>;
}

export interface PublicationGateItem {
  value: number;
  target: number;
  pass: boolean;
}

export interface BenchmarkReport {
  timestamp: string;
  profile: string;
  runtime_seconds: number;
  ocr: {
    count: number;
    mean_cer: number;
    mean_wer: number;
    field_f1: number;
    field_precision: number;
    field_recall: number;
    qa_accuracy: number;
    mean_latency_ms: number;
  };
  mode_routing: { count: number; accuracy: number };
  satellite: { count: number; feature_f1: number; feature_precision: number; feature_recall: number };
  vqa: { count: number; ft_exact_match: number; base_exact_match: number };
  doc_qa_ablation: { count: number; ocr_only_em: number; vqa_only_em: number; hybrid_em: number };
  calibration: { ece: number; brier: number; samples: number };
  publication_gate: Record<string, PublicationGateItem>;
  ready_for_publication: boolean;
}

export interface BenchmarkResponse {
  available: boolean;
  message?: string;
  report?: BenchmarkReport;
  report_md?: string;
}

export async function fetchBenchmark(): Promise<BenchmarkResponse> {
  const res = await fetch(apiUrl("/api/benchmark"), { cache: "no-store" });
  if (!res.ok) throw new Error("Benchmark fetch failed");
  return res.json() as Promise<BenchmarkResponse>;
}


export async function classifyImage(file: File): Promise<ClassifyResponse> {
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch(apiUrl("/api/classify"), { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Image classification failed (${res.status})`);
  return res.json() as Promise<ClassifyResponse>;
}

export interface RunVqaInput {
  image: File;
  question: string;
  mode: UiMode;
  lang: string;
  brightness: number;
  contrast: number;
}

export async function runVqa(input: RunVqaInput): Promise<VqaResponse> {
  const fd = new FormData();
  fd.append("image", input.image);
  fd.append("question", input.question);
  fd.append("mode", input.mode);
  fd.append("lang", input.lang);
  fd.append("brightness", String(input.brightness));
  fd.append("contrast", String(input.contrast));
  fd.append("detect", "true");

  const res = await fetch(apiUrl("/api/vqa"), { method: "POST", body: fd });

  let data: VqaResponse;
  try {
    data = (await res.json()) as VqaResponse;
  } catch {
    throw new Error(`VQA call failed with status ${res.status}`);
  }

  if (!res.ok) {
    throw new Error(data.errorMessage || `VQA call failed with status ${res.status}`);
  }

  return data;
}
