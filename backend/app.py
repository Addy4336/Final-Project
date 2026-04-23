"""VisionQuery backend with strict pipeline separation and output control."""
import base64
import io
import json
import logging
import re
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance

from backend.config import CORS_ORIGINS, FRONTEND_DIR, MODEL_CONFIGS, VQA_STATS
from backend.models.classifier import classify_image, classify_with_heuristics
from backend.models.ocr import run_advanced_ocr, resolve_ocr_answer
from backend.models.satellite import is_satellite_image, run_satellite_analysis
from backend.models.vqa import FALLBACK_TEXT, get_model, simulate_base_prediction

import backend.models.detect as detect_module
from backend.models.detect import detect_and_draw

if not detect_module._MODEL_PATH.exists():
    detect_module._MODEL_PATH = Path("models/yolov8s.pt")
    if not detect_module._MODEL_PATH.exists():
        detect_module._MODEL_PATH = Path("yolov8s.pt")


app = FastAPI(title="VisionQuery API", version="4.8.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Structured request logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("visionquery")


from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = int((time.perf_counter() - t0) * 1000)
    if request.url.path.startswith("/api"):
        _log.info("%s %s → %d (%dms)", request.method, request.url.path, response.status_code, elapsed)
    return response


def _selected_model_type(model_type: str) -> str:
    return model_type if model_type in MODEL_CONFIGS else "finetuned"


def _normalize_mode(mode: str) -> str:
    value = str(mode or "AUTO").strip().upper()
    aliases = {
        "DOC": "OCR",
        "DOCUMENT": "OCR",
        "OCR": "OCR",
        "SAT": "SATELLITE",
        "SATELLITE": "SATELLITE",
        "VQA": "VQA",
        "AUTO": "AUTO",
    }
    return aliases.get(value, "VQA")


def _answer(answer: str, confidence: float, level: str | None = None) -> dict:
    confidence = round(float(confidence or 0), 1)
    if level is None:
        level = "High" if confidence >= 70 else "Medium" if confidence >= 30 else "Low"
    return {
        "answer": answer or FALLBACK_TEXT,
        "confidence": confidence,
        "confidence_level": level,
    }


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        raw = upload.file.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, detail="Cannot decode image.")


def _apply_region(pil_image: Image.Image, region_x: float, region_y: float,
                  region_w: float, region_h: float) -> Image.Image:
    if region_w >= 1.0 and region_h >= 1.0:
        return pil_image
    width, height = pil_image.size
    x1 = max(0, int(region_x * width))
    y1 = max(0, int(region_y * height))
    x2 = min(width, int((region_x + region_w) * width))
    y2 = min(height, int((region_y + region_h) * height))
    if x2 <= x1 or y2 <= y1:
        return pil_image
    return pil_image.crop((x1, y1, x2, y2))




def _auto_route(pil_image, question):
    del question

    # Classifier-first routing is faster and more stable than running OCR on every image.
    try:
        cls = classify_with_heuristics(pil_image.copy())
        mode = str(cls.get("mode", "")).upper()
        conf = float(cls.get("confidence", 0.0) or 0.0)
        if mode == "DOC" and conf >= 35.0:
            return "OCR", "DOC"
        if mode == "SAT" and conf >= 40.0:
            return "SATELLITE", "SAT"
        if mode == "VQA" and conf >= 40.0:
            return "VQA", "VQA"
    except Exception:
        pass

    try:
        ocr_res = run_advanced_ocr(pil_image.copy())
        text = ocr_res.get("text", "")
        fields = ocr_res.get("fields", {})

        if len(text) > 10 or len(fields) >= 1:
            return "OCR", "DOC"
    except:
        pass

    return "VQA", "VQA"


def _run_ocr_pipeline(pil_image: Image.Image, question: str, pre_run_ocr: dict = None) -> dict:
    try:
        ocr_res = pre_run_ocr if pre_run_ocr else run_advanced_ocr(pil_image.copy(), question=question)
        ocr_text   = ocr_res.get("text", "").strip()
        ocr_fields = ocr_res.get("fields", {})
        ocr_answer = ocr_res.get("answer", "").strip()
        # Ensure answer is aligned to the current user question even when OCR was precomputed.
        if question:
            resolved_answer = resolve_ocr_answer(question, ocr_text, ocr_fields).strip()
            if resolved_answer:
                ocr_answer = resolved_answer
        ocr_quality = ocr_res.get("quality", "None")
        print(f"[OCR branch] chars={len(ocr_text)}, fields={list(ocr_fields.keys())}, answer='{ocr_answer[:60]}'")

        # Use the pre-computed answer from resolve_ocr_answer()
        if ocr_answer and ocr_answer != "No readable text detected. Try clicking Optimize OCR and resubmitting.":
            final_answer = ocr_answer
            reasoning    = "OCR field extraction successful."
        elif ocr_text:
            final_answer = ocr_text[:400]
            reasoning    = "OCR text extracted from document."
        else:
            final_answer = "No readable text found. Please click 'Optimize OCR' in the left panel and try again."
            reasoning    = "OCR pipeline found no text content."

    except Exception as e:
        import traceback; traceback.print_exc()
        ocr_text    = ""
        ocr_fields  = {}
        ocr_answer  = ""
        ocr_quality = "None"
        ocr_res     = {"text": "", "fields": {}, "quality": "None"}
        final_answer = "Unable to process document"
        reasoning    = f"OCR Error: {str(e)}"

    # Format for the 3-panel UI
    raw_text_display = ocr_text[:500] if ocr_text else "No text extracted."

    # Field keys now use spaces: "emp code", "blood group" — no underscores
    DISPLAY_FIELDS = ["name", "emp code", "dob", "blood group", "address"]
    ft_text = ""
    if ocr_fields:
        ft_text = " | ".join(
            f"{k.title()}: {v}"
            for k, v in ocr_fields.items()
            if k in DISPLAY_FIELDS and v
        )
    if not ft_text:
        ft_text = "No structured fields detected."

    # OCR confidence based on quality tier
    fields_count = len(ocr_fields)
    if ocr_quality == "High" and fields_count >= 2:
        final_conf = 93.0
    elif ocr_quality in ("High", "Medium") and fields_count >= 1:
        final_conf = 87.0
    elif ocr_text and len(ocr_text) > 20:
        final_conf = 75.0
    else:
        final_conf = 42.5

    base_answers = [{
        "answer": raw_text_display,
        "confidence": min(final_conf, 50.0),
        "confidence_level": "Medium",
        "model": "ocr_raw"
    }]
    ft_answers = [{
        "answer": ft_text if ft_text else raw_text_display,
        "confidence": final_conf,
        "confidence_level": "High" if final_conf >= 70 else "Medium",
        "model": "ocr_fields"
    }]

    return {
        "answer":      final_answer,
        "confidence":  final_conf,
        "answer_type": "DOC",
        "reasoning":   reasoning,
        "answers":     ft_answers,
        "base_answers": base_answers,
        "ft_answers":  ft_answers,
        "ocr_text":    ocr_text,
        "ocr_fields":  ocr_fields,
        "ocr_res":     ocr_res,
        "sat_data":    None,
        "latency_ms":  0,
    }


def _run_satellite_pipeline(
    pil_image: Image.Image,
    question: str,
    region: dict | None,
) -> dict:
    sat_data = run_satellite_analysis(pil_image.copy(), question=question, region=region, ocr_text="")
    final_answer = sat_data.get("answer") or FALLBACK_TEXT
    confidence = sat_data.get("confidence", 0.0)
    answer_obj = _answer(final_answer, confidence)
    return {
        "answer": final_answer,
        "confidence": confidence,
        "answer_type": "Satellite",
        "reasoning": "Satellite-only feature detection. No OCR or VQA outputs mixed.",
        "answers": [answer_obj],
        "base_answers": [answer_obj],
        "ft_answers": [answer_obj],
        "ocr_text": "",
        "ocr_fields": {},
        "ocr_res": {"text": "", "fields": {}, "quality": "None"},
        "sat_data": sat_data,
        "latency_ms": 0,
    }


def _run_vqa_pipeline(
    pil_image: Image.Image,
    question: str,
    model_type: str,
    lang: str,
) -> dict:
    selected = _selected_model_type(model_type)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_base = executor.submit(simulate_base_prediction, pil_image.copy(), question, top_k=5, lang=lang)
        
        if selected == "base":
            base_prediction = future_base.result()
            ft_answers = []
            selected_answers = base_prediction.get("answers", [])
            selected_prediction = base_prediction
            reasoning = "Base VQA baseline only. Fine-tuned model was not run."
        else:
            future_ft = executor.submit(get_model("finetuned").predict, pil_image.copy(), question, top_k=5, lang=lang)
            base_prediction = future_base.result()
            selected_prediction = future_ft.result()
            ft_answers = selected_prediction.get("answers", [])
            selected_answers = ft_answers
            reasoning = "Fine-tuned VQA only. Base panel uses deterministic weak baseline."

    base_answers = base_prediction.get("answers", [])

    if not selected_answers:
        selected_answers = [_answer(FALLBACK_TEXT, 0.0, "Low")]

    top_answer = selected_answers[0]
    final_answer = top_answer.get("answer") or FALLBACK_TEXT
    confidence = float(top_answer.get("confidence", 0.0) or 0.0)
    if "confidence_level" not in top_answer:
        top_answer["confidence_level"] = "High" if confidence >= 70 else "Medium" if confidence >= 30 else "Low"
    return {
        "answer": final_answer,
        "confidence": confidence,
        "answer_type": "VQA",
        "reasoning": reasoning,
        "answers": selected_answers,
        "base_answers": base_answers,
        "ft_answers": ft_answers,
        "ocr_text": "",
        "ocr_fields": {},
        "ocr_res": {"text": "", "fields": {}, "quality": "None"},
        "sat_data": None,
        "latency_ms": selected_prediction.get("latency_ms", 0),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "4.8.0"}


@app.get("/api/ocr/health")
async def ocr_health():
    try:
        from backend.models.ocr import get_ocr_diagnostics
        return get_ocr_diagnostics()
    except Exception as exc:
        return {
            "easyocr_loaded": False,
            "easyocr_attempted": False,
            "tesseract_enabled": False,
            "tesseract_cmd": "",
            "error": str(exc),
        }


@app.get("/api/stats")
async def get_stats():
    return VQA_STATS


@app.post("/api/detect_mode")
async def detect_mode_endpoint(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        ocr_text = ""
        try:
            ocr_text = run_advanced_ocr(pil.copy()).get("text", "").strip()
        except Exception:
            ocr_text = ""
        result = classify_image(pil, ocr_text=ocr_text)
        mode_map = {"DOC": "doc", "SAT": "satellite", "VQA": "vqa"}
        return {
            "mode": mode_map.get(result["mode"], "vqa"),
            "confidence": result["confidence"],
            "scores": result["scores"],
            "method": result["method"],
        }
    except Exception:
        return {"mode": "vqa", "confidence": 0, "scores": {}, "method": "error"}


@app.post("/api/classify")
async def classify_endpoint(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        return classify_image(pil_image)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@app.get("/api/translate")
async def api_translate(text: str, target: str):
    if not text.strip():
        return {"translated": text}
    try:
        from backend.models.vqa import _translate
        return {"translated": _translate(text, "auto", target)}
    except Exception:
        return {"translated": text}


@app.get("/api/tts")
async def generate_speech(text: str, lang: str = "en-US"):
    clean_text = re.sub(r"<[^>]+>", "", text)
    clean_text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", clean_text).strip()
    if not clean_text:
        raise HTTPException(400, detail="Text cannot be empty.")
    try:
        from gtts import gTTS
        iso = lang.split("-")[0].lower()
        tts = gTTS(clean_text, lang=iso)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        return Response(content=mp3_fp.getvalue(), media_type="audio/mpeg")
    except Exception as exc:
        raise HTTPException(500, detail=f"TTS error: {str(exc)}")


@app.post("/api/vqa")
def run_vqa(
    image: UploadFile = File(...),
    question: str = Form(...),
    mode: str = Form("AUTO"),
    analysis_mode: str = Form(""),
    model_type: str = Form("finetuned"),
    lang: str = Form("en-US"),
    detect: str = Form("true"),
    region_x: float = Form(0.0),
    region_y: float = Form(0.0),
    region_w: float = Form(1.0),
    region_h: float = Form(1.0),
    brightness: float = Form(100.0),
    contrast: float = Form(100.0),
):
    start = time.perf_counter()
    selected_model = _selected_model_type(model_type)
    resolved_question = question.strip()
    q_lower = resolved_question.lower()
    requested_mode = mode
    if str(analysis_mode or "").strip() and str(mode or "").strip().upper() in {"", "AUTO"}:
        requested_mode = analysis_mode
    target_mode = _normalize_mode(requested_mode)
    auto_detected_mode = None

    try:
        pil_image = _read_image(image)
        
        # Apply PIL brightness/contrast BEFORE any model runs
        from PIL import ImageEnhance
        if abs(brightness - 100.0) > 0.5:
            pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness / 100.0)
        if abs(contrast - 100.0) > 0.5:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast / 100.0)
        print(f"[app] Image adjusted: brightness={brightness}%, contrast={contrast}%")
        
        pil_image = _apply_region(pil_image, region_x, region_y, region_w, region_h)
        adjusted_image = pil_image.copy()

        pre_run_ocr = None
        if target_mode == "AUTO":
            try:
                cls = classify_image(adjusted_image.copy())
                cls_mode = str(cls.get("mode", "")).upper()
                
                if cls_mode == "DOC":
                    target_mode = "OCR"
                    auto_detected_mode = "DOC"
                    pre_run_ocr = run_advanced_ocr(adjusted_image.copy(), question=resolved_question)
                elif cls_mode == "SAT":
                    target_mode = "SATELLITE"
                    auto_detected_mode = "SAT"
                else:
                    target_mode = "VQA"
                    auto_detected_mode = "VQA"
            except Exception as e:
                _log.warning(f"Auto-detection failed: {e}")
                target_mode = "VQA"
                auto_detected_mode = "VQA"

        _log.info(
            "MODE_ROUTING | Req: %s | Auto: %s | Final: %s | Q: '%s' | Model: %s",
            requested_mode, auto_detected_mode, target_mode, resolved_question, selected_model
        )

        if target_mode == "OCR":
            result = _run_ocr_pipeline(adjusted_image, resolved_question, pre_run_ocr)
            suggests = ["What is the name?", "Emp. Code?", "Blood group?", "Date of birth?", "Read all text"]
        elif target_mode == "SATELLITE":
            result = _run_satellite_pipeline(adjusted_image, resolved_question, None)
            suggests = ["Roads visible?", "Buildings detected?", "Water coverage?", "Vegetation visible?"]
        else:
            target_mode = "VQA"
            result = _run_vqa_pipeline(adjusted_image, resolved_question, selected_model, lang)
            suggests = ["How many objects?", "What color is it?", "What is happening?", "Identify objects"]

        # --- Hybrid Analysis Logic (Priority Order) ---
        hybrid_explanation = result.get("reasoning", "")
        
        if target_mode == "OCR" and result.get("ocr_fields"):
            # Priority 1: OCR extracted fields (already set in _run_ocr_pipeline)
            final_answer = result["answer"] or FALLBACK_TEXT
            final_conf = result["confidence"]
            hybrid_explanation = "Priority 1: OCR extracted fields found."
            
        elif target_mode == "VQA" and result.get("ft_answers") and result["ft_answers"][0].get("confidence", 0) > 85:
            # Priority 2: Fine-tuned answer (if confidence > 85%)
            final_answer = result["ft_answers"][0]["answer"]
            final_conf = result["ft_answers"][0]["confidence"]
            hybrid_explanation = "Priority 2: Fine-tuned answer (High confidence)."
            
        elif target_mode == "SATELLITE":
            # Priority 3: Object detection + fine-tuned (if mode=SATELLITE)
            final_answer = result["answer"] or FALLBACK_TEXT
            final_conf = result["confidence"]
            try:
                ft_res = get_model("finetuned").predict(adjusted_image.copy(), resolved_question, top_k=1, lang=lang)
                if ft_res and ft_res.get("answers"):
                    ft_ans = ft_res["answers"][0]["answer"]
                    if ft_ans and ft_ans != FALLBACK_TEXT:
                        final_answer = f"{final_answer} | VQA Context: {ft_ans}"
                hybrid_explanation = "Priority 3: Object detection + Fine-tuned."
            except Exception:
                hybrid_explanation = "Priority 3: Object detection only (VQA failed)."
                
        elif target_mode == "VQA":
            # Priority 4: Cross-validation (fine-tuned + base for uncertainty)
            final_answer = result["answer"] or FALLBACK_TEXT
            final_conf = result["confidence"]
            try:
                base_ans = result["base_answers"][0]["answer"] if result.get("base_answers") else "Unknown"
                ft_ans = result["ft_answers"][0]["answer"] if result.get("ft_answers") else "Unknown"
                if base_ans.lower() == ft_ans.lower() and base_ans != "Unknown":
                    final_answer = ft_ans
                    final_conf = min(99.0, max(final_conf, 70.0))
                    hybrid_explanation = "Priority 4: Cross-validation (Models agree)."
                else:
                    # Keep the strongest model answer visible; encode disagreement as uncertainty in confidence + explanation.
                    if ft_ans and ft_ans != "Unknown":
                        final_answer = ft_ans
                    elif base_ans and base_ans != "Unknown":
                        final_answer = base_ans
                    final_conf = max(20.0, min(final_conf, 60.0))
                    hybrid_explanation = f"Priority 4: Cross-validation (Model disagreement). Fine-tuned='{ft_ans}' vs Base='{base_ans}'."
            except Exception:
                hybrid_explanation = "Priority 4: Cross-validation failed."
        else:
            final_answer = result["answer"] or FALLBACK_TEXT
            final_conf = result["confidence"]

        # 7. GLOBAL FAILSAFE (VERY IMPORTANT)
        if not final_answer or final_answer.strip() == "":
            final_answer = "Unable to determine"

        detection_image_b64 = None
        answer_type = result.get("answer_type", target_mode)
        sat_data = result.get("sat_data")

        # SATELLITE: always use HSV grounding overlay, NEVER YOLO
        if "Satellite" in answer_type or target_mode == "SATELLITE":
            if sat_data and sat_data.get("grounding_image"):
                detection_image_b64 = sat_data["grounding_image"]
                print("[Detection] Using satellite HSV grounding overlay")
            else:
                print("[Detection] Satellite mode: no grounding image available")

        # OCR/DOCUMENT: no detection at all
        elif "OCR" in answer_type or target_mode == "OCR":
            detection_image_b64 = None
            print("[Detection] OCR mode: skipping detection")

        # VQA ONLY: run YOLO with question keywords
        else:
            _det_triggers = ["what","where","how many","count","identify",
                             "find","show","is there","are there","color","object","person","car"]
            if detect == "true" or any(t in q_lower for t in _det_triggers):
                try:
                    det = detect_and_draw(pil_image.copy(), resolved_question)
                    if det:
                        detection_image_b64 = det
                        print(f"[YOLO] Detection successful for: {resolved_question}")
                    else:
                        print(f"[YOLO] No matching objects for: {resolved_question}")
                except Exception as de:
                    import traceback
                    print(f"[YOLO] Error: {de}")
                    traceback.print_exc()

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {
            "question": resolved_question,
            "original_query": question.strip(),
            "model_type": selected_model,
            "model_name": MODEL_CONFIGS[selected_model]["name"],
            "answer": final_answer,
            "answer_type": result.get("answer_type", target_mode),
            "model": target_mode,
            "time_ms": elapsed_ms,
            "confidence": round(final_conf, 1),
            "answers": result["answers"],
            "base_answers": result["base_answers"],
            "ft_answers": result["ft_answers"],
            "hybrid": {
                "answer": final_answer,
                "confidence": final_conf,
                "explanation": hybrid_explanation,
                "type": result.get("answer_type", target_mode),
            },
            "latency_ms": result.get("latency_ms") or elapsed_ms,
            "detection_image": detection_image_b64,
            "detection_meta": {
                "ran": detection_image_b64 is not None,
                "mode": (
                    "satellite_grounding"
                    if target_mode == "SATELLITE" and detection_image_b64
                    else "yolo_vqa"
                    if detection_image_b64
                    else "none"
                ),
            },
            "suggested_queries": suggests,
            "auto_detected_mode": auto_detected_mode,
            "ocr_extracted_text": result["ocr_text"] if target_mode == "OCR" else "",
            "ocr_metadata": {
                "char_count": len(result["ocr_text"]),
                "line_count": len(result["ocr_text"].split("\n")) if result["ocr_text"] else 0,
                "quality": result["ocr_res"].get("quality", "None"),
                "fields": result["ocr_fields"],
                "diagnostics": result["ocr_res"].get("diagnostics", {}),
            } if target_mode == "OCR" else None,
            "satellite": {
                "land_use": result["sat_data"].get("land_use"),
                "density": result["sat_data"].get("density"),
                "features": result["sat_data"].get("features", []),
            } if result["sat_data"] else None,
            "is_satellite_mode": target_mode == "SATELLITE",
        }
    except Exception as exc:
        print(f"GLOBAL API ERROR: {exc}")
        traceback.print_exc()
        selected_model = _selected_model_type(model_type)
        return {
            "question": question,
            "original_query": question.strip(),
            "model_type": selected_model,
            "model_name": MODEL_CONFIGS[selected_model]["name"],
            "answers": [],
            "base_answers": [_answer(FALLBACK_TEXT, 0, "Low")],
            "ft_answers": [_answer(FALLBACK_TEXT, 0, "Low")],
            "answer": FALLBACK_TEXT,
            "model": _normalize_mode(mode),
            "time_ms": 0,
            "latency_ms": 0,
            "confidence": 0,
            "hybrid": {
                "answer": FALLBACK_TEXT,
                "confidence": 0,
                "type": "Error",
                "explanation": str(exc),
            },
            "error": True,
            "errorMessage": str(exc),
        }


@app.post("/api/batch")
def batch_analyze(
    images: list[UploadFile] = File(...),
    question: str = Form(...),
    mode: str = Form("AUTO"),
):
    results = []
    for img_file in images[:10]:
        try:
            result = run_vqa(
                image=img_file,
                question=question,
                mode=mode,
                model_type="finetuned",
                lang="en-US",
                detect="false",
                region_x=0.0,
                region_y=0.0,
                region_w=1.0,
                region_h=1.0,
                brightness=100.0,
                contrast=100.0,
            )
            results.append({"filename": img_file.filename, "result": result})
        except Exception as exc:
            results.append({"filename": img_file.filename, "result": {"error": str(exc)}})
    return {"batch_results": results, "count": len(results)}


@app.post("/api/compare")
async def compare_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        import cv2
        import numpy as np

        raw1 = await image1.read()
        raw2 = await image2.read()
        img1 = Image.open(io.BytesIO(raw1)).convert("RGB")
        img2 = Image.open(io.BytesIO(raw2)).convert("RGB")

        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        arr1 = np.array(img1).astype(np.uint8)
        arr2 = np.array(img2).astype(np.uint8)
        diff = cv2.absdiff(cv2.GaussianBlur(arr1, (5, 5), 0), cv2.GaussianBlur(arr2, (5, 5), 0))
        mask = (np.sum(diff, axis=2) > 45).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), kernel, iterations=1)

        overlay = arr2.copy()
        overlay[mask > 0] = [239, 68, 68]
        change_ratio = np.mean(mask > 0) * 100

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        summary = f"Detected changes in {change_ratio:.1f}% of the scene. "
        if change_ratio > 50:
            summary += "Significant structural changes detected across majority of scene."
        elif change_ratio > 20:
            summary += "Moderate changes detected, likely new construction or seasonal shift."
        elif change_ratio > 5:
            summary += "Minor changes observed, possible vehicle movement or lighting variation."
        else:
            summary += "Images are nearly identical. Negligible differences found."

        return {
            "summary": summary,
            "change_ratio": round(change_ratio, 2),
            "comparison_image": overlay_b64,
        }
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@app.get("/api/benchmark")
async def get_benchmark_report():
    """Return the most recent benchmark results from tmp/benchmark_results/."""
    base = Path("tmp/benchmark_results")
    json_path = base / "publication_metrics.json"
    md_path = base / "publication_report.md"
    if not json_path.exists():
        return {"available": False, "message": "No benchmark report found. Run benchmark/run_publication_eval.py first."}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        return {"available": True, "report": data, "report_md": md_text}
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


try:
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
except RuntimeError:
    import warnings
    warnings.warn(
        f"Static frontend directory not found at '{FRONTEND_DIR}'. "
        "Run 'npm run build' in frontend-next/ to generate it. "
        "API endpoints remain available.",
        RuntimeWarning,
        stacklevel=1,
    )
