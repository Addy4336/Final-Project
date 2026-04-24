"""
VisionQuery – CLIP-based Image Classifier.

Automatically classifies images into one of three categories:
  - DOCUMENT  (ID cards, forms, receipts, text-heavy images)
  - SATELLITE (aerial views, maps, terrain, overhead imagery)
  - GENERAL   (everyday scenes, objects, people, etc.)

Uses OpenAI CLIP (via HuggingFace transformers) for zero-shot classification.
Falls back to OCR + HSV heuristics if CLIP is unavailable.
"""
import time
import threading
import numpy as np
from PIL import Image

# ── Try loading CLIP ──────────────────────────────────────────────────
_CLIP_AVAILABLE = False
_clip_model = None
_clip_processor = None
_clip_lock = threading.Lock()

try:
    from transformers import CLIPProcessor, CLIPModel
    _CLIP_AVAILABLE = True
except ImportError:
    pass

# ── Try loading OpenCV for fallback ───────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ── CLIP labels for zero-shot classification ──────────────────────────
# These prompts must be highly specific to avoid misclassification:
#   - DOC: emphasize FLAT scanned/printed documents, not scenes with text
#   - SAT: emphasize overhead/top-down perspective
#   - VQA: emphasize real-world 3D scenes with depth and people/objects
_CLIP_LABELS = [
    "a flat scanned document page, official ID card, printed form, receipt, or a close-up of text on paper",
    "a satellite image, aerial top-down view of terrain, overhead map, or bird's eye view of buildings and roads",
    "a real-world photograph of people, animals, objects, street scenes, rooms, nature, or everyday activities",
]

_LABEL_MAP = {0: "DOC", 1: "SAT", 2: "VQA"}


def _load_clip():
    """Lazy-load CLIP model (singleton)."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return
    with _clip_lock:
        if _clip_model is not None:
            return
        try:
            _clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            _clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            _clip_model.eval()
            print("[Classifier] CLIP model loaded successfully.")
        except Exception as e:
            print(f"[Classifier] CLIP load failed: {e}")


def classify_with_clip(pil_image: Image.Image) -> dict:
    """
    Classify image using CLIP zero-shot.
    Returns: {"mode": "DOC"|"SAT"|"VQA", "confidence": float, "scores": dict, "method": "clip"}
    """
    if not _CLIP_AVAILABLE:
        return None

    _load_clip()
    if _clip_model is None or _clip_processor is None:
        return None

    try:
        import torch
        t0 = time.perf_counter()

        inputs = _clip_processor(
            text=_CLIP_LABELS,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = _clip_model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        scores = {_LABEL_MAP[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
        best_idx = int(np.argmax(probs))
        mode = _LABEL_MAP[best_idx]
        confidence = float(probs[best_idx]) * 100

        latency = round((time.perf_counter() - t0) * 1000)

        return {
            "mode": mode,
            "confidence": round(confidence, 1),
            "scores": scores,
            "method": "clip",
            "latency_ms": latency,
        }
    except Exception as e:
        print(f"[Classifier] CLIP inference failed: {e}")
        return None


def classify_with_heuristics(pil_image: Image.Image, ocr_text: str = "") -> dict:
    """
    Fallback classifier using OCR text length + HSV color analysis.
    Returns: {"mode": "DOC"|"SAT"|"VQA", "confidence": float, "scores": dict, "method": "heuristic"}
    """
    t0 = time.perf_counter()

    doc_score = 0.0
    sat_score = 0.0
    vqa_score = 30.0  # default baseline

    # ── OCR signal ────────────────────────────────────────────────────
    text_len = len(ocr_text.strip()) if ocr_text else 0
    if text_len > 100:
        doc_score += 80
    elif text_len > 50:
        doc_score += 60
    elif text_len > 15:
        doc_score += 40

    # ── HSV analysis for satellite ────────────────────────────────────
    if _CV2_AVAILABLE:
        try:
            arr = np.array(pil_image.resize((256, 256))).astype(np.uint8)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            total_px = 256 * 256

            # Document visual priors: bright page + dark glyph-like strokes + low saturation.
            white_bg_mask = cv2.inRange(arr, np.array([220, 220, 220]), np.array([255, 255, 255]))
            dark_text_mask = cv2.inRange(arr, np.array([0, 0, 0]), np.array([95, 95, 95]))
            white_bg_ratio = np.sum(white_bg_mask > 0) / total_px
            dark_text_ratio = np.sum(dark_text_mask > 0) / total_px
            sat_mean = float(np.mean(hsv[:, :, 1]))

            if white_bg_ratio > 0.40 and dark_text_ratio > 0.004:
                doc_score += 35
            if white_bg_ratio > 0.55 and sat_mean < 55:
                doc_score += 30
            if 0.004 < dark_text_ratio < 0.22 and sat_mean < 70:
                doc_score += 20

            # Natural feature masks
            water_mask = cv2.inRange(hsv, np.array([85, 30, 20]), np.array([145, 255, 255]))
            veg_mask = cv2.inRange(hsv, np.array([30, 30, 20]), np.array([90, 255, 255]))
            built_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 220]))

            water_r = np.sum(water_mask > 0) / total_px
            veg_r = np.sum(veg_mask > 0) / total_px
            built_r = np.sum(built_mask > 0) / total_px

            combined = water_r + veg_r
            if combined > 0.30:
                sat_score += 40
            if veg_r > 0.15:
                sat_score += 20
            if water_r > 0.08:
                sat_score += 15

            # Texture: low saturation std → top-down
            sat_std = np.std(hsv[:, :, 1])
            if sat_std < 50:
                sat_score += 15

            # Edge density
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_px
            if 0.05 < edge_density < 0.40:
                sat_score += 10

        except Exception:
            pass

    # ── VQA is the fallback ───────────────────────────────────────────
    vqa_score = max(30.0, 100 - doc_score - sat_score)

    # Normalize
    total = doc_score + sat_score + vqa_score
    if total > 0:
        doc_score = (doc_score / total) * 100
        sat_score = (sat_score / total) * 100
        vqa_score = (vqa_score / total) * 100

    scores = {"DOC": round(doc_score, 1), "SAT": round(sat_score, 1), "VQA": round(vqa_score, 1)}
    best = max(scores, key=scores.get)

    latency = round((time.perf_counter() - t0) * 1000)

    return {
        "mode": best,
        "confidence": scores[best],
        "scores": scores,
        "method": "heuristic",
        "latency_ms": latency,
    }


def classify_image(pil_image: Image.Image, ocr_text: str = "") -> dict:
    """
    Main classifier entry point. Tries CLIP first, falls back to heuristics.
    Applies OCR-override gate: if significant text detected, boosts DOC score.

    Returns:
        {
            "mode": "DOC" | "SAT" | "VQA",
            "confidence": float (0-100),
            "scores": {"DOC": float, "SAT": float, "VQA": float},
            "method": "clip" | "heuristic" | "clip+heuristic",
            "latency_ms": int,
        }
    """
    clip_result = classify_with_clip(pil_image)
    heuristic_result = classify_with_heuristics(pil_image, ocr_text=ocr_text)

    # If CLIP is unavailable, use heuristics only
    if clip_result is None:
        return heuristic_result

    # If CLIP is highly confident (>55%), trust it
    if clip_result["confidence"] > 55.0:
        result = clip_result
    else:
        # Low CLIP confidence — blend with heuristics
        # Average the scores from both methods
        blended_scores = {}
        for key in ("DOC", "SAT", "VQA"):
            clip_s = clip_result["scores"].get(key, 0)
            heur_s = heuristic_result["scores"].get(key, 0)
            blended_scores[key] = round(clip_s * 0.4 + heur_s * 0.6, 1)

        best = max(blended_scores, key=blended_scores.get)
        total_latency = clip_result.get("latency_ms", 0) + heuristic_result.get("latency_ms", 0)
        result = {
            "mode": best,
            "confidence": blended_scores[best],
            "scores": blended_scores,
            "method": "clip+heuristic",
            "latency_ms": total_latency,
        }

    # OCR-override gate: if strong OCR text exists, ensure DOC wins
    text_len = len(ocr_text.strip()) if ocr_text else 0
    if text_len > 60 and result["mode"] != "DOC":
        # Only override if SAT score isn't dominant (prevents misclassifying satellite with text)
        if result["scores"].get("SAT", 0) < 55:
            result["scores"]["DOC"] = max(result["scores"].get("DOC", 0), 75.0)
            result["mode"] = "DOC"
            result["confidence"] = result["scores"]["DOC"]
            result["method"] += "+ocr_override"

    return result
