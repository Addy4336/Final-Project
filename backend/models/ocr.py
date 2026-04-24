"""
VisionQuery – Robust Production-Ready Document OCR.
Hybrid EasyOCR + Tesseract pipeline with multi-stage preprocessing.

Core Contract:
  - Preprocessing: Resize, grayscale, threshold, contrast boost
  - Field Extraction: Name, Employee Code, DOB, Address, Blood Group
  - Pattern Matching: Regex patterns for structured data
  - Fallback: Always return first meaningful lines if fields not found
  - NEVER return "No readable text detected"
"""
import re
import io
import shutil
import cv2
import numpy as np
import traceback
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# Imports with fallbacks
pytesseract = None
try:
    import pytesseract
    # Configure Tesseract path if available.
    _tesseract_cmd = None
    for _tp in (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ):
        if _tp.exists():
            _tesseract_cmd = str(_tp)
            break
    if not _tesseract_cmd:
        _tesseract_cmd = shutil.which("tesseract")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
        try:
            _ = pytesseract.get_tesseract_version()
            _TESSERACT_ENABLED = True
        except Exception:
            _TESSERACT_ENABLED = False
    else:
        _TESSERACT_ENABLED = False
except ImportError:
    _TESSERACT_ENABLED = False

# Global reader instance, initialized lazily
_EASYOCR_READER = None
_EASYOCR_LOADED = False

def get_easyocr_reader():
    global _EASYOCR_READER, _EASYOCR_LOADED
    if _EASYOCR_LOADED:
        return _EASYOCR_READER
    try:
        import easyocr
        # Initializing reader (gpu=False as per environment checks)
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=False)
        _EASYOCR_LOADED = True
    except Exception as e:
        print(f"[OCR] EasyOCR failed to load: {e}")
        _EASYOCR_LOADED = True  # Mark as tried
    return _EASYOCR_READER


def get_ocr_diagnostics():
    """Expose OCR engine readiness for API metadata and debugging."""
    tesseract_cmd = ""
    if _TESSERACT_ENABLED and pytesseract is not None:
        tesseract_cmd = str(getattr(pytesseract.pytesseract, "tesseract_cmd", "") or "")
    return {
        "easyocr_loaded": _EASYOCR_READER is not None,
        "easyocr_attempted": _EASYOCR_LOADED,
        "tesseract_enabled": bool(_TESSERACT_ENABLED),
        "tesseract_cmd": tesseract_cmd,
    }

# Field keys matching app.py expectations (space-separated, lowercase)
FIELD_ORDER = ["name", "emp code", "dob", "blood group", "address"]

# ──────────────────────────────────────────────────────────────────────
# PREPROCESSING – Resize, Grayscale, Threshold, Contrast Boost
# ──────────────────────────────────────────────────────────────────────

def _ensure_min_dpi(img_cv, min_width=1200):
    """Upscale small images so OCR engines have enough pixel detail."""
    h, w = img_cv.shape[:2]
    if w < min_width:
        scale = min_width / w
        img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
    return img_cv


def _to_grayscale(img_cv):
    """Convert to grayscale if needed."""
    if len(img_cv.shape) == 3:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return img_cv


def _contrast_boost(gray):
    """CLAHE contrast boost on grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_strategies(img_cv):
    """
    Build multiple preprocessed variants.
    Returns dict of name → grayscale numpy array.
    """
    # Step 1: Resize for better OCR
    img_cv = _ensure_min_dpi(img_cv)

    # Step 2: Grayscale
    gray = _to_grayscale(img_cv)

    # Step 3: Contrast boost (CLAHE)
    boosted = _contrast_boost(gray)

    # Step 4: Threshold variants
    blur = cv2.GaussianBlur(boosted, (3, 3), 0)

    # Otsu
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive Gaussian
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    # Adaptive Mean (good for uneven lighting)
    adaptive_mean = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 6
    )

    # Inverted Otsu (white text on dark background)
    inv_otsu = cv2.bitwise_not(otsu)

    # Sharpened grayscale (no threshold)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(boosted, -1, sharp_kernel)

    return {
        "gray":          gray,
        "boosted":       boosted,
        "otsu":          otsu,
        "adaptive":      adaptive,
        "adaptive_mean": adaptive_mean,
        "inv_otsu":      inv_otsu,
        "sharpened":     sharpened,
    }


# ──────────────────────────────────────────────────────────────────────
# OCR ENGINES
# ──────────────────────────────────────────────────────────────────────

def run_easy_ocr(img_cv):
    """Run EasyOCR on the provided image."""
    reader = get_easyocr_reader()
    if not reader:
        return ""
    try:
        results = reader.readtext(img_cv, detail=1, paragraph=False)
        # Sort by vertical position (top to bottom) then left to right
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        lines = []
        last_y = -999
        for bbox, text, conf in results:
            if conf < 0.15:
                continue
            y = bbox[0][1]
            if abs(y - last_y) > 15:
                lines.append(text)
            else:
                if lines:
                    lines[-1] += " " + text
                else:
                    lines.append(text)
            last_y = y
        return "\n".join(lines)
    except Exception as e:
        print(f"[EasyOCR] Error: {e}")
        return ""


def run_tesseract(img_cv, psm=6):
    """Run Tesseract with a specific PSM mode."""
    if not _TESSERACT_ENABLED:
        return ""
    try:
        config = f"--oem 3 --psm {psm}"
        return pytesseract.image_to_string(img_cv, config=config)
    except Exception:
        return ""


def run_tesseract_multi_psm(img_cv):
    """Run Tesseract with multiple PSM modes and return all results."""
    results = []
    for psm in [6, 4, 3, 11]:
        text = run_tesseract(img_cv, psm)
        if text.strip():
            results.append(text)
    return results


# ──────────────────────────────────────────────────────────────────────
# TEXT CLEANING (conservative – do NOT destroy names)
# ──────────────────────────────────────────────────────────────────────

def clean_ocr_text(text):
    """Normalize text without destroying alphabetic content."""
    if not text:
        return ""

    # Normalize whitespace within lines (keep newlines)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Collapse multiple spaces/tabs
        line = re.sub(r'[ \t]+', ' ', line)
        # Remove pure garbage lines (< 2 chars, only symbols)
        if len(line) < 2:
            continue
        if re.fullmatch(r'[\W_]+', line):
            continue
        lines.append(line)

    return "\n".join(lines)


def _score_text(text):
    """Score raw OCR text quality. Higher = better."""
    if not text:
        return 0
    clean = text.strip()
    length = len(clean)
    alpha = len(re.findall(r'[A-Za-z]', clean))
    digits = len(re.findall(r'[0-9]', clean))
    words = len(re.findall(r'[A-Za-z]{2,}', clean))
    lines = len([l for l in clean.splitlines() if len(l.strip()) > 2])
    # Reward recognizable field labels
    label_hits = 0
    for pat in [r'(?i)\bname\b', r'(?i)\bemp', r'(?i)\bdob\b', r'(?i)\bblood\b',
                r'(?i)\baddress\b', r'(?i)\bdate\b', r'(?i)\bcode\b']:
        if re.search(pat, clean):
            label_hits += 1
    return length + (alpha * 1.5) + (digits * 2) + (words * 3) + (lines * 5) + (label_hits * 50)


# ──────────────────────────────────────────────────────────────────────
# FIELD EXTRACTION – Pattern Matching + Label Parsing
# ──────────────────────────────────────────────────────────────────────

# ----- Blood Group -----
_BLOOD_GROUP_RE = re.compile(
    r'\b(A|B|AB|O)\s*[+-]\b'             # A+, B-, AB+, O-
    r'|'
    r'\b(A|B|AB|O)\s*(pos|neg)\w*\b',    # A positive, B negative
    re.IGNORECASE
)

_BLOOD_GROUP_STANDALONE = re.compile(
    r'(?<!\w)(A\+|A\-|B\+|B\-|AB\+|AB\-|O\+|O\-)(?!\w)'
)

# ----- DOB -----
_DOB_PATTERNS = [
    re.compile(r'\b(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})\b'),    # DD/MM/YYYY or DD-MM-YY
    re.compile(r'\b(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\b'),      # YYYY/MM/DD
    re.compile(r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{2,4})\b', re.I),
]

# ----- Emp Code -----
_EMP_CODE_PATTERNS = [
    re.compile(r'\b(TERF\d+)\b', re.I),                       # TERF119
    re.compile(r'\b([A-Z]{2,5}\d{3,})\b'),                    # ABC123, TERF119
    re.compile(r'\b(\d{2,}[A-Z]{2,}\d+)\b'),                  # 22EC101
    re.compile(r'\b(E[.-]?\d{4,})\b', re.I),                  # E-12345
    re.compile(r'\b([A-Z]{1,3}[-/]\d{3,})\b'),                # EC/1234
]

# ----- Name -----
_NAME_LABEL_RE = re.compile(
    r'(?i)(?:name|employee\s*name|full\s*name)\s*[:\-–—]?\s*(.+)',
)

# ----- Address -----
_ADDR_LABEL_RE = re.compile(
    r'(?i)(?:address|addr|residence|resides?\s+at)\s*[:\-–—]?\s*(.+)',
)


def extract_fields(lines, full_text=""):
    """
    Extract structured fields: Name, Employee Code, DOB, Address, Blood Group.
    Uses both label-based parsing and pattern matching.
    """
    fields = {}
    if not full_text:
        full_text = "\n".join(lines)

    # ── 1. LABEL-BASED EXTRACTION (look for "Label: Value" patterns) ──

    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        # Name
        if "name" not in fields:
            m = _NAME_LABEL_RE.search(line_clean)
            if m:
                val = m.group(1).strip()
                # Clean: remove trailing non-alpha
                val = re.sub(r'[^A-Za-z .\'-]+$', '', val).strip()
                if len(val) >= 3:
                    fields["name"] = val

        # DOB
        if "dob" not in fields:
            if re.search(r'(?i)\b(?:dob|d\.?\s*o\.?\s*b|date\s*of\s*birth|birth\s*date)\b', line_clean):
                for pat in _DOB_PATTERNS:
                    m = pat.search(line_clean)
                    if m:
                        fields["dob"] = m.group(0).strip()
                        break

        # Emp Code
        if "emp code" not in fields:
            if re.search(r'(?i)\b(?:emp|employee|code|id|roll)\b', line_clean):
                for pat in _EMP_CODE_PATTERNS:
                    m = pat.search(line_clean)
                    if m:
                        fields["emp code"] = m.group(1).strip()
                        break

        # Blood Group
        if "blood group" not in fields:
            if re.search(r'(?i)\bblood\b', line_clean):
                m = _BLOOD_GROUP_RE.search(line_clean)
                if m:
                    fields["blood group"] = m.group(0).strip().upper()
                else:
                    m = _BLOOD_GROUP_STANDALONE.search(line_clean)
                    if m:
                        fields["blood group"] = m.group(0).strip()

        # Address
        if "address" not in fields:
            m = _ADDR_LABEL_RE.search(line_clean)
            if m:
                addr_val = m.group(1).strip()
                # Address may span multiple lines
                j = i + 1
                while j < len(lines) and j < i + 4:
                    next_line = lines[j].strip()
                    # Stop if next line looks like a new label
                    if re.match(r'(?i)(name|dob|emp|blood|code|id)\b', next_line):
                        break
                    if next_line:
                        addr_val += " " + next_line
                    j += 1
                if len(addr_val) >= 5:
                    fields["address"] = addr_val.strip()

    # ── 2. PATTERN-ONLY EXTRACTION (fallback if labels weren't found) ──

    # DOB fallback: find any date in full text
    if "dob" not in fields:
        for pat in _DOB_PATTERNS:
            m = pat.search(full_text)
            if m:
                fields["dob"] = m.group(0).strip()
                break

    # Emp Code fallback
    if "emp code" not in fields:
        for pat in _EMP_CODE_PATTERNS:
            m = pat.search(full_text)
            if m:
                fields["emp code"] = m.group(1).strip()
                break

    # Blood Group fallback
    if "blood group" not in fields:
        m = _BLOOD_GROUP_RE.search(full_text)
        if m:
            fields["blood group"] = m.group(0).strip().upper()
        else:
            m = _BLOOD_GROUP_STANDALONE.search(full_text)
            if m:
                fields["blood group"] = m.group(0).strip()

    # Name fallback: Use heuristics on first meaningful lines
    if "name" not in fields:
        # Strategy A: Find a line with 2+ capitalized words (typical for names)
        for line in lines[:10]:
            line_s = line.strip()
            # Skip lines that look like labels or codes
            if re.search(r'(?i)(dob|emp|blood|address|code|id|date|group)', line_s):
                continue
            # Skip lines with too many digits
            if len(re.findall(r'\d', line_s)) > len(line_s) * 0.3:
                continue
            # Match: "Firstname Lastname" or "FIRSTNAME LASTNAME"
            name_m = re.match(r'^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})$', line_s)
            if name_m:
                fields["name"] = name_m.group(1)
                break
            # Match ALL CAPS name
            name_m = re.match(r'^([A-Z][A-Z .\'-]{3,40})$', line_s)
            if name_m and len(line_s.split()) >= 2:
                fields["name"] = name_m.group(1).strip()
                break

    return fields


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run_advanced_ocr(pil_image, question=""):
    """
    Robust OCR Pipeline.
    Returns: {text, fields, answer, quality, confidence, ...}

    Core guarantees:
      1. Preprocessing: resize → grayscale → threshold → contrast boost
      2. Multi-engine: EasyOCR + Tesseract with multiple PSM modes
      3. Best-of-N text selection
      4. Structured field extraction
      5. NEVER returns "No readable text detected"
    """
    try:
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # ── 1. Build preprocessed variants ──
        variants = preprocess_strategies(img_cv)

        candidates = []

        # ── 2. EasyOCR on key variants ──
        for key in ["boosted", "gray", "sharpened"]:
            text = run_easy_ocr(variants[key])
            if text.strip():
                candidates.append(text)

        # ── 3. EasyOCR on original (resized but unthresholded) ──
        resized_original = _ensure_min_dpi(img_cv)
        text = run_easy_ocr(resized_original)
        if text.strip():
            candidates.append(text)

        # ── 4. Tesseract on threshold variants ──
        for key in ["otsu", "adaptive", "adaptive_mean", "boosted", "sharpened"]:
            results = run_tesseract_multi_psm(variants[key])
            candidates.extend(results)

        # ── 5. Tesseract on inverted (for white-on-dark text) ──
        inv_results = run_tesseract_multi_psm(variants["inv_otsu"])
        candidates.extend(inv_results)

        # ── 6. Select best candidate ──
        if not candidates:
            # Last resort: raw EasyOCR on input
            text = run_easy_ocr(img_cv)
            if text.strip():
                candidates.append(text)

        if candidates:
            best_text = max(candidates, key=_score_text)
            best_text = clean_ocr_text(best_text)
        else:
            best_text = ""

        # ── 7. Retry with aggressive upscale if too short ──
        if len(best_text.strip()) < 15:
            print("[OCR] Short result, retrying with 3x upscale + sharpening...")
            big = cv2.resize(img_cv, None, fx=3.0, fy=3.0,
                             interpolation=cv2.INTER_CUBIC)
            big_gray = _to_grayscale(big)
            big_boosted = _contrast_boost(big_gray)

            retry_texts = [
                run_easy_ocr(big_boosted),
                run_tesseract(big_boosted, psm=6),
                run_tesseract(big_boosted, psm=3),
            ]
            for rt in retry_texts:
                if _score_text(rt) > _score_text(best_text):
                    best_text = clean_ocr_text(rt)

        # ── 8. Extract structured fields ──
        lines = best_text.splitlines()
        fields = extract_fields(lines, best_text)

        # ── 9. Resolve answer for question ──
        answer = resolve_ocr_answer(question, best_text, fields)

        # Quality tier
        field_count = len(fields)
        text_len = len(best_text.strip())
        if field_count >= 3 and text_len > 50:
            quality = "High"
        elif field_count >= 1 or text_len > 20:
            quality = "Medium"
        elif text_len > 0:
            quality = "Low"
        else:
            quality = "None"

        return {
            "text":       best_text,
            "fields":     fields,
            "answer":     answer,
            "quality":    quality,
            "confidence": quality.lower(),
            "char_count": len(best_text),
            "line_count": len(lines),
            "strategy":   "hybrid_robust_v2",
            "diagnostics": get_ocr_diagnostics(),
        }

    except Exception as e:
        traceback.print_exc()
        # NEVER return "No readable text detected" — return error context instead
        return {
            "text": "", "fields": {}, "answer": f"OCR processing error: {str(e)}",
            "quality": "None", "confidence": "none", "char_count": 0, "line_count": 0,
            "error": str(e), "diagnostics": get_ocr_diagnostics(),
        }


# ──────────────────────────────────────────────────────────────────────
# QUESTION → ANSWER RESOLUTION
# ──────────────────────────────────────────────────────────────────────

def resolve_ocr_answer(question, text, fields):
    """
    Answer questions from extracted fields.
    Fallback: always return first meaningful lines, NEVER empty.
    """
    q = str(question or "").lower().strip()

    # If no question, return all fields summary or raw text
    if not q or q in ("", "read all text", "read text", "extract text",
                       "what does it say", "what is written"):
        return _build_all_fields_answer(text, fields)

    # ── Map question keywords → field names ──
    KEYWORD_MAP = {
        "name":        ["name", "who", "whose", "person", "employee name", "full name"],
        "emp code":    ["code", "emp", "employee code", "emp code", "id", "roll",
                        "employee id", "emp id", "enrollment", "number"],
        "dob":         ["dob", "birth", "born", "date of birth", "birthday",
                        "d.o.b", "date"],
        "blood group": ["blood", "blood group", "blood type", "b.g", "bg",
                        "group"],
        "address":     ["address", "addr", "live", "lives", "residence",
                        "where", "location", "city", "resides"],
    }

    # Score each field by keyword match strength
    best_field = None
    best_score = 0
    for field_name, keywords in KEYWORD_MAP.items():
        score = 0
        for kw in keywords:
            if kw in q:
                # Longer keyword matches are more specific
                score += len(kw)
        if score > best_score:
            best_score = score
            best_field = field_name

    if best_field and best_field in fields:
        val = fields[best_field]
        if val:
            return f"{best_field.title()}: {val}"

    # If we matched a field but it wasn't extracted, say so
    if best_field and best_field not in fields:
        # Still provide what we have
        if fields:
            available = " | ".join(f"{k.title()}: {v}" for k, v in fields.items())
            return f"{best_field.title()} not found. Available: {available}"

    # Generic fallback — return all fields or first meaningful lines
    return _build_all_fields_answer(text, fields)


def _build_all_fields_answer(text, fields):
    """Build a complete answer from fields or raw text. Never empty."""
    if fields:
        parts = []
        for key in FIELD_ORDER:
            if key in fields and fields[key]:
                parts.append(f"{key.title()}: {fields[key]}")
        if parts:
            return " | ".join(parts)

    # Fallback: return first meaningful lines of raw text
    if text and text.strip():
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # Return first 5 meaningful lines (never nothing)
        meaningful = lines[:5]
        if meaningful:
            return "\n".join(meaningful)

    return "Document text could not be fully extracted. Please try adjusting brightness/contrast and resubmitting."
