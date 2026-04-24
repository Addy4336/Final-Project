"""
VisionQuery – backend object detection service.
Uses YOLOv8n (nano, ~6 MB) for fast, lightweight detection.
The first call downloads yolov8n.pt automatically to models/yolov8n.pt.

Highlighting strategy (keyword-focused):
  - ONLY objects whose YOLO class name overlaps with keywords extracted from
    the user's QUESTION are drawn with bright green bounding boxes.
  - Non-matching objects are completely hidden so the overlay stays clean,
    showing the user exactly what they asked about.
  - Stop-words (what, is, the, are, a, an, in, of, …) are stripped before
    matching so "What color is the dog?" correctly highlights 'dog'.
  - Synonym / alias mapping covers common question words →  YOLO class names
    (e.g. "person" ↔ "man"/"woman"/"people", "car" ↔ "vehicle", etc.)
"""
import io
import os
import re
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# Switch to yolov8s.pt (small) for significantly better accuracy than yolov8n (nano)
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "yolov8s.pt"
_yolo_model = None

# Common English stop-words to ignore during keyword extraction
_STOP_WORDS = {
    "what","is","are","the","a","an","in","of","on","at","to","do","does",
    "how","many","much","color","colour","which","there","this","that","these",
    "those","any","some","has","have","can","be","been","being","was","were",
    "will","with","for","from","by","about","into","than","then","when","where",
    "who","why","show","me","tell","give","find","see","look","did","does",
    "it","its","their","they","them","we","our","you","your","he","she","him","her","his","hers",
    "describe","picture","image","photo","photograph","shown","visible","present",
    "name","type","kind","identify","explain","list","count",
}

# Synonym / alias mapping: question word → set of YOLO class names it can match
_SYNONYMS = {
    "person": {"person"},
    "man": {"person"}, "woman": {"person"}, "people": {"person"},
    "boy": {"person"}, "girl": {"person"}, "human": {"person"},
    "child": {"person"}, "kid": {"person"}, "baby": {"person"},
    "car": {"car", "truck"}, "vehicle": {"car", "truck", "bus"},
    "automobile": {"car"}, "bike": {"bicycle", "motorcycle"},
    "cycle": {"bicycle"}, "motorbike": {"motorcycle"},
    "phone": {"cell phone"}, "cellphone": {"cell phone"},
    "mobile": {"cell phone"}, "laptop": {"laptop"},
    "computer": {"laptop"}, "monitor": {"tv"},
    "television": {"tv"}, "screen": {"tv", "laptop"},
    "animal": {"cat", "dog", "horse", "sheep", "cow", "elephant",
               "bear", "zebra", "giraffe", "bird"},
    "pet": {"cat", "dog"}, "puppy": {"dog"}, "kitten": {"cat"},
    "plane": {"airplane"}, "aeroplane": {"airplane"},
    "aircraft": {"airplane"}, "jet": {"airplane"},
    "couch": {"couch"}, "sofa": {"couch"},
    "dining": {"dining table"}, "table": {"dining table"},
    "food": {"banana", "apple", "sandwich", "orange", "broccoli",
             "carrot", "hot dog", "pizza", "donut", "cake"},
    "fruit": {"banana", "apple", "orange"},
    "drink": {"bottle", "wine glass", "cup"},
    "glass": {"wine glass", "cup"}, "cup": {"cup"},
    "bottle": {"bottle"}, "bag": {"backpack", "handbag", "suitcase"},
    "scissors": {"scissors"}, "scissor": {"scissors"},
    "tool": {"scissors", "knife"}, "knife": {"knife"},
    "chair": {"chair", "couch"}, "seat": {"chair", "couch"},
    "table": {"dining table"}, "desk": {"dining table"},
    "bed": {"bed"}, "toilet": {"toilet"},
    "kitchen": {"oven", "microwave", "refrigerator", "sink"},
    "fruit": {"banana", "apple", "orange"},
    "appliance": {"oven", "microwave", "refrigerator", "toaster"},
    "sports": {"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "frisbee", "skis", "snowboard"},
    "ball": {"sports ball"},
    "building": {"house", "office", "tower", "building", "structure", "roof", "shed"},
    "road": {"road", "street", "highway", "path", "lane", "track", "way"},
    "water": {"river", "lake", "ocean", "pond", "water", "sea", "stream", "canal"},
    "land": {"field", "ground", "grass", "terrain", "soil", "dirt", "area"},
    "forest": {"tree", "woods", "forest", "jungle", "vegetation", "canopy"},
}


def _get_model():
    global _yolo_model
    if _yolo_model is None and _YOLO_AVAILABLE:
        _yolo_model = YOLO(str(_MODEL_PATH))  # downloads on first call
    return _yolo_model


def _extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from a question/answer string.
    Strips punctuation, lowercases, and removes stop-words.
    """
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    kws = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return kws if kws else [text.lower().strip()]


def _keyword_matches_class(keywords: list[str], cls_name: str) -> bool:
    """
    Check if any keyword matches the YOLO class name, either directly,
    via singular/plural fuzzy matching, or via the synonym table.
    """
    cls_lower = cls_name.lower()
    # Normalize class name (e.g. "cell phone" -> "cellphone")
    cls_norm = cls_lower.replace(" ", "")
    
    for kw in keywords:
        kw_norm = kw.replace(" ", "")
        
        # 1. Direct or fuzzy plural match
        if kw_norm == cls_norm or kw_norm + "s" == cls_norm or cls_norm + "s" == kw_norm:
            return True
            
        # 2. Direct substring match
        if kw_norm in cls_norm or cls_norm in kw_norm:
            return True
            
        # 3. Synonym expansion
        aliases = _SYNONYMS.get(kw)
        if aliases:
            # Check if any alias matches the class (fuzzily)
            for alias in aliases:
                a_norm = alias.lower().replace(" ", "")
                if a_norm == cls_norm or a_norm + "s" == cls_norm or cls_norm + "s" == a_norm:
                    return True
                    
        # 4. Reverse synonym check: if the class name is a known synonym
        for syn_key, syn_vals in _SYNONYMS.items():
            if cls_lower == syn_key or cls_lower in syn_vals:
                if kw in syn_key or syn_key in kw:
                    return True
    return False


def detect_and_draw(pil_image: Image.Image, keyword_text: str) -> str | None:
    """
    Run YOLOv8 on pil_image.
    - ONLY draw objects whose class name matches keywords from keyword_text.
    - Non-matching objects are completely hidden.
    Returns a base64-encoded PNG string, or None on failure.
    """
    model = _get_model()
    if model is None:
        return None

    try:
        # Increase confidence threshold to 0.3 for clearer results
        results = model(pil_image, conf=0.3, verbose=False)[0]
    except Exception:
        return None

    if results is None or results.boxes is None or len(results.boxes) == 0:
        return None

    # Translate to English if text is non-English for proper keyword matching
    try:
        from backend.models.vqa import _translate
        # Extract language code if possible, or just use 'auto'
        keyword_text_en = _translate(keyword_text, "auto", "en")
    except Exception:
        keyword_text_en = keyword_text

    keywords = _extract_keywords(keyword_text_en)

    # First pass: collect only matched boxes
    matched_boxes = []
    for box in results.boxes:
        cls_id   = int(box.cls[0].item())
        cls_name = model.names[cls_id].lower()
        conf     = float(box.conf[0].item())
        coords   = [int(v) for v in box.xyxy[0].tolist()]

        if _keyword_matches_class(keywords, cls_name):
            matched_boxes.append((cls_name, conf, coords))

    # Sort by confidence descending
    matched_boxes.sort(key=lambda x: x[1], reverse=True)

    # If nothing matched the keyword, fallback to top detected boxes
    if not matched_boxes:
        for box in results.boxes:
            cls_id   = int(box.cls[0].item())
            cls_name = model.names[cls_id].lower()
            conf     = float(box.conf[0].item())
            coords   = [int(v) for v in box.xyxy[0].tolist()]
            matched_boxes.append((cls_name, conf, coords))
        matched_boxes.sort(key=lambda x: x[1], reverse=True)

    # Limit to top 5 objects to avoid clutter
    matched_boxes = matched_boxes[:5]

    if not matched_boxes:
        return None

    img_draw = pil_image.copy().convert("RGBA")
    overlay  = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    # Highlight colour
    box_color   = (99, 241, 130, 255)
    fill_color  = (99, 241, 130, 32)
    label_bg    = (15, 15, 20, 220)
    text_color  = (255, 255, 255, 255)

    for cls_name, conf, (x1, y1, x2, y2) in matched_boxes:
        # Thick bounding box
        thickness = 3
        for t in range(thickness):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=box_color
            )

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        # Label background + text
        label = f"{cls_name} {conf:.0%}"
        lx, ly = x1, max(y1 - 22, 0)
        draw.rectangle([lx, ly, lx + len(label) * 7 + 8, ly + 20], fill=label_bg)
        draw.text((lx + 4, ly + 3), label, fill=text_color)

    composited = Image.alpha_composite(img_draw, overlay).convert("RGB")

    buf = io.BytesIO()
    composited.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
