"""
VisionQuery â€“ Satellite Image Analysis Extension Layer.

Lightweight, rule-based satellite/aerial image analysis using OpenCV and NumPy.
Provides: object detection, land use classification, density analysis,
visual grounding, satellite mode detection, zoom-based region analysis,
OCR integration, and explanation output.

This module is an EXTENSION LAYER. It does NOT modify or replace any
existing VQA, OCR, or detection logic.
"""
import io
import base64
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFont


# â”€â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# HSV ranges for satellite feature detection
_HSV_RANGES = {
    "water": {
        "lower": np.array([85, 30, 20]),
        "upper": np.array([145, 255, 255]),
        "color": (41, 128, 185),       # Blue overlay
        "label": "Water Body",
    },
    "vegetation": {
        "lower": np.array([30, 30, 20]),
        "upper": np.array([90, 255, 255]),
        "color": (39, 174, 96),         # Green overlay
        "label": "Vegetation / Forest",
    },
    "bare_land": {
        "lower": np.array([10, 20, 70]),
        "upper": np.array([35, 180, 250]),
        "color": (160, 120, 60),        # Brown overlay
        "label": "Bare Land / Soil",
    },
    "built_up": {
        "lower": np.array([0, 0, 80]),
        "upper": np.array([180, 50, 220]),
        "color": (192, 57, 43),         # Red overlay
        "label": "Urban / Built-up Area",
    },
}

# Minimum evidence required before a feature is allowed into the answer.
_MIN_FEATURE_RATIO = {
    "water": 0.10,
    "vegetation": 0.10,
    "built_up": 0.10,
}
_MIN_ROAD_LINES = 2

# Analysis resolution (resize for speed)
_ANALYSIS_SIZE = (400, 400)


# â”€â”€â”€ Satellite Mode Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def is_satellite_image(pil_image: Image.Image) -> dict:
    """
    Heuristic detection of whether an image is satellite/aerial.
    """
    if not _CV2_AVAILABLE:
        return {"is_satellite": False, "confidence": 0, "reasoning": "OpenCV not available"}

    try:
        arr = np.array(pil_image.resize(_ANALYSIS_SIZE)).astype(np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        total_px = _ANALYSIS_SIZE[0] * _ANALYSIS_SIZE[1]

        # Feature coverage
        water_mask = cv2.inRange(hsv, _HSV_RANGES["water"]["lower"], _HSV_RANGES["water"]["upper"])
        veg_mask = cv2.inRange(hsv, _HSV_RANGES["vegetation"]["lower"], _HSV_RANGES["vegetation"]["upper"])
        land_mask = cv2.inRange(hsv, _HSV_RANGES["bare_land"]["lower"], _HSV_RANGES["bare_land"]["upper"])
        built_mask = cv2.inRange(hsv, _HSV_RANGES["built_up"]["lower"], _HSV_RANGES["built_up"]["upper"])

        water_r = np.sum(water_mask > 0) / total_px
        veg_r = np.sum(veg_mask > 0) / total_px
        land_r = np.sum(land_mask > 0) / total_px
        built_r = np.sum(built_mask > 0) / total_px
        combined_natural = water_r + veg_r + land_r

        # Edge density (satellite images typically have lots of fine edges)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_px

        # Reject blank/document-like scenes, but avoid false negatives on
        # low-texture aerial scenes that still have strong natural coverage.
        white_bg_mask = cv2.inRange(arr, np.array([240, 240, 240]), np.array([255, 255, 255]))
        white_bg_ratio = np.sum(white_bg_mask > 0) / total_px
        if (white_bg_ratio > 0.60 and combined_natural < 0.08 and built_r < 0.08) or \
           (edge_density < 0.02 and combined_natural < 0.12 and built_r < 0.08):
            return {"is_satellite": False, "confidence": 0, "reasoning": "Standard document or empty background detected"}

        # Scoring
        score = 0
        reasons = []

        if combined_natural > 0.30:
            score += 25
            reasons.append(f"Natural features confirmed")

        if veg_r > 0.10: score += 15
        if water_r > 0.05: score += 15
        if built_r > 0.10: score += 10

        # Texture analysis for aerial view
        sat_std = np.std(hsv[:, :, 1])
        if sat_std < 50:
            score += 15
            reasons.append("Top-down texture pattern detected")

        if 0.05 < edge_density < 0.40:
            score += 10

        return {
            "is_satellite": score >= 35, # Slightly lower threshold for "over-detect" sensitivity
            "confidence": score,
            "reasoning": "; ".join(reasons) if reasons else "General scene"
        }
    except Exception as e:
        return {"is_satellite": False, "confidence": 0, "reasoning": f"Detection error: {str(e)}"}


# â”€â”€â”€ Satellite Object Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_satellite_objects(pil_image: Image.Image, region: dict = None) -> dict:
    """
    Detect basic satellite elements with conditional sensitivity.
    """
    if not _CV2_AVAILABLE:
        return {"features": [], "summary": "OpenCV not available"}

    try:
        # Crop to region if specified
        if region:
            w_img, h_img = pil_image.size
            x = int(region.get("x", 0) * w_img)
            y = int(region.get("y", 0) * h_img)
            w = int(region.get("w", 1.0) * w_img)
            h = int(region.get("h", 1.0) * h_img)
            pil_image = pil_image.crop((x, y, x + w, y + h))

        arr = np.array(pil_image.resize(_ANALYSIS_SIZE)).astype(np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        total_px = _ANALYSIS_SIZE[0] * _ANALYSIS_SIZE[1]

        features = []

        # HSV-based features (clear natural/built targets only)
        for feat_key, params in _HSV_RANGES.items():
            if feat_key == "bare_land":
                continue
            mask = cv2.inRange(hsv, params["lower"], params["upper"])
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Pattern verification for Buildings
            if feat_key == "built_up":
                # Check for structures / residential blocks
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_rects = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 30: continue # Lowered from 80
                    valid_rects.append(cnt)
                
                coverage = np.sum(mask > 0) / total_px
                if coverage >= 0.10: # >10% coverage required
                    features.append({
                        "type": feat_key,
                        "coverage_pct": round(coverage * 100, 1),
                        "contour_count": len(valid_rects),
                        "label": "Buildings / Urban structures",
                        "confidence": round(min(96, 60 + coverage * 1000 + len(valid_rects) * 0.5), 1),
                    })
            else:
                coverage = np.sum(mask > 0) / total_px
                if coverage >= _MIN_FEATURE_RATIO[feat_key]:
                    features.append({
                        "type": feat_key,
                        "coverage_pct": round(coverage * 100, 1),
                        "contour_count": 0, # Placeholder for natural
                        "label": params["label"],
                        "confidence": round(min(95, 60 + coverage * 500), 1),
                    })

        # Strict Road Detection (Linear structures)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=80, maxLineGap=10)
        
        if lines is not None:
            road_mask = np.zeros(gray.shape, dtype=np.uint8)
            # Filter: keep only lines with reasonable angles (roads are mostly straight)
            filtered = []
            for line in lines:
                x1,y1,x2,y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                # Keep roughly horizontal or vertical lines (roads)
                if angle < 30 or angle > 150 or (60 < angle < 120):
                    filtered.append(line)
            
            lines = filtered if filtered else lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(road_mask, (x1, y1), (x2, y2), 255, 3)
            
            road_coverage = np.sum(road_mask > 0) / total_px
            # Cap road coverage at realistic maximum (roads rarely exceed 15%)
            road_coverage = min(road_coverage, 0.15)
            
            # FIX 3: ROAD DETECTION (length > 100px)
            road_count = 0
            if filtered:
                for line in filtered:
                    x1,y1,x2,y2 = line[0]
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if dist > 100:
                        road_count += 1

            if road_count >= _MIN_ROAD_LINES and road_coverage >= 0.10: 
                features.append({
                    "type": "road",
                    "coverage_pct": round(road_coverage * 100, 1),
                    "contour_count": road_count,
                    "label": "Roads / Infrastructure",
                    "confidence": round(min(94, 62 + road_count * 5 + road_coverage * 300), 1),
                })

        # Remove duplicate types if any
        seen = set()
        clean_features = []
        for f in features:
            if f["type"] not in seen:
                clean_features.append(f)
                seen.add(f["type"])

        return {"features": clean_features, "summary": "Analysis complete"}

    except Exception:
        return {"features": [], "summary": "Analysis error"}


# â”€â”€â”€ Land Use Classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def classify_land_use(features: list) -> dict:
    """
    Classify land use based on detected features.
    """
    if not features:
        return {"classification": "Mixed Terrain", "confidence": 0, "reasoning": "No features"}

    coverage = {f["type"]: f["coverage_pct"] for f in features}
    built = coverage.get("built_up", 0)
    veg = coverage.get("vegetation", 0)
    water = coverage.get("water", 0)
    land = coverage.get("bare_land", 0)
    road = coverage.get("road", 0)

    if built > 20 or road > 5:
        return {"classification": "Urban / Built-up", "confidence": 85, "reasoning": "Developed infrastructure detected"}
    if veg > 20:
        return {"classification": "Agricultural / Vegetation", "confidence": 90, "reasoning": "Dominant vegetation coverage"}
    if water > 15:
        return {"classification": "Coastal / Waterbody", "confidence": 90, "reasoning": "Significant water surface"}
    
    return {"classification": "Mixed Terrain", "confidence": 60, "reasoning": "Diverse spectral signatures detected"}


# â”€â”€â”€ Area Density Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def analyze_density(features: list) -> dict:
    """
    Estimate area density based on structures.
    """
    built = next((f["coverage_pct"] for f in features if f["type"] == "built_up"), 0)
    road = next((f["coverage_pct"] for f in features if f["type"] == "road"), 0)
    
    score = built * 2 + road * 5
    if score > 50: return {"density": "High Development", "score": score}
    if score > 15: return {"density": "Moderate Development", "score": score}
    return {"density": "Low / Sparse", "score": score}


# â”€â”€â”€ Visual Grounding (Overlay Generation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_visual_grounding(pil_image: Image.Image, target_features: list = None) -> str:
    """
    Draw overlays on detected satellite features.
    """
    if not _CV2_AVAILABLE:
        return None

    try:
        arr = np.array(pil_image.convert("RGB")).astype(np.uint8)
        resized = cv2.resize(arr, _ANALYSIS_SIZE)
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        
        overlay_img = pil_image.convert("RGBA").copy()
        overlay = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        scale_x = arr.shape[1] / _ANALYSIS_SIZE[0]
        scale_y = arr.shape[0] / _ANALYSIS_SIZE[1]

        for feat_key, params in _HSV_RANGES.items():
            mask = cv2.inRange(hsv, params["lower"], params["upper"])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 150: continue
                x, y, w, h = cv2.boundingRect(cnt)
                draw.rectangle([int(x*scale_x), int(y*scale_y), int((x+w)*scale_x), int((y+h)*scale_y)], 
                               outline=(*params["color"], 255), width=3)

        buf = io.BytesIO()
        Image.alpha_composite(overlay_img, overlay).convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


# â”€â”€â”€ Zoom-Based Region Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def analyze_region(pil_image: Image.Image, region: dict) -> dict:
    """
    Analyze only a selected region of the satellite image.
    """
    obj_result = detect_satellite_objects(pil_image, region=region)
    land_use = classify_land_use(obj_result["features"])
    density = analyze_density(obj_result["features"])

    return {
        "region": region,
        "objects": obj_result,
        "land_use": land_use,
        "density": density,
        "explanation": generate_explanation(obj_result["features"], land_use, density),
    }


# â”€â”€â”€ Explanation Output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_explanation(features: list, land_use: dict = None, density: dict = None) -> str:
    """
    Generate a short reasoning line combining all satellite analysis results.
    """
    if not features:
        return "Mixed terrain. No specific objects >10% coverage detected."

    parts = []
    # Output Format requirements: Object name, coverage %, confidence score
    feat_mentions = [f"{f['label']} ({f['coverage_pct']}%, conf: {f['confidence']})" for f in features]
    if feat_mentions:
        parts.append(f"Detected: {', '.join(feat_mentions)}")

    if land_use and land_use.get("classification") != "Mixed Terrain":
        parts.append(f"Area specialized as {land_use['classification']}")
    else:
        parts.append("Area represents Mixed Terrain")

    explanation = " | ".join(parts) + "."
    return explanation


# â”€â”€â”€ Smart Question Routing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_SAT_QUESTION_KEYWORDS = {
    "road": ["road", "highway", "street", "path", "lane", "route", "interstate", "freeway"],
    "building": ["building", "structure", "house", "roof", "construction", "tower", "shed"],
    "water": ["water", "river", "lake", "ocean", "pond", "stream", "canal", "reservoir"],
    "vegetation": ["vegetation", "forest", "tree", "green", "crop", "field", "grass", "jungle"],
    "land": ["land", "terrain", "soil", "ground", "earth", "area", "surface"],
    "density": ["density", "dense", "crowded", "sparse", "spread", "packed"],
    "classification": ["type", "classify", "urban", "rural", "industrial", "agricultural", "area type"],
    "satellite": ["satellite", "aerial", "bird eye", "overhead", "map", "geographic", "geospatial"],
}


def route_satellite_question(question: str) -> dict:
    """
    Determine if a question is satellite-specific.
    """
    q_lower = question.lower()
    target_features = []
    for category, keywords in _SAT_QUESTION_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            if category in ("road", "building", "water", "vegetation", "land"):
                target_features.append(category)

    return {
        "is_satellite_q": len(target_features) > 0,
        "target_features": target_features
    }


# â”€â”€â”€ Full Satellite Analysis Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_satellite_analysis(pil_image: Image.Image, question: str = "",
                           region: dict = None, ocr_text: str = "") -> dict:
    """
    Entrypoint for complete satellite analysis.
    """
    obj_result = detect_satellite_objects(pil_image, region=region)
    features = obj_result["features"]
    land_use = classify_land_use(features)
    density = analyze_density(features)
    routing = route_satellite_question(question)

    answer = _build_satellite_answer(question, features, land_use, density, routing, ocr_text)
    grounding = generate_visual_grounding(pil_image)

    return {
        "answer": answer,
        "confidence": _compute_confidence(features, land_use),
        "features": features,
        "land_use": land_use,
        "density": density,
        "explanation": "Satellite-only feature detection.",
        "grounding_image": grounding,
        "routing": routing
    }


def _build_satellite_answer(question: str, features: list, land_use: dict,
                             density: dict, routing: dict, ocr_text: str) -> str:
    """
    Build a contextual, non-generic answer based strictly on conditional detection.
    """
    del routing, ocr_text
    q_lower = question.lower()
    strong = [
        f for f in features
        if f.get("confidence", 0) >= 60 and f.get("type") in {"road", "built_up", "vegetation", "water"}
    ]

    if not strong:
        return "Mixed Terrain"

    display = {
        "road": "Road",
        "built_up": "Buildings",
        "vegetation": "Vegetation",
        "water": "Water",
    }

    target_map = {
        "road": ["road", "street", "highway", "lane", "route"],
        "built_up": ["building", "buildings", "structure", "house", "roof", "urban"],
        "vegetation": ["vegetation", "forest", "tree", "green", "crop", "field", "grass"],
        "water": ["water", "river", "lake", "ocean", "pond", "canal", "reservoir"],
    }
    target_type = None
    for feature_type, keywords in target_map.items():
        if any(keyword in q_lower for keyword in keywords):
            target_type = feature_type
            break

    if any(k in q_lower for k in ["type", "classify", "land use", "terrain", "area type"]):
        if land_use.get("classification") == "Unknown":
            return "Unable to determine clearly"
        return f"{land_use['classification']} terrain with {density['density']} density."

    if any(k in q_lower for k in ["coverage", "percent", "percentage", "%"]):
        if target_type:
            target = next((f for f in strong if f["type"] == target_type), None)
            if target:
                return f"{display[target_type]} detected at {target['coverage_pct']}% coverage."
            return "Unable to determine clearly"
        return ", ".join(f"{display[f['type']]} {f['coverage_pct']}%" for f in strong[:3])
    
    if any(k in q_lower for k in ["how many", "count", "number"]):
        if target_type:
            target = next((f for f in strong if f["type"] == target_type), None)
            if target and target.get("contour_count", 0) > 0:
                return str(target["contour_count"])
        return "Unable to determine clearly"

    if target_type:
        target = next((f for f in strong if f["type"] == target_type), None)
        if target:
            return f"{display[target_type]}, {target['coverage_pct']}%, {target['confidence']}"
        return "Mixed Terrain"

    ans_parts = [f"{display[f['type']]}, {f['coverage_pct']}%, {f['confidence']}" for f in strong[:4]]
    return " | ".join(ans_parts)


def _compute_confidence(features: list, land_use: dict) -> float:
    """Compute overall confidence."""
    del land_use
    strong = [f for f in features if f.get("confidence", 0) >= 60]
    if not strong:
        return 0.0
    return round(min(96.0, max(f.get("confidence", 0) for f in strong)), 1)
