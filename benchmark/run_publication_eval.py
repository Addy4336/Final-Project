import json
import math
import random
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import _auto_route
from backend.models.ocr import resolve_ocr_answer, run_advanced_ocr
from backend.models.satellite import detect_satellite_objects
from backend.models.vqa import get_model, simulate_base_prediction

SEED = 42
OUTPUT_DIR = Path("tmp/benchmark_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIELD_KEYS = ["name", "emp code", "dob", "blood group", "address"]
FIELD_QUESTIONS = {
    "name": "What is the name?",
    "emp code": "What is the emp code?",
    "dob": "What is the date of birth?",
    "blood group": "What is the blood group?",
    "address": "What is the address?",
}


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"+", "-", "/", " ", "\n"}:
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def _levenshtein(seq1, seq2) -> int:
    if seq1 == seq2:
        return 0
    if len(seq1) == 0:
        return len(seq2)
    if len(seq2) == 0:
        return len(seq1)

    prev = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1, start=1):
        curr = [i]
        for j, c2 in enumerate(seq2, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if c1 == c2 else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _cer(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / max(1, len(ref))


def _wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference).split()
    hyp = _normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / max(1, len(ref))


def _bin_ece(pairs, bins: int = 10) -> float:
    if not pairs:
        return 0.0
    ece = 0.0
    n = len(pairs)
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [(c, y) for c, y in pairs if (c >= lo and c < hi) or (i == bins - 1 and c == 1.0)]
        if not bucket:
            continue
        conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        ece += abs(acc - conf) * (len(bucket) / n)
    return ece


def _brier_score(pairs) -> float:
    if not pairs:
        return 0.0
    return sum((c - y) ** 2 for c, y in pairs) / len(pairs)


def _load_font(size: int = 36):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    ]
    for cand in candidates:
        if cand.exists():
            return ImageFont.truetype(str(cand), size=size)
    return ImageFont.load_default()


def _make_document_case(index: int, variant: str, rng: random.Random):
    names = ["ALICE SHARMA", "RAHUL MEHTA", "KIRAN PATEL", "NISHA IYER"]
    bloods = ["A+", "B+", "O+", "AB-"]
    cities = ["Ahmedabad", "Vadodara", "Surat", "Rajkot"]

    name = names[index % len(names)]
    emp_code = f"TERF{100 + index}"
    dob = f"{10 + (index % 18):02d}/0{1 + (index % 8)}/{1993 + (index % 8)}"
    blood = bloods[index % len(bloods)]
    city = cities[index % len(cities)]
    address = f"{20 + index} River View Road, {city}"

    fields = {
        "name": name,
        "emp code": emp_code,
        "dob": dob,
        "blood group": blood,
        "address": address,
    }

    lines = [
        f"Name: {name}",
        f"Emp Code: {emp_code}",
        f"DOB: {dob}",
        f"Blood Group: {blood}",
        f"Address: {address}",
    ]
    full_text = "\n".join(lines)

    img = Image.new("RGB", (1280, 760), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    font = _load_font(36)
    y = 90
    for line in lines:
        draw.text((90, y), line, fill=(18, 18, 20), font=font)
        y += 100

    if variant == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    elif variant == "low_contrast":
        img = ImageEnhance.Contrast(img).enhance(0.58)
    elif variant == "shadow":
        overlay = Image.new("RGB", img.size, (30, 30, 30))
        img = Image.blend(img, overlay, alpha=0.22)
    elif variant == "jpeg":
        tmp = OUTPUT_DIR / f"_tmp_{index}_{variant}.jpg"
        img.save(tmp, quality=35)
        img = Image.open(tmp).convert("RGB")
        tmp.unlink(missing_ok=True)
    elif variant == "skew":
        img = img.rotate(3.2, expand=True, fillcolor=(248, 248, 246))

    return {
        "id": f"doc_{variant}_{index}",
        "variant": variant,
        "image": img,
        "fields": fields,
        "full_text": full_text,
    }


def _make_satellite_case(index: int, rng: random.Random):
    img = Image.new("RGB", (640, 640), (148, 128, 94))
    draw = ImageDraw.Draw(img)

    has_water = (index % 2 == 0)
    has_vegetation = True
    has_built = (index % 3 != 0)
    has_road = (index % 2 == 1)

    if has_vegetation:
        draw.rectangle([40, 40, 300, 260], fill=(62, 136, 67))
        draw.rectangle([320, 330, 620, 620], fill=(58, 130, 62))
    if has_water:
        draw.rectangle([340, 70, 610, 280], fill=(42, 98, 178))
    if has_built:
        for x in range(90, 560, 100):
            draw.rectangle([x, 350, x + 50, 405], fill=(165, 165, 170))
    if has_road:
        draw.line([20, 500, 620, 500], fill=(225, 225, 225), width=9)
        draw.line([450, 20, 450, 620], fill=(220, 220, 220), width=8)

    truth = set()
    if has_water:
        truth.add("water")
    if has_vegetation:
        truth.add("vegetation")
    if has_built:
        truth.add("built_up")
    if has_road:
        truth.add("road")

    return {
        "id": f"sat_{index}",
        "image": img,
        "truth_features": truth,
    }


def _make_vqa_color_case(color_name: str, rgb):
    img = Image.new("RGB", (320, 320), rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 240, 240], outline=(255, 255, 255), width=3)
    return {
        "id": f"vqa_{color_name}",
        "image": img,
        "question": "What color is visible in this image?",
        "expected": color_name,
    }


def evaluate_ocr(rng: random.Random, variants: list[str], cases_per_variant: int):
    cases = []
    for v in variants:
        for i in range(cases_per_variant):
            cases.append(_make_document_case(i, v, rng))

    per_case = []
    calibration_pairs = []
    total_pred = 0
    total_correct = 0
    total_gt = len(cases) * len(FIELD_KEYS)

    for case in cases:
        t0 = time.perf_counter()
        res = run_advanced_ocr(case["image"].copy(), question="Read all text")
        latency_ms = (time.perf_counter() - t0) * 1000
        text = res.get("text", "")
        fields = res.get("fields", {})
        quality = res.get("quality", "None")

        case_correct_fields = 0
        predicted_fields = 0
        for k in FIELD_KEYS:
            gt = _normalize_text(case["fields"][k])
            pred = _normalize_text(fields.get(k, ""))
            if pred:
                predicted_fields += 1
                total_pred += 1
            if pred and (pred in gt or gt in pred):
                case_correct_fields += 1
                total_correct += 1

        qa_hits = 0
        for k, q in FIELD_QUESTIONS.items():
            ans = resolve_ocr_answer(q, text, fields)
            if _normalize_text(case["fields"][k]) in _normalize_text(ans):
                qa_hits += 1

        if quality == "High" and len(fields) >= 2:
            confidence = 0.93
        elif quality in ("High", "Medium") and len(fields) >= 1:
            confidence = 0.87
        elif text and len(text) > 20:
            confidence = 0.75
        else:
            confidence = 0.425

        correctness = 1 if (case_correct_fields / len(FIELD_KEYS)) >= 0.8 else 0
        calibration_pairs.append((confidence, correctness))

        per_case.append(
            {
                "id": case["id"],
                "variant": case["variant"],
                "cer": _cer(case["full_text"], text),
                "wer": _wer(case["full_text"], text),
                "field_recall": case_correct_fields / len(FIELD_KEYS),
                "qa_accuracy": qa_hits / len(FIELD_QUESTIONS),
                "quality": quality,
                "latency_ms": latency_ms,
            }
        )

    precision = total_correct / total_pred if total_pred else 0.0
    recall = total_correct / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "cases": per_case,
        "summary": {
            "count": len(cases),
            "mean_cer": float(np.mean([c["cer"] for c in per_case])),
            "mean_wer": float(np.mean([c["wer"] for c in per_case])),
            "field_precision": precision,
            "field_recall": recall,
            "field_f1": f1,
            "qa_accuracy": float(np.mean([c["qa_accuracy"] for c in per_case])),
            "mean_latency_ms": float(np.mean([c["latency_ms"] for c in per_case])),
        },
        "calibration_pairs": calibration_pairs,
    }


def evaluate_mode_routing(rng: random.Random, per_type_cases: int):
    docs = [_make_document_case(i, "clean", rng)["image"] for i in range(per_type_cases)]
    sats = [_make_satellite_case(i, rng)["image"] for i in range(per_type_cases)]
    vqa_pool = [
        _make_vqa_color_case("red", (220, 45, 45))["image"],
        _make_vqa_color_case("orange", (230, 120, 45))["image"],
        _make_vqa_color_case("pink", (235, 110, 160))["image"],
        _make_vqa_color_case("gray", (140, 140, 140))["image"],
        _make_vqa_color_case("black", (40, 40, 40))["image"],
        _make_vqa_color_case("white", (230, 230, 230))["image"],
        _make_vqa_color_case("yellow", (220, 190, 60))["image"],
        _make_vqa_color_case("purple", (120, 70, 155))["image"],
    ]
    vqas = vqa_pool[:per_type_cases]

    cases = []
    for img in docs:
        cases.append((img, "OCR"))
    for img in sats:
        cases.append((img, "SATELLITE"))
    for img in vqas:
        cases.append((img, "VQA"))

    matrix = {"OCR": {"OCR": 0, "SATELLITE": 0, "VQA": 0}, "SATELLITE": {"OCR": 0, "SATELLITE": 0, "VQA": 0}, "VQA": {"OCR": 0, "SATELLITE": 0, "VQA": 0}}
    correct = 0
    for img, expected in cases:
        pred, _ = _auto_route(img.copy(), "What is visible in this image?")
        matrix[expected][pred] += 1
        if pred == expected:
            correct += 1

    return {
        "count": len(cases),
        "accuracy": correct / len(cases),
        "confusion_matrix": matrix,
    }


def evaluate_satellite(rng: random.Random, case_count: int):
    cases = [_make_satellite_case(i, rng) for i in range(case_count)]
    tp = 0
    fp = 0
    fn = 0
    per_case = []
    calibration_pairs = []

    for case in cases:
        det = detect_satellite_objects(case["image"].copy())
        pred_features = {f.get("type") for f in det.get("features", [])}
        gt = case["truth_features"]

        tpi = len(pred_features & gt)
        fpi = len(pred_features - gt)
        fni = len(gt - pred_features)
        tp += tpi
        fp += fpi
        fn += fni

        pred_confs = [float(f.get("confidence", 0.0)) for f in det.get("features", [])]
        conf = (sum(pred_confs) / len(pred_confs) / 100.0) if pred_confs else 0.0
        case_f1 = (2 * tpi / max(1, 2 * tpi + fpi + fni))
        correctness = 1 if case_f1 >= 0.6 else 0
        calibration_pairs.append((max(0.0, min(1.0, conf)), correctness))

        per_case.append(
            {
                "id": case["id"],
                "gt": sorted(list(gt)),
                "pred": sorted(list(pred_features)),
                "case_f1": case_f1,
                "confidence": conf,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "cases": per_case,
        "summary": {
            "count": len(cases),
            "feature_precision": precision,
            "feature_recall": recall,
            "feature_f1": f1,
        },
        "calibration_pairs": calibration_pairs,
    }


def evaluate_vqa(color_count: int):
    all_colors = [
        _make_vqa_color_case("red", (230, 50, 50)),
        _make_vqa_color_case("green", (55, 170, 70)),
        _make_vqa_color_case("blue", (50, 95, 205)),
        _make_vqa_color_case("yellow", (220, 205, 60)),
        _make_vqa_color_case("black", (35, 35, 35)),
        _make_vqa_color_case("white", (235, 235, 235)),
    ]
    color_cases = all_colors[:color_count]

    model = get_model("finetuned")
    per_case = []
    calibration_pairs = []
    ft_hits = 0
    base_hits = 0

    for case in color_cases:
        ft = model.predict(case["image"].copy(), case["question"], top_k=3, lang="en-US")
        base = simulate_base_prediction(case["image"].copy(), case["question"], top_k=3)

        ft_ans = (ft.get("answers") or [{}])[0].get("answer", "")
        ft_conf = float((ft.get("answers") or [{}])[0].get("confidence", 0.0)) / 100.0
        base_ans = (base.get("answers") or [{}])[0].get("answer", "")

        ft_ok = case["expected"] in _normalize_text(ft_ans)
        base_ok = case["expected"] in _normalize_text(base_ans)
        ft_hits += int(ft_ok)
        base_hits += int(base_ok)
        calibration_pairs.append((max(0.0, min(1.0, ft_conf)), 1 if ft_ok else 0))

        per_case.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "ft_answer": ft_ans,
                "ft_confidence": ft_conf,
                "ft_correct": ft_ok,
                "base_answer": base_ans,
                "base_correct": base_ok,
            }
        )

    return {
        "cases": per_case,
        "summary": {
            "count": len(color_cases),
            "ft_exact_match": ft_hits / len(color_cases),
            "base_exact_match": base_hits / len(color_cases),
        },
        "calibration_pairs": calibration_pairs,
    }


def evaluate_doc_qa_ablation(rng: random.Random, doc_count: int):
    model = get_model("finetuned")
    docs = [_make_document_case(i, "clean", rng) for i in range(doc_count)]
    question = "What is the name?"

    ocr_hits = 0
    vqa_hits = 0
    hybrid_hits = 0

    for case in docs:
        ocr_res = run_advanced_ocr(case["image"].copy(), question=question)
        ocr_text = ocr_res.get("text", "")
        ocr_fields = ocr_res.get("fields", {})

        ocr_ans = resolve_ocr_answer(question, ocr_text, ocr_fields)
        vqa_ans = (model.predict(case["image"].copy(), question, top_k=1).get("answers") or [{}])[0].get("answer", "")

        gt = _normalize_text(case["fields"]["name"])
        ocr_ok = gt in _normalize_text(ocr_ans)
        vqa_ok = gt in _normalize_text(vqa_ans)

        # Hybrid policy mirrors current system intent: prefer OCR when fields exist.
        hybrid_ans = ocr_ans if ocr_fields else vqa_ans
        hybrid_ok = gt in _normalize_text(hybrid_ans)

        ocr_hits += int(ocr_ok)
        vqa_hits += int(vqa_ok)
        hybrid_hits += int(hybrid_ok)

    n = len(docs)
    return {
        "count": n,
        "ocr_only_em": ocr_hits / n,
        "vqa_only_em": vqa_hits / n,
        "hybrid_em": hybrid_hits / n,
    }


def run_all(profile: str = "quick"):
    rng = random.Random(SEED)
    np.random.seed(SEED)

    if profile == "quick":
        cfg = {
            "ocr_variants": ["clean", "blur"],
            "ocr_cases_per_variant": 1,
            "mode_per_type": 2,
            "sat_cases": 3,
            "vqa_colors": 3,
            "ablation_docs": 1,
        }
    else:
        cfg = {
            "ocr_variants": ["clean", "blur", "low_contrast", "shadow", "jpeg", "skew"],
            "ocr_cases_per_variant": 4,
            "mode_per_type": 8,
            "sat_cases": 12,
            "vqa_colors": 6,
            "ablation_docs": 8,
        }

    t0 = time.perf_counter()
    ocr = evaluate_ocr(rng, cfg["ocr_variants"], cfg["ocr_cases_per_variant"])
    routing = evaluate_mode_routing(rng, cfg["mode_per_type"])
    sat = evaluate_satellite(rng, cfg["sat_cases"])
    vqa = evaluate_vqa(cfg["vqa_colors"])
    ablation = evaluate_doc_qa_ablation(rng, cfg["ablation_docs"])

    calibration_pairs = []
    calibration_pairs.extend(ocr["calibration_pairs"])
    calibration_pairs.extend(sat["calibration_pairs"])
    calibration_pairs.extend(vqa["calibration_pairs"])

    ece = _bin_ece(calibration_pairs, bins=10)
    brier = _brier_score(calibration_pairs)

    publication_gate = {
        "ocr_field_f1": {"value": ocr["summary"]["field_f1"], "target": 0.80, "pass": ocr["summary"]["field_f1"] >= 0.80},
        "ocr_mean_cer": {"value": ocr["summary"]["mean_cer"], "target": 0.35, "pass": ocr["summary"]["mean_cer"] <= 0.35},
        "vqa_ft_exact_match": {"value": vqa["summary"]["ft_exact_match"], "target": 0.80, "pass": vqa["summary"]["ft_exact_match"] >= 0.80},
        "mode_routing_accuracy": {"value": routing["accuracy"], "target": 0.80, "pass": routing["accuracy"] >= 0.80},
        "satellite_feature_f1": {"value": sat["summary"]["feature_f1"], "target": 0.70, "pass": sat["summary"]["feature_f1"] >= 0.70},
        "calibration_ece": {"value": ece, "target": 0.25, "pass": ece <= 0.25},
    }

    all_pass = all(item["pass"] for item in publication_gate.values())

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": SEED,
        "profile": profile,
        "runtime_seconds": time.perf_counter() - t0,
        "ocr": ocr["summary"],
        "mode_routing": routing,
        "satellite": sat["summary"],
        "vqa": vqa["summary"],
        "doc_qa_ablation": ablation,
        "calibration": {
            "ece": ece,
            "brier": brier,
            "samples": len(calibration_pairs),
        },
        "publication_gate": publication_gate,
        "ready_for_publication": all_pass,
    }

    json_path = OUTPUT_DIR / "publication_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Publication Evaluation Report",
        "",
        f"- Timestamp (UTC): {report['timestamp']}",
        f"- Seed: {SEED}",
        f"- Runtime (s): {report['runtime_seconds']:.2f}",
        "",
        "## Key Metrics",
        "",
        f"- OCR mean CER: {report['ocr']['mean_cer']:.4f}",
        f"- OCR field F1: {report['ocr']['field_f1']:.4f}",
        f"- OCR QA accuracy: {report['ocr']['qa_accuracy']:.4f}",
        f"- Mode routing accuracy: {report['mode_routing']['accuracy']:.4f}",
        f"- Satellite feature F1: {report['satellite']['feature_f1']:.4f}",
        f"- VQA fine-tuned exact match: {report['vqa']['ft_exact_match']:.4f}",
        f"- Calibration ECE: {report['calibration']['ece']:.4f}",
        f"- Brier score: {report['calibration']['brier']:.4f}",
        "",
        "## Ablation (Document Name QA)",
        "",
        f"- OCR-only EM: {report['doc_qa_ablation']['ocr_only_em']:.4f}",
        f"- VQA-only EM: {report['doc_qa_ablation']['vqa_only_em']:.4f}",
        f"- Hybrid EM: {report['doc_qa_ablation']['hybrid_em']:.4f}",
        "",
        "## Publication Gate",
        "",
    ]

    for key, item in publication_gate.items():
        state = "PASS" if item["pass"] else "FAIL"
        md_lines.append(f"- {key}: {state} (value={item['value']:.4f}, target={item['target']:.4f})")

    md_lines.extend([
        "",
        f"## Final Verdict",
        "",
        "READY" if all_pass else "NOT READY",
        "",
        f"Machine-readable report: {json_path.as_posix()}",
    ])

    md_path = OUTPUT_DIR / "publication_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({
        "report_json": json_path.as_posix(),
        "report_md": md_path.as_posix(),
        "ready_for_publication": all_pass,
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run publication-readiness evaluation.")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    run_all(profile=args.profile)
