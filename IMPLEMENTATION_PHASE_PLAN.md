# VisionQuery PhD-Grade Implementation Plan

## Goal

Evolve the current prototype into a reproducible, publication-ready multimodal reasoning system with measurable novelty in OCR + VQA fusion under real-world noise and remote-sensing complexity.

## Phase 0: Stabilization and Reproducibility (Week 1-2)

Deliverables:
- Freeze runtime setup (`requirements.txt`, environment bootstrap, OCR diagnostics).
- Add deterministic seed handling for all inference/evaluation paths.
- Define a single evaluation runner for OCR, VQA, and hybrid routes.

Success criteria:
- Fresh machine setup in less than 30 minutes.
- Run-to-run output variance documented and bounded.

## Phase 1: OCR Reliability Core (Week 2-4)

Deliverables:
- Add OCR benchmark set with diverse document noise conditions.
- Implement OCR confidence calibration (character-level and field-level).
- Add error taxonomy tags: blur, low contrast, skew, occlusion, multilingual mismatch.

Success criteria:
- Report CER/WER and field extraction F1.
- Measured gains against current hybrid OCR baseline.

## Phase 2: Unified OCR + VQA Fusion (Week 4-7)

Deliverables:
- Introduce a fusion module that combines visual embeddings and OCR token spans.
- Add question-aware text grounding (which OCR spans contributed to final answer).
- Expose evidence traces in API response for explainability.

Success criteria:
- Hybrid model outperforms OCR-only and VQA-only on document QA tasks.
- At least one statistically significant gain on benchmark subsets.

## Phase 3: Remote-Sensing Reasoning Upgrade (Week 7-10)

Deliverables:
- Replace pure HSV heuristics with a lightweight learned segmentation/classification model.
- Add geographic-scene question templates and weakly supervised labels.
- Introduce region-level confidence with uncertainty intervals.

Success criteria:
- Better precision/recall than current rule-based satellite branch.
- Human-verifiable overlays aligned with answer content.

## Phase 4: Scientific Evaluation Framework (Week 10-12)

Deliverables:
- Ablation suite: OCR-only, VQA-only, hybrid-no-attention, hybrid-full.
- Robustness tests: blur, JPEG compression, illumination shift, perspective skew.
- Calibration metrics: ECE, Brier score, confidence-accuracy curves.

Success criteria:
- Reproducible tables and plots generated from one command.
- Clear novelty claim supported by ablation and robustness evidence.

## Phase 5: UI and Human-Centered Validation (Week 12-14)

Deliverables:
- Refactor UI into modular components with explicit evidence panels.
- Add uncertainty and provenance display (answer, confidence, evidence source).
- Conduct user study (task success, trust, explanation usefulness).

Success criteria:
- Improved user task completion and confidence interpretation.
- Reduced misinterpretation of high-confidence wrong answers.

## Phase 6: Thesis and Defense Packaging (Week 14-16)

Deliverables:
- Final reproducibility package: code, data split definitions, experiment configs.
- Thesis-ready visual artifacts: architecture diagram, ablation charts, failure analysis.
- Defense script mapping problem -> gap -> method -> evidence -> contribution.

Success criteria:
- End-to-end rerun from clean clone.
- All major claims traceable to quantitative evidence.

## Non-Negotiable Engineering Standards

- Every API response includes mode, confidence, and evidence metadata.
- Every benchmark result stores commit hash and config snapshot.
- Every major model change requires ablation update.
- No hard-coded confidence inflation in final experimental branch.

## Immediate Next Sprint (This Week)

1. Add automated OCR test set runner with CER/WER/field-F1 outputs.
2. Add structured logging for mode routing and OCR engine diagnostics.
3. Normalize frontend-backend mode contracts and verify with integration tests.
4. Create first ablation report template (baseline vs hybrid).
