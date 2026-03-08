# 01. Contribution Reframing

## Current state
The project currently demonstrates strong competition engineering (best public score 0.92666), but paper acceptance requires clear scientific contribution beyond score.

## Target contribution statements (use in paper)
1. **Small-sample robust FER pipeline for pets**
   - Multi-backbone feature extraction (DINOv2/EfficientNet/CLIP) + OOF-driven stacking + safety gates.
2. **Risk-controlled decision framework**
   - Drift-aware submission policy and fallback mechanisms that reduce online regression risk.
3. **Evidence-driven module design**
   - Temperature calibration, multi-view consistency gate, and noise-aware reweighting with ablation proof.

## What to avoid
- Do not claim novelty purely from leaderboard score.
- Do not present heuristic modules without ablation and statistical significance.
- Do not hide failed variants; convert them into insight.

## Required claim-evidence mapping
For each claimed contribution, attach:
1. A table or figure id.
2. The exact script/version used.
3. Manifest and metric json path.

Example mapping line:
- Claim: calibration improves robustness.
- Evidence: Table 3 (`v20` vs `v10` on repeated 5-seed OOF, mean±std), files: `v20_oof_metrics.json`, `v10_oof_metrics.json`.

## Minimal publishable narrative
1. Problem: small-sample pet facial expression recognition is unstable under distribution shift.
2. Method: robust fusion + calibration + consistency + safety controls.
3. Result: higher stability and lower regression risk under constrained-data conditions.
4. Insight: stability-aware design can outperform brute-force complexity increase in low-data settings.
