# 02. Gap Closure Execution Plan (Step-by-Step)

This is the core implementation plan for missing paper-quality components.

## Phase A: Experiment governance (Day 1)

### Step A1. Freeze canonical baselines
1. Set canonical baselines to `v10` and `v15`.
2. Record exact commits, config hashes, and output files.
3. Create `experiments_registry.csv` with columns:
   - `exp_id, script, commit, seed, mode, oof_acc, oof_macro_f1, drift_vs_v10, notes`
4. Acceptance:
   - Baseline rows are complete and reproducible.

### Step A2. Define fixed random protocol
1. Use seeds: `42, 52, 62, 72, 82` for all new analyses.
2. Fix fold strategy (5-fold stratified).
3. Lock preprocessing and test-time augmentation variants per experiment.
4. Acceptance:
   - Protocol doc exists and every experiment references it.

## Phase B: Mandatory ablations (Week 1-2)

### Step B1. Backbone ablation
1. Run single-backbone variants (DINO-only, EffNet-only, CLIP-only).
2. Run pairwise fusion variants (DINO+EffNet, DINO+CLIP, EffNet+CLIP).
3. Run full 3-backbone baseline.
4. For each, collect:
   - OOF accuracy, macro-F1, and confidence calibration metrics.
5. Acceptance:
   - One table with all rows and same evaluation protocol.

### Step B2. Module ablation (incremental)
Use fixed backbone setup and add modules one-by-one:
1. Base weighted fusion.
2. + meta stack.
3. + temperature calibration.
4. + hybrid alpha blending.
5. + TTA consistency gate.
6. + noise reweight.
7. + pseudo gate (off/on controlled).
8. Acceptance:
   - Delta table (`+/-` from previous row), with mean±std across seeds.

### Step B3. Safety ablation
1. Evaluate with and without drift fallback.
2. Evaluate with and without low-OOF fallback.
3. Measure regression rate against anchor submission (`v10`).
4. Acceptance:
   - Report regression count and ratio under each safety policy.

## Phase C: Statistical reliability (Week 2-3)

### Step C1. Multi-run reporting standard
1. For every core experiment, run 5 seeds.
2. Report mean, std, min, max.
3. Include confidence intervals where possible.
4. Acceptance:
   - Main result table uses mean±std, not single best.

### Step C2. Significance testing
1. For paired comparisons (e.g., `v10` vs `v22`), perform paired tests over fold-level scores.
2. Report p-values and effect size.
3. Avoid over-claiming if significance is weak.
4. Acceptance:
   - Statistical appendix table completed.

## Phase D: Robustness and error analysis (Week 3-4)

### Step D1. Slice analysis
1. Create stratified slices:
   - species proxy cluster,
   - head off-center severity,
   - low-light/high-blur proxy,
   - confidence bins.
2. Report per-slice performance and failure concentration.
3. Acceptance:
   - At least one figure and one table for slice-level behavior.

### Step D2. Failure-case atlas
1. Curate top 30 failure examples with model probabilities.
2. Group error causes: background leakage, occlusion, pose, ambiguous affect.
3. Add brief qualitative discussion.
4. Acceptance:
   - Figure panel + error taxonomy section ready.

## Phase E: External validity (Week 4-5)

### Step E1. Cross-dataset sanity check (recommended)
1. Select one external expression dataset with compatible license.
2. Map labels to nearest 3-class proxy where valid.
3. Evaluate zero-shot/frozen-feature transfer and lightweight adaptation.
4. Acceptance:
   - A table showing transfer behavior and limitations.

### Step E2. If no external dataset available
1. Build internal robustness benchmark by controlled perturbations:
   - brightness, blur, crop-shift, background perturbation.
2. Report robustness curves.
3. Acceptance:
   - Robustness section with controlled stress tests.

## Phase F: Reproducibility package (Week 5-6)

### Step F1. One-command pipeline
1. Provide one entry command for training/eval and one for submission generation.
2. Ensure outputs include metrics json + manifest json.
3. Acceptance:
   - Clean machine rerun reproduces published tables within tolerance.

### Step F2. Environment lock
1. Freeze `requirements.txt` with versions.
2. Document GPU/CPU fallback behavior.
3. Record expected runtime.
4. Acceptance:
   - Setup works from fresh venv.

### Step F3. Artifact integrity
1. Save model checkpoints, config hashes, and checksums.
2. Add artifact index file.
3. Acceptance:
   - Every result table row has exact artifact pointers.

## Phase G: Writing and submission (Week 6-8)

### Step G1. Paper assembly order
1. Methods and setup first.
2. Results and ablations second.
3. Discussion/limitations third.
4. Abstract and intro last.

### Step G2. Reviewer-mode pass
1. Check each claim has direct evidence.
2. Check threats-to-validity section is explicit.
3. Check ethical/data-license statement.

### Step G3. Final pack
1. Main paper PDF.
2. Supplementary (extended tables, implementation details).
3. Reproducibility zip/repo tag.

## Deliverables checklist
- [ ] Baseline lock complete
- [ ] Full ablation tables complete
- [ ] Statistical significance appendix complete
- [ ] Error analysis figure set complete
- [ ] External validity or stress-test section complete
- [ ] Reproducibility package complete
- [ ] Submission-ready manuscript complete
