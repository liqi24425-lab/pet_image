# 03. Paper Blueprint and Draft Map

Use this structure for a 6-8 page conference paper.

## 1. Introduction
Must include:
1. Problem definition and practical relevance.
2. Why small-sample pet FER is hard.
3. Gap in existing methods.
4. Your contributions (3 bullets max).

## 2. Related Work
Organize by themes:
1. Facial expression recognition and small-data settings.
2. Multi-backbone feature fusion.
3. Calibration and uncertainty-aware decision making.
4. Semi-supervised pseudo-labeling and label-noise handling.

## 3. Method
Subsections:
1. System overview diagram.
2. Backbone feature extraction.
3. OOF probe training and stacking.
4. Temperature calibration + hybrid decision.
5. TTA consistency gate and noise-aware reweighting.
6. Safety fallback policy and drift constraints.

## 4. Experimental Setup
Must explicitly state:
1. Dataset sizes and splits.
2. Preprocessing and augmentations.
3. Evaluation metrics.
4. Baselines and ablation settings.
5. Hardware/runtime.

## 5. Results
Required tables/figures:
1. Main comparison table (mean±std).
2. Ablation table (incremental).
3. Robustness/slice analysis figure.
4. Failure-case gallery.

## 6. Discussion
Include:
1. Why certain modules help/hurt.
2. Why offline and online differ.
3. Engineering lessons for low-data competitions.

## 7. Limitations and Ethics
Include:
1. Dataset bias and sample size risk.
2. Annotation uncertainty.
3. Animal-welfare and misuse considerations.

## 8. Conclusion
One paragraph on takeaway + future direction.

## Appendix / Supplementary
1. Additional ablations.
2. Hyperparameters and full config.
3. Reproduction instructions.
4. License and data statement.

## Draft map (what to write first)
1. Methods
2. Setup
3. Results
4. Discussion
5. Intro
6. Abstract
