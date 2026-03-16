# STA314H — Pet Facial Expression Classifier (宠物面部表情分类器)
🏆 **Official Team Score: 0.92666** | **Best Ablation Score: 0.93333 (minus-CLIP)**

> **Course**: STA314H Statistical Machine Learning — University of Toronto  
> **Task**: 3-class pet facial expression classification (Angry / Happy / Sad)  

This project documents the entire evolutionary journey of a high-performance image classifier for pet facial expressions. We progressed from basic statistical baselines (PCA + Logistic Regression) to robust CNNs, and finally to cutting-edge Vision Transformers with sophisticated feature fusion strategies.

---

## 🚀 The Path to 0.92666: Model Evolution & Experimental Analysis

Building this classifier was a continuous process of hypothesis, experimentation, and physiological analysis of pet expressions. Below is the complete record of our model evolution, from early statistical baselines to our final winning deep learning architecture.

### 🐣 Phase 1: Early Baselines & Feature Extraction
We started with simple feature extraction and statistical machine learning to set a baseline.

* **Score: 0.426 | `phase 2`**
  * **Strategy:** PCA (Principal Component Analysis) + Ridge Regression.
  * **Analysis:** Extracting basic eigenvalues (Eigenfaces) provided a low baseline score but proved that raw pixels need transformation.
* **Score: 0.840 | `0.84`**
  * **Strategy:** Basic CNN feature extractor + Ridge Regression.
  * **Analysis:** First leap in performance. CNNs easily outperformed PCA in extracting spatial patterns from pet faces.
* **Score: 0.760 | `0.84-advanced`**
  * **Strategy:** CNN + Ridge Regression + Gradient Descent (GD) to optimize the minimum learning rate.
  * **Analysis:** Encountered **overfitting** (Training Loss = 1), leading to a significant drop in test score.

### 🔨 Phase 2: CNN Architectures & Optimization
We transitioned to end-to-end deep learning and hyperparameter tuning.

* **Score: 0.840 (v2) | `0.84-pro`**
  * **Strategy:** CNN + Ridge Regression + Cross-Validation (CV) to optimize the `C` penalty matrix parameter.
  * **Analysis:** CV helped stabilize the model back to 0.840 by properly tuning regularization.
* **Score: 0.826 | `ultimate_baseline`**
  * **Strategy:** End-to-end CNN + CV + Gradient Descent.
  * **Analysis:** By removing the explicit Ridge layer and doing pure end-to-end GD optimization, we hit the best pure CNN baseline at the time.
* **Score: 0.833 | `pro_max`**
  * **Strategy:** End-to-end CNN + CV + GD + **KNN** (Weighted memory of local surroundings).
  * **Analysis:** Adding a non-parametric K-Nearest Neighbors constraint improved local decision boundaries slightly.
* **Score: 0.820 | `pro_max_v2`**
  * **Strategy:** Replaced ResNet-18 with **EfficientNet-B0** in the `pro_max` pipeline.
  * **Analysis:** EfficientNet is highly parameter-efficient but performed slightly worse without proper data scaling.
* **Score: 0.800 | `submission-resnet50`**
  * **Strategy:** Scaled ResNet-18 up to **ResNet-50** based on the `ultimate_baseline`.
  * **Analysis:** Severe overfitting. A deeper network without matching data volumes hurt performance.
* **Score: 0.806 | `0.84-lasso.py`**
  * **Strategy:** Swapped Ridge (L2) for Lasso (L1) on top of the original 0.84 baseline.
  * **Analysis:** Lasso's feature selection (setting weights to zero) was too aggressive for complex visual features, dropping the score.
* **Score: 0.840 | `0.84-end-to-end`**
  * **Strategy:** Dropped pre-trained heads entirely. Removed Ridge and Lasso, pure end-to-end deep learning from scratch.
  * **Analysis:** Met the pre-trained baseline, proving our architecture and training pipeline were solid.

### 🌟 Phase 3: The Deep Learning Breakthroughs

* **Score: 0.906 | `effnet-elastic` (`ultimate_god_mode.py`)**
  * **Strategy:** **EfficientNet-B5** + Semi-supervised learning (added high-confidence test pseudo-labels back into training).
  * **Conclusion & Analysis:** Confirmed that **"Feature Fusion + Semi-Supervised Learning"** direction was correct. However, we hit a bottleneck due to background context interference.
* **Score: 0.906 | `god-mode` (`mega_ensemble_v3.py`)**
  * **Strategy:** Added test data with pseudo labels, horizontally concatenated raw features (2048 + 512) into a massive 2560-dimensional space, and incorporated DINOv2-Small.
  * **Conclusion & Analysis:** Blindly expanding features caused **horizontal redundancy**. The L1 penalty in our ElasticNet head zeroed out the overlapping redundant features, yielding no actual score gain.

### 👑 Phase 4: Final Victory & Over-parameterization Trap

* **Score: 0.913 🥇 | `final_breakthrough_v4.py`**
  * **Strategy:** Upgraded to **DINOv2-Large** to capture fine facial muscle movements. Introduced a **weighted Zoom (Center Crop) Test-Time Augmentation (TTA)** to aggressively force the model to focus purely on the animal's facial features.
  * **Conclusion & Analysis:** **Accurate breakthrough!** By combining DINOv2's physiological landmark resolution with a local zoom-in strategy that stripped away distracting background noise, the model perfectly hit the biological core rule: *"Pet emotion classification highly depends on the eyes."*
* **Score: 0.900 | `v6_giant_ultimate.py`**
  * **Strategy:** Employed the most brute-force approach using **DINOv2-Giant** (1536 dimensions) and a massive 10-Crop full-image cropping strategy.
  * **Conclusion & Analysis:** **Curse of Dimensionality and Noise Backlash.** Over 4096 dimensions triggered severe overfitting. The 10-crop strategy generated patches containing non-informative background areas (wall corners, grass), introducing fatal background noise that biased the model.

### 🚀 Phase 5: Team Submission Upgrade

* **Score: 0.92666 🥇 | `submission_final_team_v10.csv`**
  * **Strategy:** Use the clean `v10` pipeline (OOF stacking + conservative pseudo-label gate + drift safety checks) as the only team submission target.
  * **Conclusion & Analysis:** Compared with `submission_final_team.csv` (0.92000), the `v10` single-submission protocol improved leaderboard performance and reduced coordination risk for shared team quota.

### 🧪 Phase 6: Backbone Upgrade Hypothesis (Evaluation)

* **Experiment:** `final_breakthrough_v11_backbone_upgrade.py`
  * **Change:** Keep DINOv2-Large, replace EfficientNet-B5 with EfficientNetV2-L, and replace CLIP-B/32 with CLIP-L/14.
* **Offline finding (OOF):**
  * DINO: ~0.900
  * EfficientNetV2-L: ~0.473 (major regression)
  * CLIP-L/14: ~0.860 (improved over CLIP-B/32)
* **Decision:** Do **not** promote v11 to team submission yet. Keep `submission_final_team_v10.csv` as the default until the CNN branch is replaced/fixed in a more stable way.

### 🧪 Phase 7: Effective Fusion Validation (`v12`)

* **Experiment:** `final_breakthrough_v12_effective_hf_fusion.py`
  * **Change:** Keep `DINOv2-L + EfficientNet-B5` stable branch, upgrade CLIP from `ViT-B/32` to `ViT-L/14`, keep OOF stacking + conservative pseudo-label gate unchanged.
* **Offline finding (OOF):**
  * DINO: `0.9022` acc / `0.9021` macro-F1 (stable)
  * EffNet-B5: `0.8711` acc / `0.8710` macro-F1 (stable)
  * CLIP-L/14: `0.8489` acc / `0.8486` macro-F1 (improved vs v10 CLIP `0.7689`)
  * Stack head: `0.9044` acc / `0.9044` macro-F1 (no gain over v10 stack)
* **Decision:** `v12` is analysis-only for now, **not promoted** to team default submit file.
* **Why EffNetV2-L was removed:** in `v11`, EffNetV2-L OOF dropped to ~`0.473`, causing ensemble instability and lowering stack quality. We rolled back to EffNet-B5 because it is materially more reliable on this 450-sample regime.

### 🧪 Phase 8: Hybrid Stack Stabilization (`v13`)

* **Experiment:** `final_breakthrough_v13_hybrid_stack.py`
  * **Change:** Keep `DINOv2-L + EffNet-B5 + CLIP-L/14`, replace direct stack decision with a true-OOF-calibrated hybrid head:
    * compute `weighted_oof` from model-level OOF probabilities,
    * compute `meta_true_oof` with fold-wise meta prediction,
    * tune `alpha` on OOF for `hybrid = alpha * meta + (1 - alpha) * weighted`.
* **Offline finding (OOF):**
  * weighted OOF: `0.9067`
  * meta true OOF: `0.8956`
  * hybrid OOF (`alpha=0.25`): `0.9067`, macro-F1 `0.9066`
* **Leaderboard check (2026-03-06):** `submission_final_team_v13.csv` scored `0.92000`, below `v10` (`0.92666`).
* **Decision:** keep `v13` as analysis-only; `v10` remains the only team default submit file.

### 🧪 Phase 9: Robust Prior + Reliability Reweight (`v14`, offline)

* **Experiment:** `final_breakthrough_v14_robust_prior.py`
* **Change:** reliability-based sample reweighting + Bayesian-style prior adaptation at inference.
* **Offline finding:** stack OOF dropped to `0.9000`; drift vs `v10` increased.
* **Decision:** reject `v14` (analysis-only, no submission).

### 🧪 Phase 10: Seed-Ensemble Probe Stabilization (`v15`, offline)

* **Experiment:** `final_breakthrough_v15_seed_ensemble.py`
* **Change:** keep `v10` backbone/data flow, but ensemble linear probes over seeds `(42, 52, 62)` for OOF and test probabilities.
* **Offline finding:** stack OOF `0.9178`, macro-F1 `0.9177`; prediction drift vs `v10` only `0.33%` (2/300 samples).
* **Leaderboard check (2026-03-06):** `submission_final_team_v15.csv` scored `0.92666`, tied with `v10`.
* **Decision:** `v15` is stable but not superior; still a plateau run, not a new best.

### 🧪 Phase 11: Adaptive ROI TTA for Off-Center Faces (`v16`, offline)

* **Experiment:** `final_breakthrough_v16_adaptive_roi.py`
* **Change:** replace center-zoom TTA with adaptive ROI crop (`roi`, `roi_flip`) that can shift away from image center.
* **Offline finding:** stack OOF remained `0.9178` (same as `v15`), no measurable gain.
* **Decision:** keep `v16` as analysis-only; off-center ROI handling is valid but not yet a score breakthrough.

### 🧪 Phase 12: Full-Pseudo Validation (`v17`, submitted and rejected)

* **Experiment:** `final_breakthrough_v17_species_adapter.py`
* **Change:** Full TTA + multi-seed probes + forced pseudo-label branch.
* **Offline finding:** stack OOF looked strong (`~0.924`), but pseudo selection ratio was very high (`~50%`) once enabled.
* **Leaderboard check (2026-03-07):** `submission_final_team_v17.csv` scored `0.91333`, significantly below `v10/v15` (`0.92666`).
* **Decision:** reject pseudo-on path as default. For this dataset, pseudo labels amplify noise and hurt public LB.

### 🧪 Phase 13: Calibrated Hybrid Without Pseudo (`v20`, offline candidate)

* **Experiment:** `final_breakthrough_v20_calibrated_hybrid.py`
* **Change:** keep 3 backbones + 5-fold/3-seed OOF, disable pseudo by default, add:
  * per-backbone temperature calibration on OOF probabilities,
  * hybrid decision head (`meta_proba` + weighted ensemble) tuned by OOF alpha.
* **Offline finding:**
  * stack OOF: `0.9244` acc / `0.9243` macro-F1
  * prediction drift vs `submission_final_team.csv`: `1/300` sample
  * prediction drift vs `v10`: `2/300` samples
* **Status:** high-stability candidate for next single submission attempt.

### 🧪 Phase 14: Aggressive SigLIP + ROI/TTA Gate (`v21`, offline)

* **Experiment:** `final_breakthrough_v21_aggressive_siglip_roi_gate.py`
* **Change:** aggressive branch with:
  * SigLIP vision backbone (fallback to CLIP-B/32 if loading fails),
  * ROI / ROI-flip TTA views,
  * per-sample TTA confidence gating,
  * calibrated hybrid head (same no-pseudo policy).
* **Offline finding:**
  * CLIP-branch OOF improved (`~0.8556`, up from `~0.8356` in v20).
  * Stack OOF stayed around `0.9244` (no clear gain over v20).
  * Drift vs `v10` increased (`8/300` changed), so direct submission risk is higher.
* **Risk-controlled derivative:** `submission_final_team_v21_safe.csv`
  * Rule: only keep flips where `v20` and `v21` agree and both differ from `v10`.
  * Result: only `1/300` label changed (`img_000660.jpg`), for low-quota test.

### 🧪 Phase 15: Literature-Driven v22 (Noise + Multi-view Consistency)

* **Experiment:** `final_breakthrough_v22_lit_noise_consistency.py`
* **Change:** add two literature-backed modules while keeping the v10/v20 backbone family:
  * **Noise-aware reweight** (inspired by LA-Net): down-weight uncertain / likely noisy training samples using OOF confidence + entropy.
  * **Multi-view consistency gate** (inspired by multi-view FER SSL): split TTA into global vs ROI groups and gate conflicts by confidence margin.
* **Paper sources (explicit):**
  * LA-Net: [arXiv:2307.09023](https://arxiv.org/abs/2307.09023)
  * Multi-Task Multi-Modal SSL for FER: [arXiv:2404.10904](https://arxiv.org/abs/2404.10904)
* **Traceability file:** `V22_METHOD_SOURCES_CN.md` (method-to-code mapping).

### 🧪 Phase 16: Anchor Consensus Correction (`v23`, low-risk execution)

* **Experiment:** `final_breakthrough_v23_anchor_consensus.py`
* **Change:** no retraining; start from the best anchor `v10` and only accept label flips supported by multiple later variants (`v15/v16/v20/v22`).
* **Rules:**
  * `strong`: apply flip if >=3/4 references agree and differ from `v10`.
  * `ultra`: apply flip if 4/4 references agree and differ from `v10`.
* **Generated candidates:**
  * `submission_final_team_v23_consensus_anchor.csv` (2 flips)
  * `submission_final_team_v23_ultra_consensus.csv` (1 flip)
* **Purpose:** use team submission quota rationally by testing tiny, consensus-backed perturbations instead of high-drift architecture jumps.

### 🧪 Phase 17: Atomic Consensus Ladder (`v24`, bold but controlled)

* **Experiment:** `final_breakthrough_v24_atomic_consensus.py`
* **Change:** keep `v10` as anchor, aggregate votes from all `submission_final_team_v*.csv` variants with reliability weighting:
  * weight uses OOF quality + drift penalty + stability bonus,
  * apply only flips with strong weighted support and multi-version agreement.
* **Generated candidates:**
  * `submission_final_team_v24_atomic_top1.csv` (only strongest flip)
  * `submission_final_team_v24_atomic_top2.csv` (top-2 strongest flips)
* **Why this method:** it is a multi-version strong-consistency strategy that is more innovative than simple majority vote, while still controlling drift and submission risk.

### 🧪 Phase 18: Lightweight MoE Router Validation (`v26`, full run)

* **Experiment:** `final_breakthrough_v26_lightweight_moe.py`
* **What changed:**
  * fixed CPU branch logic so full stack path can still run under lightweight feature mode,
  * kept v22-family hybrid stack (`weighted + meta + TTA gate + noise reweight`),
  * added MoE-style router decision over three experts (`weighted`, `meta`, `consistency`).
* **Offline result (2026-03-08):**
  * `stack_oof_acc = 0.9200`, `stack_oof_macro_f1 = 0.9200`,
  * router candidate `0.9178` was rejected by OOF criterion,
  * final output drift vs `submission_final_team.csv`: `18/300`.
* **Decision:** not promoted to team default; keep `submission_final_team_v10.csv` / `submission_final_team_v15.csv` as safer top-tier anchors.

### 🧪 Phase 19: E5 Backbone Removal Ablation Batch (2026-03-15)

* **Experiment package:** `ablation_runs_20260315/run_ablation_package.py`
* **Leaderboard-side finding:**
  * `ABL_20260315_E5_full_3backbones.csv`: `0.92666`
  * `ABL_20260315_E5_minus_clip.csv`: `0.93333` (new best in this ablation batch)
  * `ABL_20260315_E5_minus_effnet.csv`: `0.92666`
  * `ABL_20260315_E5_minus_dino.csv`: `0.89333`
* **Strict same-split OOF re-check (seed=42, 5-fold):**
  * source: `ablation_runs_20260315/outputs/stage4_e5_backbone_table.csv`
  * weighted OOF: full-3B `0.9067` vs minus-CLIP `0.9067` (tie)
  * stacked OOF: full-3B `0.9022` vs minus-CLIP `0.9089` (`+0.0067`)
  * interpretation: minus-CLIP appears promising but should be treated as configuration-dependent evidence; not a blanket claim that CLIP is always harmful.

## 🧩 Feature Extractor / Classifier / Method Mapping

The table below keeps the current README narrative unchanged, and explicitly adds (for each model family/version) the feature extraction method, classifier head, and extra methods used.

| Version / Script | Feature Extractor (how features are extracted) | Classifier (head / decision) | Extra methods used |
|---|---|---|---|
| `phase 2` (PCA baseline) | Raw pixel -> PCA principal components (Eigenfaces style) | Ridge-style linear classifier/regressor | Standardization + CV-style tuning |
| `0.84.py` | `ResNet18` penultimate features (512-d, frozen pretrained backbone) | `LogisticRegression` (`L2` Ridge; optional `L1`) | `StandardScaler` |
| `0.84-pro.py` | `ResNet18` frozen feature extractor | `LogisticRegression` (`L2`) with `GridSearchCV` on `C` | 5-fold CV |
| `0.84-pro-clip-elastic-raw.py` | `CLIP ViT-B/32` `image_embeds` (512-d) | `LogisticRegression` with `penalty='elasticnet'` | 2D grid search (`C`, `l1_ratio`) |
| `0.84-pro-clip-elastic-rmbg.py` | `CLIP ViT-B/32` image features on background-removed images | `LogisticRegression` with ElasticNet | Background removal + grid search |
| `0.84-end-to-end` / `0.84-advanced` | End-to-end CNN training (ResNet family) | CNN `Linear` softmax head (cross-entropy) | stronger augmentation / schedule |
| `ultimate_baseline.py` | Fine-tuned `ResNet18`, then `fc -> Identity` for 512-d features | `LogisticRegression` (`L2`) | `GridSearchCV` |
| `pro_max.py` | Same `ResNet18` 512-d features as `ultimate_baseline` | `LogisticRegression` (`L2`) + KNN-like retrieval blend | Visual RAG / similarity fusion |
| `pro_max_v2.py` | Fine-tuned `EfficientNet-B0`, then 1280-d penultimate features | `LogisticRegression` (`L2`) + KNN-like retrieval blend | MixUp + RandomErasing + RAG fusion |
| `ultimate_god_mode.py` | `EffNet-B5` + `CLIP` feature fusion | ElasticNet `LogisticRegression` | Pseudo-label expansion |
| `mega_ensemble_v3.py` | `EffNet-B5` + `DINOv2-S` + `CLIP` concatenated features | ElasticNet `LogisticRegression` | Pseudo labels + high-dim fusion |
| `final_breakthrough_v4.py` | `DINOv2-L` + `EffNet-B5` + `CLIP-B/32` multi-backbone extraction | Per-backbone linear probe (`nn.Linear`) + ElasticNet `LogisticRegression` meta-learner | OOF stacking + weighted TTA + consensus pseudo-label |
| `v6_giant_ultimate.py` | `DINOv2-G` high-dimensional + multi-crop extraction | Same stack family as v4 (linear probes + stacked decision) | 10-crop aggressive TTA |
| `final_breakthrough_v10.py` | `DINOv2-L` + `EffNet-B5` + `CLIP-B/32` | Per-backbone linear probe + ElasticNet logistic meta stack | OOF stacking + conservative pseudo gate + drift safety checks |
| `final_breakthrough_v11_backbone_upgrade.py` | `DINOv2-L` + `EffNetV2-L` + `CLIP-L/14` | Same as v10 (linear probes + ElasticNet meta stack) | Backbone upgrade ablation |
| `final_breakthrough_v12_effective_hf_fusion.py` | `DINOv2-L` + `EffNet-B5` + `CLIP-L/14` | Same as v10 (linear probes + ElasticNet meta stack) | Effective fusion validation |
| `final_breakthrough_v13_hybrid_stack.py` | Same 3-backbone extraction as v12 | ElasticNet meta model + weighted ensemble hybrid (`alpha * meta + (1-alpha) * weighted`) | True-OOF alpha tuning |
| `final_breakthrough_v14_robust_prior.py` | Same 3-backbone extraction | Linear probes + ElasticNet meta stack | Reliability reweight + robust prior adaptation |
| `final_breakthrough_v15_seed_ensemble.py` | Same 3-backbone extraction | Multi-seed linear probes + ElasticNet meta stack | Seed ensemble (`42/52/62`) |
| `final_breakthrough_v16_adaptive_roi.py` | Same backbone family, with adaptive ROI views | Same stack family (linear probes + meta logistic) | Adaptive ROI TTA gate |
| `final_breakthrough_v17_species_adapter.py` | Same backbone family + species features | Hybrid stack with species-adapter features | Full pseudo route + species adapter |
| `final_breakthrough_v20_calibrated_hybrid.py` | Same 3-backbone extraction | Per-backbone logistic probes + fixed `LogisticRegression` meta learner (`lbfgs`) | Temperature calibration + hybrid blending |
| `final_breakthrough_v21_aggressive_siglip_roi_gate.py` | `DINOv2` + `EffNet` + `SigLIP` (fallback `CLIP-B/32`) | Per-backbone logistic probes + fixed logistic meta learner | ROI/ROI-flip TTA + confidence gating + calibration |
| `final_breakthrough_v22_lit_noise_consistency.py` | Same as v21 family | Per-backbone logistic probes + fixed logistic meta learner | Noise-aware reweight + multi-view consistency gate |
| `final_breakthrough_v23_anchor_consensus.py` | No new feature extraction (submission-level post-process) | No trained classifier; deterministic consensus flip rules | Anchor consensus from `v10/v15/v16/v20/v22` |
| `final_breakthrough_v24_atomic_consensus.py` | No new feature extraction (submission-level post-process) | No trained classifier; reliability-weighted vote ladder | Atomic top-k low-drift consensus flips |
| `final_breakthrough_v25_moe_router.py` | 3-backbone extraction (`DINOv2` + `EffNet` + `CLIP`) | Hybrid stack + MoE router (`LogisticRegression`) over experts (`weighted`, `meta`, `consistency`) | Temperature calibration + router gating + noise-aware refit |
| `final_breakthrough_v26_lightweight_moe.py` | Same as v25, with CPU/lightweight branch support | Hybrid stack + MoE router (accepted/rejected by OOF criterion) | Lightweight stack path + calibration + optional pseudo gate |

## 🧾 Kaggle Public Score Register (Screenshot-Verified)

Snapshot source: team Kaggle submissions page screenshot (recorded on 2026-03-08).

| Submission File | Public Score | Notes |
|---|---:|---|
| `submission_final_team_v22.csv` | `0.91333` | Byran Li |
| `submission_v5_ensemble.csv` | `0.91333` | Ruiqii Liu |
| `submission_final_team_v21.csv` | `0.90000` | Byran Li |
| `submission_final_team_v20.csv` | `0.92000` | Byran Li |
| `submission_final_team_v17.csv` | `0.91333` | Byran Li |
| `submission_final_team_v16.csv` | `0.92000` | Byran Li |
| `submission_final_team_v15.csv` | `0.92666` | Byran Li (best tier, tied) |
| `submission_final_team_v13.csv` | `0.92000` | Byran Li |
| `submission_final_team_v10.csv` | `0.92666` | Byran Li (best tier, tied) |

## 📚 Version-Wise Method Sources (Mandatory)

To satisfy team traceability requirements, every version now has an explicit method-source annotation (paper-driven or engineering-only):

- `VERSION_METHOD_SOURCES.md` (full version index, English, current)
- `V22_METHOD_SOURCES_CN.md` (v22 detailed mapping; Chinese)

`VERSION_METHOD_SOURCES.md` is maintained in strict per-script mode (from early baseline to v26), including: implemented method, arXiv source, directness level, and rigor notes.

---

## 🛠 Project Structure & Early Phases
For historical tracking, the repository includes all our preliminary experiments:
- **Phase 1-2**: EDA, PCA + Ridge/Lasso baseline (`phase1_eda.py`, `phase2_baseline.py`)
- **Phase 3**: End-to-end baseline CNNs (ResNet18, EfficientNet-B0) with Label Smoothing and Cosine Annealing.
- **Phase 4-5**: Mixup, CutMix, 5-Fold Cross Validation (`phase5_advanced_training.py`)
- **Phase 6**: Statistical Diagnostics (MC Dropout Uncertainty, Grad-CAM, Error Correlation Analysis).

## ⚙️ How to Reproduce the Team Submission
To generate the current team-default Kaggle submission:

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run the team mainline script
python final_breakthrough_v10.py

# 3. Use the team-final file (single submission target)
# submission_final_team_v10.csv
```

## 📌 Next Actions (Team Plan)

1. Keep `submission_final_team_v10.csv` as the default submission file (same best score as `v15`).
2. Stop spending submissions on same-family variants (`v10/v13/v15`) until a new method family is validated offline.
3. Move to new directions below with strict offline gates before any submission.

## 🔭 Next Directions (Post-Plateau)

1. **Face-region refinement before backbone extraction**
Use pet-face ROI detection / tighter center-region policy to suppress background shortcut features.
2. **Species-aware heads or adapters**
Add lightweight species routing (dog/cat/other) and then expression head, reducing cross-species interference.
3. **Hard-sample denoising loop**
Use disagreement samples (`v10` vs `v13/v15`) as a fixed hard set for targeted relabel checking or confidence-aware reweighting.

## 🧭 v10 Experiment Protocol (Single Submission)

`final_breakthrough_v10.py` is the new clean mainline for iterative improvement.

Core outputs:

- `submission_final_team_v10.csv` (only submission candidate)
- `v10_oof_metrics.json` (metrics + risk gates)
- `v10_run_manifest.json` (deterministic config + runtime metadata)

Team submission rule:

1. Only submit `submission_final_team_v10.csv`.
2. Treat all other generated files as `analysis-only`.
3. Promote a run only if OOF remains stable and label drift is within gate.

Single-variable experiment matrix:

| Iteration | Variable changed | Default | Candidate values |
|---|---|---|---|
| V10-A | TTA model-weight search range | current grid | narrow around best |
| V10-B | Pseudo confidence threshold | 0.92 | 0.90 / 0.93 |
| V10-C | Pseudo entropy threshold | 0.23 | 0.20 / 0.25 |
| V10-D | Pseudo sample weight | 0.30 | 0.20 / 0.40 |

Dual-track HF workflow (analysis-only):

1. Use HF Jobs to run expansion experiments.
2. Use Trackio to log OOF and pseudo-label diagnostics.
3. Append run summaries to `hf/v10_experiments_registry.csv`.
4. Use Dataset Viewer read-only APIs to inspect remote experiment tables.

## 🧪 Label Noise & Bias Discussion (for Kaggle Report)

This task has unavoidable subjectivity: pet expressions can be ambiguous, and labels such as Angry/Happy/Sad may be noisy around borderline cases. We addressed this with three safeguards:

1. **Noise-robust supervision**: use label smoothing and weighted cross-entropy in first-level probes to avoid over-confident fitting to potentially noisy labels.
2. **Consistency pseudo-labeling**: add pseudo labels only when all three backbones (DINOv2-Large, EfficientNet-B5, CLIP) agree after 4-way TTA averaging, with strict confidence and entropy thresholds.
3. **OOF-based calibration**: use 5-fold out-of-fold stacking to reduce overfitting from high-dimensional feature fusion and to improve probability calibration.

Potential bias sources include background/lighting artifacts and cross-species visual diversity. We mitigated them with center-focused TTA (zoom/flip variants), multi-backbone feature diversity, and strict pseudo-label gating.
