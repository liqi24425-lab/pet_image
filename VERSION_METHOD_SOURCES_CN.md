# Full Version Literature Traceability Table (Strict Mode)

Updated: 2026-03-07  
Method: For each version, we first inspected the actual implementation in code, then matched it to the closest arXiv source. If no direct paper-level mapping exists, we explicitly label it as `Engineering` (or classical non-arXiv method).

---

## 1) Unified Paper Baseline Library (Used for Per-Version Mapping)

All entries below were checked via arXiv search and verification.

1. **Deep Residual Learning for Image Recognition**  
   Authors: Kaiming He et al.  
   arXiv: `1512.03385`  
   Submitted: `2015-12-10`  
   Link: https://arxiv.org/abs/1512.03385

2. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**  
   Authors: Mingxing Tan, Quoc V. Le  
   arXiv: `1905.11946`  
   Submitted: `2019-05-29`  
   Link: https://arxiv.org/abs/1905.11946

3. **EfficientNetV2: Smaller Models and Faster Training**  
   Authors: Mingxing Tan, Quoc V. Le  
   arXiv: `2104.00298`  
   Submitted: `2021-04-01`  
   Link: https://arxiv.org/abs/2104.00298

4. **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**  
   Authors: Alec Radford et al.  
   arXiv: `2103.00020`  
   Submitted: `2021-02-26`  
   Link: https://arxiv.org/abs/2103.00020

5. **DINOv2: Learning Robust Visual Features without Supervision**  
   Authors: Maxime Oquab et al.  
   arXiv: `2304.07193`  
   Submitted: `2023-04-14`  
   Link: https://arxiv.org/abs/2304.07193

6. **Sigmoid Loss for Language Image Pre-Training (SigLIP)**  
   Authors: Zhai et al.  
   arXiv: `2303.15343`  
   Submitted: `2023-03-27`  
   Link: https://arxiv.org/abs/2303.15343

7. **On Calibration of Modern Neural Networks**  
   Authors: Chuan Guo et al.  
   arXiv: `1706.04599`  
   Submitted: `2017-06-14`  
   Link: https://arxiv.org/abs/1706.04599

8. **mixup: Beyond Empirical Risk Minimization**  
   Authors: Hongyi Zhang et al.  
   arXiv: `1710.09412`  
   Submitted: `2017-10-25`  
   Link: https://arxiv.org/abs/1710.09412

9. **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**  
   Authors: Yun et al.  
   arXiv: `1905.04899`  
   Submitted: `2019-05-12`  
   Link: https://arxiv.org/abs/1905.04899

10. **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence**  
    Authors: Sohn et al.  
    arXiv: `2001.07685`  
    Submitted: `2020-01-21`  
    Link: https://arxiv.org/abs/2001.07685

11. **Self-Training with Noisy Student improves ImageNet classification**  
    Authors: Qizhe Xie et al.  
    arXiv: `1911.04252`  
    Submitted: `2019-11-10`  
    Link: https://arxiv.org/abs/1911.04252

12. **U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection**  
    Authors: Qin et al.  
    arXiv: `2005.09007`  
    Submitted: `2020-05-18`  
    Link: https://arxiv.org/abs/2005.09007

13. **LA-Net: Landmark-Guided Attention Network for FER**  
    Authors: Yu Sun et al.  
    arXiv: `2307.09023`  
    Submitted: `2023-07-18`  
    Link: https://arxiv.org/abs/2307.09023

14. **Multi-Task Multi-Modal Self-Supervised Learning for Facial Expression Recognition**  
    Authors: Chun Xiang et al.  
    arXiv: `2404.10904`  
    Submitted: `2024-04-17`  
    Link: https://arxiv.org/abs/2404.10904

15. **Focal Loss for Dense Object Detection**  
    Authors: Lin et al.  
    arXiv: `1708.02002`  
    Submitted: `2017-08-07`  
    Link: https://arxiv.org/abs/1708.02002

Additional non-arXiv classics:  
- Stacking (Wolpert, 1992)  
- Ridge / Lasso / KNN / KMeans (classical methods, not tied to one arXiv paper)

---

## 2) Per-Version, Per-Script Literature Mapping (Baseline to v22)

Level definitions:  
- `Direct`: the core implemented block directly corresponds to the cited paper method.  
- `Indirect`: paper-level idea is borrowed, but implementation is a lightweight engineering variant.  
- `Engineering`: no reliable one-paper direct anchor; primarily pipeline-level experimentation.

| Version Script | Core Implementation (from code) | Paper Source(s) | Mapping Level | Rigor Note |
|---|---|---|---|---|
| `0.84.py` | ResNet features + Logistic/Ridge/Lasso | ResNet `1512.03385` | Direct | Linear heads are classical methods |
| `0.84-pro.py` | ResNet + regularized linear head | ResNet `1512.03385` | Direct | Hyperparameter search is engineering |
| `0.84-pro-clip-elastic-raw.py` | CLIP features + ElasticNet LR | CLIP `2103.00020` | Direct | ElasticNet part is classical statistics |
| `0.84-pro-clip-elastic-rmbg.py` | CLIP + background removal + ElasticNet | CLIP `2103.00020`, U^2-Net `2005.09007` (nearest) | Indirect | Background-removal implementation is pipeline-specific |
| `0.84-end-to-end.py` | End-to-end ResNet + label smoothing | ResNet `1512.03385` | Direct | Label smoothing treated as general training trick |
| `0.84_adavanced.py` | ResNet + stronger training schedule | ResNet `1512.03385`, mixup `1710.09412` (if enabled) | Indirect | Depends on active code path |
| `ultimate_baseline.py` | ResNet baseline pipeline | ResNet `1512.03385` | Direct | Remaining pieces are engineering |
| `ultimate_baseline_v2.py` | EfficientNet baseline branch | EfficientNet `1905.11946` | Direct | Head/training knobs are engineering |
| `pro_max.py` | ResNet + KNN correction | ResNet `1512.03385` | Indirect | KNN is classical |
| `pro_max_v2.py` | EfficientNet + mixup + KNN | EfficientNet `1905.11946`, mixup `1710.09412` | Indirect | KNN is classical |
| `ultimate_god_mode.py` | EffNet + CLIP + pseudo expansion | EfficientNet `1905.11946`, CLIP `2103.00020`, FixMatch `2001.07685` | Indirect | Pseudo-label gate is custom |
| `mega_ensemble_v3.py` | EffNet + DINOv2 + CLIP + pseudo fusion | EfficientNet `1905.11946`, DINOv2 `2304.07193`, CLIP `2103.00020`, Noisy Student `1911.04252` | Indirect | Large feature concatenation is engineering |
| `final_breakthrough_v4.py` | 3-backbone fusion + TTA + pseudo | EfficientNet `1905.11946`, DINOv2 `2304.07193`, CLIP `2103.00020`, FixMatch `2001.07685` | Indirect | TTA details are custom |
| `stage1_pseudo_label.py` | Consistency pseudo-label pipeline | FixMatch `2001.07685`, Noisy Student `1911.04252` | Indirect | Threshold rules are competition-specific |
| `final_breakthrough_v10.py` | OOF stacking + conservative pseudo gate | FixMatch `2001.07685`, Noisy Student `1911.04252` + Stacking(1992) | Indirect | Stacking source is non-arXiv classic |
| `final_breakthrough_v11_backbone_upgrade.py` | EffNetV2-L / CLIP-L upgrade test | EfficientNetV2 `2104.00298`, CLIP `2103.00020`, DINOv2 `2304.07193` | Direct | Regression was validated offline |
| `final_breakthrough_v12_effective_hf_fusion.py` | Stable backbones + CLIP-L fusion | DINOv2 `2304.07193`, EfficientNet `1905.11946`, CLIP `2103.00020` | Direct | Weight tuning is engineering |
| `final_breakthrough_v13_hybrid_stack.py` | Hybrid decision head (`meta + weighted`) | Calibration `1706.04599` (probability stabilization idea) | Indirect | Custom blend logic |
| `final_breakthrough_v14_robust_prior.py` | Prior reweighting / robust inference | No single precise paper anchor | Engineering | Kept as engineering to avoid forced citation |
| `final_breakthrough_v15_seed_ensemble.py` | Multi-seed probe ensemble | Ensemble-learning practice | Engineering | No mandatory arXiv anchor |
| `final_breakthrough_v16_adaptive_roi.py` | Adaptive ROI TTA | LA-Net `2307.09023` (local-region attention idea) | Indirect | ROI search strategy is custom |
| `final_breakthrough_v17_species_adapter.py` | Species adapter + full/pseudo route | DINOv2 `2304.07193`, FixMatch `2001.07685` | Indirect | KMeans adapter is classical |
| `final_breakthrough_v18_hardset_boost.py` | Hard-sample boost | Focal Loss `1708.02002` (hard-example reweighting idea) | Indirect | Not the exact focal-loss implementation |
| `final_breakthrough_v19_roi_gate.py` | ROI gate + TTA gate | Multi-view consistency idea (`2404.10904`) | Indirect | Engineering gate, not 1:1 paper method |
| `final_breakthrough_v20_calibrated_hybrid.py` | Temperature scaling + hybrid head | Calibration `1706.04599` | Direct | Calibration matches paper-level principle |
| `final_breakthrough_v21_aggressive_siglip_roi_gate.py` | SigLIP + ROI/TTA gate | SigLIP `2303.15343`, Calibration `1706.04599` | Indirect | Gate mechanism is custom |
| `final_breakthrough_v22_lit_noise_consistency.py` | Noise-aware reweight + multi-view consistency gate | LA-Net `2307.09023`, Multi-Task Multi-Modal SSL FER `2404.10904` | Indirect (explicitly mapped) | See `V22_METHOD_SOURCES_CN.md` |
| `final_breakthrough_v23_anchor_consensus.py` | Anchor-based consensus label correction over existing submissions | Ensemble/consensus post-processing (engineering) | Engineering | No new feature extractor; deterministic low-drift label reconciliation |
| `final_breakthrough_v24_atomic_consensus.py` | Reliability-weighted multi-version atomic consensus (top-k flip ladder) | Ensemble/consensus post-processing (engineering) | Engineering | Uses weighted support + agreement constraints to build low-drift submit candidates |
| `remove_bg.py` | Background-removal preprocessing | U^2-Net `2005.09007` (nearest) | Indirect | If rembg/SAM backend changes, citation must be updated |
| `step1_rembg_purify.py` | Background removal + data purification | U^2-Net `2005.09007` (nearest) | Indirect | Pipeline-level preprocessing |
| `direction_benchmark_v20.py` | Directional benchmark script | Reuses v17-v20 method families | Engineering | Not an independent paper method |

---

## 3) Mandatory Rigor Rules (Going Forward)

1. Every new `vXX` script must add one row here before submission.  
2. Each row must include: implementation summary, source papers, mapping level, and rigor note.  
3. If no direct paper exists, explicitly mark `Engineering` and explain why.  
4. If a version regresses online, log the rejection in both README and this table.
