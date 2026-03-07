# 全版本文献溯源总表（严格版）

更新时间：2026-03-07  
检索方式：对每个版本先做代码对比（脚本级），再用 arXiv 检索最接近的论文来源；若不存在直接论文来源，明确标注“工程实验/经典非 arXiv 方法”。

---

## 1) 统一论文基线库（用于逐版本映射）

以下条目均来自 arXiv 检索并核对：

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

补充说明（非 arXiv 经典）：  
- Stacking (Wolpert, 1992)  
- Ridge / Lasso / KNN / KMeans（经典统计与机器学习方法，非特定 arXiv 单篇来源）

---

## 2) 逐版本逐脚本文献映射（从早期到 v22）

说明：  
- `直接`：代码核心模块直接对应论文方法。  
- `间接`：思想借鉴，代码是轻量工程化改写。  
- `工程`：没有明确论文锚点，主要是工程调参/策略试验。

| 版本脚本 | 核心实现（代码对比） | 文献来源（明确） | 关联级别 | 严谨备注 |
|---|---|---|---|---|
| `0.84.py` | ResNet 特征 + Logistic/Ridge/Lasso | ResNet `1512.03385` | 直接 | 线性头部分为经典方法（非 arXiv 单篇绑定） |
| `0.84-pro.py` | ResNet + 正则化线性头 | ResNet `1512.03385` | 直接 | 超参搜索为工程行为 |
| `0.84-pro-clip-elastic-raw.py` | CLIP 特征 + ElasticNet LR | CLIP `2103.00020` | 直接 | ElasticNet 部分属经典统计学习 |
| `0.84-pro-clip-elastic-rmbg.py` | CLIP + 背景去除 + ElasticNet | CLIP `2103.00020`, U^2-Net `2005.09007`（近邻） | 间接 | 背景去除脚本未声明具体模型时按近邻方法标注 |
| `0.84-end-to-end.py` | ResNet end-to-end + label smoothing | ResNet `1512.03385` | 直接 | label smoothing 为通用训练技巧 |
| `0.84_adavanced.py` | ResNet + 更强训练策略 | ResNet `1512.03385`, mixup `1710.09412`（若启用） | 间接 | 需以代码开关为准（有则直接，无则工程） |
| `ultimate_baseline.py` | ResNet 基线管线 | ResNet `1512.03385` | 直接 | 其他为工程调参 |
| `ultimate_baseline_v2.py` | EfficientNet-B0/B5 基线 | EfficientNet `1905.11946` | 直接 | 线性头与调参属工程 |
| `pro_max.py` | ResNet + KNN 修正 | ResNet `1512.03385` | 间接 | KNN 为经典方法 |
| `pro_max_v2.py` | EfficientNet + mixup + KNN | EfficientNet `1905.11946`, mixup `1710.09412` | 间接 | KNN 为经典方法 |
| `ultimate_god_mode.py` | EffNet + CLIP + pseudo 扩充 | EfficientNet `1905.11946`, CLIP `2103.00020`, FixMatch `2001.07685` | 间接 | pseudo 使用的是“高置信门控”工程变体 |
| `mega_ensemble_v3.py` | EffNet + DINOv2 + CLIP 融合 + pseudo | EfficientNet `1905.11946`, DINOv2 `2304.07193`, CLIP `2103.00020`, Noisy Student `1911.04252` | 间接 | 大规模特征拼接属于工程策略 |
| `final_breakthrough_v4.py` | 三骨干融合 + TTA + pseudo | EfficientNet `1905.11946`, DINOv2 `2304.07193`, CLIP `2103.00020`, FixMatch `2001.07685` | 间接 | TTA 细节为工程实现 |
| `stage1_pseudo_label.py` | 一致性伪标签管线 | FixMatch `2001.07685`, Noisy Student `1911.04252` | 间接 | 一致性阈值是比赛定制 |
| `final_breakthrough_v10.py` | OOF stacking + 保守 pseudo gate | FixMatch `2001.07685`, Noisy Student `1911.04252` + Stacking(1992) | 间接 | stacking 为经典非 arXiv 理论来源 |
| `final_breakthrough_v11_backbone_upgrade.py` | EffNetV2-L / CLIP-L 升级尝试 | EfficientNetV2 `2104.00298`, CLIP `2103.00020`, DINOv2 `2304.07193` | 直接 | 实测退化已在 README 记录 |
| `final_breakthrough_v12_effective_hf_fusion.py` | 稳定骨干 + CLIP-L 融合 | DINOv2 `2304.07193`, EfficientNet `1905.11946`, CLIP `2103.00020` | 直接 | 融合权重搜索为工程 |
| `final_breakthrough_v13_hybrid_stack.py` | meta + weighted 混合头 | Calibration `1706.04599`（概率稳定思想） | 间接 | 核心是工程化混合决策 |
| `final_breakthrough_v14_robust_prior.py` | 先验重加权/鲁棒推理 | 无单一精确论文锚点 | 工程 | 保留“工程实验”标注，避免过度引用 |
| `final_breakthrough_v15_seed_ensemble.py` | 多 seed probe ensemble | 集成学习通用方法 | 工程 | 无单一 arXiv 必须来源 |
| `final_breakthrough_v16_adaptive_roi.py` | 自适应 ROI TTA | LA-Net `2307.09023`（局部区域关注思想） | 间接 | ROI 搜索策略是工程实现 |
| `final_breakthrough_v17_species_adapter.py` | species adapter + full/pseudo | DINOv2 `2304.07193`, FixMatch `2001.07685` | 间接 | `kmeans` 物种代理为经典方法 |
| `final_breakthrough_v18_hardset_boost.py` | hard sample boost | Focal Loss `1708.02002`（难例重加权思想） | 间接 | 当前实现非原版 focal loss，属工程近似 |
| `final_breakthrough_v19_roi_gate.py` | ROI gate + TTA gate | Multi-view consistency（参考 `2404.10904`） | 间接 | 属工程版门控，不等同原论文 |
| `final_breakthrough_v20_calibrated_hybrid.py` | 温度缩放 + hybrid 头 | Calibration `1706.04599` | 直接 | 温度缩放实现与论文思想一致 |
| `final_breakthrough_v21_aggressive_siglip_roi_gate.py` | SigLIP + ROI/TTA gate | SigLIP `2303.15343`, Calibration `1706.04599` | 间接 | 门控是工程改造，非 SigLIP 原任务 |
| `final_breakthrough_v22_lit_noise_consistency.py` | 噪声重加权 + 多视角一致性门控 | LA-Net `2307.09023`, Multi-Task Multi-Modal SSL FER `2404.10904` | 间接（有明确映射） | 详见 `V22_METHOD_SOURCES_CN.md` |
| `remove_bg.py` | 背景去除预处理 | U^2-Net `2005.09007`（近邻） | 间接 | 若实际调用 rembg/SAM，需补充对应论文 |
| `step1_rembg_purify.py` | 去背景+数据净化 | U^2-Net `2005.09007`（近邻） | 间接 | 预处理策略为工程 |
| `direction_benchmark_v20.py` | 方向性 benchmark 汇总脚本 | 复用 v17-v20 方法集 | 工程 | 用于离线对照，不是独立方法论文 |

---

## 3) 严谨性规则（后续必须执行）

1. 新增任何 `vXX` 文件，必须在本表新增一行。  
2. 每行必须给出：`核心实现`、`文献来源`、`关联级别`、`严谨备注`。  
3. 如果无法找到直接论文，必须写“工程”并说明原因；不得硬贴论文。  
4. 若线上分数退化，必须在 README 与本表同步写“已验证退化，不建议提交”。  
