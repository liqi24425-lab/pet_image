# 全版本方法来源标注（Version -> Method Source）

说明：
- `文献驱动`：该版本核心思路有明确论文来源。
- `工程实验`：主要是本仓库内经验改造/调参/融合策略，没有直接单篇论文对应。
- 本表用于满足“每一版都可追溯来源”的协作要求。

## A. 早期基线与 0.84 系列

| 版本/脚本 | 方法摘要 | 来源类型 | 主要来源 |
|---|---|---|---|
| `0.84.py` / `0.84-pro.py` / `0.84-end-to-end.py` | CNN 特征 + 线性分类 | 文献驱动 | ResNet: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| `0.84-pro-clip-elastic-raw.py` / `...rmbg.py` | CLIP 特征 + 线性头 | 文献驱动 | CLIP: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| `0.84_adavanced.py` | 更强正则/训练技巧 | 工程实验 | 无单一固定论文（经验调参） |
| `ultimate_baseline.py` / `ultimate_baseline_v2.py` | end-to-end baseline | 工程实验 | 基于通用 CNN 训练范式 |
| `pro_max.py` / `pro_max_v2.py` | baseline + 局部策略/KNN | 工程实验 | 无单一固定论文 |

## B. 突破主线（v4 之后）

| 版本/脚本 | 方法摘要 | 来源类型 | 主要来源 |
|---|---|---|---|
| `final_breakthrough_v4.py` | DINOv2 + EffNet + CLIP 融合 + TTA | 文献驱动 | DINOv2: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193), EfficientNet: [arXiv:1905.11946](https://arxiv.org/abs/1905.11946), CLIP: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| `final_breakthrough_v10.py` | 5-fold OOF stacking + 保守 pseudo gate | 文献驱动 + 工程实验 | Stacking (Wolpert, 1992), Pseudo-labeling/Self-training: [arXiv:1901.09151](https://arxiv.org/abs/1901.09151) |
| `final_breakthrough_v11_backbone_upgrade.py` | EffNetV2-L + CLIP-L/14 升级尝试 | 文献驱动 | EfficientNetV2: [arXiv:2104.00298](https://arxiv.org/abs/2104.00298), CLIP |
| `final_breakthrough_v12_effective_hf_fusion.py` | 保守骨干 + CLIP-L 融合 | 文献驱动 + 工程实验 | CLIP + OOF 融合经验 |
| `final_breakthrough_v13_hybrid_stack.py` | `meta + weighted` 混合头 | 工程实验 | 无单一固定论文（集成校准经验） |
| `final_breakthrough_v14_robust_prior.py` | 先验重加权与鲁棒推理 | 工程实验 | 无单一固定论文 |
| `final_breakthrough_v15_seed_ensemble.py` | 多种子 probe 集成 | 工程实验 | 集成学习常规实践 |
| `final_breakthrough_v16_adaptive_roi.py` | 自适应 ROI TTA | 工程实验 | 与注意力/局部区域思想相关，但实现为经验策略 |
| `final_breakthrough_v17_species_adapter.py` | species adapter + full/pseudo 路线 | 文献驱动 + 工程实验 | 伪标签与多任务/适配思想参考 FER 近年论文 |
| `final_breakthrough_v18_hardset_boost.py` | hard-sample boost | 工程实验 | 难例重加权经验 |
| `final_breakthrough_v19_roi_gate.py` | ROI gate + TTA gate | 工程实验 | 多视角门控经验 |
| `final_breakthrough_v20_calibrated_hybrid.py` | 温度校准 + hybrid 决策头 | 文献驱动 + 工程实验 | Temperature Scaling: [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| `final_breakthrough_v21_aggressive_siglip_roi_gate.py` | SigLIP + ROI/TTA gate 激进版 | 文献驱动 + 工程实验 | SigLIP: [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) |
| `final_breakthrough_v22_lit_noise_consistency.py` | 噪声鲁棒重加权 + 多视角一致性门控 | 文献驱动 | LA-Net: [arXiv:2307.09023](https://arxiv.org/abs/2307.09023), Multi-Task Multi-Modal SSL FER: [arXiv:2404.10904](https://arxiv.org/abs/2404.10904) |

## C. 当前协作规则（文献标注）

1. 新版本提交前，必须在本文件新增一行，标注 `来源类型` 与 `主要来源链接`。  
2. 若为纯工程策略，必须写明“工程实验”并解释不引用单篇论文的原因。  
3. 若线上结果退化，需在 README 与本文件同步记录“结论：不建议提交”。  

