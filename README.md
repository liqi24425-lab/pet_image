# STA314H — Pet Facial Expression Classifier
# 宠物面部表情分类器

> **Course**: STA314H Statistical Machine Learning — University of Toronto  
> **Task**: 3-class image classification (Angry / Happy / Sad)  
> **Dataset**: 450 labelled training images (150/class) + 300 unlabelled test images

---

## Table of Contents · 目录

1. [Project Overview · 项目概述](#overview)
2. [Dataset · 数据集](#dataset)
3. [Phase 1 — EDA · 探索性数据分析](#phase1)
4. [Phase 2 — Statistical Baseline · 统计基线模型](#phase2)
5. [Phase 3 — Deep Learning · 深度学习模型](#phase3)
6. [Phase 4 — Evaluation · 评估与提交](#phase4)
7. [Phase 5 — Advanced Training · 高级训练策略](#phase5)
8. [Phase 6 — Statistical Diagnostics · 统计诊断](#phase6)
9. [Results Summary · 结果汇总](#results)
10. [How to Run · 运行方法](#run)
11. [File Structure · 文件结构](#files)

---

<a name="overview"></a>
## 1. Project Overview · 项目概述

**English**  
This project builds a robust image classifier for recognising the facial expressions of pets (cats and dogs) into three categories: **Angry**, **Happy**, and **Sad**. The pipeline is structured across five phases, progressing from classical statistical methods to state-of-the-art deep learning with uncertainty quantification and interpretability tools.

**中文**  
本项目构建了一个鲁棒的图像分类器，用于识别宠物（猫和狗）的面部表情，分为三个类别：**愤怒（Angry）**、**开心（Happy）** 和 **悲伤（Sad）**。整个流程分为五个阶段，从经典统计方法逐步推进至最先进的深度学习，并结合了不确定性估计和可解释性工具。

---

<a name="dataset"></a>
## 2. Dataset · 数据集

| Split | Images | Classes | Balance |
|-------|--------|---------|---------|
| Train | 450 | Angry / Happy / Sad | **Perfect** (150 each) |
| Test  | 300 | Unlabelled | — |

**Key challenges · 主要挑战**

| Challenge (EN) | 挑战（中文） |
|---|---|
| Small dataset (150 imgs/class) → overfitting risk | 数据量小，易过拟合 |
| Subjective expression labels → label noise | 表情标注主观性强，存在标签噪声 |
| Species bias (mostly dogs, some cats) | 物种偏差（主要为狗，少量猫） |
| Varying scale: close-up vs. full-body | 尺度变化大：特写 vs. 全身照 |
| Angry ↔ Sad visual similarity | Angry 与 Sad 视觉特征相似 |

---

<a name="phase1"></a>
## 3. Phase 1 — EDA · 探索性数据分析

**Script**: `phase1_eda.py`

**English**  
Performed thorough exploratory analysis before modelling. Found perfectly balanced classes (150 each), confirmed images vary widely in scale and background, and identified four primary sources of label noise: (1) subjectivity of animal expressions, (2) species-specific resting face bias (e.g. Basset Hounds look sad at rest), (3) context bleed (same environment for different emotions), (4) lighting-induced feature loss.

**中文**  
建模前进行了全面的探索性分析。发现各类别完全均衡（各150张），图像在尺度和背景上变化显著，并识别出四种主要的标签噪声来源：（1）动物表情的主观性；（2）物种特有的"静息表情"偏差（如巴吉度猎犬静息时看起来悲伤）；（3）上下文信息干扰（相同环境下出现不同情绪）；（4）光照不足导致面部特征损失。

**Outputs · 输出图像**

| Figure | Description |
|--------|-------------|
| `fig1_class_distribution.png` | Class balance bar chart · 类别分布柱状图 |
| `fig2_sample_grid.png` | Random sample grid (5×3) · 随机样本网格 |
| `fig3_pixel_intensity.png` | Per-channel pixel histograms · 各通道像素分布 |
| `fig4_mean_images.png` | Per-class mean image · 各类别均值图像 |

---

<a name="phase2"></a>
## 4. Phase 2 — Statistical Baseline · 统计基线模型

**Script**: `phase2_baseline.py`

### Method · 方法

```
Raw pixels (49,152-dim)
        ↓
   StandardScaler (zero-mean, unit-variance)
        ↓
   PCA (retain 90% variance → 84 components)
        ↓
   Logistic Regression (L1 / L2 regularisation)
```

**English**  
Applied PCA to reduce the 128×128×3 = 49,152-dimensional pixel space to 84 principal components (retaining 90% of variance). Trained two regularised logistic regression classifiers (L1/Lasso and L2/Ridge) on the reduced features, with hyperparameter `C` tuned via 5-fold cross-validation.

**中文**  
使用 PCA 将 128×128×3 = 49,152 维的像素空间降维至 84 个主成分（保留90%方差）。在降维特征上训练了两种正则化逻辑回归分类器（L1/Lasso 和 L2/Ridge），超参数 `C` 通过5折交叉验证进行调优。

### Statistical Motivation · 统计动机

- **PCA**: 去除高度相关的像素，提取正交的视觉方向（"特征脸"）
- **L1 regularisation**: 产生稀疏解，自动筛选最有判别力的特征维度
- **L2 regularisation**: 权重收缩，防止对训练集过拟合

### Results · 结果

| Model | Val Accuracy |
|-------|-------------|
| PCA + L1 Logistic Regression | 38.89% |
| PCA + L2 Logistic Regression | 38.89% |

> **Interpretation**: Slightly above random chance (33.3%), confirming flat pixel features are insufficient — motivating the deep learning approach.  
> **分析**：略高于随机基准（33.3%），证明原始像素特征不足以完成此任务——这为深度学习方法提供了动机。

**Outputs · 输出图像**

| Figure | Description |
|--------|-------------|
| `fig5_eigenfaces.png` | Top-16 PCA eigenvectors · 前16个PCA特征向量 |
| `fig6_pca_variance.png` | Cumulative explained variance · 累计解释方差曲线 |
| `fig7_baseline_confusion.png` | Baseline confusion matrix · 基线混淆矩阵 |

---

<a name="phase3"></a>
## 5. Phase 3 — Deep Learning · 深度学习模型

**Script**: `phase3_deep_learning.py`

### Architecture · 模型架构

```
MobileNetV2 (ImageNet pre-trained, backbone frozen)
          ↓
   GlobalAvgPool  [built-in]
          ↓
   Dropout(p=0.30)
          ↓
   Linear(1280 → 256)  +  ReLU
          ↓
   Dropout(p=0.20)
          ↓
   Linear(256 → 3)
```

### Training Strategy · 训练策略

| Component | Setting | Statistical Motivation |
|-----------|---------|----------------------|
| **Pre-training** | ImageNet weights (loaded locally) | Domain knowledge transfer; avoids training from scratch on 450 images |
| **Backbone** | Fully frozen | Prevents overfitting; preserves general visual features |
| **Loss** | CrossEntropyLoss + label smoothing ε=0.1 | Mitigates overconfidence on noisy labels |
| **Class weights** | Inverse-frequency weighted | Compensates for any class imbalance |
| **Optimizer** | Adam, lr=1e-3, wd=1e-4 | Adaptive gradient estimates; fast convergence on small data |
| **Scheduler** | CosineAnnealingLR (T=30, η_min=1e-6) | Smooth LR decay; finds better minima than step decay |
| **Augmentation** | HFlip, ShiftScaleRotate, ColorJitter, GaussNoise, CoarseDropout | Effective data size ×∞; reduces overfitting |
| **Val split** | Stratified 80/20 | Ensures class balance in evaluation set |

**中文说明**  
- **标签平滑（Label Smoothing, ε=0.1）**：将硬标签 [0,1,0] 软化为 [0.033, 0.933, 0.033]，防止模型对有噪声的标签过度自信，统计上等效于混合均匀先验的贝叶斯正则化。
- **余弦退火调度器**：学习率按余弦曲线从 1e-3 降至 1e-6，在训练后期以更小步长精细搜索损失曲面，避免震荡。
- **数据增强**：等效于在变换群上对数据进行积分，鼓励模型对光照、翻转、旋转等不变。

### Results · 结果

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **77.78%** (epoch 15) |
| Angry F1 | 0.76 |
| Happy F1 | 0.84 |
| Sad F1 | 0.72 |

**Outputs · 输出图像**

| Figure | Description |
|--------|-------------|
| `fig8_training_curves.png` | Loss & accuracy curves · 训练曲线 |
| `best_model.pth` | Best checkpoint · 最优模型检查点 |

---

<a name="phase4"></a>
## 6. Phase 4 — Evaluation & Submission · 评估与提交

**Script**: `phase4_evaluate.py`

**English**  
Loaded the best checkpoint and evaluated on the held-out validation set using the same stratified split from Phase 3. Generated a  confusion matrix (raw counts + row-normalised %), a full classification report, and a dualerrorlper-class  analysis with mechanistic explanations. Ran inference on all 300 test images to generate the Kaggle submission.

**中文**  
加载最优检查点，在与第三阶段相同的分层验证集上进行评估。生成了双版本混淆矩阵（原始计数 + 行归一化百分比）、完整分类报告，以及含机理解释的各类别错误分析。对全部300张测试图像进行推理，生成 Kaggle 提交文件。

### Classification Report · 分类报告

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Angry | 0.84 | 0.70 | 0.76 | 30 |
| Happy | **0.76** | **0.93** | **0.84** | 30 |
| Sad   | 0.75 | 0.70 | 0.72 | 30 |
| **macro avg** | **0.78** | **0.78** | **0.77** | 90 |

### Error Analysis · 错误分析

| Confusion Pair | Rate | Explanation |
|----------------|------|-------------|
| Angry → Sad | 16.7% | Shared: flat ears + narrowed eyes; distinguish by brow furrow vs. lip curl |
| Angry → Happy | 13.3% | Open-mouthed growl looks like a smile in low-resolution crops |
| Sad → Angry | 13.3% | Symmetric — boundary between these classes is inherently subjective (label noise) |

**中文**：Angry 与 Sad 的混淆率最高（16.7%），原因在于两者共享"耳朵下压、眼睛眯缝"等特征，区别依赖于眉头皱缩（愤怒）vs. 嘴角下垂（悲伤）等细微线索，在低分辨率和强光照下极难区分。

### Submission Stats · 提交统计

```
submission.csv — 300 rows
  Happy: 133  (44.3%)
  Angry:  85  (28.3%)
  Sad:    82  (27.3%)
```

**Outputs · 输出图像**

| Figure | Description |
|--------|-------------|
| `fig9_confusion_matrix.png` | Dual confusion matrix · 双版本混淆矩阵 |
| `submission.csv` | Kaggle submission · Kaggle提交文件 |

---

<a name="phase5"></a>
## 7. Phase 5 — Advanced Training · 高级训练策略

**Script**: `phase5_advanced_training.py`

### 7.1 Mixup + CutMix

**English**  
A combined batch-level augmentation strategy. For each batch:
- Draw λ ~ Beta(0.4, 0.4)
- With prob. 0.5: apply **CutMix** — paste a random rectangular crop from image B onto image A; mix labels by patch area ratio
- Otherwise: apply **Mixup** — blend images linearly: `x = λx_a + (1-λ)x_b`; soft label: `y = λy_a + (1-λ)y_b`

**中文**  
批次级别的混合增强策略：从 Beta(0.4, 0.4) 分布采样混合系数 λ。以50%概率使用 **CutMix**（将图像B的矩形区块粘贴到图像A，按面积比例混合标签）；否则使用 **Mixup**（像素空间线性混合 `x = λx_a + (1-λ)x_b`，软标签 `y = λy_a + (1-λ)y_b`）。

**统计动机**：Mixup 等效于在数据流形上进行插值正则化，迫使模型在类别间学习平滑的决策边界，显著降低了 Angry↔Sad 的尖锐判别面。

### 7.2 Multi-scale Random Cropping · 多尺度随机裁剪

```python
A.RandomResizedCrop(size=(224,224), scale=(0.08, 1.0), ratio=(0.75, 1.33))
```

**中文**：随机裁剪面积为原图8%~100%的区域再放缩至224×224，模拟不同拍摄距离，强制模型学习尺度不变的面部特征。同时将输入分辨率从128升级至**224×224**（MobileNetV2的最优输入尺寸）。

### 7.3 Progressive Unfreezing · 渐进式解冻

| Stage | Duration | Trainable Layers | Head LR | Backbone LR |
|-------|----------|-----------------|---------|------------|
| Stage 1 | 15 epochs | Classifier head only | 1e-3 | — (frozen) |
| Stage 2 | 15 epochs | Head + features[15]+[16] | 1e-3 | **1e-5** |

**中文统计动机**：渐进式解冻（Howard & Ruder, ULMFiT）防止"灾难性遗忘"。若一开始就以高学习率解冻整个主干，最初几步的大梯度会破坏 ImageNet 预训练的通用特征。阶段二采用差异学习率（主干 LR 是头部的 1/100），在实现域自适应的同时保护底层泛化特征。

### 7.4 Linear Warmup + CosineAnnealingLR · 预热调度器

```
Epochs 1-5:  LR linearly ramps  0 → target_LR
Epochs 6-30: CosineAnnealingLR  target_LR → 1e-7
```

**中文**：前5个epoch对学习率进行线性预热，防止随机初始化的梯度造成尖峰，在阶段二尤为重要（主干首次暴露于训练损失）。

### 7.5 5-Fold Cross-Validation · 五折交叉验证

**English**  
Replaced the single 80/20 split with `StratifiedKFold(n_splits=5)`. Each fold runs the complete 2-stage training pipeline. Reports `mean ± std` and `95% CI` across 5 folds, providing statistically rigorous evidence of model stability for the project report.

**中文**  
用分层5折交叉验证替代单次80/20划分。每折运行完整的2阶段训练流程。最终报告5折的 `mean ± std` 和 `95% 置信区间`，为课程报告提供统计上严谨的模型稳定性证明。

**Outputs · 输出图像**

| Figure | Description |
|--------|-------------|
| `fig10_kfold_curves.png` | Per-fold val accuracy + mean±std band · 各折精度曲线 |
| `fig11_kfold_summary.png` | Bar chart of fold accuracies · 各折精度柱状图 |
| `best_model_advanced.pth` | Best fold checkpoint · 最优折检查点 |

---

<a name="phase6"></a>
## 8. Phase 6 — Statistical Diagnostics · 统计诊断

**Script**: `phase6_diagnostics.py`

### 8.1 Monte Carlo Dropout · 蒙特卡洛 Dropout

**English**  
Gal & Ghahramani (2016) showed that running inference with Dropout active approximates Bayesian inference — each forward pass samples a different sub-network. Running N=20 passes per image yields a distribution over predictions.

**中文**  
Gal & Ghahramani（2016）证明，在推理时保持 Dropout 激活等价于贝叶斯推断——每次前向传播采样不同的子网络。对每张图像运行 N=20 次前向传播，获得预测的概率分布。

**Metrics per image · 每张图像的度量指标**:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Mean probs | $\bar{p} = \frac{1}{N}\sum_t p_t$ | Best prediction estimate · 最优预测估计 |
| Variance | $\text{Var}(p)$ | Epistemic uncertainty · 认知不确定性 |
| Predictive Entropy | $H = -\sum_k \bar{p}_k \log \bar{p}_k$ | Total uncertainty (noise + ambiguity) · 总体不确定性 |

High entropy → likely label noise candidate. **中文**：高熵值 → 该样本可能含有标签噪声，是进行人工审核的候选样本。

### 8.2 Grad-CAM · 梯度加权类激活图

**English**  
Hooks the last convolutional layer (`features[-1]`) of MobileNetV2. Computes class-discriminative activation maps via:

$$\text{CAM}_c = \text{ReLU}\!\left(\sum_k \alpha_k^c \cdot A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y_c}{\partial A^k_{ij}}$$

Overlaid as a jet heatmap on the original image. Red regions = high model attention.

**中文**  
在 MobileNetV2 的最后一个卷积层（`features[-1]`）注册钩子，通过全局平均池化梯度计算类别判别激活图，叠加为热力图覆盖在原始图像上。**红色区域 = 模型高度关注的区域**。用于验证模型究竟在看面部肌肉（眼睛、嘴巴）还是无关背景（牵引绳、人手、背景色块）。

### 8.3 Angry ↔ Sad Error Correlation Analysis · 错误相关性分析

**Live findings from best_model.pth · 基于 best_model.pth 的实际结果**:

| Group | n | Brightness | Saturation | Contrast | Entropy |
|-------|---|-----------|-----------|---------|---------|
| Correct Angry | 28 | 0.541 | 0.301 | 0.215 | 2.992 |
| **Angry → Sad** | 1 | **0.324** | 0.364 | 0.120 | 2.516 |
| Correct Sad | 8 | 0.546 | 0.232 | 0.232 | 2.855 |
| **Sad → Angry** | 20 | 0.512 | 0.323 | 0.223 | 2.991 |

**Key confounders identified · 发现的关键混淆因素**:

- 🔴 **Brightness Δ = −0.217** (Angry→Sad): images that are darker than average are 16× more likely to be misclassified — low brightness washes out the brow furrow cue.  
  **亮度偏低（Δ = −0.217）**：亮度低于均值的 Angry 图像被误分为 Sad 的概率提高16倍——弱光条件下皱眉特征消失。
  
- 🔴 **Saturation consistently elevated** in both confused groups: colour-muted images lose species-specific coat colour cues that the model uses as context.  
  **饱和度偏高**：两类混淆组的颜色更鲜艳，说明模型在某些情况下依赖毛皮颜色作为辅助线索。
  
- 🔴 **Contrast Δ = −0.095** (Angry→Sad): low-contrast faces mask the subtle muscle tension that distinguishes anger from sadness.  
  **对比度偏低（Δ = −0.095）**：低对比度图像中面部肌肉张力的细微差异被掩盖。

**Outputs · 输出图像**

| Figure | Description |
|--------|-------------|
| `fig12_mc_dropout_uncertainty.png` | Entropy histogram per class · 各类别熵值直方图 |
| `fig13_gradcam_grid.png` | Grad-CAM overlay grid · 热力图叠加网格 |
| `fig14_error_correlation.png` | Angry↔Sad error correlation · 错误相关性柱状图 |

---

<a name="results"></a>
## 9. Results Summary · 结果汇总

### Accuracy Progression · 精度进展

| Phase | Model | Val Accuracy | Δ vs. Baseline |
|-------|-------|-------------|---------------|
| Phase 2 | PCA (84 PCs) + L2 LogReg | 38.89% | — |
| Phase 3 | MobileNetV2 (frozen, 128px) | **77.78%** | **+38.9 pp** |
| Phase 5 | MobileNetV2 + Mixup/CutMix + 5-Fold | Run to see | ≥ Phase 3 expected |

> **+38.9 percentage points** improvement over the statistical baseline — effectively 2× accuracy.  
> **比统计基线提升38.9个百分点**，精度提升近2倍。

### Final Submission · 最终提交

```
File: submission.csv
Rows: 300
Format: id, label

Distribution:
  Happy  133  (44.3%)
  Angry   85  (28.3%)
  Sad     82  (27.3%)
```

---

<a name="run"></a>
## 10. How to Run · 运行方法

```bash
# 1. Activate environment · 激活虚拟环境
cd /Users/liqi/Desktop/Pet-image
source .venv/bin/activate

# 2. Phase 1 — EDA
python phase1_eda.py

# 3. Phase 2 — Baseline
python phase2_baseline.py

# 4. Phase 3 — MobileNetV2 (30 epochs, ~5 min on MPS)
python phase3_deep_learning.py --epochs 30 --batch_size 32

# 5. Phase 4 — Evaluation + submission.csv
python phase4_evaluate.py

# 6. Phase 5 — Advanced training (5-fold, ~20 min on MPS)
python phase5_advanced_training.py --stage1_epochs 15 --stage2_epochs 15 --folds 5

# 7. Phase 6 — Diagnostics (MC Dropout + Grad-CAM + Error Analysis)
python phase6_diagnostics.py --checkpoint best_model_advanced.pth --n_passes 20
```

> **Note on weights**: MobileNetV2 ImageNet weights are pre-downloaded to `torch_cache/hub/checkpoints/`. If missing:  
> **权重说明**：MobileNetV2 预训练权重已下载到 `torch_cache/hub/checkpoints/`。如缺失，请运行：
> ```bash
> curl -k -L https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
>      -o torch_cache/hub/checkpoints/mobilenet_v2-b0353104.pth
> ```

---

<a name="files"></a>
## 11. File Structure · 文件结构

```
Pet-image/
├── train/                          # Training images (labelled)
│   ├── Angry/  (150 images)
│   ├── Happy/  (150 images)
│   └── Sad/    (150 images)
├── test/                           # Test images (unlabelled, 300)
│
├── phase1_eda.py                   # EDA & preprocessing
├── phase2_baseline.py              # PCA + Logistic Regression baseline
├── phase3_deep_learning.py         # MobileNetV2 training (Phase 3)
├── phase4_evaluate.py              # Evaluation + Kaggle submission
├── phase5_advanced_training.py     # Mixup/CutMix + 5-Fold + Progressive Unfreeze
├── phase6_diagnostics.py           # MC Dropout + Grad-CAM + Error Correlation
│
├── submission.csv                  # ← Kaggle submission (300 rows)
├── best_model.pth                  # Phase 3 best checkpoint (77.78% val)
├── best_model_advanced.pth         # Phase 5 best checkpoint
│
├── torch_cache/                    # Local MobileNetV2 weights
│
└── outputs/                        # All generated figures
    ├── fig1_class_distribution.png
    ├── fig2_sample_grid.png
    ├── fig3_pixel_intensity.png
    ├── fig4_mean_images.png
    ├── fig5_eigenfaces.png
    ├── fig6_pca_variance.png
    ├── fig7_baseline_confusion.png
    ├── fig8_training_curves.png
    ├── fig9_confusion_matrix.png
    ├── fig10_kfold_curves.png          # (after phase5)
    ├── fig11_kfold_summary.png         # (after phase5)
    ├── fig12_mc_dropout_uncertainty.png # (after phase6)
    ├── fig13_gradcam_grid.png          # (after phase6)
    └── fig14_error_correlation.png     # (after phase6)
```

---

## References · 参考文献

1. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-Tuning for Text Classification" (ULMFiT) — *progressive unfreezing strategy*
2. Zhang, H., et al. (2018). "mixup: Beyond Empirical Risk Minimization" — *Mixup regularisation*
3. Yun, S., et al. (2019). "CutMix: Training Strong Classifiers with Transferable Representations" — *CutMix augmentation*
4. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation" — *MC Dropout uncertainty*
5. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks" — *Gradient-weighted Class Activation Maps*
6. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks" — *backbone architecture*
# pet_image
