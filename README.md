# STA314H — Pet Facial Expression Classifier (宠物面部表情分类器)
🏆 **Final Score: 0.913**

> **Course**: STA314H Statistical Machine Learning — University of Toronto  
> **Task**: 3-class pet facial expression classification (Angry / Happy / Sad)  

This project documents the entire evolutionary journey of a high-performance image classifier for pet facial expressions. We progressed from basic statistical baselines (PCA + Logistic Regression) to robust CNNs, and finally to cutting-edge Vision Transformers with sophisticated feature fusion strategies.

---

## 🚀 The Path to 0.913: Model Evolution & Experimental Analysis

Building this classifier was a continuous process of hypothesis, experimentation, and physiological analysis of pet expressions. Below are the 4 major breakthrough and experimental phases that guided us to the final winning score.

### 1. 奠定强基座 (Score: 0.906) —— `ultimate_god_mode.py`
**Strategy:**
Feature fusion between **EfficientNet** (designed to capture local facial textures) and **CLIP** (providing global semantic understanding), combined with high-confidence pseudo-labeling for data augmentation.

**Conclusion & Analysis:**
This approach confirmed that the **"Feature Fusion + Semi-Supervised Learning"** direction was absolutely correct. However, we eventually hit a performance bottleneck. The root cause was that CLIP's global semantic embeddings introduced severe background context interference (anthropomorphic bias), which distracted the model from purely focusing on the facial expressions themselves.

### 2. 遭遇横向冗余 (Score: 0.906) —— `mega_ensemble_v3.py`
**Strategy:**
Attempted to blindly increase feature dimensionality by adding **DINOv2-Small** (384 dimensions) into the ensemble.

**Conclusion & Analysis:**
This expansion proved to be **ineffective**. The newly introduced shallow features from DINOv2-Small highly overlapped with the existing EfficientNet representations. As a result of the L1 regularization (the "razor effect") in our ElasticNet classification head, these redundant features were directly filtered out, failing to provide any additional incremental information or performance gain.

### 3. 垂直深挖夺冠 (Score: 0.913) 🥇 —— `final_breakthrough_v4.py`
**Strategy:**
Upgraded the vision backbone to **DINOv2-Large** to capture fine facial muscle movements. Crucially, we introduced a **weighted Zoom (Center Crop) Test-Time Augmentation (TTA)** to aggressively force the model to focus purely on the animal's facial features.

**Conclusion & Analysis:**
**Accurate breakthrough!** DINOv2-Large provided an exceptionally high-quality "physiological landmark" resolution. By combining this with a local zoom-in strategy (Zoom TTA) that stripped away distracting edge background noise, the model perfectly hit the biological core rule of this dataset: *"Pet emotion classification highly depends on the eyes."* This extreme focus on the eyes and muzzle without environmental distraction led directly to our peak competition score.

### 4. 过度参数化翻车 (Score: 0.900) —— `v6_giant_ultimate.py`
**Strategy:**
Employed the most brute-force parameter approach using **DINOv2-Giant** (1536 dimensions) along with a massive 10-Crop full-image cropping strategy.

**Conclusion & Analysis:**
**Suffered from the Curse of Dimensionality and Noise Backlash.** With our relatively small dataset size, indiscriminately expanding the feature space to over 4096 dimensions triggered severe overfitting. Furthermore, the 10-crop strategy generated patches containing non-informative background areas (like wall corners or patches of grass). These irrelevant crops introduced fatal background noise that fundamentally biased and misled the model's decision-making process.

---

## 🛠 Project Structure & Early Phases
For historical tracking, the repository includes all our preliminary experiments:
- **Phase 1-2**: EDA, PCA + Ridge/Lasso baseline (`phase1_eda.py`, `phase2_baseline.py`)
- **Phase 3**: End-to-end baseline CNNs (ResNet18, EfficientNet-B0) with Label Smoothing and Cosine Annealing.
- **Phase 4-5**: Mixup, CutMix, 5-Fold Cross Validation (`phase5_advanced_training.py`)
- **Phase 6**: Statistical Diagnostics (MC Dropout Uncertainty, Grad-CAM, Error Correlation Analysis).

## ⚙️ How to Reproduce the Winning Submission
To generate the final Kaggle submission with the peak 0.913 score:

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run the winning inference script
python final_breakthrough_v4.py
```
