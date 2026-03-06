# STA314H — Pet Facial Expression Classifier (宠物面部表情分类器)
🏆 **Final Score: 0.913**

> **Course**: STA314H Statistical Machine Learning — University of Toronto  
> **Task**: 3-class pet facial expression classification (Angry / Happy / Sad)  

This project documents the entire evolutionary journey of a high-performance image classifier for pet facial expressions. We progressed from basic statistical baselines (PCA + Logistic Regression) to robust CNNs, and finally to cutting-edge Vision Transformers with sophisticated feature fusion strategies.

---

## 🚀 The Path to 0.913: Model Evolution & Experimental Analysis

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
* **Score: 0.806 | `0.84-lasso`**
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
