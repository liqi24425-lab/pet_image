import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🔥 启动终极屠龙方案！(2560维特征融合 + 迭代伪标签 + TTA) Using: {device}")

# =====================================================
# 1️⃣ 严谨的 Transforms 设定 (区分 EffNet 和 CLIP 的专属归一化)
# =====================================================
# EffNet 专属 (ImageNet 标准)
eff_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# CLIP 专属 (OpenAI 标准)
clip_norm = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

# 原图提取 Pipeline
eff_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), eff_norm])
clip_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), clip_norm])

# TTA 镜像提取 Pipeline (强制水平翻转)
eff_tta_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), eff_norm])
clip_tta_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), clip_norm])

# =====================================================
# 2️⃣ 加载双模型 (EffNetB5 + CLIP)
# =====================================================
print("\n🚀 正在加载双擎特征提取器...")
model_eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
model_eff.classifier[1] = nn.Identity() # 暴露 2048 维特征
model_eff = model_eff.to(device).eval()

model_clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
model_clip = model_clip.to(device).eval()

# 辅助函数：提取融合特征
def extract_super_features(img_dir, trans_eff, trans_clip, desc, is_test=False):
    # 构建临时 Dataset
    class TempDataset(Dataset):
        def __init__(self, directory, transform):
            self.dir = directory
            self.transform = transform
            if is_test:
                self.files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.png','.jpg','.jpeg')))
            else:
                self.dataset = datasets.ImageFolder(directory)
        def __len__(self): return len(self.files) if is_test else len(self.dataset)
        def __getitem__(self, idx):
            if is_test:
                path = os.path.join(self.dir, self.files[idx])
                img = Image.open(path).convert("RGB")
                return self.transform(img), self.files[idx]
            else:
                img, label = self.dataset[idx]
                return self.transform(img), label

    loader_eff = DataLoader(TempDataset(img_dir, trans_eff), batch_size=32, shuffle=False)
    loader_clip = DataLoader(TempDataset(img_dir, trans_clip), batch_size=32, shuffle=False)
    
    X_eff, X_clip, y_labels = [], [], []
    with torch.no_grad():
        for (img_e, target), (img_c, _) in tqdm(zip(loader_eff, loader_clip), total=len(loader_eff), desc=desc):
            feat_e = model_eff(img_e.to(device)).cpu().numpy()
            feat_c = model_clip(pixel_values=img_c.to(device)).image_embeds.cpu().numpy()
            X_eff.append(feat_e)
            X_clip.append(feat_c)
            y_labels.extend(target if not is_test else target)
            
    # 🔥 核心：横向拼接 2048 + 512 = 2560 维超级特征
    X_super = np.hstack([np.vstack(X_eff), np.vstack(X_clip)])
    return X_super, np.array(y_labels)

# =====================================================
# 3️⃣ 提取 2560 维超级特征矩阵
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 1: 提取 2560 维全量特征矩阵")
print("="*50)

# 确保读取原始干净的 train 文件夹
X_train_base, y_train_base = extract_super_features("train", eff_transform, clip_transform, "提取原始训练集", is_test=False)

test_dir = "test/test" if os.path.exists("test/test") else "test"
X_test_norm, test_names = extract_super_features(test_dir, eff_transform, clip_transform, "提取测试集(原图)", is_test=True)
X_test_flip, _ = extract_super_features(test_dir, eff_tta_transform, clip_tta_transform, "提取测试集(镜像)", is_test=True)

print(f"✅ 特征组装完毕！训练集维度: {X_train_base.shape}")

# =====================================================
# 4️⃣ 迭代伪标签挖掘 (Iterative Pseudo-Labeling)
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 2: 在 2560 维空间中挖掘高置信度伪标签")
print("="*50)

base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=2000))
])
print("正在训练初代引路模型...")
base_pipeline.fit(X_train_base, y_train_base)

# 预测测试集寻找高分题
probs = base_pipeline.predict_proba(X_test_norm)
max_probs = np.max(probs, axis=1)
preds_base = np.argmax(probs, axis=1)

# 🎯 设定 95% 置信度阈值
pseudo_indices = np.where(max_probs >= 0.95)[0]
X_pseudo = X_test_norm[pseudo_indices]
y_pseudo = preds_base[pseudo_indices]

print(f"🎉 寻宝成功！在 2560 维超级模型眼下，找到了 {len(pseudo_indices)} 张极高置信度的考场真题！")

# 将伪标签矩阵并入训练矩阵，完成无监督域适应
X_train_ultimate = np.vstack([X_train_base, X_pseudo])
y_train_ultimate = np.concatenate([y_train_base, y_pseudo])

# =====================================================
# 5️⃣ 终极 ElasticNet 剃刀与网格搜索
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 3: 训练融合了伪标签的终极 ElasticNet 剃刀")
print("="*50)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='elasticnet', solver='saga', max_iter=3000))
])

param_grid = {
    "clf__C": np.logspace(-2, 2, 8),
    "clf__l1_ratio": [0.1, 0.5, 0.8] 
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_ultimate, y_train_ultimate)

print(f"⭐ 终极 C 值: {grid.best_params_['clf__C']:.4f}")
print(f"⭐ 终极 L1 剃刀比例: {grid.best_params_['clf__l1_ratio']:.2f}")
print(f"⭐ 含金量拉满的 CV 平均准确率: {grid.best_score_:.4f}")

# =====================================================
# 6️⃣ 融合 TTA 预测并生成最终文件
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 4: TTA 测试时增强推理")
print("="*50)

best_model = grid.best_estimator_

print("1/2 计算测试集原图概率...")
P_norm = best_model.predict_proba(X_test_norm)

print("2/2 计算测试集镜像图概率...")
P_flip = best_model.predict_proba(X_test_flip)

# 💡 TTA 概率融合
P_final = (P_norm + P_flip) / 2.0
preds_final = np.argmax(P_final, axis=1)

# 映射回标签名称
train_dataset_for_classes = datasets.ImageFolder("train")
idx_to_class = {v: k for k, v in train_dataset_for_classes.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds_final]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_god_mode.csv", index=False)

print("\n🏆 绝杀达成！submission_god_mode.csv 已经生成。祝你在 Leaderboard 上一骑绝尘！")