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
print(f"💎 开启最终突破方案：DINOv2-Large + Multi-Scale TTA! Using: {device}")

# =====================================================
# 1️⃣ 动态多尺度 TTA 预处理
# =====================================================
def get_tta_transforms():
    # 基础归一化
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # 定义 4 种 TTA 策略：原图、水平翻转、中心局部放大、镜像放大
    strategies = {
        "orig": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm]),
        "flip": transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
        "zoom": transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), norm]),
        "zoom_flip": transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm])
    }
    return strategies

# =====================================================
# 2️⃣ 加载“视觉核武器”
# =====================================================
print("\n🔥 正在加载 DINOv2-Large (1024D) & EffNet-B5 (2048D)...")
# EffNet-B5 (自带 SE 注意力，聚焦眼部 [cite: 7])
model_eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT).to(device).eval()
model_eff.classifier[1] = nn.Identity()

# DINOv2-Large (Meta 2024-2025 王者，专注生理地标 [cite: 52])
model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device).eval()

# CLIP (专注全局语义理解 [cite: 3])
model_clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

def extract_mega_features(img_dir, transform_key, desc, is_test=False):
    trans = get_tta_transforms()[transform_key]
    # CLIP 需要单独的归一化，这里为了速度统一使用 ImageNet 归一化，DINO/EffNet 收益更大
    
    class QuickDataset(Dataset):
        def __init__(self, directory):
            self.is_test = is_test
            if is_test:
                self.files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.png','.jpg','.jpeg')))
                self.dir = directory
            else:
                self.ds = datasets.ImageFolder(directory)
        def __len__(self): return len(self.files) if self.is_test else len(self.ds)
        def __getitem__(self, idx):
            img = Image.open(os.path.join(self.dir, self.files[idx])).convert("RGB") if self.is_test else self.ds[idx][0]
            label = self.files[idx] if self.is_test else self.ds[idx][1]
            return trans(img), label

    loader = DataLoader(QuickDataset(img_dir), batch_size=16, shuffle=False)
    f_eff, f_clip, f_dino, labels = [], [], [], []
    
    with torch.no_grad():
        for img, target in tqdm(loader, desc=f"{desc} [{transform_key}]"):
            img_dev = img.to(device)
            f_eff.append(model_eff(img_dev).cpu().numpy())
            f_dino.append(model_dino(img_dev).cpu().numpy())
            # CLIP 稍微兼容一下归一化差异
            f_clip.append(model_clip(pixel_values=img_dev).image_embeds.cpu().numpy())
            labels.extend(target)
            
    # 拼接维度: 2048 + 1024 + 512 = 3584 维
    return np.hstack([np.vstack(f_eff), np.vstack(f_dino), np.vstack(f_clip)]), np.array(labels)

# =====================================================
# 3️⃣ 阶段一：特征提取 (原图 + 伪标签递归)
# =====================================================
print("\n" + "="*30 + " 阶段 1: 提取 3584D 终极特征 " + "="*30)
# 我们直接读取之前脚本生成的 train_augmented 文件夹 (那里面已经包含 185 张高质量真题)
train_path = "train_augmented" if os.path.exists("train_augmented") else "train"
X_train, y_train = extract_mega_features(train_path, "orig", "提取训练集")

test_path = "test/test" if os.path.exists("test/test") else "test"
X_test_orig, test_ids = extract_mega_features(test_path, "orig", "TTA-1", is_test=True)
X_test_flip, _ = extract_mega_features(test_path, "flip", "TTA-2", is_test=True)
X_test_zoom, _ = extract_mega_features(test_path, "zoom", "TTA-3", is_test=True)

# =====================================================
# 4️⃣ 阶段二：ElasticNet 终极校准
# =====================================================
print("\n" + "="*30 + " 阶段 2: 3584D 空间深度搜索 " + "="*30)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='elasticnet', solver='saga', max_iter=4000))
])

param_grid = {
    "clf__C": [0.5, 1.0, 5.0],
    "clf__l1_ratio": [0.4, 0.6] # 聚焦于最平衡的区域
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"⭐ 突破性 CV 准确率: {grid.best_score_:.4f}")

# =====================================================
# 5️⃣ 阶段三：Multi-Scale Soft Voting 推理
# =====================================================
print("\n" + "="*30 + " 阶段 3: 多尺度 TTA 投票决策 " + "="*30)
prob1 = grid.best_estimator_.predict_proba(X_test_orig)
prob2 = grid.best_estimator_.predict_proba(X_test_flip)
prob3 = grid.best_estimator_.predict_proba(X_test_zoom)

# 动态加权投票 (原图和翻转图权重稍大)
final_prob = 0.4 * prob1 + 0.4 * prob2 + 0.2 * prob3
final_preds = np.argmax(final_prob, axis=1)

classes = datasets.ImageFolder("train").classes
submission = pd.read_csv("sample_submission.csv")
submission["label"] = [classes[i] for i in final_preds]
submission.to_csv("submission_v4_breakthrough.csv", index=False)
print("\n🏆 终极绝杀文件 submission_v4_breakthrough.csv 已生成。")