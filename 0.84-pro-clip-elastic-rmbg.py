import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import CLIPModel
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🔥 Starting CLIP + ElasticNet (物理去噪版)! Using device: {device}")

# =====================================================
# 1️⃣ CLIP 专属 Transforms (极其重要：严格匹配 CLIP 的预处理)
# =====================================================
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

static_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std)
])

# =====================================================
# 2️⃣ 加载【物理去噪后】的数据集
# =====================================================
train_dir = "train_rmbg" # ⚠️ 指向抠图后的文件夹
full_train_dataset = datasets.ImageFolder(train_dir, transform=static_transform)
static_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=False, num_workers=0)

# =====================================================
# 3️⃣ 加载 Hugging Face CLIP 模型
# =====================================================
print("\n🚀 加载 OpenAI CLIP (ViT-B/32) 模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# =====================================================
# 4️⃣ 提取 512 维高阶语义特征
# =====================================================
X_features, y_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(static_train_loader, desc="Extracting Train Features (RMBG)"):
        # CLIP 的 get_image_features 直接输出 512 维的纯正语义向量
        features = model.get_image_features(pixel_values=images.to(device))
        X_features.append(features.cpu().numpy())
        y_labels.append(labels.numpy())

X_train = np.vstack(X_features)
y_train = np.concatenate(y_labels)

# =====================================================
# 5️⃣ ElasticNet (L1 + L2) 剃刀分类器与二维网格搜索
# =====================================================
print("\n🚀 构建 ElasticNet 剃刀分类器...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='elasticnet', solver='saga', max_iter=3000))
])

# 二维搜索：寻找最佳惩罚力度 C，以及 L1 的占比 (l1_ratio)
param_grid = {
    "clf__C": np.logspace(-2, 2, 10),
    "clf__l1_ratio": [0.1, 0.5, 0.9] # 0.9 代表极其残酷的 L1 剃刀(杀掉大部分特征)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"⭐ 最优 C 值 (总体惩罚): {grid.best_params_['clf__C']:.4f}")
print(f"⭐ 最优 L1 比例 (剃刀烈度): {grid.best_params_['clf__l1_ratio']:.2f}")
print(f"⭐ 5折 CV 平均准确率: {grid.best_score_:.4f}")

# =====================================================
# 6️⃣ 盲测集推理
# =====================================================
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        self.image_dir, self.transform = image_dir, transform
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        return self.transform(Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")), self.image_files[idx]

test_dir = "test_rmbg/test" if os.path.exists("test_rmbg/test") else "test_rmbg" # ⚠️ 同样指向抠图后的测试集
test_loader = DataLoader(TestDataset(test_dir, transform=static_transform), batch_size=32, shuffle=False)

X_test_features = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Extracting Test Features (RMBG)"):
        features = model.get_image_features(pixel_values=images.to(device))
        X_test_features.append(features.cpu().numpy())
X_test = np.vstack(X_test_features)

preds = grid.best_estimator_.predict(X_test)
idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}

submission = pd.read_csv("sample_submission.csv")
submission["label"] = [idx_to_class[i] for i in preds]
submission.to_csv("submission_clip_rmbg.csv", index=False)
print("\n🎉 物理消噪版 submission_clip_rmbg.csv 生成完毕！")