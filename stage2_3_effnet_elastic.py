import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# 引入强大的 EfficientNet-B5 (原生自带 SE 注意力模块)
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🔥 阶段二&三：SE 注意力特征提取 + ElasticNet! Using: {device}")

# =====================================================
# 1. 静态 Transforms (不需要任何随机增强，保证特征绝对稳定)
# =====================================================
static_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================
# 2. 加载【包含伪标签扩增】的新数据集
# =====================================================
train_dir = "train_augmented" # ⚠️ 指向我们刚刚生成的新文件夹
full_train_dataset = datasets.ImageFolder(train_dir, transform=static_transform)
static_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"📦 成功加载扩增数据集，总图像数量: {len(full_train_dataset)}")

# =====================================================
# 3. 加载 EfficientNet-B5 并暴露高阶特征层
# =====================================================
print("\n🚀 加载 EfficientNet-B5 (内嵌 SE 注意力块)...")
model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
# 将分类头替换为 Identity，直接输出卷积块提取的 2048 维全局特征
model.classifier[1] = nn.Identity()
model = model.to(device)
model.eval()

# =====================================================
# 4. 提取 2048 维自适应焦点特征
# =====================================================
X_features, y_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(static_train_loader, desc="提取训练集特征 (含伪标签)"):
        features = model(images.to(device))
        X_features.append(features.cpu().numpy())
        y_labels.append(labels.numpy())

X_train = np.vstack(X_features)
y_train = np.concatenate(y_labels)

# =====================================================
# 5. ElasticNet 终极“剃刀”分类器
# =====================================================
print("\n🚀 构建 ElasticNet 剃刀分类器...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='elasticnet', solver='saga', max_iter=3000))
])

param_grid = {
    "clf__C": np.logspace(-2, 2, 10),
    "clf__l1_ratio": [0.1, 0.5, 0.9] # L1 比例：用于砍掉残留的环境噪声特征
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"⭐ 最优 C 值 (总体惩罚): {grid.best_params_['clf__C']:.4f}")
print(f"⭐ 最优 L1 比例 (剃刀烈度): {grid.best_params_['clf__l1_ratio']:.2f}")
print(f"⭐ 5折 CV 平均准确率: {grid.best_score_:.4f}")

# =====================================================
# 6. 测试集推理生成结果
# =====================================================
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        self.image_dir, self.transform = image_dir, transform
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        return self.transform(Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")), self.image_files[idx]

test_dir = "test/test" if os.path.exists("test/test") else "test"
test_loader = DataLoader(TestDataset(test_dir, transform=static_transform), batch_size=32, shuffle=False)

X_test_features = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="提取盲测集特征"):
        features = model(images.to(device))
        X_test_features.append(features.cpu().numpy())
X_test = np.vstack(X_test_features)

preds = grid.best_estimator_.predict(X_test)
idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}

submission = pd.read_csv("sample_submission.csv")
submission["label"] = [idx_to_class[i] for i in preds]
submission.to_csv("submission_effnet_pseudo.csv", index=False)
print("\n🎉 方案落地完成！submission_effnet_pseudo.csv 生成完毕！")