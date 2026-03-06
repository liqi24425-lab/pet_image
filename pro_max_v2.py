import ssl
# 强行绕过 Mac SSL 证书验证
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🔥 Starting Pro Max V2 Pipeline! Using device: {device}")

# =====================================================
# 1️⃣ 数据增强与 Transform (加入 Random Erasing)
# =====================================================
train_aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(), # ToTensor 必须在 RandomErasing 之前
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), # 🔥 防过拟合神器：50%概率随机擦除
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

static_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================
# 2️⃣ Phase 1: 端到端深度微调 (EfficientNet + MixUp)
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 1: 端到端微调 EfficientNet-B0 (加入 MixUp 增强)")
print("="*50)

train_dir = "train"
full_train_dataset = datasets.ImageFolder(train_dir, transform=train_aug_transform)
class_names = full_train_dataset.classes
num_classes = len(class_names)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])
val_ds.dataset.transform = static_transform 

train_loader_aug = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

# 🔥 换血：加载更强的 EfficientNet-B0
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# EfficientNet 专用的差异化学习率配置
optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5},    # 底层给极小学习率
    {'params': model.classifier.parameters(), 'lr': 1e-3}   # 分类头给大学习率
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

epochs = 15
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader_aug:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 🔥 MixUp 核心逻辑：打乱并融合图片与标签
        alpha_mixup = 0.2
        lam = np.random.beta(alpha_mixup, alpha_mixup) if alpha_mixup > 0 else 1
        index = torch.randperm(images.size(0)).to(device)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        
        outputs = model(mixed_images)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_acc = correct / total
    scheduler.step(val_acc)
    print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_finetuned_effnet.pth")

# =====================================================
# 3️⃣ Phase 2: 全局静态特征提取 (EfficientNet 1280维特征)
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 2: 使用最强权重，离线提取全量训练集特征")
print("="*50)

model.load_state_dict(torch.load("best_finetuned_effnet.pth"))
# 把最后的线性层换成透明通道，直接输出 1280 维特征矩阵
model.classifier[1] = nn.Identity()
model.eval()

static_train_dataset = datasets.ImageFolder(train_dir, transform=static_transform)
static_train_loader = DataLoader(static_train_dataset, batch_size=32, shuffle=False, num_workers=0)

X_features = []
y_labels = []

with torch.no_grad():
    for images, labels in tqdm(static_train_loader, desc="Extracting Train Features"):
        images = images.to(device)
        features = model(images)  # 输出维度: (batch, 1280)
        X_features.append(features.cpu().numpy())
        y_labels.append(labels.numpy())

X_train = np.vstack(X_features)
y_train = np.concatenate(y_labels)
print(f"✅ 特征提取完毕! X_train 维度: {X_train.shape}")

# =====================================================
# 4️⃣ Phase 3: 严谨的统计学建模 (Ridge + GridSearchCV)
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 3: 构建 Ridge Logistic 回归与严谨的交叉验证")
print("="*50)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000))
])

param_grid = {"clf__C": np.logspace(-3, 3, 15)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
print("正在运行 5-Fold GridSearchCV 搜索最优超参数...")
grid.fit(X_train, y_train)

print(f"⭐ 最优 C 值: {grid.best_params_['clf__C']:.4f}")
print(f"⭐ 5折交叉验证平均准确率: {grid.best_score_:.4f}")

best_ml_model = grid.best_estimator_

# =====================================================
# 5️⃣ Phase 4: 盲测集推理 (引入视觉 RAG 检索增强)
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 4: 盲测集推理 (Visual RAG 融合)")
print("="*50)

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        return self.transform(Image.open(img_path).convert("RGB")), self.image_files[idx]

test_dir = "test/test" if os.path.exists("test/test") else "test"
test_dataset = TestDataset(test_dir, transform=static_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

X_test_features = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Extracting Test Features"):
        images = images.to(device)
        features = model(images)
        X_test_features.append(features.cpu().numpy())

X_test = np.vstack(X_test_features)

# ---------------------------------------------------------
# 🧠 视觉 RAG 核心逻辑
# ---------------------------------------------------------
print("正在执行知识库检索与决策融合...")
P_ridge = best_ml_model.predict_proba(X_test) 

def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-8)

X_train_norm = l2_normalize(X_train)
X_test_norm = l2_normalize(X_test)
sim_matrix = np.dot(X_test_norm, X_train_norm.T)

k = 5       
alpha = 0.8 
P_knn = np.zeros((X_test.shape[0], num_classes))

for i in range(X_test.shape[0]):
    top_k_indices = np.argsort(sim_matrix[i])[-k:]
    top_k_labels = y_train[top_k_indices]
    for label in top_k_labels:
        P_knn[i, label] += 1.0 / k

P_final = alpha * P_ridge + (1 - alpha) * P_knn
final_preds = np.argmax(P_final, axis=1)

idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in final_preds]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_pro_max_v2.csv", index=False)

print(f"\n🎉 绝杀达成！submission_pro_max_v2.csv 已经生成。")