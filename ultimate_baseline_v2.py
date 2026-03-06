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
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🔥 Starting Ultimate Pipeline V2! Using device: {device}")

# =====================================================
# 1️⃣ 数据增强与 Transform
# =====================================================
# 💡 升级 1：加入 ColorJitter 防止对特定光线过拟合，加入 RandomErasing 强迫看全局
train_aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)), # 涂黑小块区域
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

static_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试时增强 (TTA) 用的翻转 Transform
tta_flip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0), # 强制水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================
# 2️⃣ Phase 1: 端到端深度微调
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 1: 端到端微调 EfficientNet-B0")
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

# 💡 升级 2：加载 EfficientNet-B0
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

# 💡 升级 3：标签平滑 (Label Smoothing) 核心防过拟合神技
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-3) # 加大 L2 正则化

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

epochs = 15
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader_aug:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
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
# 3️⃣ Phase 2: 全局静态特征提取 
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 2: 提取 1280 维全局静态特征")
print("="*50)

model.load_state_dict(torch.load("best_finetuned_effnet.pth"))
model.classifier[1] = nn.Identity()
model.eval()

static_train_dataset = datasets.ImageFolder(train_dir, transform=static_transform)
static_train_loader = DataLoader(static_train_dataset, batch_size=32, shuffle=False, num_workers=0)

X_features, y_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(static_train_loader, desc="Extracting Train Features"):
        X_features.append(model(images.to(device)).cpu().numpy())
        y_labels.append(labels.numpy())

X_train = np.vstack(X_features)
y_train = np.concatenate(y_labels)

# =====================================================
# 4️⃣ Phase 3: Ridge 回归与 CV
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 3: 严谨的 Ridge 交叉验证")
print("="*50)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000))
])

param_grid = {"clf__C": np.logspace(-3, 3, 15)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"⭐ 最优 C 值: {grid.best_params_['clf__C']:.4f}")
print(f"⭐ 5折交叉验证平均准确率 (挤掉泡沫后的真实预估): {grid.best_score_:.4f}")

best_ml_model = grid.best_estimator_

# =====================================================
# 5️⃣ Phase 4: 测试集推理与 TTA 融合
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 4: 盲测集推理与 TTA(测试时增强) 概率融合")
print("="*50)

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        self.image_dir, self.transform = image_dir, transform
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        return self.transform(Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")), self.image_files[idx]

test_dir = "test/test" if os.path.exists("test/test") else "test"

# 辅助函数：提取某一种 Transform 下的特征并预测概率
def get_test_probs(transform_pipeline):
    loader = DataLoader(TestDataset(test_dir, transform=transform_pipeline), batch_size=32, shuffle=False)
    features = []
    with torch.no_grad():
        for images, _ in loader:
            features.append(model(images.to(device)).cpu().numpy())
    X_test = np.vstack(features)
    return best_ml_model.predict_proba(X_test)

print("1. 正在获取原图预测概率...")
prob_normal = get_test_probs(static_transform)

print("2. 正在获取翻转图像预测概率 (TTA)...")
prob_flipped = get_test_probs(tta_flip_transform)

# 💡 升级 4：TTA 概率融合 (50% 原图判定 + 50% 镜像判定)
prob_final = (prob_normal + prob_flipped) / 2.0
preds = np.argmax(prob_final, axis=1)

idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_ultimate_v2.csv", index=False)

print("\n🎉 完美收官！submission_ultimate_v2.csv 已经生成。")