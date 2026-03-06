import ssl
# 强行绕过 Mac SSL 证书验证，防止下载权重报错
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🔥 Starting Ultimate Pipeline! Using device: {device}")

# =====================================================
# 1️⃣ 数据增强与 Transform
# =====================================================
# 用于 Phase 1 深度学习微调的动态增强（防过拟合）
train_aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 用于 Phase 2 特征提取 & Phase 4 测试集推理的静态变换（绝对不能有随机性）
static_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================
# 2️⃣ Phase 1: 端到端深度微调 (Advanced 核心逻辑)
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 1: 端到端微调 ResNet18 (打造专属特征提取器)")
print("="*50)

train_dir = "train"
full_train_dataset = datasets.ImageFolder(train_dir, transform=train_aug_transform)
class_names = full_train_dataset.classes
num_classes = len(class_names)

# 划分出验证集用于监控微调过程
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])
val_ds.dataset.transform = static_transform # 验证集切回静态变换

train_loader_aug = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

epochs = 12
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
        torch.save(model.state_dict(), "best_finetuned_resnet18.pth")

# =====================================================
# 3️⃣ Phase 2: 全局静态特征提取 (Pro 核心逻辑起始点)
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 2: 使用最强权重，离线提取全量训练集特征")
print("="*50)

# 加载我们刚刚微调出的最强权重
model.load_state_dict(torch.load("best_finetuned_resnet18.pth"))
# 神奇的一步：把用来分类的全连接层替换成“透明通道” (Identity)
# 这样模型前向传播到最后，输出的直接就是 512 维的高阶特征矩阵
model.fc = nn.Identity()
model.eval()

# ⚠️ 注意：提取特征时，我们必须使用全量训练集，且绝对不能 Shuffle，不能做随机增强
static_train_dataset = datasets.ImageFolder(train_dir, transform=static_transform)
static_train_loader = DataLoader(static_train_dataset, batch_size=32, shuffle=False, num_workers=0)

X_features = []
y_labels = []

with torch.no_grad():
    for images, labels in tqdm(static_train_loader, desc="Extracting Train Features"):
        images = images.to(device)
        features = model(images)  # 输出维度: (batch, 512)
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

# 穷举 15 个数量级的 C 值，寻找最完美的分类决策面
param_grid = {"clf__C": np.logspace(-3, 3, 15)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)

print("正在运行 5-Fold GridSearchCV 搜索最优超参数...")
grid.fit(X_train, y_train)

print(f"⭐ 交叉验证完美找到的最优 C 值: {grid.best_params_['clf__C']:.4f}")
print(f"⭐ 5折交叉验证平均准确率 (稳如泰山的泛化预估): {grid.best_score_:.4f}")

best_ml_model = grid.best_estimator_

# =====================================================
# 5️⃣ Phase 4: 测试集特征提取与推理预测
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 4: 盲测集推理与生成 Kaggle 提交文件")
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

# 自动兼容你的文件夹层级，如果 test/test 不存在就找 test
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

# 用通过网格搜索得到的“最强大脑”进行最后的定夺
preds = best_ml_model.predict(X_test)

idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_ultimate.csv", index=False)

print("\n🎉 完美收官！submission_ultimate.csv 已经生成。去 Kaggle 上拿你的最高分吧！")