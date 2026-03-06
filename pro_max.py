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
# 5️⃣ Phase 4: 高阶视觉 RAG 推理与预测融合
# =====================================================
print("\n" + "="*50)
print("🚀 PHASE 4: 盲测集推理 (引入视觉 RAG 检索增强)")
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
# 🧠 视觉 RAG 核心逻辑开始
# ---------------------------------------------------------
print("正在执行知识库检索 (Visual RAG)...")

# 1. 获取 Ridge 模型的全局先验预测概率
P_ridge = best_ml_model.predict_proba(X_test) # 维度: (N_test, num_classes)

# 2. 向量 L2 归一化 (为余弦相似度做准备)
def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-8)

X_train_norm = l2_normalize(X_train)
X_test_norm = l2_normalize(X_test)

# 3. 计算余弦相似度矩阵 (神来之笔：一次矩阵乘法完成所有图片的相似度对比)
# sim_matrix 维度: (N_test, N_train)
sim_matrix = np.dot(X_test_norm, X_train_norm.T)

# 4. RAG 超参数
k = 5       # 检索最相似的 5 张历史图片
alpha = 0.8 # 融合权重：80% 听 Ridge 大模型的，20% 听 RAG 检索的相似案例
P_knn = np.zeros((X_test.shape[0], num_classes))

# 5. Top-k 投票计算
for i in range(X_test.shape[0]):
    # 获取第 i 张测试图片在训练集中最相似的 k 个索引 (从小到大排序的最后 k 个)
    top_k_indices = np.argsort(sim_matrix[i])[-k:]
    # 获取这 k 个相似图片的真实标签
    top_k_labels = y_train[top_k_indices]
    
    # 统计 k 近邻的标签频率，转换为概率分布
    for label in top_k_labels:
        P_knn[i, label] += 1.0 / k

# 6. 决策融合 (Alpha Blending)
P_final = alpha * P_ridge + (1 - alpha) * P_knn

# 最终预测结果：取融合后概率最大的类别
final_preds = np.argmax(P_final, axis=1)

# ---------------------------------------------------------
# 🧠 视觉 RAG 核心逻辑结束
# ---------------------------------------------------------

idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in final_preds]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_rag_ultimate.csv", index=False)

print(f"\n🎉 完美收官！结合视觉 RAG 的预测文件 submission_rag_ultimate.csv 已经生成。")
print(f"参数提示: 当前 RAG 检索池 = {X_train.shape[0]}张图, 检索 k={k}, 融合权重 Alpha={alpha}")