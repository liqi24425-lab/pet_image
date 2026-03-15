import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ===============================
# 0️⃣ Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 1️⃣ Transforms
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# 2️⃣ Load training data
# ===============================
train_dir = "train"
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))

# ===============================
# 3️⃣ Load pretrained ResNet18
# ===============================
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 去掉最后分类层
model = nn.Sequential(*list(model.children())[:-1])

# 冻结参数
for param in model.parameters():
    param.requires_grad = False

model = model.to(device)
model.eval()

# ===============================
# 4️⃣ 提取训练特征
# ===============================
X_features = []
y_labels = []

with torch.no_grad():
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.view(outputs.size(0), -1)

        X_features.append(outputs.cpu().numpy())
        y_labels.append(labels.numpy())

X_train = np.vstack(X_features)
y_train = np.concatenate(y_labels)

print("Feature shape:", X_train.shape)

# ===============================
# 5️⃣ 5-Fold CV 搜索最佳 C
# ===============================
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = 0
best_C = None

for C in C_values:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty='l1',
            C=C,
            max_iter=5000,
            solver='saga',
            n_jobs=-1
        ))
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    mean_score = scores.mean()
    print(f"C={C} | Mean CV Accuracy={mean_score:.4f} | Std={scores.std():.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_C = C

print("\nBest C found:", best_C)
print("Best CV Accuracy:", best_score)

# ===============================
# 6️⃣ 用最佳 C 重新训练最终模型
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = LogisticRegression(
    penalty='l1',
    C=best_C,
    max_iter=5000,
    solver='saga',
    n_jobs=-1
)

clf.fit(X_train_scaled, y_train)
print("Final model trained.")

# ===============================
# 7️⃣ Load test data
# ===============================
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]

test_dataset = TestDataset("test/test", transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# ===============================
# 8️⃣ 提取 test 特征
# ===============================
X_test_features = []
image_ids = []

with torch.no_grad():
    for images, names in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.view(outputs.size(0), -1)

        X_test_features.append(outputs.cpu().numpy())
        image_ids.extend(names)

X_test = np.vstack(X_test_features)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 9️⃣ 预测
# ===============================
preds = clf.predict(X_test_scaled)

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds]

# ===============================
# 🔟 生成 submission
# ===============================
submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission084-lasso.csv", index=False)

print("submission084-lasso.csv saved!")
