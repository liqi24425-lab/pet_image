import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
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
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=False,  # ⚠ 特征提取时不要打乱
                          num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of training images:", len(train_dataset))

# ===============================
# 3️⃣ Load pretrained ResNet
# ===============================

model = models.resnet18(pretrained=True)

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

        outputs = model(images)  # (batch, 512, 1, 1)
        outputs = outputs.view(outputs.size(0), -1)  # (batch, 512)

        X_features.append(outputs.cpu().numpy())
        y_labels.append(labels.numpy())

X_train = np.vstack(X_features)
y_train = np.concatenate(y_labels)

print("Feature shape:", X_train.shape)

# ===============================
# 5️⃣ 标准化特征
# ===============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ===============================
# 6️⃣ Lasso / Ridge Logistic
# ===============================

# L2 (Ridge)
clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
)

# 如果想用 L1：
# clf = LogisticRegression(
#     penalty='l1',
#     C=1.0,
#     max_iter=1000,
#     solver='saga',
# )

clf.fit(X_train_scaled, y_train)

print("Training finished.")

# ===============================
# 7️⃣ Test Dataset
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
test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=False,
                         num_workers=0)

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

# 标准化（用训练集 scaler）
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
submission.to_csv("submission.csv", index=False)

print("submission.csv saved!")

# 定义 pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty='l2',
        C=1.0,
        max_iter=1000,
        solver='lbfgs'
    ))
])

# 5-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X_train, y_train,
                         cv=cv,
                         scoring='accuracy')

print("5-Fold Accuracy per fold:", scores)
print("Mean Accuracy:", scores.mean())
print("Std:", scores.std())
