import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =====================================================
# 0️⃣ Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# 1️⃣ Transforms
# =====================================================

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

# =====================================================
# 2️⃣ Load Training Data
# =====================================================

train_dir = "train"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=False,
                          num_workers=0)

print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))

# =====================================================
# 3️⃣ Load Pretrained ResNet18 (Feature Extractor)
# =====================================================

model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])

for param in model.parameters():
    param.requires_grad = False

model = model.to(device)
model.eval()

# =====================================================
# 4️⃣ Extract Train Features
# =====================================================

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

# =====================================================
# 5️⃣ Ridge Logistic + GridSearchCV
# =====================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=3000
    ))
])

param_grid = {
    "clf__C": np.logspace(-3, 3, 15)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("Running 5-fold CV for C selection...")
grid.fit(X_train, y_train)

print("Best C:", grid.best_params_["clf__C"])
print("Best CV Accuracy:", grid.best_score_)

best_model = grid.best_estimator_

# =====================================================
# 6️⃣ Load Test Dataset
# =====================================================

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

# =====================================================
# 7️⃣ Extract Test Features
# =====================================================

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

# =====================================================
# 8️⃣ Predict Using Best Model
# =====================================================

preds = best_model.predict(X_test)

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds]

# =====================================================
# 9️⃣ Create Submission
# =====================================================

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission1.csv", index=False)

print("submission.csv saved successfully!")


# =====================================================
# 🔎 Nested Cross-Validation (Unbiased Generalization Estimate)
# =====================================================

from sklearn.model_selection import StratifiedKFold, cross_val_score

print("\nRunning Nested 5-fold CV...")

# 外层 CV（真正评估泛化误差）
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 内层 CV（只用于调 C）
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=inner_cv,
    scoring="accuracy",
    n_jobs=-1
)

nested_scores = cross_val_score(
    nested_grid,
    X_train,
    y_train,
    cv=outer_cv,
    scoring="accuracy",
    n_jobs=-1
)

print("Nested 5-Fold Accuracy per fold:", nested_scores)
print("Nested Mean Accuracy:", nested_scores.mean())
print("Nested Std:", nested_scores.std())
