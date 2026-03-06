import os
import copy
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from tqdm import tqdm

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
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===============================
# 2️⃣ Load Dataset
# ===============================
dataset = datasets.ImageFolder("train", transform=train_transform)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Classes:", dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ===============================
# 3️⃣ Model
# ===============================
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

# 解冻最后一个 block
for param in model.layer4.parameters():
    param.requires_grad = True

# 替换分类头
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

# ===============================
# 4️⃣ Loss & Optimizer
# ===============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3,
    factor=0.5
)

# ===============================
# 5️⃣ Training Loop
# ===============================
best_val_acc = 0
best_model = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(20):
    model.train()
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 5:
        print("Early stopping triggered")
        break

# 恢复表现最好的那一轮权重
model.load_state_dict(best_model)

# 💡 核心修改点：把这个 0.84 的最强权重永久保存到硬盘！
torch.save(model.state_dict(), "best_resnet18.pth")
print("\n✅ 最优模型权重已被保存为 'best_resnet18.pth'！\n")

# ===============================
# 6️⃣ Test Prediction
# ===============================
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

test_dataset = TestDataset("test/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
preds = []
image_ids = []

with torch.no_grad():
    for images, names in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(1).cpu().numpy()
        preds.extend(predictions)
        image_ids.extend(names)

idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_end2end.csv", index=False)

print("submission_end2end.csv saved!")