import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 1. 动态数据增强 (Transforms)
# ===============================
# 训练集：加入更多增强策略，因为端到端可以动态消化这些方差
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集/测试集：绝对不能有随机性
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===============================
# 2. 加载并划分数据集
# ===============================
train_dir = "train"
# Kaggle 通常只有 train 和 test，我们需要自己切分一个验证集来监控泛化能力
full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
class_names = full_train_dataset.classes
num_classes = len(class_names)

# 80% 训练, 20% 验证
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 强制覆盖验证集的 transform，去除随机性
val_dataset.dataset.transform = val_test_transform 

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Classes: {class_names}")
print(f"Training images: {train_size} | Validation images: {val_size}")

# ===============================
# 3. 构建微调模型
# ===============================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 换头手术：将原本输出 1000 类的全连接层，替换为我们自己的 num_classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# ===============================
# 4. 差异化学习率配置
# ===============================
# 原本的特征提取层给极小的学习率，新加的全连接层给大一点的学习率
optimizer = optim.AdamW([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4) # weight_decay 在深度学习中等价于 L2 正则化

criterion = nn.CrossEntropyLoss()
# 学习率调度器：如果验证集准确率不再提升，自动降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# ===============================
# 5. 端到端训练循环
# ===============================
epochs = 15
best_val_acc = 0.0

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_train_loss = train_loss / total
    epoch_train_acc = correct / total
    
    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_val_loss = val_loss / total
    epoch_val_acc = correct / total
    
    scheduler.step(epoch_val_acc)
    
    print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
    # 保存最佳模型
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_resnet18.pth")
        print(">>> Best model saved!")
    print("-" * 40)

# ===============================
# 6. 生成 Kaggle Submission
# ===============================
print("Loading best model for inference...")
model.load_state_dict(torch.load("best_resnet18.pth"))
model.eval()

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

test_dataset = TestDataset("test", transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

preds = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Predicting Test Set"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        preds.extend(predicted.cpu().numpy())

# 映射回字符串标签
idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_finetuned.csv", index=False)

print("submission_finetuned.csv saved! Ready for Kaggle.")