import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("🚀 正在极速生成最终预测文件...")

# 1. 准备静态与 TTA Transform
static_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tta_flip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = "train"
full_train_dataset = datasets.ImageFolder(train_dir, transform=static_transform)
num_classes = len(full_train_dataset.classes)

# 2. 组装模型并加载你辛苦炼好的权重
model = efficientnet_b0()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("best_finetuned_effnet.pth", map_location=device))
model.classifier[1] = nn.Identity() # 切回特征提取模式
model = model.to(device)
model.eval()

# 3. 极速提取训练集特征
static_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=False, num_workers=0)
X_features, y_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(static_train_loader, desc="提取历史特征"):
        X_features.append(model(images.to(device)).cpu().numpy())
        y_labels.append(labels.numpy())
X_train = np.vstack(X_features)
y_train = np.concatenate(y_labels)

# 4. 极速训练 Ridge 模型
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000, C=0.0193)) # 直接使用之前测出的最优 C 值
])
pipeline.fit(X_train, y_train)

# 5. TTA 测试集推理
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        self.image_dir, self.transform = image_dir, transform
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        return self.transform(Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")), self.image_files[idx]

test_dir = "test/test" if os.path.exists("test/test") else "test"

def get_test_probs(transform_pipeline):
    loader = DataLoader(TestDataset(test_dir, transform=transform_pipeline), batch_size=32, shuffle=False)
    features = []
    with torch.no_grad():
        for images, _ in loader:
            features.append(model(images.to(device)).cpu().numpy())
    return pipeline.predict_proba(np.vstack(features))

print("正在计算原图与 TTA 翻转概率...")
prob_final = (get_test_probs(static_transform) + get_test_probs(tta_flip_transform)) / 2.0
preds = np.argmax(prob_final, axis=1)

# 6. 安全保存 (防崩溃处理)
idx_to_class = {v: k for k, v in full_train_dataset.class_to_idx.items()}
pred_labels = [idx_to_class[i] for i in preds]

try:
    submission = pd.read_csv("sample_submission.csv")
    submission["label"] = pred_labels
    submission.to_csv("submission_ultimate_v2.csv", index=False)
    print("🎉 绝杀达成！submission_ultimate_v2.csv 已经完美生成！")
except FileNotFoundError:
    print("⚠️ 仍然没有找到 sample_submission.csv！我将为你强制生成一个无模板的 CSV。请检查格式是否符合 Kaggle 要求。")
    pd.DataFrame({"id": TestDataset(test_dir).image_files, "label": pred_labels}).to_csv("submission_ultimate_v2_fallback.csv", index=False)