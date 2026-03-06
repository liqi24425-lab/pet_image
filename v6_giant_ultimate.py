import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"👑 启动 V6 终极进化方案：DINOv2-Giant + 10-Crop TTA + 伪标签 3.0! Using: {device}")

# =====================================================
# 1️⃣ 10-Crop TTA 专用 Transform
# =====================================================
def get_10crop_transform():
    # 使用 ImageNet 归一化，兼容 EffNet 和 DINOv2
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224), # 生成中心、四个角及其镜像，共10张图
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([norm(crop) for crop in crops]))
    ])

# =====================================================
# 2️⃣ 加载“视觉天花板” DINOv2-Giant
# =====================================================
print("\n🔥 正在激活：DINOv2-Giant (1536D) + EffNet-B5 (2048D) + CLIP (512D)...")
model_eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT).to(device).eval()
model_eff.classifier[1] = nn.Identity()

# 加载 Giant 模型，其对捕捉细微肌肉收缩具有顶级解析力
model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device).eval()

model_clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

def extract_mega_features_v6(img_dir, desc, is_test=False):
    # 基础 transform 用于训练集特征提取
    base_trans = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class UnifiedDS(Dataset):
        def __init__(self, directory):
            self.is_test = is_test
            if is_test:
                self.files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.png','.jpg','.jpeg')))
                self.dir = directory
            else:
                self.ds = datasets.ImageFolder(directory)
        def __len__(self): return len(self.files) if self.is_test else len(self.ds)
        def __getitem__(self, idx):
            if self.is_test:
                path = os.path.join(self.dir, self.files[idx])
                return base_trans(Image.open(path).convert("RGB")), self.files[idx]
            img, label = self.ds[idx]
            return base_trans(img), label

    loader = DataLoader(UnifiedDS(img_dir), batch_size=8, shuffle=False)
    f_all, labels = [], []
    
    with torch.no_grad():
        for img, target in tqdm(loader, desc=desc):
            img_dev = img.to(device)
            # 组装 4096 维超级特征
            e = model_eff(img_dev).cpu().numpy()
            d = model_dino(img_dev).cpu().numpy()
            c = model_clip(pixel_values=img_dev).image_embeds.cpu().numpy()
            f_all.append(np.hstack([e, d, c]))
            labels.extend(target)
            
    return np.vstack(f_all), np.array(labels)

# =====================================================
# 3️⃣ 阶段一：递归伪标签 3.0 (Recursive Pseudo-Labeling)
# =====================================================
print("\n" + "="*50 + "\n🚀 PHASE 1: 递归伪标签 3.0 数据扩增\n" + "="*50)
# 首先基于 4096 维特征训练一个“专家”引路模型
X_train_orig, y_train_orig = extract_mega_features_v6("train", "提取原始训练集特征")
test_dir = "test/test" if os.path.exists("test/test") else "test"
X_test_base, test_files = extract_mega_features_v6(test_dir, "扫描测试集生成伪标签", is_test=True)

expert = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=2000))])
expert.fit(X_train_orig, y_train_orig)

# 递归挖掘：利用 0.91+ 级别的特征空间挑选高确信度样本
probs = expert.predict_proba(X_test_base)
max_p, preds = np.max(probs, axis=1), np.argmax(probs, axis=1)

# 筛选置信度 > 95% 的“隐藏真题”
idx_pseudo = np.where(max_p >= 0.95)[0]
print(f"🔍 专家模型发现了 {len(idx_pseudo)} 张高置信度伪标签图片！")

X_augmented = np.vstack([X_train_orig, X_test_base[idx_pseudo]])
y_augmented = np.concatenate([y_train_orig, preds[idx_pseudo]])

# =====================================================
# 4️⃣ 阶段二：终极 ElasticNet 剃刀训练
# =====================================================
print("\n" + "="*50 + "\n🚀 PHASE 2: 4096 维超级特征训练\n" + "="*50)
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='elasticnet', solver='saga', max_iter=4000))
])

# 针对高维空间，增强 L1 比例以聚焦核心面部特征
param_grid = {"clf__C": [0.1, 1.0, 10.0], "clf__l1_ratio": [0.5, 0.8]}
grid = GridSearchCV(final_pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_augmented, y_augmented)
print(f"⭐ 终极 CV 准确率: {grid.best_score_:.4f}")

# =====================================================
# 5️⃣ 阶段三：10-Crop TTA 终极推理
# =====================================================
print("\n" + "="*50 + "\n🚀 PHASE 3: 10-Crop TTA 深度扫描\n" + "="*50)
crop_trans = get_10crop_transform()
final_test_probs = []

with torch.no_grad():
    for f in tqdm(test_files, desc="执行 10-Crop 推理"):
        img_path = os.path.join(test_dir, f)
        crops = crop_trans(Image.open(img_path).convert("RGB")).to(device) # (10, 3, 224, 224)
        
        # 对 10 个裁剪样本分别提取特征并预测
        e = model_eff(crops).cpu().numpy()
        d = model_dino(crops).cpu().numpy()
        c = model_clip(pixel_values=crops).image_embeds.cpu().numpy()
        mega_f = np.hstack([e, d, c])
        
        # 10 次预测取平均值，确保锁定眼部区域
        p_crops = grid.best_estimator_.predict_proba(mega_f)
        final_test_probs.append(p_crops.mean(axis=0))

final_preds = np.argmax(final_test_probs, axis=1)
classes = datasets.ImageFolder("train").classes
submission = pd.read_csv("sample_submission.csv")
submission["label"] = [classes[i] for i in final_preds]
submission.to_csv("submission_v6_ultimate.csv", index=False)
print("\n🏆 V6 终极文件 submission_v6_ultimate.csv 已生成。祝你冲击 0.93+！")