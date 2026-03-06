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
print(f"👑 启动 MEGA-ENSEMBLE 终极进化引擎！\n特征空间: EffNetB5(2048) + CLIP(512) + DINOv2(384) = 2944D")

# =====================================================
# 1️⃣ 专属归一化与数据加载
# =====================================================
# 统一使用 ImageNet 归一化（DINOv2 和 EffNet 均兼容）
standard_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
clip_norm = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

def get_trans(is_clip=False, flip=False):
    t_list = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    if flip: t_list.append(transforms.RandomHorizontalFlip(p=1.0))
    t_list.append(transforms.ToTensor())
    t_list.append(clip_norm if is_clip else standard_norm)
    return transforms.Compose(t_list)

# =====================================================
# 2️⃣ 加载三引擎特征提取器
# =====================================================
print("\n🔥 正在激活三擎大脑...")
model_eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT).to(device).eval()
model_eff.classifier[1] = nn.Identity()

model_clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

# 引入视觉霸主 DINOv2
model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

def extract_mega_features(img_dir, desc, is_test=False, flip=False):
    class QuickDataset(Dataset):
        def __init__(self, directory):
            self.is_test = is_test
            if is_test:
                self.files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.png','.jpg','.jpeg')))
                self.dir = directory
            else:
                self.ds = datasets.ImageFolder(directory)
        def __len__(self): return len(self.files) if self.is_test else len(self.ds)
        def __getitem__(self, idx):
            img = Image.open(os.path.join(self.dir, self.files[idx])).convert("RGB") if self.is_test else self.ds[idx][0]
            label = self.files[idx] if self.is_test else self.ds[idx][1]
            return get_trans(is_clip=False, flip=flip)(img), get_trans(is_clip=True, flip=flip)(img), label

    loader = DataLoader(QuickDataset(img_dir), batch_size=32, shuffle=False)
    f_eff, f_clip, f_dino, labels = [], [], [], []
    
    with torch.no_grad():
        for img_s, img_c, target in tqdm(loader, desc=desc):
            f_eff.append(model_eff(img_s.to(device)).cpu().numpy())
            f_clip.append(model_clip(pixel_values=img_c.to(device)).image_embeds.cpu().numpy())
            f_dino.append(model_dino(img_s.to(device)).cpu().numpy())
            labels.extend(target)
            
    return np.hstack([np.vstack(f_eff), np.vstack(f_clip), np.vstack(f_dino)]), np.array(labels)

# =====================================================
# 3️⃣ 阶段一：初次特征组装
# =====================================================
print("\n" + "="*30 + " 阶段 1: 组装 2944 维特征 " + "="*30)
X_train_raw, y_train_raw = extract_mega_features("train", "提取原始训练集")
test_path = "test/test" if os.path.exists("test/test") else "test"
X_test_norm, test_ids = extract_mega_features(test_path, "提取测试集(原图)", is_test=True)
X_test_flip, _ = extract_mega_features(test_path, "提取测试集(TTA)", is_test=True, flip=True)

# =====================================================
# 4️⃣ 阶段二：递归伪标签 2.0 (Recursive Pseudo-Labeling)
# =====================================================
print("\n" + "="*30 + " 阶段 2: 递归伪标签 2.0 " + "="*30)
# 第一轮：用 2944D 训练“强老师”
teacher = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=2000))])
teacher.fit(X_train_raw, y_train_raw)

# 老师去测试集“圈重点”
probs = teacher.predict_proba(X_test_norm)
max_p = np.max(probs, axis=1)
preds = np.argmax(probs, axis=1)

# 选出 > 95% 置信度的真题
idx_pseudo = np.where(max_p >= 0.95)[0]
print(f"🔍 老师通过 2944 维空间锁定了 {len(idx_pseudo)} 张测试集真题（预期比之前更多）")

X_combined = np.vstack([X_train_raw, X_test_norm[idx_pseudo]])
y_combined = np.concatenate([y_train_raw, preds[idx_pseudo]])

# =====================================================
# 5️⃣ 阶段三：终极 ElasticNet 训练
# =====================================================
print("\n" + "="*30 + " 阶段 3: 终极 ElasticNet 剃刀训练 " + "="*30)
final_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(penalty='elasticnet', solver='saga', max_iter=3000))])
param_grid = {"clf__C": np.logspace(-1, 2, 8), "clf__l1_ratio": [0.2, 0.5, 0.8]}
grid = GridSearchCV(final_pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_combined, y_combined)

print(f"⭐ 最佳准确率 (CV): {grid.best_score_:.4f}")

# =====================================================
# 6️⃣ 阶段四：TTA 融合预测
# =====================================================
print("\n" + "="*30 + " 阶段 4: TTA 终极推理 " + "="*30)
p1 = grid.best_estimator_.predict_proba(X_test_norm)
p2 = grid.best_estimator_.predict_proba(X_test_flip)
final_p = (p1 + p2) / 2.0
final_preds = np.argmax(final_p, axis=1)

classes = datasets.ImageFolder("train").classes
submission = pd.read_csv("sample_submission.csv")
submission["label"] = [classes[i] for i in final_preds]
submission.to_csv("submission_mega_ensemble.csv", index=False)
print("\n🏆 绝杀文件 submission_mega_ensemble.csv 已生成！这是技术的终点。")