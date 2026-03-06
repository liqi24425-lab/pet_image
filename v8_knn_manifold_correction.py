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
from PIL import Image
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# 0️⃣ 系统设定
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🗡️ 启动 V8 绝杀方案：V4 基座 + KNN 流形距离纠错! Using: {device}")

# ===============================
# 1️⃣ 加载 V4 最强基座引擎
# ===============================
print("\n🔥 正在激活：EffNet-B5 (2048D) + DINOv2-Large (1024D)...")
model_eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT).to(device).eval()
model_eff.classifier[1] = nn.Identity()

# 回归表现最稳的 Large 版本
model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device).eval()

# 使用标准的 ImageNet 归一化提取纯视觉特征
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
base_trans = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    norm
])

def extract_pure_features(img_dir, desc, is_test=False):
    class QuickDS(Dataset):
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

    loader = DataLoader(QuickDS(img_dir), batch_size=16, shuffle=False)
    f_list, labels = [], []
    
    with torch.no_grad():
        for img, target in tqdm(loader, desc=desc):
            img_dev = img.to(device)
            # 拼接 2048 + 1024 = 3072D 纯视觉空间
            e = model_eff(img_dev).cpu().numpy()
            d = model_dino(img_dev).cpu().numpy()
            f_list.append(np.hstack([e, d]))
            labels.extend(target)
            
    return np.vstack(f_list), np.array(labels)

# ===============================
# 2️⃣ 提取特征 (构建距离空间)
# ===============================
print("\n" + "="*30 + " 阶段 1: 提取 3072 维流形空间特征 " + "="*30)
# 为了最高纯度，我们直接使用原始训练集进行 KNN 匹配
train_path = "train" 
X_train, y_train = extract_pure_features(train_path, "提取训练集纯净特征")

test_path = "test/test" if os.path.exists("test/test") else "test"
X_test, test_files = extract_pure_features(test_path, "提取测试集纯净特征", is_test=True)

# ===============================
# 3️⃣ 基础分类器预测 (复现 0.913 的超平面)
# ===============================
print("\n" + "="*30 + " 阶段 2: 基础线性超平面预测 " + "="*30)
# 使用 ElasticNet 给出基准预测和概率
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=3000)
clf.fit(X_train_scaled, y_train)

base_probs = clf.predict_proba(X_test_scaled)
base_preds = np.argmax(base_probs, axis=1)
base_conf = np.max(base_probs, axis=1) # 获取分类器的置信度

# ===============================
# 4️⃣ KNN 流形距离强行纠错 (Kaggle 绝杀)
# ===============================
print("\n" + "="*30 + " 阶段 3: KNN 最近邻强制纠错 " + "="*30)
# 在未缩放的原始高维特征空间中计算余弦相似度，因为距离度量更真实
similarities = cosine_similarity(X_test, X_train)

final_preds = base_preds.copy()
correction_count = 0

# 设定 KNN 参数
K_NEIGHBORS = 5        # 看最近的 5 张图
CONFIDENCE_THRES = 0.85 # 如果分类器把握低于 85%，就允许被邻居纠错
VOTE_THRES = 4         # 5个邻居里至少要有 4 个统一意见，才敢推翻分类器

class_map = datasets.ImageFolder("train").classes

for i in range(len(X_test)):
    # 找到与当前测试图最相似的 Top-K 训练集图片的索引
    top_k_indices = np.argsort(similarities[i])[::-1][:K_NEIGHBORS]
    # 获取这 K 个邻居的真实标签
    top_k_labels = y_train[top_k_indices]
    
    # 统计邻居中各个标签的得票数
    votes = np.bincount(top_k_labels, minlength=len(class_map))
    majority_label = np.argmax(votes)
    majority_votes = votes[majority_label]
    
    # 触发纠错机制的条件
    if base_conf[i] < CONFIDENCE_THRES and majority_votes >= VOTE_THRES and majority_label != base_preds[i]:
        print(f"⚠️ 纠错触发！图片 {test_files[i]} | 原分类: {class_map[base_preds[i]]} ({base_conf[i]:.2f}) -> KNN强改: {class_map[majority_label]} (邻居票数: {majority_votes}/{K_NEIGHBORS})")
        final_preds[i] = majority_label
        correction_count += 1

print(f"\n✅ KNN 纠错完毕。共强行推翻了 {correction_count} 个处于边界的模糊判断。")

# ===============================
# 5️⃣ 导出最终提交文件
# ===============================
submission = pd.read_csv("sample_submission.csv")
submission["label"] = [class_map[i] for i in final_preds]
submission.to_csv("submission_v8_knn_corrected.csv", index=False)
print("🏆 V8 极限纠错版 submission_v8_knn_corrected.csv 已生成！请提交！")