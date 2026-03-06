import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import timm 
# 🛠️ 修复点 1：导入 AutoImageProcessor
from transformers import AutoModel, AutoImageProcessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"☢️ 启动 V9 TITAN 屠榜方案：原始图像 + 四大天王 + Zoom TTA! Using: {device}")

# =====================================================
# 1️⃣ 加载 HF 最新四大天王 (直接应对复杂背景)
# =====================================================
print("\n🔥 正在从云端召唤视觉四大天王...")
# 1. SwinV2: 基于移动窗口，对复杂的边缘和毛发极其敏感
model_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0).to(device).eval()
# 2. ConvNeXt-V2: 现代最强纯卷积，提供极强的平移不变性，抗过拟合
model_conv = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0).to(device).eval()
# 3. DINOv2-Large: 捕捉细微肌肉动作 (V4的夺冠功臣)
model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device).eval()
# 4. SigLIP: Google最强图文对齐模型
siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
# 🛠️ 修复点 2：使用 AutoImageProcessor 绕过 Tokenizer Bug
siglip_proc = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")

# =====================================================
# 2️⃣ TTA 策略：原图 vs. 强迫看脸 (Zoom)
# =====================================================
strict_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

trans_orig = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(), strict_norm
])
trans_zoom = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(), strict_norm
])

def extract_titan_features(img_dir, transform_mode, desc, is_test=False):
    trans = trans_orig if transform_mode == 'orig' else trans_zoom
    
    class RawDS(Dataset):
        def __init__(self, directory):
            self.is_test = is_test
            if is_test:
                self.files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.png','.jpg','.jpeg')))
                self.dir = directory
            else:
                self.ds = datasets.ImageFolder(directory)
        def __len__(self): return len(self.files) if self.is_test else len(self.ds)
        def __getitem__(self, idx):
            path = os.path.join(self.dir, self.files[idx]) if self.is_test else self.ds.imgs[idx][0]
            label = self.files[idx] if self.is_test else self.ds[idx][1]
            img = Image.open(path).convert("RGB")
            
            # 🛠️ 修复核心：直接在 Dataset 里用 SigLIP 处理器把原图转成 Tensor
            # squeeze(0) 是为了去掉 batch 维度，让 DataLoader 稍后重新正常打包
            siglip_tensor = siglip_proc(images=img, return_tensors="pt").pixel_values.squeeze(0)
            
            # 现在返回的两个都是纯纯的 PyTorch Tensor 了！
            return trans(img), siglip_tensor, label 

    loader = DataLoader(RawDS(img_dir), batch_size=8, shuffle=False)
    f_list, labels = [], []
    
    with torch.no_grad():
        # 🛠️ 配合修改：这里接收的直接是 siglip_tensor
        for img_tensor, siglip_tensor, target in tqdm(loader, desc=f"{desc} [{transform_mode}]"):
            img_dev = img_tensor.to(device)
            siglip_dev = siglip_tensor.to(device)
            
            # 提取前三大天王的特征
            f_swin = model_swin(img_dev).cpu().numpy()
            f_conv = model_conv(img_dev).cpu().numpy()
            f_dino = model_dino(img_dev).cpu().numpy()
            
            # 提取 SigLIP 特征 (直接传入 pixel_values)
            # 提取 SigLIP 特征 (从 BaseModelOutputWithPooling 中解包 Tensor)
            sig_out = siglip_model.get_image_features(pixel_values=siglip_dev)
            
            # 智能拆包逻辑：兼容不同版本的 transformers 库
            if not isinstance(sig_out, torch.Tensor):
                if hasattr(sig_out, 'image_embeds'):
                    sig_tensor = sig_out.image_embeds
                elif hasattr(sig_out, 'pooler_output'):
                    sig_tensor = sig_out.pooler_output
                else:
                    sig_tensor = sig_out[0]
            else:
                sig_tensor = sig_out
                
            # 拿到纯正的 Tensor 后再转 NumPy
            f_siglip = sig_tensor.cpu().numpy()
            
            # 组合成 3840 维巨型特征
            f_list.append(np.hstack([f_swin, f_conv, f_dino, f_siglip]))
            labels.extend(target)
            
    return np.vstack(f_list), np.array(labels)

# =====================================================
# 3️⃣ 提取原始图像特征 
# =====================================================
print("\n" + "="*30 + " 阶段 1: 提取 3840 维神级特征 " + "="*30)
train_dir = "train_augmented" if os.path.exists("train_augmented") else "train"
test_dir = "test/test" if os.path.exists("test/test") else "test"

X_train, y_train = extract_titan_features(train_dir, "orig", "提取训练集")
X_test_orig, test_files = extract_titan_features(test_dir, "orig", "提取测试集-全景", is_test=True)
X_test_zoom, _ = extract_titan_features(test_dir, "zoom", "提取测试集-放大", is_test=True)

# =====================================================
# 4️⃣ 终极决战：强 L1 惩罚过滤背景噪音
# =====================================================
print("\n" + "="*30 + " 阶段 2: 超高维 ElasticNet 剃刀裁决 " + "="*30)
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty='elasticnet', solver='saga', max_iter=4000))
])

param_grid = {"clf__C": [0.1, 1.0, 5.0], "clf__l1_ratio": [0.5, 0.8]}
grid = GridSearchCV(final_pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"⭐ V9 TITAN 交叉验证准确率: {grid.best_score_:.4f}")

# =====================================================
# 5️⃣ 多尺度软投票推理 (Soft Voting)
# =====================================================
print("\n" + "="*30 + " 阶段 3: 多尺度 TTA 决策 " + "="*30)
prob_orig = grid.best_estimator_.predict_proba(X_test_orig)
prob_zoom = grid.best_estimator_.predict_proba(X_test_zoom)

final_prob = 0.5 * prob_orig + 0.5 * prob_zoom
final_preds = np.argmax(final_prob, axis=1)

# =====================================================
# 6️⃣ 导出成绩单
# =====================================================
classes = datasets.ImageFolder("train").classes
submission = pd.read_csv("sample_submission.csv")
submission["label"] = [classes[i] for i in final_preds]
submission.to_csv("submission_v9_titan_raw.csv", index=False)
print("🏆 恭喜！最强屠榜文件 submission_v9_titan_raw.csv 生成完毕！")