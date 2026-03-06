import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🔥 阶段一：启动伪标签 (Pseudo-Labeling) 数据扩增引擎! Using: {device}")

# =====================================================
# 1. 重建你的 0.84 ResNet18 模型架构
# =====================================================
# ⚠️ 注意：运行此脚本前，请确保你保存了 0.84 模型的权重 (例如命名为 best_resnet18.pth)
# 如果之前代码没保存，可以在之前训练完加一句：torch.save(model.state_dict(), "best_resnet18.pth")

# 先获取类别名称和数量
original_train_dataset = datasets.ImageFolder("train")
class_names = original_train_dataset.classes
num_classes = len(class_names)

model = resnet18()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
# 加载权重 (请确保当前目录下有这个文件)
try:
    model.load_state_dict(torch.load("best_resnet18.pth", map_location=device))
    print("✅ 成功加载 0.84 版本的 ResNet18 权重！")
except FileNotFoundError:
    print("❌ 找不到 best_resnet18.pth！请先在之前的代码末尾加上 torch.save(model.state_dict(), 'best_resnet18.pth') 并运行一次。")
    exit()

model = model.to(device)
model.eval()

# =====================================================
# 2. 读取无标签的测试集
# =====================================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png')))
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.image_files[idx])
        return self.transform(Image.open(path).convert("RGB")), path, self.image_files[idx]

test_dir = "test/test" if os.path.exists("test/test") else "test"
test_loader = DataLoader(TestDataset(test_dir, transform=test_transform), batch_size=32, shuffle=False)

# =====================================================
# 3. 核心逻辑：设定阈值，筛选高置信度样本
# =====================================================
CONFIDENCE_THRESHOLD = 0.95 # 极其严苛的阈值：95% 把握才算数
pseudo_count = 0

# 创建一个新的扩增训练集文件夹
aug_train_dir = "train_augmented"
os.makedirs(aug_train_dir, exist_ok=True)
for c in class_names:
    os.makedirs(os.path.join(aug_train_dir, c), exist_ok=True)

# 先把原本的 train 数据复制过去
print("\n正在拷贝原始训练集...")
for c in class_names:
    src_dir = os.path.join("train", c)
    dst_dir = os.path.join(aug_train_dir, c)
    for file in os.listdir(src_dir):
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

print("\n🚀 正在扫描测试集并生成伪标签...")
with torch.no_grad():
    for images, paths, filenames in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        # 将原始 Logits 转换为 0~1 的概率分布
        probs = F.softmax(outputs, dim=1)
        
        max_probs, preds = torch.max(probs, dim=1)
        
        for i in range(len(max_probs)):
            if max_probs[i].item() >= CONFIDENCE_THRESHOLD:
                # 如果置信度大于 95%，执行物理拷贝
                pred_class = class_names[preds[i].item()]
                src_path = paths[i]
                # 加个前缀防止重名
                dst_path = os.path.join(aug_train_dir, pred_class, f"pseudo_{filenames[i]}") 
                shutil.copy(src_path, dst_path)
                pseudo_count += 1

print(f"\n🎉 扩增完成！成功从测试集中白嫖了 {pseudo_count} 张高置信度图片！")
print(f"📁 新的数据集已存放在 [{aug_train_dir}] 文件夹下。")