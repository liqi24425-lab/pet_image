import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义完全相同的测试集 Transform
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 定义测试集 Dataset 类
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

# ⚠️ 注意这里：请把 "test/test" 改成你实际存放测试图片的文件夹名字，比如 "test"
test_dir = "test" 
test_dataset = TestDataset(test_dir, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 3. 加载我们刚刚训练好的最强权重
num_classes = 3 # 你的输出日志显示有 ['Angry', 'Happy', 'Sad'] 3个类别
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# 4. 开始预测
preds = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Predicting Test Set"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        preds.extend(predicted.cpu().numpy())

# 5. 映射回字符串标签并保存
# 注意：这里的映射顺序必须和训练集类别的字母顺序一致
idx_to_class = {0: 'Angry', 1: 'Happy', 2: 'Sad'}
pred_labels = [idx_to_class[i] for i in preds]

submission = pd.read_csv("sample_submission.csv")
submission["label"] = pred_labels
submission.to_csv("submission_finetuned_88.csv", index=False)

print("完美！submission_finetuned_88.csv 已经生成，快去 Kaggle 提交吧！")