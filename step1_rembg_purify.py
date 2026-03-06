import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

print("🧹 启动数据洗髓程序：U-2-Net 语义级背景剥离...")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 支持遍历子文件夹（适配训练集的按类别存放）
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                # 保持目录结构
                rel_path = os.path.relpath(root, input_dir)
                out_sub_dir = os.path.join(output_dir, rel_path)
                if not os.path.exists(out_sub_dir):
                    os.makedirs(out_sub_dir)
                    
                # 将后缀统一改为 .jpg
                out_file_name = os.path.splitext(file)[0] + ".jpg"
                output_path = os.path.join(out_sub_dir, out_file_name)
                
                try:
                    # 1. 读取原图
                    input_img = Image.open(input_path).convert("RGBA")
                    # 2. 智能抠图，剥离背景
                    subject_img = remove(input_img)
                    # 3. 创建纯黑背景 (黑色对网络而言是 0 激活，最不会产生干扰)
                    black_bg = Image.new("RGBA", subject_img.size, (0, 0, 0, 255))
                    # 4. 将抠出的主体合成到纯黑背景上
                    black_bg.paste(subject_img, mask=subject_img)
                    # 5. 转换为 RGB 并保存
                    black_bg.convert("RGB").save(output_path, "JPEG", quality=95)
                except Exception as e:
                    print(f"处理 {file} 失败: {e}")

# 执行洗髓
print("正在清洗训练集...")
process_directory("train", "train_rembg")

print("正在清洗测试集...")
# 根据你的实际路径调整
test_in = "test/test" if os.path.exists("test/test") else "test"
test_out = "test_rembg/test" if os.path.exists("test/test") else "test_rembg"
process_directory(test_in, test_out)

print("✅ 洗髓完成！纯净数据已保存在 train_rembg 和 test_rembg。")