import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

def process_directory(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"⚠️ 找不到目录 {input_dir}，请检查路径。")
        return

    # 预先收集所有图片路径，方便 tqdm 显示进度
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append((root, file))

    if not image_paths:
        print(f"⚠️ 在 {input_dir} 中没有找到图片。")
        return

    print(f"\n📁 发现 {len(image_paths)} 张图片 in {input_dir}. 开始物理去噪...")
    
    for root, file in tqdm(image_paths, desc=f"Processing {input_dir}"):
        input_path = os.path.join(root, file)
        
        # 保持原有的子文件夹结构映射到新的 output_dir
        rel_path = os.path.relpath(root, input_dir)
        output_folder = os.path.join(output_dir, rel_path)
        os.makedirs(output_folder, exist_ok=True)
        
        # 统一保存为 .png 格式以防压缩损失
        output_filename = os.path.splitext(file)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # 1. 打开原图并强制转换为包含 Alpha 通道的 RGBA
            input_image = Image.open(input_path).convert("RGBA")
            
            # 2. 核心：U-Net 强力抠图
            output_image = remove(input_image)
            
            # 3. 创建一张纯黑底图 (0, 0, 0)
            background = Image.new("RGBA", output_image.size, (0, 0, 0, 255))
            
            # 4. 将抠出来的宠物叠加到黑底上，并转回标准的 RGB
            clean_image = Image.alpha_composite(background, output_image).convert("RGB")
            
            # 5. 保存纯净版数据
            clean_image.save(output_path, "PNG")
        except Exception as e:
            print(f"❌ 处理 {input_path} 时出错: {e}")

if __name__ == "__main__":
    print("🚀 启动数据源头净化 Pipeline (rembg U-Net)...")
    
    # 1. 处理训练集：输出到 train_rmbg
    process_directory("train", "train_rmbg")
    
    # 2. 处理测试集：输出到 test_rmbg (自动兼容 test/test 嵌套结构)
    test_in = "test/test" if os.path.exists("test/test") else "test"
    test_out = "test_rmbg/test" if os.path.exists("test/test") else "test_rmbg"
    process_directory(test_in, test_out)
    
    print("\n✅ 恭喜！所有图片的背景噪音已被彻底清除！")
    print("📁 你的干净数据集现在躺在 [train_rmbg] 和 [test_rmbg] 文件夹里了。")