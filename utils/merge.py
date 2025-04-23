import cv2
import os
import math
import numpy as np
from natsort import natsorted

def create_image_mosaic(input_dir, output_path, output_size=1024):
    """
    参数说明：
    input_dir: 输入图片文件夹路径
    output_path: 输出图片保存路径
    output_size: 最终输出图片的边长尺寸（默认1024）
    """
    # 读取并排序图片文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir) 
            if f.lower().endswith(valid_exts)]
    files = natsorted(files)  # 自然排序
    
    if not files:
        raise ValueError("输入文件夹中没有图片文件")

    # 加载所有图片
    images = []
    for filename in files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    if not images:
        raise ValueError("没有有效的可读图片")

    # 计算拼接参数
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))  # 网格尺寸
    total_tiles = grid_size ** 2                 # 总需要图片数
    
    # 填充图片数组（从第一张开始按照顺序重新填充）
    if num_images < total_tiles:
        for i in range(total_tiles - num_images):
            images.append(images[i % num_images].copy())

    # 计算每个拼贴图尺寸
    tile_size = output_size // grid_size
    final_size = tile_size * grid_size
    
    # 预处理所有图片（保持比例缩放并居中裁剪）
    processed = []
    for img in images:
        h, w = img.shape[:2]
        
        # 计算缩放比例
        scale = max(tile_size/w, tile_size/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放并裁剪
        resized = cv2.resize(img, (new_w, new_h))
        y = (new_h - tile_size) // 2
        x = (new_w - tile_size) // 2
        cropped = resized[y:y+tile_size, x:x+tile_size]
        processed.append(cropped)

    # 创建拼接画布
    mosaic = np.zeros((final_size, final_size, 3), dtype=np.uint8)
    
    # 填充拼贴图
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            y = i * tile_size
            x = j * tile_size
            mosaic[y:y+tile_size, x:x+tile_size] = processed[idx]
    
    # 最终尺寸校准
    if final_size != output_size:
        mosaic = cv2.resize(mosaic, (output_size, output_size))
    
    cv2.imwrite(output_path, mosaic)
    print(f"拼接完成，输出尺寸：{output_size}x{output_size}")

# 使用示例
create_image_mosaic(
    input_dir=r"data\changjiehe\10\caijian",
    output_path=r"data\changjiehe\merge_cjh\10.jpg",
    output_size=1024  # 可调整输出尺寸
)