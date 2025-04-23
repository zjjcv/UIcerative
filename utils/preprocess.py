import cv2
import numpy as np
import os
from tqdm import tqdm  # 进度条显示

def batch_process_endoscopy_images(input_folder, output_folder, target_size=640):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    file_list = [f for f in os.listdir(input_folder) 
                if f.lower().endswith(valid_exts)]
    
    # 处理进度条
    for filename in tqdm(file_list, desc="Processing Images"):
        try:
            # 原始文件路径
            input_path = os.path.join(input_folder, filename)
            
            # 处理流程
            img = cv2.imread(input_path)
            if img is None:
                continue
                
            # 裁剪处理（复用之前的逻辑）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            max_contour = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(max_contour)
            
            # 扩展边界
            expand = 15
            x = max(0, x-expand)
            y = max(0, y-expand)
            w = min(img.shape[1]-x, w+expand*2)
            h = min(img.shape[0]-y, h+expand*2)
            
            cropped = img[y:y+h, x:x+w]
            
            # 去除文字
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0,0,200], dtype=np.uint8)
            upper_white = np.array([255,30,255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)
            result = cv2.inpaint(cropped, mask, 3, cv2.INPAINT_TELEA)
            
            # 调整尺寸
            resized = cv2.resize(result, (target_size, target_size), 
                               interpolation=cv2.INTER_AREA)
            
            # 保存结果
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized)
            
        except Exception as e:
            print(f"\n处理 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_dir = r"data\UC\UC9"
    output_dir = r"data\UC\UC9\caijian"
    
    batch_process_endoscopy_images(
        input_folder=input_dir,
        output_folder=output_dir,
        target_size=640
    )