import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
from . import dataloader
from . import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import cv2

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'snapshots', 'dehazer.pth')

def compute_mean_gradient(image):
    """计算图像的平均梯度"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(gradient_magnitude)

def dehaze_image(image_path, dehaze_net, output_path):
    """对单张图片进行去雾并返回处理时间"""
    start_time = time.time()  # 记录开始时间

    # 加载并处理图片
    data_hazy = Image.open(image_path)
    original_image = np.asarray(data_hazy)
    data_hazy = (original_image / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float().permute(2, 0, 1).unsqueeze(0).cuda()

    # 去雾操作
    clean_image = dehaze_net(data_hazy)
    clean_image = clean_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    clean_image = (clean_image * 255).clip(0, 255).astype(np.uint8)
    end_time = time.time()  # 记录结束时间
    
    # 获取文件名
    filename = os.path.basename(image_path)
    # 保存去雾后的图片
    # 从backend/dehazing/回到backend/，然后进入results/dehazed/
    # output_path = os.path.join(os.path.dirname(CURRENT_DIR), 'results', 'dehazed', 'dehazed_' + filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR))

    # 计算原始和去雾后的平均梯度
    original_gradient = compute_mean_gradient(original_image)
    dehazed_gradient = compute_mean_gradient(clean_image)

    processing_time = end_time - start_time  # 计算处理时间

    return original_gradient, dehazed_gradient, processing_time

def start_dehaze(input_path, output_path):
    """对输入路径的图片进行去雾处理
    
    Args:
        input_path: 输入图片的路径
        
    Returns:
        dict: 包含去雾评估指标的字典，包括：
            - original_gradient: 原始图片的平均梯度
            - dehazed_gradient: 去雾后图片的平均梯度
            - processing_time: 处理时间
            - improvement_ratio: 梯度改善比例
    """
    try:
        dehaze_net = net.dehaze_net().cuda()
        
        # 检查模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
            
        dehaze_net.load_state_dict(torch.load(MODEL_PATH))
        
        # 处理单张图片
        original_gradient, dehazed_gradient, processing_time = dehaze_image(input_path, dehaze_net, output_path)
        
        # 计算改善比例
        improvement_ratio = (dehazed_gradient - original_gradient) / original_gradient if original_gradient != 0 else 0
        
        # 构造返回的指标字典
        metrics = {
            'original_gradient': float(original_gradient),
            'dehazed_gradient': float(dehazed_gradient),
            'processing_time': float(processing_time),
            'improvement_ratio': float(improvement_ratio)
        }
        
        print(f"去雾处理完成:")
        print(f"原始梯度: {original_gradient:.4f}")
        print(f"去雾后梯度: {dehazed_gradient:.4f}")
        print(f"处理时间: {processing_time:.4f} 秒")
        print(f"改善比例: {improvement_ratio:.2%}")
        
        return metrics
        
    except Exception as e:
        print(f"去雾处理失败: {str(e)}")
        return None

if __name__ == '__main__':
    # 初始化去雾模型
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))

    # 获取待处理的图片列表
    test_list = glob.glob("/root/autodl-tmp/pipeline/frames/processed_rgb_smoked3/*")

    # 用于存储每张图片处理时间的列表
    original_gradients = []
    dehazed_gradients = []
    processing_times = [] 

    # 处理每张图片
    for image in test_list:
        original_gradient, dehazed_gradient, processing_time = dehaze_image(image, dehaze_net)
        original_gradients.append(original_gradient)
        dehazed_gradients.append(dehazed_gradient)
        processing_times.append(processing_time)
        print(f"{image}: Original Mean Gradient = {original_gradient:.2f}, Dehazed Mean Gradient = {dehazed_gradient:.2f}, Processing Time = {processing_time:.4f} s")
        print(image, "done!")

    # 计算并输出平均处理时间
    if original_gradients and dehazed_gradients and processing_times:
        average_original_gradient = sum(original_gradients) / len(original_gradients)
        average_dehazed_gradient = sum(dehazed_gradients) / len(dehazed_gradients)
        average_time = sum(processing_times) / len(processing_times)
        print(f"平均每张图片的原始梯度: {average_original_gradient:.4f}")
        print(f"平均每张图片的去烟梯度: {average_dehazed_gradient:.4f}")
        print(f"平均每张图片的去烟用时: {average_time:.4f} 秒")
    else:
        print("未处理任何图片。")