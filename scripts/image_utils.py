import os
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

def save_tensor_as_image(tensor_img, save_path, min_max=(-1, 1), prefix="img"):
    """
    将tensor图像保存为PNG文件
    
    Args:
        tensor_img: 图像tensor，形状为[B,C,H,W]或[C,H,W]
        save_path: 保存路径（包含文件名）
        min_max: tensor的值范围
        prefix: 文件名前缀
    """
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 处理tensor维度
    if tensor_img.dim() == 4:  # [B, C, H, W]
        img_tensor = tensor_img[0]  # 取第一个batch
    else:  # [C, H, W]
        img_tensor = tensor_img
    
    # 将tensor从[min_max]范围转换到[0,1]
    img_tensor = (img_tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # 转换为numpy数组并调整维度
    img_np = img_tensor.detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # [C,H,W] -> [H,W,C]
    
    # 转换到[0,255]范围
    img_np = (img_np * 255).astype(np.uint8)
    
    # 转换颜色空间 RGB -> BGR (OpenCV需要BGR格式)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 保存图像
    cv2.imwrite(save_path, img_bgr)
    print(f"Image saved to: {save_path}")

def save_denoised_images_batch(denoised_tensors, save_dir="x_r", prefix="denoised"):
    """
    批量保存去噪图像
    
    Args:
        denoised_tensors: 去噪图像tensor列表
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, tensor_img in enumerate(denoised_tensors):
        filename = f"{prefix}_{i:04d}.png"
        save_path = os.path.join(save_dir, filename)
        save_tensor_as_image(tensor_img, save_path) 