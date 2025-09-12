#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICNet噪声调制集成测试脚本
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from icnet import icnet
from noise_modules import NoiseLevelMLP, FeatureWiseAffine

class Args:
    """模拟参数类"""
    def __init__(self):
        self.n_feats = 64

def test_noise_modules():
    """测试噪声调制模块"""
    print("测试噪声调制模块...")
    
    # 测试参数
    batch_size = 2
    n_feats = 64
    time_steps = torch.randn(batch_size)
    
    # 测试NoiseLevelMLP
    noise_mlp = NoiseLevelMLP(n_feats)
    noise_features = noise_mlp(time_steps)
    print(f"NoiseLevelMLP输出形状: {noise_features.shape}")
    assert noise_features.shape == (batch_size, n_feats), f"期望形状: ({batch_size}, {n_feats}), 实际形状: {noise_features.shape}"
    
    # 测试FeatureWiseAffine
    feature_affine = FeatureWiseAffine(n_feats, n_feats)
    test_features = torch.randn(batch_size, n_feats, 32, 32)
    modulated_features = feature_affine(test_features, noise_features)
    print(f"FeatureWiseAffine输出形状: {modulated_features.shape}")
    assert modulated_features.shape == test_features.shape, f"特征形状应该保持不变"
    
    print("✓ 噪声调制模块测试通过")

def test_icnet_integration():
    """测试ICNet噪声调制集成"""
    print("\n测试ICNet噪声调制集成...")
    
    # 创建模拟参数
    args = Args()
    
    # 创建网络
    model = icnet.LFSRNet(args)
    
    # 测试输入
    batch_size = 1
    height, width = 400, 600
    x = torch.randn(batch_size, 3, height, width)
    diff_img = torch.randn(batch_size, 3, height, width)
    time = torch.randn(batch_size)
    
    print(f"输入图像形状: {x.shape}")
    print(f"差分图像形状: {diff_img.shape}")
    print(f"时间步形状: {time.shape}")
    
    # 前向传播
    try:
        with torch.no_grad():
            output = model(x, diff_img, time)
        print(f"输出图像形状: {output.shape}")
        assert output.shape == x.shape, f"输出形状应该与输入相同"
        print("✓ ICNet噪声调制集成测试通过")
        
    except Exception as e:
        print(f"✗ ICNet测试失败: {e}")
        raise

def test_gradient_flow():
    """测试梯度流"""
    print("\n测试梯度流...")
    
    args = Args()
    model = icnet.LFSRNet(args)
    
    # 测试输入
    x = torch.randn(1, 3, 100, 150, requires_grad=True)
    diff_img = torch.randn(1, 3, 100, 150, requires_grad=True)
    time = torch.randn(1, requires_grad=True)
    
    # 前向传播
    output = model(x, diff_img, time)
    
    # 计算损失
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"输入图像梯度: {x.grad is not None}")
    print(f"差分图像梯度: {diff_img.grad is not None}")
    print(f"时间步梯度: {time.grad is not None}")
    
    # 检查噪声调制层梯度
    noise_mlp_grad = model.noise_level_mlp.noise_level_mlp[1].weight.grad
    print(f"噪声MLP梯度: {noise_mlp_grad is not None}")
    
    print("✓ 梯度流测试通过")

def main():
    """主测试函数"""
    print("开始ICNet噪声调制集成测试...")
    
    try:
        # 测试噪声调制模块
        test_noise_modules()
        
        # 测试ICNet集成
        test_icnet_integration()
        
        # 测试梯度流
        test_gradient_flow()
        
        print("\n🎉 所有测试通过！ICNet噪声调制集成成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)