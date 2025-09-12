import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码模块，用于将噪声级别编码为特征向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class Swish(nn.Module):
    """Swish激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeatureWiseAffine(nn.Module):
    """特征级仿射变换，用于噪声调制"""
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            noise_feature = self.noise_func(noise_embed).view(batch, -1, 1, 1)
            x = x + noise_feature
        return x

class NoiseLevelMLP(nn.Module):
    """噪声级别MLP，将时间步编码为噪声特征"""
    def __init__(self, dim):
        super(NoiseLevelMLP, self).__init__()
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, time):
        return self.noise_level_mlp(time)