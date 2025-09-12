# ICNet 噪声调制集成说明

## 概述

本文档说明了如何在ICNet网络中集成噪声调制功能，参考DRNrt_Retinex的实现方式。

## 主要修改

### 1. 新增噪声调制模块 (`noise_modules.py`)

- **PositionalEncoding**: 位置编码模块，将噪声级别编码为特征向量
- **Swish**: Swish激活函数
- **FeatureWiseAffine**: 特征级仿射变换，用于噪声调制
- **NoiseLevelMLP**: 噪声级别MLP，将时间步编码为噪声特征

### 2. ICNet网络修改

#### 初始化阶段 (`__init__`)
```python
# 噪声调制相关组件
self.noise_level_mlp = NoiseLevelMLP(args.n_feats)

# 为不同层级添加噪声调制层
self.t_enc_layer1 = FeatureWiseAffine(args.n_feats, args.n_feats)
self.t_enc_layer2 = FeatureWiseAffine(args.n_feats, args.n_feats)
self.t_enc_layer3 = FeatureWiseAffine(args.n_feats, args.n_feats)
self.t_dec_layer1 = FeatureWiseAffine(args.n_feats, args.n_feats)
self.t_dec_layer2 = FeatureWiseAffine(args.n_feats, args.n_feats)
self.t_dec_layer3 = FeatureWiseAffine(args.n_feats, args.n_feats)
```

#### 前向传播阶段 (`forward`)
```python
# 噪声调制：将时间步编码为噪声特征
t = self.noise_level_mlp(time)

# 在各个关键层级应用噪声调制
feature = self.t_enc_layer1(feature, t)  # 编码器第一层
inp4 = self.t_enc_layer2(inp4, t)        # 编码器第二层
x3 = self.t_enc_layer3(x3, t)            # 编码器第三层
x2 = self.t_dec_layer3(x2, t)            # 解码器第三层
x1 = self.t_dec_layer2(x1, t)            # 解码器第二层
res = self.t_dec_layer1(res, t)          # 解码器第一层
```

## 噪声调制原理

### 1. 时间步编码
- 输入时间步 `time` 通过 `NoiseLevelMLP` 编码为噪声特征向量 `t`
- 使用位置编码和MLP网络将标量时间步转换为高维特征

### 2. 特征级调制
- 使用 `FeatureWiseAffine` 将噪声特征调制到网络特征上
- 通过仿射变换 `x = x + noise_feature` 实现噪声信息注入

### 3. 多层级调制
- 在编码器和解码器的关键层级都应用噪声调制
- 确保噪声信息在整个网络中有效传播

## 使用方法

### 1. 导入模块
```python
from icnet.icnet import LFSRNet
from icnet.noise_modules import NoiseLevelMLP, FeatureWiseAffine
```

### 2. 创建网络
```python
# 假设 args.n_feats = 64
model = LFSRNet(args)
```

### 3. 前向传播
```python
# 输入参数
# x: 输入图像 (B, 3, H, W)
# diff_img: 差分图像 (B, 3, H, W)  
# time: 时间步 (B,)

output = model(x, diff_img, time)
```

## 与DRNrt_Retinex的对比

| 特性 | DRNrt_Retinex | ICNet (修改后) |
|------|---------------|----------------|
| 噪声MLP | ✓ | ✓ |
| 位置编码 | ✓ | ✓ |
| 特征级仿射 | ✓ | ✓ |
| 多层级调制 | ✓ | ✓ |
| 编码器调制 | ✓ | ✓ |
| 解码器调制 | ✓ | ✓ |

## 优势

1. **渐进式噪声注入**: 在多个层级逐步注入噪声信息
2. **特征自适应**: 噪声调制根据特征内容自适应调整
3. **网络兼容性**: 保持原有网络结构，仅添加噪声调制功能
4. **训练稳定性**: 使用成熟的噪声调制技术，保证训练稳定

## 注意事项

1. 确保 `args.n_feats` 参数正确设置
2. 时间步 `time` 应该是标量或一维张量
3. 噪声调制会增加网络参数量和计算量
4. 建议在训练时使用适当的噪声调度策略