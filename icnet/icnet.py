import torch.nn as nn
import torch
from icnet import common,fenet,refineblock
from icnet import biablock
from icnet.noise_modules import NoiseLevelMLP, FeatureWiseAffine
import torch.nn.functional as F
from einops import rearrange
import numbers

# Retinex监督相关组件
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class Illum_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Illum_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illum):
        b, c, h, w = x.shape
        bi, ci, hi, wi = illum.shape
        assert b == bi and c == ci and h == hi and w == wi, "Input and illumination should have the same dimensions"
        q = self.q_dwconv(self.q(illum))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class Illum_Guided_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Illum_Guided_TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.normL = LayerNorm(dim, LayerNorm_type)
        self.attn = Illum_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, illum):
        x = x + self.attn(self.norm1(x), self.normL(illum))
        x = x + self.ffn(self.norm2(x))
        return x

class Retinex_Supervision_Attn(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, out_channels_L=1, out_channels_R=3):
        super(Retinex_Supervision_Attn, self).__init__()
        self.L_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(dim, out_channels_L, kernel_size=1, stride=1, padding=0))
        self.R_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels_R, kernel_size=1, stride=1, padding=0),
            nn.Tanh()  # 确保输出在[-1,1]范围内
        )
        self.L_rein = nn.Sequential(nn.Conv2d(out_channels_L, dim, kernel_size=3, stride=1, padding=1),
                                    nn.GELU())
        self.R_rein = nn.Conv2d(out_channels_R, dim, kernel_size=3, stride=1, padding=1)
        self.illum_attn = Illum_Guided_TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, x):
        L = self.L_conv(x)
        R = self.R_conv(x)
        L_rein = self.L_rein(L.detach())
        R_rein = self.R_rein(R.detach())
        x = self.illum_attn(x, L_rein)
        x = x + 0.1*R_rein
        debug_tensor_stats("x", x)
        debug_tensor_stats("L", L)
        debug_tensor_stats("R", R)
        return x, L, R

class LightNet(nn.Module):
    def __init__(self, args):
        super(LightNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=args.n_feats, kernel_size=1)

    def forward(self, x):
        return x
class GNet(nn.Module):
    def __init__(self, args):
        super(GNet, self).__init__()

        n_feats = 64
        kernel_size = 3
        self.args = args
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=kernel_size, stride=1, padding=1), nn.ReLU(True)])
        self.conv1 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1, padding=1),
                                     nn.ReLU(True)])

        self.conv2 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                     nn.ReLU(True),
                                     nn.AvgPool2d(2)])#AvgPool2d是下采样1/2
        self.conv3 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                     nn.ReLU(True),
                                     nn.AvgPool2d(2)

        ])
        self.conv4 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                     nn.ReLU(True),
                                     nn.AvgPool2d(2)])

        self.tail = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                    nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size,
                                              stride=1,
                                              padding=1),
                                    nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size,
                                              stride=1,
                                              padding=1),
                                    nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size,
                                              stride=1,
                                              padding=1),
                                    ])

    def forward(self, x):#x(1,3,2496,2496)

        head = self.head(x)#(1，64，256，256) //(64,2496,2496)
        guidance = []
        f = self.conv1(head)#(1,64,256,256) //(64,2496,2496)
        f1 = self.tail[0](f)#(1,1,256,256)//(1,2496,2496)
        guidance.append(f1)
        f = self.conv2(f)#(1,64,128,128) //(64,1248,1248)
        f1 = self.tail[1](f)#(1,1,128,128))//(1,1248,1248)
        guidance.append(f1)
        f = self.conv3(f)#(1,64,64,64)//(64,624,624)
        f1 = self.tail[2](f)#(1,1,64,64)//(1,624,624)
        guidance.append(f1)
        # f = self.conv4(f)#(1,64,32,32)
        # f1 = self.tail[3](f)#(1,1,32,32)
        # guidance.append(f1)

        return guidance

def debug_tensor_stats(name, t):
    """
    打印张量的基本统计信息:
      - min/max
      - mean/std
      - 各通道均值
    """
    if t is None:
        print(f"{name}: None")
        return
    # 保证是 CPU 上的 numpy，避免太卡
    t_cpu = t.detach().cpu()
    print(f"[{name}] shape={tuple(t_cpu.shape)} "
          f"min={t_cpu.min().item():.4f} max={t_cpu.max().item():.4f} "
          f"mean={t_cpu.mean().item():.4f} std={t_cpu.std().item():.4f}")
    if t_cpu.ndim == 4 and t_cpu.shape[1] <= 10:  # 常见 BCHW
        ch_mean = t_cpu.mean(dim=(0, 2, 3))
        print(f"    per-channel mean: {ch_mean.numpy()}")


class LFSRNet(nn.Module):
    def __init__(self, args):
        super(LFSRNet, self).__init__()
        n_blocks = 8
        self.FENet = fenet.FENet(args)
        self.GNet = GNet(args)
        self.args = args
        
        # 噪声调制相关组件
        self.noise_level_mlp = NoiseLevelMLP(args.n_feats)
        
        ## light FSRNet
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=9, out_channels=args.n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True)])
        
        # 为不同层级添加噪声调制层
        self.t_enc_layer1 = FeatureWiseAffine(args.n_feats, args.n_feats)
        self.t_enc_layer2 = FeatureWiseAffine(args.n_feats, args.n_feats)
        self.t_enc_layer3 = FeatureWiseAffine(args.n_feats, args.n_feats)
        self.t_dec_layer1 = FeatureWiseAffine(args.n_feats, args.n_feats)
        self.t_dec_layer2 = FeatureWiseAffine(args.n_feats, args.n_feats)
        self.t_dec_layer3 = FeatureWiseAffine(args.n_feats, args.n_feats)
        
        self.down1 =  common.invUpsampler(scale=2, n_feats=args.n_feats)
        self.down_stage1 = biablock.Biablock(args)
        self.down2 =  common.invUpsampler(scale=2, n_feats=args.n_feats)
        self.down_stage2 = biablock.Biablock(args)
        self.down3 = common.invUpsampler(scale=2, n_feats=args.n_feats)
        self.down_stage3 = biablock.Biablock(args)

        self.up21 = common.Upsampler_module(scale=2, n_feats=args.n_feats)
        self.up2_stage1 = biablock.Biablock(args)
        self.up22 = common.Upsampler_module(scale=2, n_feats=args.n_feats)

        self.up2_stage2 = biablock.Biablock(args)
        self.up23 = common.Upsampler_module(scale=2, n_feats=args.n_feats)
        self.up2_stage3 = biablock.Biablock(args)
        self.tail =  nn.Conv2d(in_channels=args.n_feats, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.refine_grid = nn.Sequential(*[refineblock.RB(args),refineblock.RB(args),refineblock.RB(args),refineblock.RB(args),refineblock.RB(args)
                                           ])
        
        # 添加Retinex监督层
        self.retinex_attn_level1 = Retinex_Supervision_Attn(dim=args.n_feats, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.retinex_attn_level2 = Retinex_Supervision_Attn(dim=args.n_feats, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.retinex_attn_level3 = Retinex_Supervision_Attn(dim=args.n_feats, num_heads=4, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.retinex_attn_level_refinement = Retinex_Supervision_Attn(dim=args.n_feats, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

    def forward(self, x,diff_img, time):

        t = self.noise_level_mlp(time)
        #取前三个通道
        img=x[:, :3, :, :]
        x= torch.cat((x, diff_img), 1)
        B,C,H,W=x.shape
        
        # 特征提取和引导图生成
        grid = self.FENet(img)#
        guidance = self.GNet(img)#生成各个尺度的引导图G
        save_x = img 
        debug_tensor_stats("save_x", save_x)
        
        # 初始特征提取和噪声调制
        feature = self.head(x)#
        feature = self.t_enc_layer1(feature, t)  # 第一层噪声调制
        
        # 编码器路径 - 下采样阶段
        #######400x600 -> 200x300
        x4 = self.down1(feature)
        grid = grid.view(grid.shape[0], 64, 4, int(H/8),  int(W/8))
        inp4 = self.down_stage1(x4, guidance[1], grid)
        inp4 = self.t_enc_layer2(inp4, t)  # 第二层噪声调制
        
        # 200x300 -> 100x150
        x3 = self.down2(inp4)#x3(1, 64, 100, 150) 
        grid = grid.view(grid.shape[0], 64, int(H/4), int(W/4))#(1,64,100,150))
        gh, gw = grid.shape[-2], grid.shape[-1]
        grid = self.refine_grid[0](grid, x3, gh, gw) #torch.Size([1, 64, 16, 16])#Illumination Refinement#(64,624,624)
        x3 = self.t_enc_layer3(x3, t)  # 第三层噪声调制

        # 瓶颈层处理 - 添加Retinex监督
        ###100, 150
        grid = grid.view(grid.shape[0], 64, 4, int(H/8),  int(W/8))
        inp3, inp3_l, inp3_r = self.retinex_attn_level3(x3)  # 第三层Retinex监督 - 在进入主模块之前
        # inp3=x3
        inp3 = self.down_stage2(inp3, guidance[2], grid)
        inp2 = inp3+ x3
        
        # 解码器路径 - 上采样阶段
        x2= self.up21(inp2)###(1, 64, 200, 300)
        x2 = self.t_dec_layer3(x2, t)  # 解码器第三层噪声调制
        grid = grid.view(grid.shape[0],64,  int(H/4), int(W/4))
        gh, gw = grid.shape[-2], grid.shape[-1]
        grid = self.refine_grid[1](grid, x2, gh, gw)

        # 100x150 -> 200x300 - 添加Retinex监督
        grid = grid.view(grid.shape[0], 64, 4, int(H/8),  int(W/8))
        inp2, inp2_l, inp2_r = self.retinex_attn_level2(x2)  # 第二层Retinex监督 - 在进入主模块之前
        debug_tensor_stats("inp2", inp2)
        debug_tensor_stats("inp2_l", inp2_l)
        debug_tensor_stats("inp2_r", inp2_r)    
        # inp2=x2
        inp2= self.up2_stage1(inp2, guidance[1], grid)###(64,1248,1248)
        inp4 = inp2 + x4
        x1= self.up22(inp4)#(1,64,2496,2496)
        x1 = self.t_dec_layer2(x1, t)  # 解码器第二层噪声调制
        grid = grid.view(grid.shape[0],64, int(H/4), int(W/4))
        gh, gw = grid.shape[-2], grid.shape[-1]
        grid = self.refine_grid[2](grid, x1, gh, gw)

        # 最终输出 - 添加Retinex监督
        ###200x300 -> 400x600
        grid = grid.view(grid.shape[0], 64,4, int(H/8),  int(W/8))
        res, res_l, res_r = self.retinex_attn_level1(x1)  # 第一层Retinex监督 - 在进入主模块之前
        # res=x1
        res = self.up2_stage2(res, guidance[0], grid)
        res = self.t_dec_layer1(res, t)  # 解码器第一层噪声调制
        
        # 最终精化阶段的Retinex监督
        res, res_refine_l, res_refine_r = self.retinex_attn_level_refinement(res)
        res_out = self.tail(res)
        debug_tensor_stats("tail(res)", res_out)
        sr = 0.5*res_out + save_x
        debug_tensor_stats("sr", sr)
        # 返回结果和Retinex监督信息
        return sr, [res_refine_l, res_l, inp2_l, inp3_l], [res_refine_r, res_r, inp2_r, inp3_r]
        # return sr

def make_model(args):
    return LFSRNet(args)
