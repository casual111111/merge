import torch.nn as nn
from model import refineblock
import torch.nn.functional as F
class FENet(nn.Module):
    def __init__(self, args):
        super(FENet, self).__init__()

        n_feats = 64
        kernel_size = 3
        self.args = args
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=kernel_size, stride=1, padding=1), nn.ReLU(True)])

        self.down_stage1 = FENetf(args) #encoder + Illumination Estimation
    def forward(self, x):##

        # x = F.interpolate(x, scale_factor=1/16, mode="bicubic")#双三次下采样(1,3,16,16)  
        head = self.head(x)#特征提取模块，卷积加激活函数(64,624,624) 
        grid = self.down_stage1(head)#Encoder)
        return grid #(64,624,624)
class FENetf(nn.Module):
    def __init__(self, args):
        super(FENetf, self).__init__()
        self.args = args
        self.enhance = nn.Sequential(nn.Conv2d(args.n_feats, args.n_feats, 5, stride=1, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(args.n_feats, args.n_feats, 3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(args.n_feats, args.n_feats, 1, stride=1, padding=0), )
        embed_dim = 8
        self.block = refineblock.SingleFusionTransformer(dim=embed_dim, num_heads=2)
        self.input_proj = refineblock.InputProj(in_channel=args.n_feats, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = refineblock.OutputProj(in_channel=embed_dim, out_channel=args.n_feats, kernel_size=3, stride=1)

    def forward(self, x):#(64,624,624)
        x = self.input_proj(x)#卷积加激活，展平加转置（B，C,HW）,归一化(64,389376)
        x1 = self.block(x, x)#（Ecoder）自注意力+全连接层
        x = self.output_proj(x1)#形状映射回去(B,C,H,W)(64,624,624)
        x = self.enhance(x)#（IIumination Estimation）三层卷积(64,624,624)
        return x

