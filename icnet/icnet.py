import torch.nn as nn
import torch
from icnet import common,fenet,refineblock
from icnet import biablock
from icnet.noise_modules import NoiseLevelMLP, FeatureWiseAffine
import torch.nn.functional as F
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
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=args.n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True)])
        
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

    def forward(self, x,diff_img, time):

    #####///////////256x256///////////////////////////////////////////////////
        #x是暗光HR(x(1,3,256,256)) ///（1，3，2496，2496）
        # x_down=x #//（3，624，624）
        # x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False) #scale_factor=4
        # grid = self.FENet(x_down)#（Encoder +Illumination Estimation=B）(1.64,16,16)//(64,624,624)//（64，64，64）
        # guidance = self.GNet(x)#生成各个尺度的引导图G
        # save_x = x  
        # feature = self.head(x)#F0SR（64，2496，2496）
        
        # #######128/////1248
        # x4 = self.down1(feature)#(1,64,1248,1248)///
        # grid = grid.view(grid.shape[0], 64, 4, 32, 32)#B(1,64,4,312,312)
        # inp4 = self.down_stage1(x4, guidance[1], grid)#torch.Size([1, 64, 128, 128]) #B->) Super-Resolution + 3D + SlcIllumination Adjustment//（64,1248,1248）
        # x3 = self.down2(inp4)#(1,64,64,64) //(64,624,624)
        # grid = grid.view(grid.shape[0], 64, 64, 64)#(64,624,624)
        # grid = self.refine_grid[0](grid, x3) #torch.Size([1, 64, 16, 16])#Illumination Refinement#(64,624,624)


        # ###64///624
        # grid = grid.view(grid.shape[0], 64, 4, 32, 32)#(64,4,312,312)
        # inp3 = self.down_stage2(x3, guidance[2], grid) ####(64,624,624)
        # inp2 = inp3+ x3
        # x2= self.up21(inp2)###(64,1248,1248)
        # grid = grid.view(grid.shape[0], 64, 64, 64)
        # grid = self.refine_grid[1](grid, x2)
        

        # #//1248
        # grid = grid.view(grid.shape[0], 64, 4, 32, 32)
        # inp2= self.up2_stage1(x2, guidance[1], grid)###(64,1248,1248)
        # inp4 = inp2 + x4
        # x1= self.up22(inp4)#(1,64,2496,2496)
        # grid = grid.view(grid.shape[0], 64, 64, 64)
        # grid = self.refine_grid[2](grid, x1)

        #  ###2496
        # grid = grid.view(grid.shape[0], 64, 4, 32, 32)
        # res = self.up2_stage2(x1, guidance[0], grid)
        # sr = self.tail(res) + save_x
        # return sr
        #////////////////////////////256x256/////////////////////////////////////



        #///////////////////////600x400/////////////////////////////////////
        # 噪声调制：将时间步编码为噪声特征
        t = self.noise_level_mlp(time)
        img=x
        x= torch.cat((x, diff_img), 1)
        B,C,H,W=x.shape
        
        # 特征提取和引导图生成
        grid = self.FENet(img)#（Encoder +Illumination Estimation=B）(1.64,16,16)//(64,624,624)//（64，64，64）
        guidance = self.GNet(img)#生成各个尺度的引导图G
        save_x = x  
        
        # 初始特征提取和噪声调制
        feature = self.head(x)#F0SR (1, 64, 400, 600)///
        feature = self.t_enc_layer1(feature, t)  # 第一层噪声调制
        
        # 编码器路径 - 下采样阶段
        #######400x600 -> 200x300
        x4 = self.down1(feature)#x4(1, 64, 200, 300)///
        grid = grid.view(grid.shape[0], 64, 4, int(H/8),  int(W/8))#grid[1, 64, 4, 50, 75]
        inp4 = self.down_stage1(x4, guidance[1], grid)#guidance(1,1,)#B->) Super-Resolution + 3D + SlcIllumination Adjustment//（64,1248,1248）
        inp4 = self.t_enc_layer2(inp4, t)  # 第二层噪声调制
        
        # 200x300 -> 100x150
        x3 = self.down2(inp4)#x3(1, 64, 100, 150) 
        grid = grid.view(grid.shape[0], 64, int(H/4), int(W/4))#(1,64,100,150))
        gh, gw = grid.shape[-2], grid.shape[-1]
        grid = self.refine_grid[0](grid, x3, gh, gw) #torch.Size([1, 64, 16, 16])#Illumination Refinement#(64,624,624)
        x3 = self.t_enc_layer3(x3, t)  # 第三层噪声调制

        # 瓶颈层处理
        ###100, 150
        grid = grid.view(grid.shape[0], 64, 4, int(H/8),  int(W/8))#(64,4,312,312)
        inp3 = self.down_stage2(x3, guidance[2], grid) ####(64,624,624)
        inp2 = inp3+ x3
        
        # 解码器路径 - 上采样阶段
        x2= self.up21(inp2)###(1, 64, 200, 300)
        x2 = self.t_dec_layer3(x2, t)  # 解码器第三层噪声调制
        grid = grid.view(grid.shape[0],64,  int(H/4), int(W/4))
        gh, gw = grid.shape[-2], grid.shape[-1]
        grid = self.refine_grid[1](grid, x2, gh, gw)

        # 100x150 -> 200x300
        grid = grid.view(grid.shape[0], 64, 4, int(H/8),  int(W/8))
        inp2= self.up2_stage1(x2, guidance[1], grid)###(64,1248,1248)
        inp4 = inp2 + x4
        x1= self.up22(inp4)#(1,64,2496,2496)
        x1 = self.t_dec_layer2(x1, t)  # 解码器第二层噪声调制
        grid = grid.view(grid.shape[0],64, int(H/4), int(W/4))
        gh, gw = grid.shape[-2], grid.shape[-1]
        grid = self.refine_grid[2](grid, x1, gh, gw)

        # 最终输出
        ###200x300 -> 400x600
        grid = grid.view(grid.shape[0], 64,4, int(H/8),  int(W/8))
        res = self.up2_stage2(x1, guidance[0], grid)
        res = self.t_dec_layer1(res, t)  # 解码器第一层噪声调制
        sr = self.tail(res) + save_x
        return sr

        
        # x6 = self.down3(inp2)#(1,64,32,32)
        # grid = grid.view(grid.shape[0], 64, 16, 16)
        # grid = self.refine_grid[1](grid, x6) ############
       
        # ####32
        # grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        # x = self.down_stage3(x6, guidance[3], grid)
        # inp3 = x + x6#(1,64,32,32)
        # x = self.up21(inp3)##(1,64,64,64) 
        # grid = grid.view(grid.shape[0], 64, 16, 16)
        # grid = self.refine_grid[2](grid, x)

        # ###64
        # grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        # x = self.up2_stage1(x, guidance[2], grid)
        # inp4 = x + x5
        # x = self.up22(inp4)#(1,64,128,128)
        # grid = grid.view(grid.shape[0], 64, 16, 16)
        # grid = self.refine_grid[3](grid, x)
        # ###128
        # grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        # x = self.up2_stage2(x, guidance[1], grid)
        # inp5 = x + x4
        # x = self.up23(inp5)#(1,64,256,256)
        # grid = grid.view(grid.shape[0], 64, 16, 16)
        # grid = self.refine_grid[4](grid, x)

        # ###256
        # grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        # res = self.up2_stage3(x, guidance[0], grid)
        # sr = self.tail(res) + save_x
        # return sr
def make_model(args):
    return LFSRNet(args)
