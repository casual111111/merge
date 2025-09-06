import os
import gc
from option import args
import torch
import torch.optim as optim
import torch.nn as nn
import model
from data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import util

# 修复1：添加save_every属性（如果不存在）
if not hasattr(args, 'save_every'):
    args.save_every = 0  # 默认不保存中间模型

# 初始化模型
net = model.get_model(args).to(args.device)

# 创建日志目录
from datetime import datetime
timestamp = datetime.now().strftime('%m%d_%H%M')
log_dir = os.path.join('./logs', args.writer_name, timestamp)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# 准备数据集
traindata = dataset.Data(root=os.path.join(args.dir_data, args.data_train), args=args, train=True)
valdata = dataset.Data(root=os.path.join(args.dir_data, args.data_val), args=args, train=False)
trainset = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=4)
valset = DataLoader(valdata, batch_size=1, shuffle=False, num_workers=1)

# 损失函数和优化器
criterion1 = nn.L1Loss()
optimizer = optim.Adam(params=net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

# 创建模型保存目录
model_dir = os.path.join(args.save_path, args.writer_name, 'model')
os.makedirs(model_dir, exist_ok=True)

# 初始化最佳指标
best_psnr = 0.0
best_ssim = 0.0

# 训练循环
for epoch in range(args.epochs):
    # ================= 训练阶段 =================
    net.train()
    train_loss = 0.0
    
    # 训练进度条
    train_bar = tqdm(trainset, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', ncols=100)
    for batch, (lr, hr, _) in enumerate(train_bar):
        lr, hr = util.prepare(lr), util.prepare(hr)
        
        # 前向传播
        sr = net(lr)
        l1_loss = criterion1(sr, hr)
        train_loss += l1_loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        l1_loss.backward()
        optimizer.step()
        
        # 更新进度条
        train_bar.set_postfix(loss=f"{l1_loss.item():.4f}")
    
    # 记录训练损失
    avg_train_loss = train_loss / len(trainset)
    print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss * 255:.3f}")
    writer.add_scalar('train_loss', avg_train_loss * 255, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
    # ================= 验证阶段 =================
    net.eval()
    val_psnr_dic = 0.0
    val_ssim_dic = 0.0
    
    # 验证进度条 - 关键修改：添加torch.no_grad()
    val_bar = tqdm(valset, desc=f'Epoch {epoch+1}/{args.epochs} [Validate]', ncols=100)
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for batch, (lr, hr, filename) in enumerate(val_bar):
            lr, hr = util.prepare(lr), util.prepare(hr)
            
            # 推理
            sr = net(lr)
            
            # 修复2：确保张量形状匹配
            # 有些模型可能输出带批处理维度的结果
            if sr.dim() == 4 and sr.size(0) == 1:
                sr = sr.squeeze(0)
            if hr.dim() == 4 and hr.size(0) == 1:
                hr = hr.squeeze(0)
            
            # 计算指标 - 移到CPU上计算
            hr_cpu = hr.cpu()
            sr_cpu = sr.cpu()
            
            # 修复3：添加形状检查
            print(f"HR shape: {hr_cpu.shape}, SR shape: {sr_cpu.shape}")  # 调试输出
            
            # 修复4：使用正确的维度计算指标
            # 如果图像是多通道的，确保计算指标时考虑通道维度
            psnr_c, ssim_c = util.calc_metrics(hr_cpu, sr_cpu)
            val_psnr_dic += psnr_c
            val_ssim_dic += ssim_c
            
            # 更新进度条
            val_bar.set_postfix(psnr=f"{psnr_c:.2f}", ssim=f"{ssim_c:.4f}")
            
            # 及时释放不再需要的变量
            del lr, hr, sr, hr_cpu, sr_cpu
    
    # 强制垃圾回收
    torch.cuda.empty_cache()
    gc.collect()
    
    # 计算平均指标
    avg_psnr = val_psnr_dic / len(valset)
    avg_ssim = val_ssim_dic / len(valset)
    
    # 记录验证指标
    print(f"Epoch: {epoch+1}, Val PSNR: {avg_psnr:.3f}, Val SSIM: {avg_ssim:.3f}")
    writer.add_scalar("val_psnr_DIC", avg_psnr, epoch)
    writer.add_scalar("val_ssim_DIC", avg_ssim, epoch)
    
    # 保存最佳模型
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(net.state_dict(),
                   os.path.join(args.save_path, args.writer_name, 'model', 'epoch_best_{}.pth'.format(epoch + 1)))
        print(f"Saved new best PSNR model: {best_psnr:.3f}")
    
    if avg_ssim > best_ssim:
        best_ssim = avg_ssim
        torch.save(net.state_dict(),
                   os.path.join(args.save_path, args.writer_name, 'model', 'epoch_best_{}.pth'.format(epoch + 1)))
        print(f"Saved new best SSIM model: {best_ssim:.3f}")


# 关闭writer
writer.close()
print("Training completed!")