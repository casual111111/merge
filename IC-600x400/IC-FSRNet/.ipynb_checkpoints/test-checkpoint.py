import os
from option import args
import torch
import model
from data import dataset
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time  # 导入时间模块
import re  # 导入正则表达式模块
from tqdm import tqdm  # 导入 tqdm
import util


# 初始化模型和数据
net = model.get_model(args)
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
testdata = dataset.Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
testset = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)

# 加载预训练模型
pretrained_dict = torch.load('/root/autodl-tmp/low-lightSR/IC-FSRDENet-main/IC-FSRNet/experiment/lolX4/model/epoch_best_359.pth')
net.load_state_dict(pretrained_dict)

# 创建保存文件夹，添加时间戳
timestamp = time.strftime("%m%d_%H%M", time.localtime())  # 获取当前时间戳
save_name = f"test_{timestamp}"  # 文件夹名称添加时间戳
os.makedirs(os.path.join(args.save_path, args.writer_name, save_name), exist_ok=True)

# 计算 PSNR 和 SSIM
total_psnr = 0
total_ssim = 0
num_samples = 0

# 确保只在一开始解析一次 args
# 计算并输出测试集的 PSNR 和 SSIM
with torch.no_grad():
    net.eval()
    if args.dir_data == "/root/autodl-tmp/dataset/LLLR/RELLISUR/RELLISUR_256":
        # 遍历测试集并计算每张图片的 PSNR 和 SSIM（去掉进度条）
        for batch, (lr, hr, filename, suffix) in enumerate(testset):  # 去掉 tqdm
            lr, hr = util.prepare(lr), util.prepare(hr)
            sr = net(lr)

            # 计算 PSNR 和 SSIM
            psnr_c, ssim_c = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu())  # 计算当前批次的 PSNR 和 SSIM
            total_psnr += psnr_c
            total_ssim += ssim_c
            num_samples += 1
            print(f"Image {filename[0][:-4]}{suffix}.png: PSNR = {psnr_c:.3f}, SSIM = {ssim_c:.3f}")
            # 保存超分辨率图像，并在文件名中添加后缀
            save_path = os.path.join(args.save_path, args.writer_name, save_name, f'{filename[:-4]}{suffix}.png')
            torchvision.utils.save_image(sr[0], save_path)

    else:
        for batch, (lr, hr, filename) in enumerate(testset):
            lr, hr = util.prepare(lr), util.prepare(hr)
            sr = net(lr)

            # 计算 PSNR 和 SSIM
            psnr_c, ssim_c = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu())  # 计算当前批次的 PSNR 和 SSIM
            total_psnr += psnr_c
            total_ssim += ssim_c
            num_samples += 1
            print(f"Image {filename[0][:-4]}.png: PSNR = {psnr_c:.3f}, SSIM = {ssim_c:.3f}")
            save_path = os.path.join(args.save_path, args.writer_name, save_name, f'{filename[0][:-4]}.png')
            torchvision.utils.save_image(sr[0], save_path)

    # 计算平均 PSNR 和 SSIM
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    # 输出并保存结果
    print(f"Average PSNR: {avg_psnr:.3f}")
    print(f"Average SSIM: {avg_ssim:.3f}")

    # 保存平均 PSNR 和 SSIM 到 TensorBoard
    writer.add_scalar("test_psnr", avg_psnr, 0)
    writer.add_scalar("test_ssim", avg_ssim, 0)

    print('Test Over')
