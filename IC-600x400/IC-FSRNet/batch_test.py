import os
import glob
import time
import json
from option import args
import torch
import model
from data import dataset
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import re
from tqdm import tqdm
import util


def test_single_model(model_path, args, testset, save_results=True):
    """
    测试单个模型权重文件
    
    Args:
        model_path: 权重文件路径
        args: 配置参数
        testset: 测试数据集
        save_results: 是否保存结果图像
    
    Returns:
        dict: 包含测试结果的字典
    """
    print(f"\n正在测试模型: {os.path.basename(model_path)}")
    
    # 初始化模型
    net = model.get_model(args)
    
    try:
        # 加载权重
        pretrained_dict = torch.load(model_path, map_location=args.device)
        net.load_state_dict(pretrained_dict)
        net.to(args.device)
        net.eval()
        
        # 创建保存文件夹
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        timestamp = time.strftime("%m%d_%H%M", time.localtime())
        save_name = f"batch_test_{model_name}_{timestamp}"
        
        if save_results:
            save_dir = os.path.join(args.save_path, args.writer_name, save_name)
            os.makedirs(save_dir, exist_ok=True)
        
        # 计算指标
        total_psnr = 0
        total_ssim = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch, (lr, hr, filename) in enumerate(tqdm(testset, desc=f"测试 {model_name}")):
                lr, hr = util.prepare(lr), util.prepare(hr)
                sr = net(lr)
                
                # 计算 PSNR 和 SSIM
                psnr_c, ssim_c = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu())
                total_psnr += psnr_c
                total_ssim += ssim_c
                num_samples += 1
                
                # 保存结果图像
                if save_results:
                    save_path = os.path.join(save_dir, f'{filename[0][:-4]}.png')
                    torchvision.utils.save_image(sr[0], save_path)
        
        # 计算平均指标
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        
        print(f"模型 {model_name}: PSNR = {avg_psnr:.3f}, SSIM = {avg_ssim:.4f}")
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'num_samples': num_samples,
            'save_dir': save_dir if save_results else None
        }
        
    except Exception as e:
        print(f"测试模型 {model_path} 时出错: {str(e)}")
        return {
            'model_name': os.path.splitext(os.path.basename(model_path))[0],
            'model_path': model_path,
            'error': str(e)
        }


def batch_test_models(model_dir, args, save_results=True, save_summary=True):
    """
    批量测试模型权重文件
    
    Args:
        model_dir: 权重文件目录
        args: 配置参数
        save_results: 是否保存结果图像
        save_summary: 是否保存测试总结
    """
    print("开始批量测试模型...")
    print(f"权重文件目录: {model_dir}")
    
    # 准备测试数据
    testdata = dataset.Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
    testset = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
    
    # 获取所有权重文件
    model_pattern = os.path.join(model_dir, "*.pth")
    model_files = glob.glob(model_pattern)
    model_files.sort()  # 按文件名排序
    
    if not model_files:
        print(f"在目录 {model_dir} 中没有找到 .pth 文件")
        return
    
    print(f"找到 {len(model_files)} 个权重文件")
    
    # 测试结果列表
    results = []
    
    # 批量测试
    for i, model_path in enumerate(model_files):
        print(f"\n进度: {i+1}/{len(model_files)}")
        result = test_single_model(model_path, args, testset, save_results)
        results.append(result)
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    # 生成测试总结
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print("\n" + "="*50)
    print("批量测试完成!")
    print(f"成功测试: {len(successful_results)} 个模型")
    print(f"失败测试: {len(failed_results)} 个模型")
    
    if successful_results:
        # 按PSNR排序
        successful_results.sort(key=lambda x: x['avg_psnr'], reverse=True)
        
        print("\n测试结果排名 (按PSNR排序):")
        print("-" * 80)
        print(f"{'排名':<4} {'模型名称':<20} {'PSNR':<8} {'SSIM':<8}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results, 1):
            print(f"{i:<4} {result['model_name']:<20} {result['avg_psnr']:<8.3f} {result['avg_ssim']:<8.4f}")
        
        # 找出最佳模型
        best_model = successful_results[0]
        print(f"\n最佳模型: {best_model['model_name']}")
        print(f"最佳PSNR: {best_model['avg_psnr']:.3f}")
        print(f"最佳SSIM: {best_model['avg_ssim']:.4f}")
    
    if failed_results:
        print("\n失败的模型:")
        for result in failed_results:
            print(f"- {result['model_name']}: {result['error']}")
    
    # 保存测试总结
    if save_summary:
        timestamp = time.strftime("%m%d_%H%M", time.localtime())
        summary_file = os.path.join(args.save_path, args.writer_name, f"batch_test_summary_{timestamp}.json")
        
        summary = {
            'timestamp': timestamp,
            'total_models': len(model_files),
            'successful_tests': len(successful_results),
            'failed_tests': len(failed_results),
            'results': results,
            'best_model': best_model if successful_results else None
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试总结已保存到: {summary_file}")


if __name__ == "__main__":
    # 设置权重文件目录
    model_dir = "/root/autodl-tmp/low-lightSR/IC-600x400/IC-FSRNet/experiment/LOL_save/model"
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"错误: 权重文件目录不存在: {model_dir}")
        exit(1)
    
    # 开始批量测试
    batch_test_models(model_dir, args, save_results=True, save_summary=True) 