from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.functional
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F
import os.path as osp
from tqdm import tqdm
import torch
import cv2
import os
cv2.setNumThreads(1)
from scripts.utils import pad_tensor_back
from types import SimpleNamespace
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from icnet.icnet import LFSRNet


@ARCH_REGISTRY.register()
class ICNet(nn.Module):
    def __init__(self, n_feats=64, device='cuda', **kwargs):
        super().__init__()
        # 将 YAML 参数转为 LFSRNet 所需的 args 对象
        args = SimpleNamespace(n_feats=n_feats, device=device, **kwargs)
        self.net = LFSRNet(args)

    def forward(self, x):
        return self.net(x)


@MODEL_REGISTRY.register()
class DRNetModel(BaseModel):
    def __init__(self, opt):
        super(DRNetModel, self).__init__(opt)

        # define sr3-u-net network
        self.sr3unet = build_network(opt['network_sr3unet'])#扩散模型的unet网络
        self.sr3unet = self.model_to_device(self.sr3unet)
        opt['network_ddpm']['denoise_fn'] = self.sr3unet

        #define icnet network
        self.icnet = build_network(opt['network_icnet'])#icnet网络
        self.icnet = self.model_to_device(self.icnet)
        # 如果需要在扩散模型里访问icnet，直接挂属性（可选）
        # if getattr(self, 'icnet', None) is not None:
        #     bare = self.ddpm.module if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)) else self.ddpm
        #     bare.icnet = self.icnet
        # define consistent-u-net network
        # self.consistent_unet = build_network(opt['network_consistentunet'])#专家网络结合retinex网络
        # self.consistent_unet = self.model_to_device(self.consistent_unet)
        # opt['network_ddpm']['restore_fn'] = self.consistent_unet

        self.ddpm = build_network(opt['network_ddpm'])#扩散模型采样
        self.ddpm = self.model_to_device(self.ddpm)

        # 关键：把 icnet 挂到 ddpm 上，供 ddpm_arch 调用 self.icnet(...)
        if getattr(self, 'icnet', None) is not None:
            bare = self.ddpm.module if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)) else self.ddpm
            bare.icnet = self.icnet

        # self.decom_net = build_network(opt['network_decom'])#retinex网络
        # self.decom_net = self.model_to_device(self.decom_net)

        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_ddpm_model = self.ddpm.module
        else:
            self.bare_ddpm_model = self.ddpm

        self.bare_ddpm_model.set_new_noise_schedule(schedule_opt=opt['ddpm_schedule'], device=self.device)
        self.bare_ddpm_model.set_loss(device=self.device)
        self.print_network(self.ddpm)

        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

        if self.is_train:
            self.init_training_settings()

        logger = get_root_logger()
        is_train_mode = not self.opt['val'].get('test_flag', False)
        # if is_train_mode:
        #     assert os.path.exists(self.opt['network_decom']['path']), logger.info("The decom-network weights is not exsit.")
        #     load_decom_path = self.opt['network_decom']['path']
        #     if isinstance(self.decom_net, (DataParallel, DistributedDataParallel)):
        #         logger.info(self.decom_net.module.load_state_dict(torch.load(load_decom_path, map_location='cuda')['model'], strict=True))
        #     else:
        #         logger.info(self.decom_net.load_state_dict(torch.load(load_decom_path, map_location='cuda')['model'], strict=True))
        #     logger.info("Load the decom-net weight success! Setting --------> eval mode")
        # else:
        #     logger.info("Begin to eval the model!")

        # if isinstance(self.decom_net, (DataParallel, DistributedDataParallel)):
        #     self.decom_net.module.requires_grad_(False)
        #     self.decom_net.module.eval()
        # else:
        #     self.decom_net.requires_grad_(False)
        #     self.decom_net.eval()

        # load_path = self.opt['path'].get('pretrain_network_ddpm', None)
        # if load_path is not None:
        #     if isinstance(self.decom_net, (DataParallel, DistributedDataParallel)):
        #         param_key = self.opt['path'].get('param_key_ddpm', 'params')
        #         self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_ddpm', True), param_key)
        #     else:
        #         param_key = self.opt['path'].get('param_key_ddpm', 'params')
        #         self.load_bare_network(self.bare_ddpm_model, load_path, self.opt['path'].get('strict_load_ddpm', True), param_key)

    def init_training_settings(self):
        self.ddpm.train()

        self.best_val_psnr = 0.0
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        net_g_reg_ratio = 1
        normal_params = []
        normal_params = list(self.ddpm.parameters())
        optim_params_g = normal_params
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0.9**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

    def load_network(self, net, load_path, strict=False, param_key='params'):
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')

        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=False)

    def load_bare_network(self, net, load_path, strict=True, param_key='params'):
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if "module." in k:
                load_net[k.replace("module.", "")] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'pad_left' in data:
            self.pad_left = data['pad_left'].to(self.device)
            self.pad_right = data['pad_right'].to(self.device)
            self.pad_top = data['pad_top'].to(self.device)
            self.pad_bottom = data['pad_bottom'].to(self.device)
    
    def norm_minus1_1(self, x):
        return (x - 0.5) * 2
    
    def norm_0_1(self, x):
        return (x + 1) * 0.5

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        pred_noise, noise, x_recon_out = self.ddpm(self.norm_minus1_1(self.gt), self.norm_minus1_1(self.lq),
                  train_type=self.opt['train'].get('train_type', None),
                  different_t_in_one_batch=self.opt['train'].get('different_t_in_one_batch', None),
                  clip_noise=self.opt['train'].get('clip_noise', None),
                  t_range=self.opt['train'].get('t_range', None),
                  frozen_denoise=self.opt['train'].get('frozen_denoise', None))

        # if isinstance(out, tuple):
        #     pred_noise, noise, x_recon_out = out[:3]
        # else:
        #     pred_noise, noise, x_recon_out = None, None, out

        # supervised_l_list = [F.interpolate(output, size=self.gt.shape[2:], mode='bilinear', align_corners=False) for output in supervised_l_list]
        # supervised_r_list = [F.interpolate(output, size=self.gt.shape[2:], mode='bilinear', align_corners=False) for output in supervised_r_list]

        # r_gt, l_gt = self.decom_net(self.gt)

        x_recon_out = self.norm_0_1(x_recon_out)

        l_total = 0
        loss_dict = OrderedDict()

        l_diff = F.l1_loss(pred_noise, noise)
        l_l1 = F.l1_loss(x_recon_out, self.gt)
        # l_retinex_l = [F.l1_loss(l, l_gt) for l in supervised_l_list]
        # l_retinex_r = [F.l1_loss(r, r_gt) for r in supervised_r_list]
        # l_retinex = 0.5 * sum(l_retinex_l) + sum(l_retinex_r)

        l_total += l_l1  + l_diff 
        loss_dict['l_l1'] = l_l1
        loss_dict['l_diff'] = l_diff
        # loss_dict['l_retinex'] = l_retinex
        # loss_dict['l_MoE'] = l_MoE
        loss_dict['l_total'] = l_total
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def color_loss(self, lq, gt):
        lq_color_norm = torch.nn.functional.normalize(lq, p=2, dim=3)
        gt_color_norm = torch.nn.functional.normalize(gt, p=2, dim=3)
        l_color = torch.mean(1 - torch.sum(lq_color_norm * gt_color_norm, dim=3, keepdim=True))
        return l_color
    
    def gradient_loss(self, lq, gt):
        lq_grad = torch.abs(self.gradient(lq))
        gt_grad = torch.abs(self.gradient(gt))
        l_grad = torch.mean(torch.abs(lq_grad - gt_grad))
        return l_grad
    
    def gradient(self, image):
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        image_gray = F.pad(image_gray, self.padding, mode='replicate')
        gradient_x = F.conv2d(image_gray, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image_gray, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)

    def test(self):
        """Robust test() that matches training I/O and works whether ddim_sample returns:
        - Tensor (sr)
        - Tuple (sr, [l_list], [r_list]) from old Retinex branch
        - Dict with keys like 'sr', 'pred_noise', 'noise', 'x_recon'
        """
        with torch.no_grad():
            self.bare_ddpm_model.eval()

            # 读取验证期可选项
            val_opt = self.opt.get('val', {}) if hasattr(self, 'opt') else {}
            out = self.bare_ddpm_model.ddim_sample(self.norm_minus1_1(self.lq),
                                                    continous=self.opt['val'].get('ret_process', False), 
                                                    ddim_timesteps=self.opt['val'].get('ddim_timesteps', 25),
                                                    return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                    return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                    ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                    ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                    pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                    clip_noise=self.opt['val'].get('clip_noise', False),
                                                    return_all=self.opt['val'].get('ret_all', False))

            # 先清空缓存字段
            self.diff_sr = None
            self.pred_noise = None
            self.noise = None
            self.x_recon = None

            # 兼容多种返回格式
            if isinstance(out, dict):
                # 优先取 'sr' 或 'x_recon' 作为最终重建
                self.diff_sr = out.get('sr', out.get('x_recon', None))
                self.pred_noise = out.get('pred_noise', None)
                self.noise = out.get('noise', None)
                self.x_recon = out.get('x_recon', None)
                # 注释掉不需要的返回
                # self.supervised_l_list = out.get('l_list', None)
                # self.supervised_r_list = out.get('r_list', None)
            elif isinstance(out, (list, tuple)):
                # 旧接口：一般为 (sr, l_list, r_list)；也可能只有 (sr,)
                self.diff_sr = out[0]
                # 注释掉不需要的返回
                # if len(out) > 1 and isinstance(out[1], (list, tuple)):
                #     self.supervised_l_list = out[1]
                # if len(out) > 2 and isinstance(out[2], (list, tuple, torch.Tensor)):
                #     self.supervised_r_list = out[2]
            else:
                # 单张 SR Tensor
                self.diff_sr = out

            # 归一化回 [0,1]
            if self.diff_sr is not None:
                self.diff_sr = self.norm_0_1(self.diff_sr)

            # pad 还原（仅当非过程可视化）
            if hasattr(self, 'pad_left') and not val_opt.get('ret_process', False):
                if self.diff_sr is not None:
                    self.diff_sr = pad_tensor_back(self.diff_sr, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.lq = pad_tensor_back(self.lq, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.gt = pad_tensor_back(self.gt, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

            self.bare_ddpm_model.train()



    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, test=False):
        test = self.opt['val'].get('test_flag', False)
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        metric_data = dict()
        metric_data_pytorch = dict()

        pbar = tqdm(total=len(dataloader), unit='item')
        if self.opt['val'].get('split_log', False):
            self.split_results = {}
            self.split_results['LOL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            gt_img = tensor2img([visuals['gt']], min_max=(0, 1))
            lq_img = tensor2img([visuals['lq']], min_max=(0, 1))
            sr_img = tensor2img([visuals['diff_sr']], min_max=(0, 1))

            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img
            metric_data_pytorch['img'] = sr_img
            metric_data_pytorch['img2'] = gt_img

            if self.opt['rank'] == 0 and save_img:
                # 保存结果：若有分解可视化则额外保存，否则只存 SR
                vis_dir = osp.join(self.opt['path']['visualization'], img_name)
                os.makedirs(vis_dir, exist_ok=True)

                if not self.opt['val'].get('save_result_only', False):
                    save_img_path = osp.join(vis_dir, f'{img_name}_{current_iter}.png')
                    imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)

                    # 仅当存在 Retinex 产物时才保存
                    if 'l_list' in visuals and 'r_list' in visuals:
                        l_list_img = tensor2img([visuals['l_list']], min_max=(0, 1))
                        r_list_img = tensor2img([visuals['r_list']], min_max=(0, 1))
                        save_img_retinex_path = osp.join(vis_dir, f'{img_name}_{current_iter}_retinex.png')
                        imwrite(np.concatenate([l_list_img, r_list_img], axis=1), save_img_retinex_path)
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], f'{img_name}.png')
                    imwrite(sr_img, save_img_path)

            if with_metrics:
                # 计算指标
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'pytorch' in opt_['type']:
                        self.metric_results[name] += calculate_metric(metric_data_pytorch, opt_).item()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            # 释放
            del self.lq, self.gt
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            if test is not True:
                # 仅依据 psnr 选择最优（与训练保持一致的主观指标）
                for metric, value in self.metric_results.items():
                    if metric == "psnr" and value > getattr(self, 'best_val_psnr', 0.0):
                        self.best_val_psnr = value
                        self.save(epoch=0, current_iter=0, name="best_val_psnr")

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
            logger = get_root_logger()
            log_str = f'Validation {dataset_name}\n'
            for metric, value in self.metric_results.items():
                log_str += f'\t # {metric}: {value:.4f}\n'
            logger.info(log_str)
            
            if tb_logger:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['lq'] = self.lq[:, :3, :, :].detach().cpu()
        out_dict['diff_sr'] = self.diff_sr.detach().cpu()

        # 仅当存在时返回 l_list / r_list（兼容老分解分支）
        if getattr(self, 'supervised_l_list', None):
            resized_l = [
                torch.nn.functional.interpolate(
                    torch.cat((l.detach().cpu(), l.detach().cpu(), l.detach().cpu()), dim=1),
                    size=self.gt.shape[2:], mode='bilinear', align_corners=False
                ) for l in self.supervised_l_list
            ]
            out_dict['l_list'] = torch.cat(resized_l, dim=3).detach().cpu()

        if getattr(self, 'supervised_r_list', None):
            resized_r = [
                torch.nn.functional.interpolate(
                    r.detach().cpu(),
                    size=self.gt.shape[2:], mode='bilinear', align_corners=False
                ) for r in self.supervised_r_list
            ]
            out_dict['r_list'] = torch.cat(resized_r, dim=3).detach().cpu()

        return out_dict

    
    def save(self, epoch, current_iter, name=None):
        if name is None:
            self.save_network([self.ddpm], 'net_g_ddpm', current_iter, param_key=['params'])
        else:
            self.save_network([self.ddpm], 'net_g_best_ddpm_val', current_iter, param_key=['params']) 