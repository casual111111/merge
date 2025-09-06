import os
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import torch
import glob
import re

class Data(data.Dataset):
    def __init__(self, root, args, train=False):
        self.args = args
        self.train = train

        self.imgs_HR_path = os.path.join(root, 'NLHR/X4')
        self.imgs_LR_path = os.path.join(root, 'LLLR')
        
        # 获取所有高分辨率和低分辨率图像路径
        self.imgs_HR = sorted(glob.glob(os.path.join(self.imgs_HR_path, '*.png')))
        self.imgs_LR = sorted(glob.glob(os.path.join(self.imgs_LR_path, '*.png')))

        print("HR images path:", self.imgs_HR_path)
        print("LR images path:", self.imgs_LR_path)

        self.transform = transforms.ToTensor()

        if self.args.dir_data == "/root/autodl-tmp/dataset/LLLR/RELLISUR/RELLISUR_256":
            # 根据是否是测试阶段加载不同的数据
            if self.train:
                self.train_data = self._create_train_pairs()
            else:
                self.test_data = self._create_test_pairs()
        else:
            # 处理其他数据集路径的情况
            self.train = train
        


    def _create_train_pairs(self):
        """生成高分辨率和低分辨率的配对数据（训练阶段）"""
        train_pairs = []

        # 遍历每张高分辨率图片
        for hr_img in self.imgs_HR:
            hr_img_name = os.path.splitext(os.path.basename(hr_img))[0]
            
            # 使用正则表达式匹配低分辨率图片
            pattern = re.compile(f"^{hr_img_name}-(\d+\.\d+)\.png$")
            matching_lr_imgs = [img for img in self.imgs_LR if pattern.match(os.path.basename(img))]

            if len(matching_lr_imgs) == 5:
                matching_lr_imgs = sorted(matching_lr_imgs, key=lambda x: float(re.search(r'(\d+\.\d+)', os.path.basename(x)).group(0)))
                
                # 将每张 HR 图片添加到训练列表中五次
                for lr_img in matching_lr_imgs:
                    train_pairs.append((hr_img, lr_img))
            else:
                print(f"Warning: Missing or incorrect number of LR images for {hr_img_name}")

        return train_pairs

    def _create_test_pairs(self):
        """生成高分辨率和低分辨率的配对数据（测试阶段）"""
        test_pairs = []

        # 遍历每张高分辨率图片
        for hr_img in self.imgs_HR:
            hr_img_name = os.path.splitext(os.path.basename(hr_img))[0]
            
            # 使用正则表达式匹配低分辨率图片
            pattern = re.compile(f"^{hr_img_name}-(\d+\.\d+)\.png$")
            matching_lr_imgs = [img for img in self.imgs_LR if pattern.match(os.path.basename(img))]

            # 确保找到了五张低分辨率图片
            if len(matching_lr_imgs) == 5:
                matching_lr_imgs = sorted(matching_lr_imgs, key=lambda x: float(re.search(r'(\d+\.\d+)', os.path.basename(x)).group(0)))

                # 将每张 HR 图片添加到测试列表中五次
                for lr_img in matching_lr_imgs:
                    test_pairs.append((hr_img, lr_img))
            else:
                print(f"Warning: Missing or incorrect number of LR images for {hr_img_name}")

        return test_pairs

    def __getitem__(self, item):
        """从数据集中获取一对 HR 和 LR 图像"""
        if self.args.dir_data == "/root/autodl-tmp/dataset/LLLR/RELLISUR/RELLISUR_256":
            if self.train:
                hr_img, lr_img = self.train_data[item]
            else:
                hr_img, lr_img = self.test_data[item]
            
            # 打开 HR 和 LR 图像
            HR = Image.open(hr_img)
            LR = Image.open(lr_img)

            HR = np.array(HR)
            LR = np.array(LR)

            LR = np.ascontiguousarray(LR)
            HR = np.ascontiguousarray(HR)

            HR = ToTensor()(HR)
            LR = ToTensor()(LR)

            filename = os.path.basename(hr_img)

            lr_filename = os.path.basename(lr_img)  # 获取低分辨率图像的文件名
            match = re.search(r'-(\d+\.\d+)', lr_filename)  # 提取后缀（如-2.5）

            if match:
                suffix = match.group(0)  # 获取后缀（例如：-2.5）
            else:
                suffix = "unknown"  # 如果没有找到，使用默认值

            suffix = suffix.strip()  # 去除可能的空格或换行符

            return LR, HR, filename, suffix  # 返回后缀
        else:
            # 处理其他数据集路径的加载逻辑
            img_path_HR = os.path.join(self.imgs_HR_path, self.imgs_HR[item])
            img_path_LR = os.path.join(self.imgs_LR_path, os.path.basename(img_path_HR))
            LR = Image.open(img_path_LR)
            HR = Image.open(img_path_HR)

            HR = np.array(HR)
            LR = np.array(LR)

            LR = np.ascontiguousarray(LR)
            HR = np.ascontiguousarray(HR)

            HR = ToTensor()(HR)
            LR = ToTensor()(LR)

            filename = os.path.basename(img_path_HR)

            return LR, HR, filename

    def __len__(self):
        if self.args.dir_data == "/root/autodl-tmp/dataset/LLLR/RELLISUR/RELLISUR_256":
            """返回数据集的大小"""
            if self.train:
                return len(self.train_data)
            else:
                return len(self.test_data)
        else:
             return len(self.imgs_HR)  
